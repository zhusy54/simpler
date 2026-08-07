/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "intrinsic.h"
#include "../../../../../../../src/a5/runtime/host_build_graph_aicore/runtime/aicore_ready_queue_v0.h"

constexpr uint64_t PROBE_BYTES = 256 * 1024;
constexpr int32_t PROBE_STRESS_ITERS = 1024;

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ uint8_t *raw = reinterpret_cast<__gm__ uint8_t *>(tensor->buffer.addr) + tensor->start_offset;
    const uint64_t raw_address = reinterpret_cast<uint64_t>(raw);
    const uint64_t sidecar_address = (raw_address + 255) & ~UINT64_C(127);
    __gm__ void *sidecar = reinterpret_cast<__gm__ void *>(sidecar_address);
    const int32_t block_idx = get_block_idx(args);
    const int32_t block_num = get_block_num(args);

    AicoreExecutionSidecarLayoutV0 layout{};
    if (!aicore_sidecar_plan_v0(block_num, block_num, 0, &layout)) return;
    const uint64_t results_offset = layout.total_size;
    if (sidecar_address - raw_address + results_offset + static_cast<uint64_t>(block_num) * sizeof(int64_t) >
        PROBE_BYTES) {
        return;
    }

    __gm__ AicoreRunControlV0 *control = aicore_sidecar_at_v0<AicoreRunControlV0>(sidecar, layout.run_control_offset);
    __gm__ AicoreTaskControlV0 *tasks = aicore_sidecar_at_v0<AicoreTaskControlV0>(sidecar, layout.task_controls_offset);
    __gm__ AicoreReadyQueueV0 *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(sidecar, layout.aic_queue_offset);
    __gm__ AicoreReadyQueueSlotV0 *slots =
        aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(sidecar, layout.aic_queue_slots_offset);
    __gm__ int64_t *results = aicore_sidecar_at_v0<int64_t>(sidecar, results_offset);

    if (block_idx == 0) {
        __gm__ uint64_t *metadata = reinterpret_cast<__gm__ uint64_t *>(raw);
        metadata[0] = sidecar_address - raw_address;
        metadata[1] = layout.task_controls_offset;
        metadata[2] = layout.aic_queue_offset;
        metadata[3] = results_offset;
        metadata[4] = static_cast<uint64_t>(block_num);
        for (int32_t i = 0; i < block_num; ++i) {
            tasks[i].completion = 0;
            tasks[i].wake_list_head = AICORE_TASK_ID_INVALID_V0;
            tasks[i].next_waiter = AICORE_TASK_ID_INVALID_V0;
        }
        queue->slots_offset = layout.aic_queue_slots_offset;
        queue->capacity = layout.aic_queue_capacity;
        queue->mask = layout.aic_queue_capacity - 1;
        queue->enqueue_pos = 0;
        queue->dequeue_pos = 0;
        for (uint64_t i = 0; i < layout.aic_queue_capacity; ++i) {
            slots[i].sequence = static_cast<int64_t>(i);
            slots[i].task_id = AICORE_TASK_ID_INVALID_V0;
        }
#if defined(__CCE_AICORE__)
        OUT_OF_ORDER_STORE_BARRIER();
#else
        __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
        dcci(raw, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        aicore_gm_store_v0(control->startup_count, UINT64_C(1));
    } else {
        while (aicore_gm_load_v0(control->startup_count) == 0) {}
    }

    for (int32_t i = 0; i < PROBE_STRESS_ITERS; ++i) {
        aicore_gm_store_v0(tasks[block_idx].completion, static_cast<int64_t>(i));
        (void)aicore_gm_exchange_v0(tasks[block_idx].wake_list_head, static_cast<int64_t>(i));
        aicore_publish_next_waiter_v0(&tasks[block_idx], static_cast<int64_t>(i));
    }
    aicore_gm_store_v0(tasks[block_idx].completion, INT64_C(0x10000) + block_idx);
    (void)aicore_gm_exchange_v0(tasks[block_idx].wake_list_head, INT64_C(0x20000) + block_idx);
    aicore_publish_next_waiter_v0(&tasks[block_idx], INT64_C(0x30000) + block_idx);

    while (!aicore_ready_queue_push_v0(sidecar, queue, block_idx)) {}
    aicore_gm_fetch_add_v0(control->queue_push_count, UINT64_C(1));
    while (aicore_gm_load_v0(control->queue_push_count) != static_cast<uint64_t>(block_num)) {}

    int64_t task_id = AICORE_TASK_ID_INVALID_V0;
    while (!aicore_ready_queue_pop_v0(sidecar, queue, &task_id)) {}
    aicore_gm_store_v0(results[block_idx], task_id);
    aicore_gm_fetch_add_v0(control->finished_count, UINT64_C(1));
}
