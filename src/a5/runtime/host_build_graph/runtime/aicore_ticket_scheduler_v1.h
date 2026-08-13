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

#pragma once

#include "aicore_execution_sidecar_v1.h"
#include "aicore_gm_atomic.h"
#include "aicore_graph_view_v0.h"

enum class AicorePendingStateV1 : uint64_t {
    EMPTY = 0,
    BLOCKED = 1,
    READY = 2,
    ERROR = 3,
};

struct AicorePendingSlotV1 {
    int64_t task_id;
    int32_t fanin_count;
    int32_t next_fanin_index;
    int32_t waiting_producer;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t reserved;
    uint64_t stream_index;
    uint64_t claim_start_cycles;
    uint64_t claim_end_cycles;
    uint64_t pending_wait_start_cycles;
    AicoreClaimKindV1 claim_kind;
};

static_assert(sizeof(AicorePendingSlotV1) == 64, "pending slot layout changed");

inline __host__ __aicore__ void aicore_pending_clear_v1(AicorePendingSlotV1 *slot) {
    if (slot == nullptr) return;
    *slot = {
        AICORE_TASK_ID_INVALID_V1,
        0,
        0,
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        UINT16_MAX,
        UINT8_MAX,
        0,
        0,
        0,
        0,
        0,
        AicoreClaimKindV1::SEED,
    };
}

inline __aicore__ __gm__ AicoreTaskCompletionCellV1 *
aicore_completion_cell_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, int64_t task_id) {
    return aicore_sidecar_at_v1<AicoreTaskCompletionCellV1>(
        sidecar_base,
        context->completion_cells_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskCompletionCellV1)
    );
}

inline __aicore__ void aicore_record_scheduler_error_v1(
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreRootStatusV0 status,
    const AicoreReadonlyGraphV0 *graph = nullptr, __gm__ const AicoreWorkerContextV1 *context = nullptr
) {
    if (aicore_gm_compare_exchange_v0(run_control->error_claimed, UINT64_C(0), UINT64_C(1)) != 0) return;
    aicore_gm_store_v0(run_control->error_task_id, static_cast<uint64_t>(task_id));
    if (graph != nullptr) {
        aicore_gm_store_v0(run_control->error_graph_task_count, graph->task_count);
        aicore_gm_store_v0(run_control->error_descriptors_address, graph->descriptors_address);
        aicore_gm_store_v0(run_control->error_payloads_address, graph->payloads_address);
        aicore_gm_store_v0(run_control->error_task_window_mask, graph->task_window_mask);
    }
    if (context != nullptr) {
        aicore_gm_store_v0(run_control->error_core_id, static_cast<uint64_t>(context->physical_core_id));
        aicore_gm_store_v0(run_control->error_core_type, static_cast<uint64_t>(context->core_type));
    }
    aicore_gm_publish_v0(run_control->scheduler_error, static_cast<uint64_t>(status));
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_pending_initialize_v1(
    const AicoreReadonlyGraphV0 &graph, int64_t task_id, AicoreRootCoreTypeV0 expected_core_type, int32_t max_func_id,
    uint64_t stream_index, AicoreClaimKindV1 claim_kind, uint64_t claim_start_cycles, uint64_t claim_end_cycles,
    AicorePendingSlotV1 *slot
) {
    if (slot == nullptr) return AicoreRootStatusV0::INVALID_ARGUMENTS;
    AicoreTaskInfoV0 task{};
    AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
    if (status != AicoreRootStatusV0::OK) return status;
    if (task.core_type != expected_core_type || task.kernel_id >= max_func_id ||
        task.kernel_id > static_cast<int32_t>(UINT16_MAX)) {
        return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    *slot = {
        task_id,
        *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0),
        0,
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        static_cast<uint16_t>(task.kernel_id),
        static_cast<uint8_t>(task.subtask_slot),
        0,
        stream_index,
        claim_start_cycles,
        claim_end_cycles,
        0,
        claim_kind,
    };
    return AicoreRootStatusV0::OK;
}

inline __aicore__ AicorePendingStateV1 aicore_pending_readiness_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context,
    AicorePendingSlotV1 *slot, uint64_t *fanin_load_count
) {
    if (slot == nullptr || slot->task_id < 0) return AicorePendingStateV1::EMPTY;
    while (slot->next_fanin_index < slot->fanin_count) {
        int32_t producer = aicore_graph_fanin_id_v0(graph, slot->task_id, slot->next_fanin_index);
        if (producer < 0 || producer >= slot->task_id) return AicorePendingStateV1::ERROR;
        if (fanin_load_count != nullptr) ++*fanin_load_count;
        __gm__ AicoreTaskCompletionCellV1 *completion = aicore_completion_cell_at_v1(sidecar_base, context, producer);
        if (aicore_gm_load_v0(completion->completion) != static_cast<int64_t>(AicoreTaskCompletionV1::DONE)) {
            slot->waiting_producer = producer;
            return AicorePendingStateV1::BLOCKED;
        }
        ++slot->next_fanin_index;
    }
    slot->waiting_producer = static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1);
    return AicorePendingStateV1::READY;
}

inline __aicore__ bool aicore_claim_ticket_v1(
    __gm__ void *sidecar_base, __gm__ AicoreTaskStreamV1 *stream, uint64_t *stream_index, int64_t *task_id
) {
    if (stream_index == nullptr || task_id == nullptr) return false;
    uint64_t index = aicore_gm_fetch_add_v0(stream->next_index, UINT64_C(1));
    *stream_index = index;
    if (index >= stream->task_count) {
        *task_id = AICORE_TASK_ID_INVALID_V1;
        return false;
    }
    __gm__ uint32_t *task_ids = aicore_sidecar_at_v1<uint32_t>(sidecar_base, stream->task_ids_offset);
    *task_id = static_cast<int64_t>(task_ids[index]);
    return true;
}

inline __aicore__ void aicore_publish_task_completion_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, int64_t task_id
) {
    __gm__ AicoreTaskCompletionCellV1 *completion = aicore_completion_cell_at_v1(sidecar_base, context, task_id);
    (void)aicore_gm_exchange_v0(completion->completion, static_cast<int64_t>(AicoreTaskCompletionV1::DONE));
}
