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

#include <stddef.h>
#include <stdint.h>

#include "pto2_dispatch_payload.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

inline constexpr uint64_t AICORE_SIDECAR_ALIGNMENT_V0 = 128;
inline constexpr uint64_t AICORE_WORKER_CAPACITY_V0 = 108;
inline constexpr int64_t AICORE_TASK_ID_INVALID_V0 = -1;
inline constexpr int64_t AICORE_WAKE_LIST_CLOSED_V0 = -2;

struct alignas(128) AicoreTaskControlV0 {
    // Cross-core RMW and publication words occupy the first cache line.
    int64_t completion;
    int64_t wake_list_head;
    uint8_t atomic_line_padding[48];

    // This single-writer field is published with DCCI. Keeping it on the
    // second line prevents that DCCI from writing back a stale atomic word.
    int64_t next_waiter;
    uint8_t publish_line_padding[56];
};

struct alignas(16) AicoreReadyQueueSlotV0 {
    int64_t sequence;
    int64_t task_id;
};

struct alignas(128) AicoreReadyQueueV0 {
    uint64_t slots_offset;
    uint64_t capacity;
    uint64_t mask;
    uint8_t metadata_padding[104];

    uint64_t enqueue_pos;
    uint8_t enqueue_padding[120];

    uint64_t dequeue_pos;
    uint8_t dequeue_padding[120];
};

struct alignas(128) AicoreRunControlV0 {
    uint64_t attached_count;
    uint64_t classified_count;
    uint64_t active_aic_count;
    uint64_t active_aiv_count;
    uint64_t expected_task_count;
    uint64_t completed_count;
    uint64_t exit_requested;
    uint64_t finished_count;
    uint64_t startup_count;
    uint64_t queue_push_count;
    uint64_t queue_pop_count;
    uint64_t queue_miss_count;
    uint8_t padding[32];
};

struct alignas(128) AicoreWorkerContextV0 {
    int32_t core_type;
    int32_t physical_core_id;
    int32_t type_rank;
    int32_t active;
    uint64_t run_control_offset;
    uint64_t task_controls_offset;
    uint64_t aic_queue_offset;
    uint64_t aiv_queue_offset;
    uint64_t readonly_graph_address;
    uint64_t sidecar_base_address;
    uint64_t dispatch_payload_offset;
    uint64_t local_completed_delta;
    uint64_t poll_count;
    uint64_t task_count;
    uint8_t padding[32];
};

struct AicoreExecutionSidecarLayoutV0 {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t aic_queue_capacity;
    uint64_t aiv_queue_capacity;
    uint64_t run_control_offset;
    uint64_t worker_contexts_offset;
    uint64_t dispatch_payloads_offset;
    uint64_t task_controls_offset;
    uint64_t aic_queue_offset;
    uint64_t aic_queue_slots_offset;
    uint64_t aiv_queue_offset;
    uint64_t aiv_queue_slots_offset;
};

static_assert(sizeof(AicoreTaskControlV0) == 128, "AICore task control must be exactly 128 bytes");
static_assert(alignof(AicoreTaskControlV0) == 128, "AICore task control must be 128-byte aligned");
static_assert(offsetof(AicoreTaskControlV0, completion) == 0, "completion offset changed");
static_assert(offsetof(AicoreTaskControlV0, wake_list_head) == 8, "wake head offset changed");
static_assert(offsetof(AicoreTaskControlV0, next_waiter) == 64, "next waiter must occupy a separate cache line");
static_assert(sizeof(AicoreReadyQueueSlotV0) == 16, "task-id queue slot layout changed");
static_assert(sizeof(AicoreReadyQueueV0) == 384, "task-id queue header layout changed");
static_assert(offsetof(AicoreReadyQueueV0, enqueue_pos) == 128, "enqueue cursor must have its own line");
static_assert(offsetof(AicoreReadyQueueV0, dequeue_pos) == 256, "dequeue cursor must have its own line");
static_assert(sizeof(AicoreRunControlV0) == 128, "run control layout changed");
static_assert(sizeof(AicoreWorkerContextV0) == 128, "worker context layout changed");

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<AicoreTaskControlV0>);
static_assert(std::is_trivially_copyable_v<AicoreTaskControlV0>);
static_assert(std::is_standard_layout_v<AicoreReadyQueueV0>);
static_assert(std::is_trivially_copyable_v<AicoreReadyQueueV0>);
static_assert(std::is_standard_layout_v<AicoreRunControlV0>);
static_assert(std::is_trivially_copyable_v<AicoreRunControlV0>);
static_assert(std::is_standard_layout_v<AicoreWorkerContextV0>);
static_assert(std::is_trivially_copyable_v<AicoreWorkerContextV0>);
#endif

inline bool aicore_sidecar_checked_add_v0(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool aicore_sidecar_checked_mul_v0(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool aicore_sidecar_checked_align_v0(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!aicore_sidecar_checked_add_v0(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool aicore_sidecar_next_power_of_two_v0(uint64_t value, uint64_t *out) {
    if (out == nullptr) return false;
    if (value <= 1) {
        *out = 1;
        return true;
    }
    if (value > (UINT64_C(1) << 63)) return false;
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    *out = value + 1;
    return true;
}

inline bool aicore_sidecar_reserve_v0(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !aicore_sidecar_checked_align_v0(*cursor, alignment, &aligned)) {
        return false;
    }
    uint64_t end = 0;
    if (!aicore_sidecar_checked_add_v0(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool aicore_sidecar_plan_v0(
    uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, AicoreExecutionSidecarLayoutV0 *layout
) {
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count ||
        aic_task_count > task_count - aiv_task_count) {
        return false;
    }

    AicoreExecutionSidecarLayoutV0 next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    if (!aicore_sidecar_next_power_of_two_v0(aic_task_count, &next.aic_queue_capacity) ||
        !aicore_sidecar_next_power_of_two_v0(aiv_task_count, &next.aiv_queue_capacity)) {
        return false;
    }

    uint64_t cursor = 0;
    uint64_t bytes = 0;
    if (!aicore_sidecar_reserve_v0(
            &cursor, sizeof(AicoreRunControlV0), alignof(AicoreRunControlV0), &next.run_control_offset
        ) ||
        !aicore_sidecar_checked_mul_v0(AICORE_WORKER_CAPACITY_V0, sizeof(AicoreWorkerContextV0), &bytes) ||
        !aicore_sidecar_reserve_v0(&cursor, bytes, alignof(AicoreWorkerContextV0), &next.worker_contexts_offset) ||
        !aicore_sidecar_checked_mul_v0(AICORE_WORKER_CAPACITY_V0, sizeof(PTO2DispatchPayload), &bytes) ||
        !aicore_sidecar_reserve_v0(&cursor, bytes, alignof(PTO2DispatchPayload), &next.dispatch_payloads_offset) ||
        !aicore_sidecar_checked_mul_v0(task_count, sizeof(AicoreTaskControlV0), &bytes) ||
        !aicore_sidecar_reserve_v0(&cursor, bytes, alignof(AicoreTaskControlV0), &next.task_controls_offset) ||
        !aicore_sidecar_reserve_v0(
            &cursor, sizeof(AicoreReadyQueueV0), alignof(AicoreReadyQueueV0), &next.aic_queue_offset
        ) ||
        !aicore_sidecar_checked_mul_v0(next.aic_queue_capacity, sizeof(AicoreReadyQueueSlotV0), &bytes) ||
        !aicore_sidecar_reserve_v0(&cursor, bytes, alignof(AicoreReadyQueueSlotV0), &next.aic_queue_slots_offset) ||
        !aicore_sidecar_reserve_v0(
            &cursor, sizeof(AicoreReadyQueueV0), alignof(AicoreReadyQueueV0), &next.aiv_queue_offset
        ) ||
        !aicore_sidecar_checked_mul_v0(next.aiv_queue_capacity, sizeof(AicoreReadyQueueSlotV0), &bytes) ||
        !aicore_sidecar_reserve_v0(&cursor, bytes, alignof(AicoreReadyQueueSlotV0), &next.aiv_queue_slots_offset) ||
        !aicore_sidecar_checked_align_v0(cursor, AICORE_SIDECAR_ALIGNMENT_V0, &next.total_size)) {
        return false;
    }

    *layout = next;
    return true;
}

template <typename T>
inline __gm__ T *aicore_sidecar_at_v0(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

inline bool aicore_sidecar_init_v0(void *base, const AicoreExecutionSidecarLayoutV0 &layout) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (AICORE_SIDECAR_ALIGNMENT_V0 - 1)) != 0) {
        return false;
    }
    __builtin_memset(base, 0, static_cast<size_t>(layout.total_size));

    AicoreTaskControlV0 *controls = aicore_sidecar_at_v0<AicoreTaskControlV0>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].completion = 0;
        controls[i].wake_list_head = AICORE_TASK_ID_INVALID_V0;
        controls[i].next_waiter = AICORE_TASK_ID_INVALID_V0;
    }

    AicoreReadyQueueV0 *aic_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(base, layout.aic_queue_offset);
    AicoreReadyQueueV0 *aiv_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(base, layout.aiv_queue_offset);
    AicoreReadyQueueSlotV0 *aic_slots =
        aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(base, layout.aic_queue_slots_offset);
    AicoreReadyQueueSlotV0 *aiv_slots =
        aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(base, layout.aiv_queue_slots_offset);

    aic_queue->slots_offset = layout.aic_queue_slots_offset;
    aic_queue->capacity = layout.aic_queue_capacity;
    aic_queue->mask = layout.aic_queue_capacity - 1;
    aiv_queue->slots_offset = layout.aiv_queue_slots_offset;
    aiv_queue->capacity = layout.aiv_queue_capacity;
    aiv_queue->mask = layout.aiv_queue_capacity - 1;
    for (uint64_t i = 0; i < layout.aic_queue_capacity; ++i) {
        aic_slots[i].sequence = static_cast<int64_t>(i);
        aic_slots[i].task_id = AICORE_TASK_ID_INVALID_V0;
    }
    for (uint64_t i = 0; i < layout.aiv_queue_capacity; ++i) {
        aiv_slots[i].sequence = static_cast<int64_t>(i);
        aiv_slots[i].task_id = AICORE_TASK_ID_INVALID_V0;
    }
    return true;
}
