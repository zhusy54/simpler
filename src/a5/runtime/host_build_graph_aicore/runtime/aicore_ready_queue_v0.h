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

#include "aicore_gm_atomic.h"

inline __aicore__ bool
aicore_ready_queue_push_v0(__gm__ void *sidecar_base, __gm__ AicoreReadyQueueV0 *queue, int64_t task_id) {
    uint64_t pos = aicore_gm_load_v0(queue->enqueue_pos, __ATOMIC_RELAXED);
    while (true) {
        __gm__ AicoreReadyQueueSlotV0 *slots =
            aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(sidecar_base, queue->slots_offset);
        __gm__ AicoreReadyQueueSlotV0 *slot = &slots[pos & queue->mask];
        int64_t sequence = aicore_gm_load_v0(slot->sequence, __ATOMIC_ACQUIRE);
        int64_t diff = sequence - static_cast<int64_t>(pos);
        if (diff == 0) {
            uint64_t observed =
                aicore_gm_compare_exchange_v0(queue->enqueue_pos, pos, pos + 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
            if (observed == pos) {
                aicore_gm_store_v0(slot->task_id, task_id, __ATOMIC_RELAXED);
                aicore_gm_store_v0(slot->sequence, static_cast<int64_t>(pos + 1), __ATOMIC_RELEASE);
                return true;
            }
            pos = observed;
        } else if (diff < 0) {
            return false;
        } else {
            pos = aicore_gm_load_v0(queue->enqueue_pos, __ATOMIC_RELAXED);
        }
    }
}

inline __aicore__ bool
aicore_ready_queue_pop_v0(__gm__ void *sidecar_base, __gm__ AicoreReadyQueueV0 *queue, int64_t *task_id) {
    if (task_id == nullptr) return false;
    uint64_t pos = aicore_gm_load_v0(queue->dequeue_pos, __ATOMIC_RELAXED);
    while (true) {
        __gm__ AicoreReadyQueueSlotV0 *slots =
            aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(sidecar_base, queue->slots_offset);
        __gm__ AicoreReadyQueueSlotV0 *slot = &slots[pos & queue->mask];
        int64_t sequence = aicore_gm_load_v0(slot->sequence, __ATOMIC_ACQUIRE);
        int64_t diff = sequence - static_cast<int64_t>(pos + 1);
        if (diff == 0) {
            uint64_t observed =
                aicore_gm_compare_exchange_v0(queue->dequeue_pos, pos, pos + 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
            if (observed == pos) {
                *task_id = aicore_gm_load_v0(slot->task_id, __ATOMIC_ACQUIRE);
                aicore_gm_store_v0(slot->sequence, static_cast<int64_t>(pos + queue->capacity), __ATOMIC_RELEASE);
                return true;
            }
            pos = observed;
        } else if (diff < 0) {
            return false;
        } else {
            pos = aicore_gm_load_v0(queue->dequeue_pos, __ATOMIC_RELAXED);
        }
    }
}
