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

#include <stdint.h>

#include "aicore/chip_swimlane_collector_aicore.h"
#include "aicore_execution_sidecar_v0.h"
#include "aicore_gm_atomic.h"

struct AicoreTaskProfilingStateV0 {
    __gm__ ChipSwimlaneActiveHead *head;
    ChipSwimlaneAicoreLocalState local;
    bool enabled;
};

inline __host__ __aicore__ void
aicore_task_profiling_init_v0(AicoreTaskProfilingStateV0 *state, bool enabled, __gm__ ChipSwimlaneActiveHead *head) {
    if (state == nullptr) return;
    state->head = head;
    state->local = {nullptr, UINT32_MAX, 0};
    state->enabled = enabled && head != nullptr;
}

inline __host__ __aicore__ __gm__ ChipSwimlaneAicoreTaskRecord *
aicore_task_profiling_reserve_v0(AicoreTaskProfilingStateV0 *state, __gm__ AicoreWorkerContextV0 *worker) {
    if (state == nullptr || worker == nullptr || !state->enabled) return nullptr;
    aicore_gm_fetch_add_v0(worker->dfx_reserved[AICORE_PROFILE_ATTEMPTED_INDEX_V0], UINT64_C(1));
    __gm__ ChipSwimlaneAicoreTaskRecord *record = chip_swimlane_aicore_reserve_task_record(state->head, &state->local);
    if (record == nullptr) {
        aicore_gm_fetch_add_v0(worker->dfx_reserved[AICORE_PROFILE_DROPPED_INDEX_V0], UINT64_C(1));
    }
    return record;
}

inline __host__ __aicore__ void aicore_task_profiling_commit_v0(
    __gm__ ChipSwimlaneAicoreTaskRecord *record, __gm__ AicoreWorkerContextV0 *worker, uint64_t task_id,
    uint64_t receive_time, uint64_t start_time, uint64_t end_time
) {
    if (record == nullptr || worker == nullptr) return;
    chip_swimlane_aicore_commit_task_record(
        record, task_id, static_cast<uint32_t>(task_id), receive_time, start_time, end_time
    );
    aicore_gm_fetch_add_v0(worker->dfx_reserved[AICORE_PROFILE_WRITTEN_INDEX_V0], UINT64_C(1));
}
