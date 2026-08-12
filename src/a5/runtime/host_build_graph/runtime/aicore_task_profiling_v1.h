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
#include "aicore_execution_sidecar_v1.h"

struct AicoreTaskProfilingStateV1 {
    __gm__ ChipSwimlaneActiveHead *head;
    ChipSwimlaneAicoreLocalState local;
    bool enabled;
};

inline __host__ __aicore__ void
aicore_task_profiling_init_v1(AicoreTaskProfilingStateV1 *state, bool enabled, __gm__ ChipSwimlaneActiveHead *head) {
    if (state == nullptr) return;
    state->head = head;
    state->local = {nullptr, UINT32_MAX, 0};
    state->enabled = enabled && head != nullptr;
}

inline __host__ __aicore__ __gm__ ChipSwimlaneAicoreTaskRecord *
aicore_task_profiling_reserve_v1(AicoreTaskProfilingStateV1 *state) {
    if (state == nullptr || !state->enabled) return nullptr;
    return chip_swimlane_aicore_reserve_task_record(state->head, &state->local);
}

inline __host__ __aicore__ void aicore_task_profiling_commit_v1(
    __gm__ ChipSwimlaneAicoreTaskRecord *record, uint64_t task_id, uint64_t receive_time, uint64_t start_time,
    uint64_t end_time
) {
    if (record == nullptr) return;
    chip_swimlane_aicore_commit_task_record(
        record, task_id, static_cast<uint32_t>(task_id), receive_time, start_time, end_time
    );
}
