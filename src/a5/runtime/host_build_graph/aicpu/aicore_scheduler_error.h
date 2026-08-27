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

#include <atomic>
#include <cstdint>

#include "host_build_graph/runtime_status.h"

// This bridge is intentionally AICPU/host-only. AICore code publishes its
// detailed first error through SchedulerRunControl; AICPU maps that internal
// diagnostic to the existing host runtime status ABI without making this
// std::atomic helper visible to either AICore compiler.
inline int32_t aicore_scheduler_runtime_error_code(uint64_t scheduler_error) {
    return scheduler_error == 0 ? SIMPLER_ERROR_NONE : SIMPLER_ERROR_INVALID_ARGS;
}

inline bool latch_aicore_scheduler_runtime_error(std::atomic<int32_t> *sched_error_code, uint64_t scheduler_error) {
    if (sched_error_code == nullptr || scheduler_error == 0) return false;
    int32_t expected = SIMPLER_ERROR_NONE;
    return sched_error_code->compare_exchange_strong(
        expected, aicore_scheduler_runtime_error_code(scheduler_error), std::memory_order_acq_rel,
        std::memory_order_acquire
    );
}
