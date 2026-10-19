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

#include <cstdint>
#include <type_traits>

struct TaskTransitionTiming {
    uint64_t ready_cycles{0};
    uint64_t registered_cycles{0};
};

static_assert(sizeof(TaskTransitionTiming) == 16);
static_assert(std::is_trivially_copyable_v<TaskTransitionTiming> && std::is_standard_layout_v<TaskTransitionTiming>);

enum class SchedulerWaitReason : uint32_t { TaskWindow = 0, Heap = 1, DependencyPool = 2, FaninSpill = 3 };

struct ChipSwimlaneRuntimeStats {
    uint64_t run_epoch{0};
    uint64_t present{0};
    uint64_t wait_cycles[4]{};
    uint64_t wait_count[4]{};
    uint64_t max_wait_cycles[4]{};
    uint64_t dependency_peak[32]{};
};

static_assert(sizeof(ChipSwimlaneRuntimeStats) == 368);
static_assert(
    std::is_trivially_copyable_v<ChipSwimlaneRuntimeStats> && std::is_standard_layout_v<ChipSwimlaneRuntimeStats>
);
