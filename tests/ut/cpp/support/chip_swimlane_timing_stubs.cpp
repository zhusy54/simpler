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

#include "common/task_transition_timing.h"

enum class ChipSwimlaneLevel : uint32_t;

ChipSwimlaneLevel get_chip_swimlane_level() { return static_cast<ChipSwimlaneLevel>(0); }
void chip_swimlane_aicpu_gate(int, uint32_t) {}
void chip_swimlane_aicpu_dependency_peak(uint32_t, uint64_t) {}
void chip_swimlane_aicpu_wait(SchedulerWaitReason, uint64_t, uint64_t) {}
