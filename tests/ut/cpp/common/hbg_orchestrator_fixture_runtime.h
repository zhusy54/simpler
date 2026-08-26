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
#include <utility>

#include "utils/device_arena.h"

// A2A3's host orchestrator still owns a simulated SchedulerState. A5 builds the
// whole graph for the AICore scheduler and therefore needs no host scheduler.
// Keep shared orchestrator tests source-identical by deriving the old scheduler
// type from the orchestrator only when that member exists.
template <typename Orchestrator, typename = void>
struct HbgOrchestratorHasHostScheduler : std::false_type {};

template <typename Orchestrator>
struct HbgOrchestratorHasHostScheduler<
    Orchestrator,
    std::void_t<decltype(std::declval<Orchestrator &>().scheduler)>> : std::true_type {};

template <typename Orchestrator, bool HasHostScheduler = HbgOrchestratorHasHostScheduler<Orchestrator>::value>
class HbgOrchestratorFixtureRuntime;

template <typename Orchestrator>
class HbgOrchestratorFixtureRuntime<Orchestrator, false> {
public:
    bool init(Orchestrator &orch, void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks) {
        return orch.init(sm_base, gm_heap, heap_size, max_tasks);
    }

    void destroy() {}
};

template <typename Orchestrator>
class HbgOrchestratorFixtureRuntime<Orchestrator, true> {
    using Scheduler = std::remove_pointer_t<decltype(std::declval<Orchestrator &>().scheduler)>;
    using Layout = decltype(Scheduler::reserve_layout(std::declval<DeviceArena &>()));

public:
    bool init(Orchestrator &orch, void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks) {
        layout_ = Scheduler::reserve_layout(arena_);
        if (arena_.commit() == nullptr) return false;
        if (!scheduler_.init_data_from_layout(layout_, arena_, sm_base)) return false;
        scheduler_.wire_arena_pointers(layout_, arena_);
        initialized_ = true;
        return orch.init(sm_base, gm_heap, heap_size, max_tasks, &scheduler_);
    }

    void destroy() {
        if (initialized_) scheduler_.destroy();
        arena_.release();
        initialized_ = false;
    }

private:
    DeviceArena arena_;
    Scheduler scheduler_{};
    Layout layout_{};
    bool initialized_{false};
};
