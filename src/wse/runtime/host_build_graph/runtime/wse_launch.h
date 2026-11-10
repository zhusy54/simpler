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
#include "scheduler/scheduler_graph.h"

inline constexpr uint32_t WSE_PE_PER_CORE = 4;
inline constexpr uint32_t WSE_DEFAULT_CORE_COUNT = 40;

// Immutable host-published image. Mutable scheduling state belongs to the PE runtime.
struct WseNodePlan {
    uint64_t entry;
    uint32_t blocks;
    uint32_t host_done;
    uint32_t has_predicate;
    uint32_t reserved;
};

struct WseGraphDescriptor {
    SchedulerGraphView graph;
    uint64_t nodes_address;
    uint64_t expected_tasks;
};

enum class WseEventKind : uint32_t {
    CONFIG_READY,
    CORE_INIT,
    PE_READY,
    GO,
    BLOCK_BEGIN,
    BLOCK_END,
    TASK_DONE,
    QUIESCENT,
    CORE_RETIRED,
    RUN_DONE,
    STOP,
    BLOCK_SKIPPED,
    OP_PUBLISH,
    OP_WAIT,
    OP_READ,
    OP_CONSUMED
};
struct WseEvent {
    uint64_t sequence;
    uint64_t task;
    uint64_t value;
    uint32_t core;
    uint32_t pe;
    uint32_t block;
    WseEventKind kind;
    uint32_t slot;
    uint64_t generation;
};

struct WseRun;
extern "C" {
WseRun *wse_create_run(const WseGraphDescriptor *, uint32_t cores, uint64_t epoch);
void wse_destroy_run(WseRun *);
int wse_execute_pe(WseRun *, uint32_t worker);
const WseEvent *wse_run_events(const WseRun *, uint32_t worker, uint64_t *count);
uint64_t wse_completed_tasks(const WseRun *);
}
