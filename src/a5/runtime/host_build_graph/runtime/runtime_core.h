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
/**
 * Runtime core interface
 *
 * This is the main header for the runtime.
 * It provides a unified API for task graph construction and execution.
 *
 * Key Features:
 * - Bump-allocated task table and graph heap (zero allocation overhead)
 * - Lazy invalidation TensorMap for dependency discovery
 * - Manual scopes that bypass TensorMap discovery for explicit dependencies
 * - Per-task spinlocks for concurrent fanout updates
 * - Orchestrator-Scheduler decoupling via shared memory
 *
 * Usage:
 *   1. Create runtime: RuntimeContext create methods
 *   2. Build task graph in orchestration function:
 *      - begin_scope() / end_scope()
 *      - submit_task()
 *   3. Mark orchestration complete: mark_done()
 *   4. Destroy runtime
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <type_traits>

#include "utils/device_arena.h"
#include "runtime_types.h"
#include "graph_cache.h"
#include "submit_types.h"
#include "shared_memory.h"
#include "task_allocator.h"
#include "tensormap.h"
#include "orchestrator.h"

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum RuntimeMode {
    MODE_EXECUTE = 0,    // Execute tasks on workers
    MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

/**
 * Function-pointer ops table for runtime operations.
 *
 * The orchestration .so calls runtime functions through this table
 * (via orchestration_api.h inline wrappers), so it has zero link
 * dependencies on runtime .cpp files.
 */
typedef struct RuntimeContext RuntimeContext;  // forward declare for ops signatures
class HostTensorAccessor;

struct RuntimeOps {
    TaskOutputTensors (*submit_task)(RuntimeContext *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(RuntimeContext *rt);
    void (*scope_end)(RuntimeContext *rt);
    void (*orchestration_done)(RuntimeContext *rt);
    bool (*is_fatal)(RuntimeContext *rt);
    void (*report_fatal)(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(
        RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]
    );
    void (*set_tensor_data)(
        RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(RuntimeContext *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(RuntimeContext *rt, const CoreTaskArgs &args);

    // This-run core geometry latched by the host bind: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(RuntimeContext *rt);
    int32_t (*available_aiv_count)(RuntimeContext *rt);
    GraphScopeResult (*graph_begin)(RuntimeContext *rt, uint64_t graph_key, const GraphTaskArgs &args);
    bool (*graph_prepare)(RuntimeContext *rt, void *recording_handle, const GraphTaskArgs &args);
    void (*graph_abort)(RuntimeContext *rt, void *recording_handle);
    bool (*graph_end)(RuntimeContext *rt);
    void (*graph_commit)(RuntimeContext *rt);
    // Record one orchestration-side phase on the calling thread. The submission
    // segments this carries are measured in the orchestration .so, which reaches the
    // runtime only through this table. Always present in the struct so the layout does
    // not move with SIMPLER_DFX; nullptr when off.
    void (*record_orch_phase)(uint32_t kind, uint64_t start_ns, uint64_t end_ns, uint64_t detail);
};

/**
 * Runtime context
 *
 * Contains the host orchestration state exposed through RuntimeOps.
 */
struct RuntimeContext {
    // Ops table (first field — used by orchestration .so via function pointers)
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode;

    // Components
    // Host-only, and by pointer so that this header stays trivially copyable:
    // the orchestrator runs on the host and owns non-trivial scratch.
    OrchestratorState *orchestrator;

    // Mode
    RuntimeMode mode;

    // Statistics
    int64_t total_cycles;
    // Graph definitions are process-local host cache entries. The callable
    // identity prevents two orchestration DSOs from sharing the same key.
    uint64_t active_callable_hash;

    // Host views of the tensors this run staged, owned by the run that
    // registered them. Null on the AICPU path, which loads device addresses
    // directly; get_tensor_data / set_tensor_data then fail closed rather than
    // dereferencing one. Lives past the first two fields, so the orchestration
    // .so's partial RuntimeContext definition neither sees nor needs it.
    HostTensorAccessor *tensor_access;
};

// The orchestration DSO consumes the common prefix through its partial
// RuntimeContext definition, so this runtime-owned form must retain C layout.
static_assert(
    std::is_trivially_copyable_v<RuntimeContext> && std::is_standard_layout_v<RuntimeContext>,
    "RuntimeContext must retain a stable orchestration ABI prefix"
);

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Install the runtime-owned operations table before invoking orchestration.
 */
void runtime_bind_ops(RuntimeContext *rt);

/**
 * Set execution mode
 */
void runtime_set_mode(RuntimeContext *rt, RuntimeMode mode);

// =============================================================================
// Orchestration API (called by orchestration function)
// =============================================================================

/**
 * Begin a new scope
 *
 * submit_task requires at least one open scope. A MANUAL scope additionally makes
 * every submit inside it bypass TensorMap discovery and take its fanin from
 * CoreTaskArgs::set_dependencies() instead; an AUTO scope nested inside a MANUAL
 * one is rejected. The mode is read from RuntimeContext::pending_scope_mode, which
 * SIMPLER_SCOPE sets immediately before this call.
 */
void rt_scope_begin(RuntimeContext *rt);

/**
 * End current scope
 *
 * Closes the innermost scope, and when that is the outermost MANUAL one, returns
 * later submits to TensorMap discovery. A scope bounds no task or buffer lifetime
 * here: the task table is whole-graph-resident, so no task slot and no heap byte is
 * reclaimed before the run ends.
 */
void rt_scope_end(RuntimeContext *rt);

/**
 * Mark orchestration as complete
 *
 * Signals that no more tasks will be submitted.
 */
void rt_orchestration_done(RuntimeContext *rt);

/**
 * Enter fatal state explicitly from orchestration.
 */
void rt_report_fatal(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...);

/**
 * Cross-layer data access: read a tensor value by waiting for its producer.
 */
uint64_t
get_tensor_data(RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]);

/**
 * Cross-layer data access: write a value to a tensor at given indices.
 * Waits for producer completion (WAW) and all consumers (WAR) via TensorMap.
 * See set_tensor_data in orchestration_api.h for full documentation.
 */
void set_tensor_data(
    RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
);

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with orchestration_api.h (same layout, guarded).
 */
#ifndef ORCHESTRATION_CONFIG_DEFINED
#define ORCHESTRATION_CONFIG_DEFINED
struct OrchestrationConfig {
    int expected_arg_count;
};
#endif
