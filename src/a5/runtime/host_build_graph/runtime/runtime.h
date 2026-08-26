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
 * Runtime Class - Device Execution and Handshake Control
 *
 * This class manages device-side execution through AICPU-AICore handshake
 * protocol. Task graph construction is handled by RuntimeContext; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, aicpu_thread_num)
 * - simpler::hbg::Tensor pair management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * AICore workers materialize a DispatchPayload from the uploaded graph,
 * execute it, and publish its per-task completion flag.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <vector>

#include "common/core_type.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS (aicpu_allowed_cpus bound)
#include "scheduler/scheduler_types.h"
#include "dispatch_payload.h"
#include "task_args.h"
#include "tensor.h"  // EntryArgsStorage

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_ARGS 128
#define RUNTIME_MAX_WORKER 108  // 36 AIC + 72 AIV cores
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SO_SIZE (4 * 1024 * 1024)  // 4MB max for orchestration SO
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Handshake Structure - Shared between Host, AICPU, and AICore
 *
 * This structure facilitates communication and synchronization between
 * AICPU and AICore during task execution.
 *
 * Protocol State Machine:
 * 1. Initialization: AICPU sets aicpu_ready=1
 * 2. Acknowledgment: AICore sets aicore_done=core_id+1
 * 3. Execution: AICore resolves dependencies and executes graph tasks
 * 4. Shutdown: AICPU waits for all workers, then closes register windows
 *
 * Each AICore instance has its own handshake buffer to enable concurrent
 * task execution across multiple cores.
 */

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Field Access Patterns:
 * - aicpu_ready: Reserved legacy field; the current handshake does not use it
 * - aicore_done: Written by AICore, read by AICPU (final report; physical_core_id
 *   and core_type are published alongside it in the same write)
 * - task: Written by AICPU before window-open, read by AICore after window-open
 * - core_type: Written by AICore (with aicore_done), read by AICPU (CoreType::AIC or CoreType::AIV)
 * - physical_core_id: Written by AICore (with aicore_done), read by AICPU
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // Legacy layout field; unused by the current handshake
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // DispatchPayload* published before register window-open
    volatile CoreType core_type;    // Core type: CoreType::AIC or CoreType::AIV (reported by AICore with aicore_done)
    volatile uint32_t physical_core_id;  // Physical core ID (reported by AICore with aicore_done)
} __attribute__((aligned(64)));

/**
 * simpler::hbg::Tensor pair for tracking host-device memory mappings.
 * Used for copy-back during finalize.
 */
struct TensorPair {
    void *host_ptr;
    void *dev_ptr;
    size_t size;
    // false for read-only INPUT tensors: they are never written by the kernel,
    // so the end-of-run D2H copy-back is skipped. OUTPUT/INOUT/unknown
    // keep the safe default of copying back.
    bool needs_copy_back = true;
};

/**
 * Task structure - Compatibility stub for platform layer
 *
 * RT2 uses DispatchPayload instead of Task for task dispatch.
 * This stub exists only for API compatibility with device_runner.cpp.
 * Since get_task_count() returns 0, this struct is never actually used.
 */
struct Task {
    int func_id;
    uint64_t function_bin_addr;
};

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by RuntimeContext; this class only handles
 * execution control and device orchestration state.
 */
class Runtime {
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    // Execution parameters for AICPU lifecycle threads and AICore workers.
    //
    // aicpu_thread_num is the total AICPU thread count launched on this run.
    // host_build_graph builds the task graph on the host, so there is no
    // on-device orchestrator. The highest-index thread supervises the AICore
    // bootstrap publication and waits for graph completion; other threads
    // wait until teardown.
    int aicpu_thread_num;

    // Filter-style affinity gate input (a2a3 onboard). Host fills these
    // before launch from AICPU OCCUPY, and the device gate keeps threads whose
    // sched_getcpu() lands on one of the cpu_ids. The array position is the
    // deterministic exec_idx used by AicpuExecutor for scheduler-thread
    // assignment; the highest active index additionally runs the host-orch boot.
    int32_t aicpu_allowed_cpus[MAX_GATE_THREADS];
    int32_t aicpu_allowed_cpu_count;
    int32_t aicpu_launch_count;

    // kernel binary resolution: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Total tasks submitted by the host orchestrator and consumed by the
    // scheduler state.
    int32_t host_total_tasks;

    // Size of the compact shared-memory image, including argument pools.
    uint64_t sm_image_bytes;

    // Per-run mutable AICore state. Device code uses the aligned base; Host
    // cleanup retains the original allocation returned by device_malloc.
    void *scheduler_state_base;
    void *scheduler_state_allocation;
    uint64_t scheduler_state_allocation_size;
    SchedulerLayout scheduler_layout;

private:
    // Kernel binary tracking for cleanup

    void *gm_sm_ptr_;                                   // GM pointer to shared memory (device)
    simpler::hbg::EntryArgsStorage orch_args_storage_;  // Entry args, adopted on the host

    // Orchestration metadata set by the platform host (DeviceRunner) when
    // registering a callable. host_build_graph runs the orchestrator on the
    // host, so the device side no longer reads the SO bytes / symbol names —
    // but the platform registration path still writes them through these
    // setters (shared with tensormap_and_ringbuffer), so the fields and their
    // setters are part of the platform↔runtime ABI and must stay.
    uint64_t dev_orch_so_addr_;
    uint64_t dev_orch_so_size_;
    int32_t active_callable_id_;
    char device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];
    char device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];

public:
    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Accessors for the execution-parameter fields
    //
    // These exist with identical signatures on the tensormap_and_ringbuffer
    // Runtime so the shared platform layer (device_runner*.cpp, kernel.cpp) can
    // compile against either variant. hbg stores the fields flat (trb keeps
    // them in a `dev` sub-struct); the accessors hide that difference.
    // =========================================================================

    int get_worker_count() const { return worker_count; }
    void set_worker_count(int n) { worker_count = n; }
    int get_aicpu_thread_num() const { return aicpu_thread_num; }
    void set_aicpu_thread_num(int n) { aicpu_thread_num = n; }
    Handshake *get_workers() { return workers; }
    int32_t get_aicpu_allowed_cpu_count() const { return aicpu_allowed_cpu_count; }
    void set_aicpu_allowed_cpu_count(int32_t n) { aicpu_allowed_cpu_count = n; }
    int32_t get_aicpu_launch_count() const { return aicpu_launch_count; }
    void set_aicpu_launch_count(int32_t n) { aicpu_launch_count = n; }
    int32_t *get_aicpu_allowed_cpus() { return aicpu_allowed_cpus; }
    size_t aicpu_allowed_cpus_capacity() const { return sizeof(aicpu_allowed_cpus) / sizeof(aicpu_allowed_cpus[0]); }

    // =========================================================================
    // Performance Profiling
    // =========================================================================

    // =========================================================================
    // Shared-memory / orchestration argument plumbing
    // =========================================================================

    void *get_gm_sm_ptr() const { return gm_sm_ptr_; }
    const simpler::hbg::EntryArgsStorage &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Orchestration metadata written by the platform host (DeviceRunner) at
    // callable registration. Shared ABI with tensormap_and_ringbuffer; the
    // host_build_graph device side no longer reads them (host-orch builds the
    // graph on the host), so only the setter + get_active_callable_id() the
    // platform reads are exposed.
    void set_dev_orch_so(uint64_t dev_addr, uint64_t size);
    void set_active_callable_id(int32_t callable_id);
    int32_t get_active_callable_id() const;
    void set_device_orch_func_name(const char *name);
    void set_device_orch_config_name(const char *name);

    uint64_t get_function_bin_addr(int func_id) const;
    /**
     * Map a func_id onto the device address of its CoreCallable. Used by
     * DeviceRunner::bind_callable_to_runtime for each of the active callable's
     * child kernels, after clear_function_bin_addrs() has emptied the table.
     */
    void replay_function_bin_addr(int func_id, uint64_t addr);

    /**
     * Drop every func_id -> CoreCallable address mapping.
     *
     * Each mapping points into one callable's retained ChipCallable buffer,
     * which unregistering that callable frees. `bind_callable_to_runtime` calls
     * this before replaying the active callable's addresses so no entry outlives
     * the buffer it points into: the scheduler dereferences these addresses and
     * the AICore calls what it finds there.
     */
    void clear_function_bin_addrs();

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by RuntimeContext, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses DispatchPayload, not Task. Always returns nullptr. */
    Task *get_task(int) { return nullptr; }

    // Host-side tensor ledger for D2H copy-back at finalize. Populated by
    // runtime_maker.cpp from orch_args at bind time, then iterated in
    // validate_runtime_impl. Not read by AICPU/AICore — the device-side
    // Runtime image also carries the host-only std::vector control block, which
    // device code must not inspect. No fixed cap — grows with the chip-level
    // entry-tensor count.
    std::vector<TensorPair> tensor_pairs_;
};

// Number of bytes of the Runtime image that must be copied to the device.
// host_build_graph returns sizeof(Runtime) (its device image is the whole
// object); trb returns sizeof(DeviceRuntimeLaunchDesc). Defined per-runtime so
// the shared device_runner_helpers.cpp copy path stays runtime-agnostic.
size_t runtime_device_copy_size(const Runtime &rt);
