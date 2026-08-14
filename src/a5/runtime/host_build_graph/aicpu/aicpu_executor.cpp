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
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicore_lifecycle.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_time.h"
#include "../common/pto_runtime_status.h"
#include "pto_shared_memory.h"
#include "runtime.h"
#include "spin_hint.h"

#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "utils/thread_completion_gate.h"

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};

    // Parallel-handshake coordination (see AicpuExecutor::init). hs_setup_done_
    // is published by the leader once the shared pre-handshake setup is visible;
    // hs_arrived_ is the barrier counting threads that finished their core slice.
    // hs_thread_seq_ hands out a distinct [0, nthreads) index when the platform
    // exposes no affinity idx (sim, where platform_aicpu_affinity_thread_idx()
    // is -1 during init) so the threads don't all collapse to leader 0.
    std::atomic<bool> hs_setup_done_{false};
    std::atomic<bool> hs_config_done_{false};
    std::atomic<int32_t> hs_arrived_{0};
    std::atomic<int32_t> hs_release_arrived_{0};
    std::atomic<int32_t> hs_thread_seq_{0};

    int32_t aicpu_thread_num_{0};

    simpler::ThreadCompletionGate completion_gate_;
    std::atomic<bool> shutdown_ready_{false};
    std::atomic<int32_t> shutdown_signaled_{0};
    std::atomic<int32_t> run_status_{0};

    // AICPU owns only the AICore launch/teardown lifecycle. AICore owns graph
    // ticket ownership, private pending polling, and task execution.
    AicoreLifecycle aicore_lifecycle_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
};

static AicpuExecutor g_aicpu_executor;

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr || runtime->get_gm_sm_ptr() == nullptr) return 0;
    auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // All AICPU threads enter init. The per-core AICore handshake is the
    // dominant preamble cost (serial MMIO, ~217 µs of ~283 µs for 72 cores), so
    // it is parallelized: the leader (tidx 0) does the shared setup, every
    // thread handshakes a disjoint slice of cores, then the leader finishes init
    // across two barriers: discovery precedes configuration, and all register
    // releases precede init completion.
    int32_t nthreads = runtime->aicpu_thread_num;
    if (nthreads == 0) nthreads = 1;
    if (nthreads < 1 || nthreads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", nthreads);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }
    // Each thread needs a distinct index in [0, nthreads) to pick the leader and
    // partition the cores. Onboard the gate filter assigns it (exec_idx); sim's
    // gate does not, so platform_aicpu_affinity_thread_idx() is -1 here for every
    // thread — hand those a distinct index from a counter (mirrors run()'s
    // thread_idx_++ fallback) instead of collapsing them all to leader 0, which
    // would run pre_/post_handshake_init on every thread and race the shared
    // lifecycle state. Exactly nthreads threads reach init (the gate drops the
    // rest), so the counter yields a gap-free [0, nthreads).
    int32_t tidx = platform_aicpu_affinity_thread_idx();
    if (tidx < 0) tidx = hs_thread_seq_.fetch_add(1, std::memory_order_acq_rel);
    // A thread whose index still falls outside [0, nthreads) owns no core slice:
    // handshake_partition would compute lo/hi past cores_total_num_ and index
    // all_handshakes[]/core_exec_states_ out of bounds. Reject it here (mirrors
    // the bounds guard already in run()). Fail only this thread and do NOT set
    // init_failed_ — that would make the valid peers abort before their
    // hs_arrived_ increment and hang the leader at the barrier below.
    if (tidx >= nthreads) {
        LOG_ERROR("AICPU affinity thread idx %d out of range [0,%d) in init", tidx, nthreads);
        return -1;
    }
    const bool is_leader = (tidx == 0);

    if (is_leader) {
        LOG_INFO("AicpuExecutor: Initializing");
        // The 0 → 1 fixup already applied above.
        aicpu_thread_num_ = nthreads;

        hs_arrived_.store(0, std::memory_order_relaxed);
        hs_release_arrived_.store(0, std::memory_order_relaxed);
        if (aicore_lifecycle_.pre_handshake_init(runtime, aicpu_thread_num_, get_platform_regs()) != 0) {
            init_failed_.store(true, std::memory_order_release);
        }
        hs_setup_done_.store(true, std::memory_order_release);
    } else {
        while (!hs_setup_done_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    // All threads: handshake this thread's slice of cores in parallel.
    aicore_lifecycle_.handshake_partition(runtime, tidx, nthreads);

    // Barrier 1: every thread discovers its core slice before the leader builds
    // and publishes the topology-dependent execution configuration.
    hs_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        while (hs_arrived_.load(std::memory_order_acquire) < nthreads)
            SPIN_WAIT_HINT();
        completion_gate_.reset();
        if (!init_failed_.load(std::memory_order_acquire) && aicore_lifecycle_.post_handshake_init(runtime) != 0) {
            init_failed_.store(true, std::memory_order_release);
        }
        hs_config_done_.store(true, std::memory_order_release);
    } else {
        while (!hs_config_done_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();
    }

    // The DMB register is the execution gate. On success each AICPU thread
    // opens its own slice only after the shared configuration is visible. On
    // failure the same partitioned path sends EXIT and waits for every ACK.
    if (aicore_lifecycle_.release_partition(tidx, !init_failed_.load(std::memory_order_acquire)) != 0) {
        init_failed_.store(true, std::memory_order_release);
    }

    // Barrier 2: init cannot return until every successful slice is released,
    // or every failed slice has acknowledged EXIT.
    hs_release_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        while (hs_release_arrived_.load(std::memory_order_acquire) < nthreads)
            SPIN_WAIT_HINT();
        init_done_.store(true, std::memory_order_release);
        if (!init_failed_.load(std::memory_order_acquire)) LOG_INFO("AicpuExecutor: Init complete");
    } else {
        while (!init_done_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();
    }
    return init_failed_.load(std::memory_order_acquire) ? -1 : 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = (affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++);
    if (thread_idx < 0 || thread_idx >= aicpu_thread_num_ || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Thread index %d out of bounds (active=%d max=%d exec_idx=%d)", thread_idx, aicpu_thread_num_,
            PLATFORM_MAX_AICPU_THREADS, affinity_exec_idx
        );
        return -1;
    }
    if (thread_idx == aicpu_thread_num_ - 1) {
        int32_t supervisor_rc = 0;
        if (runtime->aicore_sidecar_base == nullptr || runtime->host_total_tasks < 0) {
            LOG_ERROR("A5 HBG AICore scheduler requires an initialized graph");
            supervisor_rc = -1;
        } else {
            auto *run_control = aicore_sidecar_at_v1<AicoreRunControlV1>(
                runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.run_control_offset
            );
            const uint64_t executable_task_count =
                runtime->aicore_sidecar_layout.aic_task_count + runtime->aicore_sidecar_layout.aiv_task_count;
            const bool trace_enabled = is_chip_swimlane_enabled();
            const uint64_t completion_wait_start_cycles = trace_enabled ? get_sys_cnt_aicpu() : 0;
            uint64_t all_executors_drained_cycles = 0;
            uint64_t all_tasks_resolved_cycles = 0;
            uint64_t completion_poll_count = 0;
            uint64_t completion_poll_cycles = 0;
            uint64_t error_poll_total = 0;
            uint32_t error_poll_count = 0;
            while (true) {
                const uint64_t poll_start = trace_enabled ? get_sys_cnt_aicpu() : 0;
                cache_invalidate_range(reinterpret_cast<uint8_t *>(run_control) + 128, 128);
                const bool executors_drained =
                    run_control->executor_drained_worker_count == run_control->active_worker_count;
                const bool tasks_resolved = run_control->resolved_task_count == executable_task_count;
                if (trace_enabled) {
                    const uint64_t poll_end = get_sys_cnt_aicpu();
                    ++completion_poll_count;
                    completion_poll_cycles += poll_end - poll_start;
                    if (executors_drained && all_executors_drained_cycles == 0) {
                        all_executors_drained_cycles = poll_end;
                    }
                    if (executors_drained && tasks_resolved && all_tasks_resolved_cycles == 0) {
                        all_tasks_resolved_cycles = poll_end;
                    }
                }
                if (executors_drained && tasks_resolved) {
                    break;
                }
                if (++error_poll_count == 64) {
                    error_poll_count = 0;
                    ++error_poll_total;
                    cache_invalidate_range(reinterpret_cast<uint8_t *>(run_control) + 256, 128);
                    if (run_control->scheduler_error != 0) break;
                }
                SPIN_WAIT_HINT();
            }
            cache_invalidate_range(reinterpret_cast<uint8_t *>(run_control) + 128, 256);
            if (run_control->scheduler_error != 0) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: graph execution failed at task=%" PRIu64 " status=%" PRIu64
                    " core=%" PRIu64 " type=%" PRIu64 " graph_tasks=%" PRIu64 " descriptors=0x%" PRIx64
                    " payloads=0x%" PRIx64 " mask=0x%" PRIx64,
                    run_control->error_task_id, run_control->scheduler_error, run_control->error_core_id,
                    run_control->error_core_type, run_control->error_graph_task_count,
                    run_control->error_descriptors_address, run_control->error_payloads_address,
                    run_control->error_task_window_mask
                );
                supervisor_rc = -1;
            }
            if (supervisor_rc == 0 && run_control->executed_task_count + run_control->inline_completed_count !=
                                          run_control->expected_task_count) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: count mismatch expected=%" PRIu64 " executed=%" PRIu64 " inline=%" PRIu64,
                    run_control->expected_task_count, run_control->executed_task_count,
                    run_control->inline_completed_count
                );
                supervisor_rc = -1;
            }
            if (trace_enabled) {
                run_control->completion_wait_start_cycles = completion_wait_start_cycles;
                run_control->all_executors_drained_cycles = all_executors_drained_cycles;
                run_control->all_tasks_resolved_cycles = all_tasks_resolved_cycles;
                run_control->shutdown_ready_cycles = get_sys_cnt_aicpu();
                run_control->completion_poll_count = completion_poll_count;
                run_control->completion_poll_cycles = completion_poll_cycles;
                run_control->error_poll_count = error_poll_total;
                cache_flush_range(reinterpret_cast<uint8_t *>(run_control) + 128, 128);
            }
        }
        run_status_.store(supervisor_rc, std::memory_order_release);
        shutdown_ready_.store(true, std::memory_order_release);
    } else {
        while (!shutdown_ready_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();
    }

    int32_t run_rc = run_status_.load(std::memory_order_acquire);
    aicore_lifecycle_.signal_shutdown_partition(thread_idx);
    shutdown_signaled_.fetch_add(1, std::memory_order_acq_rel);
    while (shutdown_signaled_.load(std::memory_order_acquire) < aicpu_thread_num_)
        SPIN_WAIT_HINT();
    int32_t shutdown_rc = aicore_lifecycle_.finish_shutdown_partition(thread_idx, runtime);
    if (shutdown_rc != 0 && run_rc == 0) run_rc = shutdown_rc;
    completion_gate_.arrive_and_finalize_if_last(aicpu_thread_num_, [] {});
    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    aicore_lifecycle_.deinit();

    completion_gate_.reset();
    shutdown_ready_.store(false, std::memory_order_release);
    shutdown_signaled_.store(0, std::memory_order_release);
    run_status_.store(0, std::memory_order_release);

    aicpu_thread_num_ = 0;

    LOG_INFO("DeInit: Runtime execution state reset");

    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    hs_setup_done_.store(false, std::memory_order_release);
    hs_config_done_.store(false, std::memory_order_release);
    hs_arrived_.store(0, std::memory_order_release);
    hs_release_arrived_.store(0, std::memory_order_release);
    hs_thread_seq_.store(0, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

extern "C" int32_t aicpu_prewarm_callable(Runtime *runtime) {
    // host_build_graph host-orch: the orchestration .so is dlopen'd on the HOST
    // during prepare_callable_impl and the whole task graph is built host-side,
    // so there is no device-side orchestrator .so to pre-load — prewarm is a
    // no-op. The symbol is retained because the platform onboard kernel
    // (src/a2a3/platform/onboard/aicpu/kernel.cpp) links it strongly via
    // simpler_aicpu_prewarm_callable; removing it would break the onboard link.
    (void)runtime;
    return 0;
}

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor: all threads enter init(), which handshakes the cores
 *    in parallel and barriers internally until init is complete (or a thread
 *    failed); its return value is authoritative on every thread.
 * 2. Execute tasks on managed cores
 * 3. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // init() barriers every thread internally until init is complete on the
    // leader (or a thread failed), then returns the status — so a non-zero
    // return is authoritative on all threads and no extra spin is needed.
    if (g_aicpu_executor.init(runtime) != 0) {
        LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
        return -1;
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // The finalizer publishes cleanup eligibility only after runtime destruction.
    if (g_aicpu_executor.completion_gate_.claim_cleanup()) {
        LOG_INFO("aicpu_execute: All threads finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
