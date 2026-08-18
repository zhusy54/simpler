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

#include "aicore_lifecycle.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "runtime.h"
#include "spin_hint.h"

int32_t AicoreLifecycle::pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base) {
    if (runtime == nullptr || runtime->worker_count <= 0 || runtime->worker_count > kMaxWorkers) {
        LOG_ERROR("Invalid AICore lifecycle worker count");
        return -1;
    }

    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    core_count_ = runtime->worker_count;
    aicpu_thread_num_ = aicpu_thread_num;
    regs_base_ = regs_base;
    handshake_failed_.store(false, std::memory_order_release);

    if (is_chip_swimlane_enabled()) {
        chip_swimlane_aicpu_init(core_count_);
        if (get_chip_swimlane_level() != ChipSwimlaneLevel::AICORE_TIMING) {
            LOG_ERROR(
                "A5 HBG AICore scheduler supports chip swimlane level 1 only, got %u",
                static_cast<uint32_t>(get_chip_swimlane_level())
            );
            return -1;
        }
    }
    return 0;
}

void AicoreLifecycle::handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    Handshake *handshakes = runtime->workers;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(tidx) * core_count_) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(tidx + 1) * core_count_) / nthreads);
    const uint32_t physical_core_count = platform_get_physical_cores_count();
    uint64_t *regs = reinterpret_cast<uint64_t *>(regs_base_);

    struct ReadyCore {
        int32_t worker_id;
        uint32_t physical_core_id;
        uint64_t reg_addr;
        CoreType core_type;
    };
    ReadyCore ready[kMaxWorkers]{};
    bool observed[kMaxWorkers]{};
    int32_t ready_count = 0;

    for (int32_t remaining = hi - lo; remaining > 0;) {
        for (int32_t i = lo; i < hi; ++i) {
            if (observed[i]) continue;
            Handshake *handshake = &handshakes[i];
            cache_invalidate_range(handshake, sizeof(*handshake));
            if (handshake->aicore_done == 0) {
                SPIN_WAIT_HINT();
                continue;
            }
            observed[i] = true;
            --remaining;
            uint32_t physical_core_id = handshake->physical_core_id;
            if (physical_core_id >= physical_core_count) {
                LOG_ERROR(
                    "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                    physical_core_count
                );
                handshake_failed_.store(true, std::memory_order_release);
                continue;
            }
            ready[ready_count++] = {i, physical_core_id, regs[physical_core_id], handshake->core_type};
        }
    }

    for (int32_t i = 0; i < ready_count; ++i) {
        const ReadyCore &core = ready[i];
        handshakes[core.worker_id].task = reinterpret_cast<uint64_t>(aicore_sidecar_at_v1<AicoreWorkerContextV1>(
            runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.worker_contexts_offset +
                                              static_cast<uint64_t>(core.worker_id) * sizeof(AicoreWorkerContextV1)
        ));
        cores_[core.worker_id] = {core.reg_addr, core.physical_core_id, core.core_type, nullptr};
        physical_core_ids_[core.worker_id] = core.physical_core_id;
    }
    if (hi > lo) cache_flush_range(&handshakes[lo], static_cast<size_t>(hi - lo) * sizeof(Handshake));
}

int32_t AicoreLifecycle::post_handshake_init(Runtime *runtime) {
    if (runtime == nullptr || runtime->aicore_sidecar_base == nullptr || runtime->host_total_tasks < 0 ||
        handshake_failed_.load(std::memory_order_acquire))
        return -1;

    auto *run_control = aicore_sidecar_at_v1<AicoreRunControlV1>(
        runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.run_control_offset
    );
    auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(
        runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.worker_contexts_offset
    );
    auto *lifecycle_traces = aicore_sidecar_at_v1<AicpuCoreLifecycleTraceV1>(
        runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.aicpu_lifecycle_traces_offset
    );
    cache_invalidate_range(run_control, sizeof(*run_control));
    cache_invalidate_range(contexts, static_cast<size_t>(core_count_) * sizeof(*contexts));
    cache_invalidate_range(lifecycle_traces, static_cast<size_t>(core_count_) * sizeof(*lifecycle_traces));

    int32_t aic_count = 0;
    int32_t aiv_count = 0;
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].core_type == CoreType::AIC) {
            contexts[i].type_rank = aic_count++;
        } else if (cores_[i].core_type == CoreType::AIV) {
            contexts[i].type_rank = aiv_count++;
        } else {
            LOG_ERROR("Core %d reported invalid core type", i);
            return -1;
        }
        contexts[i].core_type = static_cast<int32_t>(cores_[i].core_type);
        contexts[i].physical_core_id = static_cast<int32_t>(cores_[i].physical_core_id);
        contexts[i].active = 0;
        cores_[i].trace = &lifecycle_traces[i];
        lifecycle_traces[i].worker_id = static_cast<uint64_t>(i);
        lifecycle_traces[i].core_type = static_cast<uint64_t>(cores_[i].core_type);
        lifecycle_traces[i].physical_core_id = static_cast<uint64_t>(cores_[i].physical_core_id);
    }
    LOG_INFO("Core discovery complete: %d AIC, %d AIV", aic_count, aiv_count);

    const int32_t active_aic = static_cast<int32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(aic_count), runtime->aicore_sidecar_layout.aic_task_count)
    );
    const int32_t active_aiv_executors = static_cast<int32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(aiv_count), runtime->aicore_sidecar_layout.aiv_task_count)
    );
    const uint64_t executable_task_count =
        runtime->aicore_sidecar_layout.aic_task_count + runtime->aicore_sidecar_layout.aiv_task_count;
    const int32_t active_aiv = aiv_count == 0 ? 0 : std::max(active_aiv_executors, executable_task_count != 0 ? 1 : 0);
    if ((runtime->aicore_sidecar_layout.aic_task_count != 0 && active_aic == 0) ||
        (runtime->aicore_sidecar_layout.aiv_task_count != 0 && active_aiv_executors == 0) ||
        (executable_task_count != 0 && active_aiv == 0)) {
        LOG_ERROR(
            "A5 HBG AICore scheduler: topology cannot execute graph (tasks AIC=%" PRIu64 " AIV=%" PRIu64
            ", cores AIC=%d AIV=%d)",
            runtime->aicore_sidecar_layout.aic_task_count, runtime->aicore_sidecar_layout.aiv_task_count, aic_count,
            aiv_count
        );
        return -1;
    }

    for (int32_t i = 0; i < core_count_; ++i) {
        const bool is_aiv = cores_[i].core_type == CoreType::AIV;
        const int32_t active_count = is_aiv ? active_aiv : active_aic;
        contexts[i].active = contexts[i].type_rank < active_count ? 1 : 0;
        contexts[i].inbox_index =
            is_aiv && contexts[i].active != 0 ? static_cast<uint64_t>(contexts[i].type_rank) : UINT64_MAX;
    }
    run_control->active_worker_count = static_cast<uint64_t>(active_aic) + static_cast<uint64_t>(active_aiv);
    run_control->aic_active_worker_count = static_cast<uint64_t>(active_aic);
    run_control->aiv_active_worker_count = static_cast<uint64_t>(active_aiv);
    if (executable_task_count == 0) run_control->bootstrap_complete = 1;

    if (is_pmu_enabled()) pmu_aicpu_init(physical_core_ids_, core_count_);
    cache_flush_range(contexts, static_cast<size_t>(core_count_) * sizeof(*contexts));
    cache_flush_range(lifecycle_traces, static_cast<size_t>(core_count_) * sizeof(*lifecycle_traces));
    cache_flush_range(run_control, sizeof(*run_control));
    wmb();
    return 0;
}

int32_t AicoreLifecycle::release_partition(int32_t thread_idx, bool start_execution) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    wmb();
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (start_execution) {
            if (cores_[i].trace != nullptr) {
                cores_[i].trace->aicpu_thread_id = static_cast<uint64_t>(thread_idx);
                cores_[i].trace->register_release_cycles = get_sys_cnt_aicpu();
            }
            platform_init_aicore_regs(cores_[i].reg_addr);
        } else {
            platform_signal_aicore_exit(cores_[i].reg_addr);
        }
    }
    if (start_execution) return 0;

    int32_t rc = 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr != 0 && platform_wait_aicore_exit(cores_[i].reg_addr) != 0) rc = -1;
    }
    return rc;
}

void AicoreLifecycle::signal_shutdown_partition(int32_t thread_idx) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (cores_[i].trace != nullptr) {
            cores_[i].trace->aicpu_thread_id = static_cast<uint64_t>(thread_idx);
            cores_[i].trace->exit_signal_cycles = get_sys_cnt_aicpu();
        }
        platform_signal_aicore_exit(cores_[i].reg_addr);
    }
}

int32_t AicoreLifecycle::finish_shutdown_partition(int32_t thread_idx, Runtime *runtime) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
    int32_t rc = 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr == 0) continue;
        if (platform_wait_aicore_exit(cores_[i].reg_addr) != 0) {
            rc = -1;
        } else if (cores_[i].trace != nullptr) {
            cores_[i].trace->exit_ack_cycles = get_sys_cnt_aicpu();
        }
    }
    rmb();
    if (hi > lo && cores_[lo].trace != nullptr) {
        cache_flush_range(cores_[lo].trace, static_cast<size_t>(hi - lo) * sizeof(AicpuCoreLifecycleTraceV1));
    }

    int32_t core_ids[kMaxWorkers]{};
    int32_t count = 0;

    for (int32_t i = lo; i < hi; ++i) {
        core_ids[count++] = i;
        if (is_chip_swimlane_enabled()) {
            auto *worker = aicore_sidecar_at_v1<AicoreWorkerContextV1>(
                runtime->aicore_sidecar_base, runtime->aicore_sidecar_layout.worker_contexts_offset +
                                                  static_cast<uint64_t>(i) * sizeof(AicoreWorkerContextV1)
            );
            cache_invalidate_range(worker, sizeof(*worker));
            chip_swimlane_aicpu_set_aicore_counts(i, worker->executed_task_count == 0 ? 0 : 1, 0);
        }
    }

    if (is_chip_swimlane_enabled()) chip_swimlane_aicpu_flush(thread_idx, core_ids, count);
    if (is_pmu_enabled()) pmu_aicpu_finalize(core_ids, count);
    return rc;
}

void AicoreLifecycle::deinit() {
    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    handshake_failed_.store(false, std::memory_order_release);
    core_count_ = 0;
    aicpu_thread_num_ = 0;
    regs_base_ = 0;
}
