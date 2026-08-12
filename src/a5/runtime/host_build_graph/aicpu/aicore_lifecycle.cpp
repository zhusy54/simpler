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

#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
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
    }
    OUT_OF_ORDER_STORE_BARRIER();

    for (int32_t i = 0; i < ready_count; ++i) {
        platform_init_aicore_regs(ready[i].reg_addr);
    }
    for (int32_t i = 0; i < ready_count; ++i) {
        const ReadyCore &core = ready[i];
        cores_[core.worker_id] = {core.reg_addr, core.physical_core_id, core.core_type};
        physical_core_ids_[core.worker_id] = core.physical_core_id;
    }
    OUT_OF_ORDER_STORE_BARRIER();
}

int32_t AicoreLifecycle::post_handshake_init(Runtime *) {
    if (handshake_failed_.load(std::memory_order_acquire)) {
        emergency_shutdown();
        return -1;
    }

    int32_t aic_count = 0;
    int32_t aiv_count = 0;
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].core_type == CoreType::AIC) {
            ++aic_count;
        } else {
            ++aiv_count;
        }
    }
    LOG_INFO("Core discovery complete: %d AIC, %d AIV", aic_count, aiv_count);

    if (is_pmu_enabled()) pmu_aicpu_init(physical_core_ids_, core_count_);
    return 0;
}

int32_t AicoreLifecycle::shutdown(int32_t thread_idx, Runtime *runtime) {
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(thread_idx) * core_count_) / aicpu_thread_num_);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(thread_idx + 1) * core_count_) / aicpu_thread_num_);
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

    int32_t rc = 0;
    for (int32_t i = lo; i < hi; ++i) {
        if (cores_[i].reg_addr != 0 && platform_deinit_aicore_regs(cores_[i].reg_addr) != 0) rc = -1;
    }
    return rc;
}

void AicoreLifecycle::emergency_shutdown() {
    for (int32_t i = 0; i < core_count_; ++i) {
        if (cores_[i].reg_addr != 0) platform_deinit_aicore_regs(cores_[i].reg_addr);
    }
}

void AicoreLifecycle::deinit() {
    std::memset(cores_, 0, sizeof(cores_));
    std::memset(physical_core_ids_, 0, sizeof(physical_core_ids_));
    handshake_failed_.store(false, std::memory_order_release);
    core_count_ = 0;
    aicpu_thread_num_ = 0;
    regs_base_ = 0;
}
