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

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/platform_config.h"
#include "aicore_dependency_scheduler_v0.h"
#include "aicore_task_profiling_v0.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((always_inline)) static void publish_worker_config(__gm__ AicoreWorkerContextV0 *context) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 0; offset < 128; offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(context) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb((mem_dsb_t)0);
#else
    (void)context;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[block_idx]);
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);

    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    aicore_observe_cache_line_v0(my_hank);
    __gm__ AicoreWorkerContextV0 *worker_context = reinterpret_cast<__gm__ AicoreWorkerContextV0 *>(my_hank->task);
    aicore_observe_cache_line_v0(worker_context);
    worker_context->physical_core_id = static_cast<int32_t>(my_hank->physical_core_id);
    worker_context->core_type = static_cast<int32_t>(core_type);
    publish_worker_config(worker_context);

    const bool chip_swimlane_enabled =
        SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    AicoreTaskProfilingStateV0 task_profiling{};
    aicore_task_profiling_init_v0(
        &task_profiling, chip_swimlane_enabled, chip_swimlane_enabled ? get_chip_swimlane_aicore_head() : nullptr
    );

    __gm__ void *sidecar_base = reinterpret_cast<__gm__ void *>(worker_context->sidecar_base_address);
    __gm__ AicoreRunControlV0 *run_control =
        aicore_sidecar_at_v0<AicoreRunControlV0>(sidecar_base, worker_context->run_control_offset);
    AicoreReadonlyGraphV0 graph{
        worker_context->graph_descriptors_address,
        worker_context->graph_payloads_address,
        worker_context->graph_task_count,
        worker_context->task_window_mask,
    };
    aicore_gm_fetch_add_v0(run_control->attached_count, UINT64_C(1));

    while (aicore_gm_load_v0(run_control->startup_phase) == static_cast<uint64_t>(AicoreRunPhaseV0::ATTACH) &&
           aicore_gm_load_v0(run_control->exit_requested) == 0) {
        SPIN_WAIT_HINT();
    }

    aicore_observe_cache_line_v0(worker_context);
    if (worker_context->resolver_active != 0 && graph.task_count != 0 &&
        aicore_gm_load_v0(run_control->startup_phase) == static_cast<uint64_t>(AicoreRunPhaseV0::CLASSIFY)) {
        uint64_t classify_start = get_sys_cnt_aicore();
        aicore_observe_data_cache_v0(reinterpret_cast<__gm__ void *>(graph.descriptors_address));
        uint64_t resolver_count = run_control->resolver_count;
        uint64_t rank = static_cast<uint64_t>(worker_context->classifier_rank);
        uint64_t quotient = graph.task_count / resolver_count;
        uint64_t remainder = graph.task_count % resolver_count;
        uint64_t start = rank * quotient + (rank < remainder ? rank : remainder);
        uint64_t count = quotient + (rank < remainder ? 1 : 0);
        for (uint64_t task_id = start; task_id < start + count; ++task_id) {
            AicoreRouteResultV0 result = aicore_classify_and_route_v0(
                graph, sidecar_base, worker_context, run_control, static_cast<int64_t>(task_id)
            );
            if (result == AicoreRouteResultV0::ERROR) break;
            if (result == AicoreRouteResultV0::READY) {
                aicore_gm_fetch_add_v0(worker_context->initial_ready_count, UINT64_C(1));
            } else if (result == AicoreRouteResultV0::WAITING) {
                aicore_gm_fetch_add_v0(worker_context->initial_waiting_count, UINT64_C(1));
            }
        }
        aicore_gm_fetch_add_v0(worker_context->classify_cycles, get_sys_cnt_aicore() - classify_start);
        aicore_gm_fetch_add_v0(run_control->classified_count, UINT64_C(1));
    }

    while (aicore_gm_load_v0(run_control->startup_phase) != static_cast<uint64_t>(AicoreRunPhaseV0::RUN) &&
           aicore_gm_load_v0(run_control->startup_phase) != static_cast<uint64_t>(AicoreRunPhaseV0::EXIT)) {
        SPIN_WAIT_HINT();
    }

    aicore_observe_cache_line_v0(worker_context);
    if ((worker_context->active != 0 || worker_context->resolver_active != 0) &&
        aicore_gm_load_v0(run_control->exit_requested) == 0) {
        __gm__ AicoreReadyQueueV0 *ready_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(
            sidecar_base,
            core_type == CoreType::AIC ? worker_context->aic_queue_offset : worker_context->aiv_queue_offset
        );
        __gm__ AicoreReadyQueueV0 *completion_queue =
            aicore_sidecar_at_v0<AicoreReadyQueueV0>(sidecar_base, worker_context->completion_queue_offset);
        uint64_t local_completed_delta = 0;
        while (aicore_gm_load_v0(run_control->classification_error) == 0) {
            if (worker_context->resolver_active != 0) {
                int64_t completed_task_id = AICORE_TASK_ID_INVALID_V0;
                uint64_t wake_start = get_sys_cnt_aicore();
                if (aicore_ready_queue_pop_v0(sidecar_base, completion_queue, &completed_task_id)) {
                    __gm__ ChipSwimlaneAicoreTaskRecord *resolve_profile_record =
                        aicore_task_profiling_reserve_v0(&task_profiling, worker_context);
                    bool resolved = aicore_complete_and_wake_v0(
                        graph, sidecar_base, worker_context, run_control, completed_task_id
                    );
                    uint64_t wake_end = get_sys_cnt_aicore();
                    aicore_task_profiling_commit_resolve_v0(
                        resolve_profile_record, worker_context, static_cast<uint64_t>(completed_task_id), wake_start,
                        wake_end
                    );
                    if (!resolved) {
                        break;
                    }
                    ++local_completed_delta;
                    aicore_gm_fetch_add_v0(worker_context->resolved_task_count, UINT64_C(1));
                    aicore_gm_fetch_add_v0(worker_context->completion_pop_count, UINT64_C(1));
                    aicore_gm_fetch_add_v0(worker_context->wake_cycles, wake_end - wake_start);
                    continue;
                }
                aicore_gm_fetch_add_v0(worker_context->completion_empty_count, UINT64_C(1));
                if (local_completed_delta != 0) {
                    aicore_gm_fetch_add_v0(run_control->completed_count, local_completed_delta);
                    aicore_gm_fetch_add_v0(worker_context->resolved_flush_count, UINT64_C(1));
                    aicore_gm_fetch_add_v0(worker_context->resolved_flush_tasks, local_completed_delta);
                    local_completed_delta = 0;
                }
            }

            uint64_t ready_start = get_sys_cnt_aicore();
            int64_t task_id = AICORE_TASK_ID_INVALID_V0;
            if (worker_context->active == 0 || !aicore_ready_queue_pop_v0(sidecar_base, ready_queue, &task_id)) {
                aicore_gm_fetch_add_v0(worker_context->poll_count, UINT64_C(1));
                aicore_gm_fetch_add_v0(worker_context->ready_empty_count, UINT64_C(1));
                if (aicore_gm_load_v0(run_control->completed_count) == graph.task_count) break;
                SPIN_WAIT_HINT();
                continue;
            }

            aicore_gm_fetch_add_v0(worker_context->ready_pop_count, UINT64_C(1));
            uint64_t receive_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
            uint64_t payload_start = get_sys_cnt_aicore();
            aicore_observe_data_cache_v0(aicore_graph_descriptor_v0(graph, task_id));
            AicoreTaskInfoV0 task{};
            AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
            if (status != AicoreRootStatusV0::OK || task.kernel_id >= RUNTIME_MAX_FUNC_ID ||
                static_cast<uint64_t>(task.core_type) != static_cast<uint64_t>(core_type)) {
                aicore_record_scheduler_error_v0(
                    run_control, task_id,
                    status == AicoreRootStatusV0::OK ? AicoreRootStatusV0::UNSUPPORTED_SHAPE : status
                );
                break;
            }

            dcci(&runtime->func_id_to_addr_[task.kernel_id], SINGLE_CACHE_LINE);
            uint64_t callable_address = runtime->func_id_to_addr_[task.kernel_id];
            if (callable_address != 0) {
                dcci(reinterpret_cast<__gm__ void *>(callable_address), ENTIRE_DATA_CACHE);
            }
            __gm__ PTO2DispatchPayload *dispatch_payload =
                aicore_sidecar_at_v0<PTO2DispatchPayload>(sidecar_base, worker_context->dispatch_payload_offset);
            status = aicore_materialize_task_payload_v0(graph, task, callable_address, dispatch_payload);
            if (status != AicoreRootStatusV0::OK) {
                aicore_record_scheduler_error_v0(run_control, task_id, status);
                break;
            }

            OUT_OF_ORDER_STORE_BARRIER();
            __gm__ ChipSwimlaneAicoreTaskRecord *profile_record =
                aicore_task_profiling_reserve_v0(&task_profiling, worker_context);
            uint64_t kernel_start = get_sys_cnt_aicore();
            execute_task(dispatch_payload);
            uint64_t kernel_end = get_sys_cnt_aicore();
            aicore_publish_data_cache_v0(dispatch_payload);
            aicore_task_profiling_commit_v0(
                profile_record, worker_context, static_cast<uint64_t>(task_id), receive_time, kernel_start, kernel_end
            );
            if (!aicore_ready_queue_push_v0(sidecar_base, completion_queue, task_id)) {
                aicore_gm_fetch_add_v0(worker_context->completion_queue_full_count, UINT64_C(1));
                aicore_record_scheduler_error_v0(run_control, task_id, AicoreRootStatusV0::QUEUE_FULL);
                break;
            }
            uint64_t completion_end = get_sys_cnt_aicore();

            aicore_gm_fetch_add_v0(worker_context->completion_push_count, UINT64_C(1));
            aicore_gm_fetch_add_v0(worker_context->executed_task_count, UINT64_C(1));
            aicore_gm_fetch_add_v0(worker_context->ready_to_start_cycles, payload_start - ready_start);
            aicore_gm_fetch_add_v0(worker_context->payload_cycles, kernel_start - payload_start);
            aicore_gm_fetch_add_v0(worker_context->kernel_cycles, kernel_end - kernel_start);
            aicore_gm_fetch_add_v0(worker_context->completion_cycles, completion_end - kernel_end);
        }
    }

    while (aicore_gm_load_v0(run_control->exit_requested) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
    OUT_OF_ORDER_STORE_BARRIER();
    aicore_gm_fetch_add_v0(run_control->finished_count, UINT64_C(1));
}
