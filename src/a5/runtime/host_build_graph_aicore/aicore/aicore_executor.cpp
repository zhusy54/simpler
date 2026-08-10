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
#include "common/platform_config.h"
#include "aicore_dependency_scheduler_v0.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[block_idx]);
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    dcci(my_hank, SINGLE_CACHE_LINE);
    __gm__ AicoreWorkerContextV0 *worker_context = reinterpret_cast<__gm__ AicoreWorkerContextV0 *>(my_hank->task);
    dcci(worker_context, ENTIRE_DATA_CACHE);
    worker_context->physical_core_id = static_cast<int32_t>(my_hank->physical_core_id);
    worker_context->core_type = static_cast<int32_t>(core_type);
    OUT_OF_ORDER_STORE_BARRIER();
    dcci(worker_context, ENTIRE_DATA_CACHE, CACHELINE_OUT);

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

    if (block_idx == 0 && graph.task_count != 0) {
        uint64_t classify_start = get_sys_cnt_aicore();
        dcci(reinterpret_cast<__gm__ void *>(graph.descriptors_address), ENTIRE_DATA_CACHE);
        dcci(reinterpret_cast<__gm__ void *>(graph.payloads_address), ENTIRE_DATA_CACHE);
        for (int64_t task_id = 0; task_id < static_cast<int64_t>(graph.task_count); ++task_id) {
            AicoreRouteResultV0 result =
                aicore_classify_and_route_v0(graph, sidecar_base, worker_context, run_control, task_id);
            if (result == AicoreRouteResultV0::ERROR) break;
            if (result == AicoreRouteResultV0::READY) {
                aicore_gm_fetch_add_v0(run_control->initial_ready_count, UINT64_C(1));
            } else {
                aicore_gm_fetch_add_v0(run_control->initial_waiting_count, UINT64_C(1));
            }
        }
        aicore_gm_fetch_add_v0(run_control->classify_cycles, get_sys_cnt_aicore() - classify_start);
    }
    aicore_gm_fetch_add_v0(run_control->classified_count, UINT64_C(1));

    while (aicore_gm_load_v0(run_control->startup_count) == 0 && aicore_gm_load_v0(run_control->exit_requested) == 0) {
        SPIN_WAIT_HINT();
    }

    dcci(worker_context, ENTIRE_DATA_CACHE);
    if (worker_context->active != 0 && aicore_gm_load_v0(run_control->exit_requested) == 0) {
        __gm__ AicoreReadyQueueV0 *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(
            sidecar_base,
            core_type == CoreType::AIC ? worker_context->aic_queue_offset : worker_context->aiv_queue_offset
        );
        while (aicore_gm_load_v0(run_control->classification_error) == 0) {
            uint64_t ready_start = get_sys_cnt_aicore();
            int64_t task_id = AICORE_TASK_ID_INVALID_V0;
            if (!aicore_ready_queue_pop_v0(sidecar_base, queue, &task_id)) {
                ++worker_context->poll_count;
                aicore_gm_fetch_add_v0(run_control->empty_scan_count, UINT64_C(1));
                if (worker_context->local_completed_delta != 0) {
                    aicore_gm_fetch_add_v0(run_control->completed_count, worker_context->local_completed_delta);
                    worker_context->local_completed_delta = 0;
                }
                if (aicore_gm_load_v0(run_control->completed_count) == graph.task_count) break;
                SPIN_WAIT_HINT();
                continue;
            }

            aicore_gm_fetch_add_v0(run_control->queue_pop_count, UINT64_C(1));
            uint64_t payload_start = get_sys_cnt_aicore();
            dcci(aicore_graph_descriptor_v0(graph, task_id), ENTIRE_DATA_CACHE);
            dcci(aicore_graph_payload_v0(graph, task_id), ENTIRE_DATA_CACHE);
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
            uint64_t kernel_start = get_sys_cnt_aicore();
            execute_task(dispatch_payload);
            uint64_t kernel_end = get_sys_cnt_aicore();
            uint64_t wake_start = kernel_end;
            if (!aicore_complete_and_wake_v0(graph, sidecar_base, worker_context, run_control, task_id)) break;
            uint64_t completion_end = get_sys_cnt_aicore();

            ++worker_context->local_completed_delta;
            ++worker_context->executed_task_count;
            aicore_gm_fetch_add_v0(run_control->ready_to_start_cycles, payload_start - ready_start);
            aicore_gm_fetch_add_v0(run_control->payload_cycles, kernel_start - payload_start);
            aicore_gm_fetch_add_v0(run_control->kernel_cycles, kernel_end - kernel_start);
            aicore_gm_fetch_add_v0(run_control->completion_cycles, completion_end - kernel_end);
            aicore_gm_fetch_add_v0(run_control->wake_cycles, completion_end - wake_start);
            aicore_gm_fetch_add_v0(run_control->executed_count, UINT64_C(1));
        }
        OUT_OF_ORDER_STORE_BARRIER();
        dcci(worker_context, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    while (aicore_gm_load_v0(run_control->exit_requested) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
    OUT_OF_ORDER_STORE_BARRIER();
    aicore_gm_fetch_add_v0(run_control->finished_count, UINT64_C(1));
}
