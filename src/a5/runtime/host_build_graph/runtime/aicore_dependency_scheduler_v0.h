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

#include "aicore_execution_sidecar_v0.h"
#include "aicore_gm_atomic.h"
#include "aicore_graph_view_v0.h"
#include "aicore_ready_queue_v0.h"

enum class AicoreRouteResultV0 : uint64_t {
    READY = 0,
    WAITING = 1,
    COMPLETED = 2,
    ERROR = 3,
};

inline __aicore__ __gm__ AicoreTaskControlV0 *
aicore_task_control_at_v0(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV0 *context, int64_t task_id) {
    return aicore_sidecar_at_v0<AicoreTaskControlV0>(
        sidecar_base, context->task_controls_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskControlV0)
    );
}

inline __aicore__ void aicore_record_scheduler_error_v0(
    __gm__ AicoreRunControlV0 *run_control, int64_t task_id, AicoreRootStatusV0 status,
    const AicoreReadonlyGraphV0 *graph = nullptr, __gm__ const AicoreWorkerContextV0 *context = nullptr
) {
    // Preserve the first failure and publish its diagnostics before the status
    // flag makes AICPU stop the graph. Concurrent failures must not overwrite
    // the task id with a less useful secondary symptom.
    if (aicore_gm_compare_exchange_v0(run_control->reserved[0], UINT64_C(0), UINT64_C(1)) != 0) return;
    aicore_gm_store_v0(run_control->error_task_id, static_cast<uint64_t>(task_id));
    if (graph != nullptr) {
        aicore_gm_store_v0(run_control->reserved[1], graph->task_count);
        aicore_gm_store_v0(run_control->reserved[2], graph->descriptors_address);
        aicore_gm_store_v0(run_control->reserved[3], graph->payloads_address);
        aicore_gm_store_v0(run_control->reserved[4], graph->task_window_mask);
    }
    if (context != nullptr) {
        aicore_gm_store_v0(run_control->reserved[5], static_cast<uint64_t>(context->physical_core_id));
        aicore_gm_store_v0(run_control->reserved[6], static_cast<uint64_t>(context->core_type));
    }
    aicore_gm_publish_v0(run_control->classification_error, static_cast<uint64_t>(status));
}

inline __aicore__ AicoreRouteResultV0 aicore_classify_and_route_v0(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV0 *context,
    __gm__ AicoreRunControlV0 *run_control, int64_t task_id
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v0(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return AicoreRouteResultV0::ERROR;
    }
    __gm__ AicoreTaskControlV0 *task_control = aicore_task_control_at_v0(sidecar_base, context, task_id);
    if (aicore_gm_load_v0(task_control->completion) != 0) {
        return AicoreRouteResultV0::COMPLETED;
    }

    AicoreTaskInfoV0 task{};
    AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
    if (status != AicoreRootStatusV0::OK) {
        aicore_record_scheduler_error_v0(run_control, task_id, status, &graph, context);
        return AicoreRouteResultV0::ERROR;
    }

    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    while (true) {
        aicore_gm_fetch_add_v0(context->fanin_scan_count, UINT64_C(1));
        int64_t unmet_producer = AICORE_TASK_ID_INVALID_V0;
        for (int32_t i = 0; i < fanin_count; ++i) {
            aicore_gm_fetch_add_v0(context->fanin_edge_count, UINT64_C(1));
            int64_t producer = aicore_graph_fanin_id_v0(graph, task_id, i);
            __gm__ AicoreTaskControlV0 *producer_control = aicore_task_control_at_v0(sidecar_base, context, producer);
            if (aicore_gm_load_v0(producer_control->completion) == 0) {
                unmet_producer = producer;
                break;
            }
        }

        if (unmet_producer == AICORE_TASK_ID_INVALID_V0) {
            __gm__ AicoreReadyQueueV0 *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(
                sidecar_base,
                task.core_type == AicoreRootCoreTypeV0::AIC ? context->aic_queue_offset : context->aiv_queue_offset
            );
            if (!aicore_ready_queue_push_v0(sidecar_base, queue, task_id)) {
                aicore_gm_fetch_add_v0(context->ready_queue_full_count, UINT64_C(1));
                aicore_record_scheduler_error_v0(run_control, task_id, AicoreRootStatusV0::QUEUE_FULL, &graph, context);
                return AicoreRouteResultV0::ERROR;
            }
            aicore_gm_fetch_add_v0(context->ready_push_count, UINT64_C(1));
            return AicoreRouteResultV0::READY;
        }

        __gm__ AicoreTaskControlV0 *producer_control = aicore_task_control_at_v0(sidecar_base, context, unmet_producer);
        while (true) {
            int64_t observed = aicore_gm_load_v0(producer_control->wake_list_head);
            if (observed == AICORE_WAKE_LIST_CLOSED_V0) {
                aicore_gm_fetch_add_v0(context->wake_closed_retry_count, UINT64_C(1));
                break;
            }
            aicore_publish_next_waiter_v0(task_control, observed);
            int64_t actual = aicore_gm_compare_exchange_v0(producer_control->wake_list_head, observed, task_id);
            if (actual == observed) {
                aicore_gm_fetch_add_v0(context->wake_register_count, UINT64_C(1));
                return AicoreRouteResultV0::WAITING;
            }
        }
    }
}

inline __aicore__ bool aicore_complete_and_wake_v0(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV0 *context,
    __gm__ AicoreRunControlV0 *run_control, int64_t task_id
) {
    __gm__ AicoreTaskControlV0 *task_control = aicore_task_control_at_v0(sidecar_base, context, task_id);
    aicore_gm_store_v0(task_control->completion, INT64_C(1));
    int64_t waiter = aicore_gm_exchange_v0(task_control->wake_list_head, AICORE_WAKE_LIST_CLOSED_V0);
    aicore_gm_fetch_add_v0(context->wake_close_count, UINT64_C(1));

    while (waiter >= 0) {
        if (static_cast<uint64_t>(waiter) >= graph.task_count) {
            aicore_record_scheduler_error_v0(run_control, waiter, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
            return false;
        }
        __gm__ AicoreTaskControlV0 *waiter_control = aicore_task_control_at_v0(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v0(waiter_control);
        aicore_gm_fetch_add_v0(context->wake_reclassify_count, UINT64_C(1));
        if (aicore_classify_and_route_v0(graph, sidecar_base, context, run_control, waiter) ==
            AicoreRouteResultV0::ERROR) {
            return false;
        }
        waiter = next;
    }
    return true;
}
