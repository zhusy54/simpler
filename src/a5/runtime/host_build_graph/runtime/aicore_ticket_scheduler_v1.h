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

#include "aicore_execution_sidecar_v1.h"
#include "aicore_gm_atomic.h"
#include "aicore_graph_view_v0.h"

enum class AicorePendingStateV1 : uint64_t {
    EMPTY = 0,
    BLOCKED = 1,
    READY = 2,
    ERROR = 3,
};

enum class AicoreRouteResultV1 : uint64_t {
    READY = 0,
    WAITING = 1,
    COMPLETED = 2,
    ERROR = 3,
};

struct AicoreWakeStatsV1 {
    uint64_t fanin_state_load_count{0};
    uint64_t wake_register_count{0};
    uint64_t wake_cas_retry_count{0};
    uint64_t wake_closed_retry_count{0};
    uint64_t wake_migrate_count{0};
    uint64_t wake_close_count{0};
};

struct AicoreCompletionStatsV1 {
    uint64_t enqueue_count{0};
    uint64_t batch_count{0};
    uint64_t resolve_count{0};
    uint64_t steal_count{0};
    uint64_t link_wait_count{0};
    uint64_t link_wait_max{0};
    uint64_t completion_lag_cycles{0};
    uint64_t completion_lag_max_cycles{0};
    uint64_t ready_to_kernel_cycles{0};
    uint64_t ready_to_kernel_max_cycles{0};
};

inline __host__ __aicore__ uint64_t aicore_scheduler_cycles_v1() {
#if defined(__CCE_AICORE__)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

struct AicorePendingSlotV1 {
    int64_t task_id;
    int32_t fanin_count;
    int32_t next_fanin_index;
    int32_t waiting_producer;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t reserved;
    uint64_t stream_index;
    uint64_t claim_start_cycles;
    uint64_t claim_end_cycles;
    uint64_t pending_wait_start_cycles;
    AicoreClaimKindV1 claim_kind;
};

static_assert(sizeof(AicorePendingSlotV1) == 64, "pending slot layout changed");

inline __host__ __aicore__ void aicore_pending_clear_v1(AicorePendingSlotV1 *slot) {
    if (slot == nullptr) return;
    *slot = {
        AICORE_TASK_ID_INVALID_V1,
        0,
        0,
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        UINT16_MAX,
        UINT8_MAX,
        0,
        0,
        0,
        0,
        0,
        AicoreClaimKindV1::SEED,
    };
}

inline __host__ __aicore__ __gm__ AicoreTaskControlV1 *
aicore_task_control_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, int64_t task_id) {
    return aicore_sidecar_at_v1<AicoreTaskControlV1>(
        sidecar_base,
        context->task_controls_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskControlV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreCompletionInboxV1 *aicore_completion_inbox_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t inbox_index
) {
    return aicore_sidecar_at_v1<AicoreCompletionInboxV1>(
        sidecar_base, context->completion_inboxes_offset + inbox_index * sizeof(AicoreCompletionInboxV1)
    );
}

inline __aicore__ void aicore_publish_waiter_metadata_v1(
    __gm__ AicoreTaskControlV1 *control, int64_t next_waiter, int32_t next_fanin_index, int32_t waiting_producer
) {
    control->next_waiter = next_waiter;
    control->next_fanin_index = next_fanin_index;
    control->waiting_producer = waiting_producer;
    aicore_publish_cache_line_v0(&control->next_waiter);
}

inline __aicore__ int64_t aicore_observe_next_waiter_v1(__gm__ AicoreTaskControlV1 *control) {
    aicore_observe_cache_line_v0(&control->next_waiter);
    return control->next_waiter;
}

inline __aicore__ void aicore_record_scheduler_error_v1(
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreRootStatusV0 status,
    const AicoreReadonlyGraphV0 *graph = nullptr, __gm__ const AicoreWorkerContextV1 *context = nullptr
) {
    if (aicore_gm_compare_exchange_v0(run_control->error_claimed, UINT64_C(0), UINT64_C(1)) != 0) return;
    aicore_gm_store_v0(run_control->error_task_id, static_cast<uint64_t>(task_id));
    if (graph != nullptr) {
        aicore_gm_store_v0(run_control->error_graph_task_count, graph->task_count);
        aicore_gm_store_v0(run_control->error_descriptors_address, graph->descriptors_address);
        aicore_gm_store_v0(run_control->error_payloads_address, graph->payloads_address);
        aicore_gm_store_v0(run_control->error_task_window_mask, graph->task_window_mask);
    }
    if (context != nullptr) {
        aicore_gm_store_v0(run_control->error_core_id, static_cast<uint64_t>(context->physical_core_id));
        aicore_gm_store_v0(run_control->error_core_type, static_cast<uint64_t>(context->core_type));
    }
    aicore_gm_publish_v0(run_control->scheduler_error, static_cast<uint64_t>(status));
}

inline __host__ __aicore__ AicoreRootStatusV0 aicore_pending_initialize_v1(
    const AicoreReadonlyGraphV0 &graph, int64_t task_id, AicoreRootCoreTypeV0 expected_core_type, int32_t max_func_id,
    uint64_t stream_index, AicoreClaimKindV1 claim_kind, uint64_t claim_start_cycles, uint64_t claim_end_cycles,
    AicorePendingSlotV1 *slot
) {
    if (slot == nullptr) return AicoreRootStatusV0::INVALID_ARGUMENTS;
    AicoreTaskInfoV0 task{};
    AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
    if (status != AicoreRootStatusV0::OK) return status;
    if (task.core_type != expected_core_type || task.kernel_id >= max_func_id ||
        task.kernel_id > static_cast<int32_t>(UINT16_MAX)) {
        return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    *slot = {
        task_id,
        *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0),
        0,
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        static_cast<uint16_t>(task.kernel_id),
        static_cast<uint8_t>(task.subtask_slot),
        0,
        stream_index,
        claim_start_cycles,
        claim_end_cycles,
        0,
        claim_kind,
    };
    return AicoreRootStatusV0::OK;
}

inline __aicore__ AicoreRouteResultV1 aicore_route_task_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats, bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    int64_t current_state = aicore_gm_load_v0(control->state);
    if (current_state == static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
        return AicoreRouteResultV1::COMPLETED;
    }
    if (current_state == static_cast<int64_t>(AicoreTaskStateV1::READY)) {
        return AicoreRouteResultV1::READY;
    }
    if (current_state != static_cast<int64_t>(AicoreTaskStateV1::BLOCKED)) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }

    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    int32_t next_fanin = control->next_fanin_index;
    if (fanin_count < 0 || next_fanin < 0 || next_fanin > fanin_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_FANIN_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }

    while (next_fanin < fanin_count) {
        int32_t producer = aicore_graph_fanin_id_v0(graph, task_id, next_fanin);
        if (producer < 0 || producer >= task_id) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_FANIN_ID, &graph, context
            );
            return AicoreRouteResultV1::ERROR;
        }
        if (stats != nullptr) ++stats->fanin_state_load_count;
        __gm__ AicoreTaskControlV1 *producer_control =
            aicore_task_control_at_v1(sidecar_base, context, producer);
        if (aicore_gm_load_v0(producer_control->state) == static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
            ++next_fanin;
            continue;
        }

        while (true) {
            int64_t observed = aicore_gm_load_v0(producer_control->wake_list_head);
            if (observed == AICORE_WAKE_LIST_CLOSED_V1) {
                if (stats != nullptr) ++stats->wake_closed_retry_count;
                break;
            }
            aicore_publish_waiter_metadata_v1(control, observed, next_fanin, producer);
            int64_t actual = aicore_gm_compare_exchange_v0(producer_control->wake_list_head, observed, task_id);
            if (actual == observed) {
                if (stats != nullptr) ++stats->wake_register_count;
                return AicoreRouteResultV1::WAITING;
            }
            if (stats != nullptr) ++stats->wake_cas_retry_count;
            if (actual == AICORE_WAKE_LIST_CLOSED_V1) {
                if (stats != nullptr) ++stats->wake_closed_retry_count;
                break;
            }
        }
    }

    aicore_publish_waiter_metadata_v1(
        control, AICORE_TASK_ID_INVALID_V1, fanin_count, static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1)
    );
    if (trace_enabled) {
        control->ready_publish_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&control->next_waiter);
    }
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    return AicoreRouteResultV1::READY;
}

inline __aicore__ AicorePendingStateV1 aicore_pending_state_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, AicorePendingSlotV1 *slot
) {
    if (slot == nullptr || slot->task_id < 0) return AicorePendingStateV1::EMPTY;
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, slot->task_id);
    int64_t state = aicore_gm_load_v0(control->state);
    if (state == static_cast<int64_t>(AicoreTaskStateV1::BLOCKED)) {
        aicore_observe_cache_line_v0(&control->next_waiter);
        slot->next_fanin_index = control->next_fanin_index;
        slot->waiting_producer = control->waiting_producer;
        return AicorePendingStateV1::BLOCKED;
    }
    if (state == static_cast<int64_t>(AicoreTaskStateV1::READY)) return AicorePendingStateV1::READY;
    return AicorePendingStateV1::ERROR;
}

inline __aicore__ bool aicore_claim_ticket_v1(
    __gm__ void *sidecar_base, __gm__ AicoreTaskStreamV1 *stream, uint64_t *stream_index, int64_t *task_id
) {
    if (stream_index == nullptr || task_id == nullptr) return false;
    uint64_t index = aicore_gm_fetch_add_v0(stream->next_index, UINT64_C(1));
    *stream_index = index;
    if (index >= stream->task_count) {
        *task_id = AICORE_TASK_ID_INVALID_V1;
        return false;
    }
    __gm__ uint32_t *task_ids = aicore_sidecar_at_v1<uint32_t>(sidecar_base, stream->task_ids_offset);
    *task_id = static_cast<int64_t>(task_ids[index]);
    return true;
}

inline __aicore__ bool aicore_complete_and_wake_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    int64_t waiter = aicore_gm_exchange_v0(control->wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    if (stats != nullptr) ++stats->wake_close_count;

    while (waiter >= 0) {
        if (static_cast<uint64_t>(waiter) >= graph.task_count) {
            aicore_record_scheduler_error_v1(
                run_control, waiter, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context
            );
            return false;
        }
        __gm__ AicoreTaskControlV1 *waiter_control = aicore_task_control_at_v1(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v1(waiter_control);
        if (stats != nullptr) ++stats->wake_migrate_count;
        if (aicore_route_task_v1(graph, sidecar_base, context, run_control, waiter, stats) ==
            AicoreRouteResultV1::ERROR) {
            return false;
        }
        waiter = next;
    }
    return true;
}


inline __aicore__ bool aicore_enqueue_completion_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreCompletionStatsV1 *stats,
    bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return false;
    }
    const uint64_t active_workers = aicore_gm_load_v0(run_control->active_worker_count);
    if (active_workers == 0) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    int64_t observed = aicore_gm_compare_exchange_v0(
        control->state, static_cast<int64_t>(AicoreTaskStateV1::READY), static_cast<int64_t>(AicoreTaskStateV1::DONE)
    );
    if (observed != static_cast<int64_t>(AicoreTaskStateV1::READY)) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    if (trace_enabled) control->completion_enqueue_cycles = aicore_scheduler_cycles_v1();
    uint64_t inbox_index = static_cast<uint64_t>(task_id) % active_workers;
    __gm__ AicoreCompletionInboxV1 *inbox = aicore_completion_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t previous = aicore_gm_exchange_v0(inbox->head, task_id);
    control->completion_next = previous;
    aicore_publish_cache_line_v0(&control->next_waiter);
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool aicore_resolve_completion_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *wake_stats,
    AicoreCompletionStatsV1 *completion_stats = nullptr, bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (aicore_gm_load_v0(control->state) != static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    uint64_t resolve_start = 0;
    if (trace_enabled) {
        resolve_start = aicore_scheduler_cycles_v1();
        control->completion_resolve_start_cycles = resolve_start;
        control->resolver_worker_id = context->worker_index;
        if (control->completion_enqueue_cycles != 0 && resolve_start >= control->completion_enqueue_cycles &&
            completion_stats != nullptr) {
            uint64_t lag = resolve_start - control->completion_enqueue_cycles;
            completion_stats->completion_lag_cycles += lag;
            if (lag > completion_stats->completion_lag_max_cycles) {
                completion_stats->completion_lag_max_cycles = lag;
            }
        }
    }
    int64_t waiter = aicore_gm_exchange_v0(control->wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    if (waiter == AICORE_WAKE_LIST_CLOSED_V1) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    if (wake_stats != nullptr) ++wake_stats->wake_close_count;
    while (waiter >= 0) {
        if (static_cast<uint64_t>(waiter) >= graph.task_count) {
            aicore_record_scheduler_error_v1(
                run_control, waiter, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context
            );
            return false;
        }
        __gm__ AicoreTaskControlV1 *waiter_control = aicore_task_control_at_v1(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v1(waiter_control);
        if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        if (aicore_route_task_v1(graph, sidecar_base, context, run_control, waiter, wake_stats, trace_enabled) ==
            AicoreRouteResultV1::ERROR) {
            return false;
        }
        waiter = next;
    }
    if (trace_enabled) {
        control->completion_resolve_end_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&control->next_waiter);
    }
    return true;
}

inline __aicore__ bool aicore_service_completion_inboxes_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t *victim_cursor, AicoreWakeStatsV1 *wake_stats,
    AicoreCompletionStatsV1 *completion_stats, bool *made_progress, bool trace_enabled = false
) {
    if (made_progress != nullptr) *made_progress = false;
    const uint64_t active_workers = aicore_gm_load_v0(run_control->active_worker_count);
    if (active_workers == 0 || victim_cursor == nullptr) return true;
    uint64_t inbox_index = context->inbox_index;
    __gm__ AicoreCompletionInboxV1 *inbox = aicore_completion_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t task_id = AICORE_COMPLETION_INBOX_EMPTY_V1;
    if (aicore_gm_load_v0(inbox->head) != AICORE_COMPLETION_INBOX_EMPTY_V1) {
        task_id = aicore_gm_exchange_v0(inbox->head, AICORE_COMPLETION_INBOX_EMPTY_V1);
    }
    if (task_id == AICORE_COMPLETION_INBOX_EMPTY_V1 && active_workers > 1) {
        uint64_t victim = *victim_cursor % active_workers;
        *victim_cursor = (victim + 1) % active_workers;
        if (victim == inbox_index) {
            victim = *victim_cursor;
            *victim_cursor = (victim + 1) % active_workers;
        }
        inbox = aicore_completion_inbox_at_v1(sidecar_base, context, victim);
        if (aicore_gm_load_v0(inbox->head) != AICORE_COMPLETION_INBOX_EMPTY_V1) {
            task_id = aicore_gm_exchange_v0(inbox->head, AICORE_COMPLETION_INBOX_EMPTY_V1);
        }
        if (task_id != AICORE_COMPLETION_INBOX_EMPTY_V1 && completion_stats != nullptr) {
            ++completion_stats->steal_count;
        }
    }
    if (task_id == AICORE_COMPLETION_INBOX_EMPTY_V1) return true;
    if (completion_stats != nullptr) ++completion_stats->batch_count;
    uint64_t batch_count = 0;
    while (task_id >= 0) {
        if (static_cast<uint64_t>(task_id) >= graph.task_count) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context
            );
            return false;
        }
        __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
        int64_t next = aicore_gm_load_v0(control->completion_next);
        uint64_t waits = 0;
        while (next == AICORE_COMPLETION_LINK_UNPUBLISHED_V1 && waits < UINT64_C(1048576)) {
            ++waits;
            aicore_observe_cache_line_v0(&control->next_waiter);
            next = aicore_gm_load_v0(control->completion_next);
        }
        if (completion_stats != nullptr) completion_stats->link_wait_count += waits;
        if (completion_stats != nullptr && waits > completion_stats->link_wait_max) {
            completion_stats->link_wait_max = waits;
        }
        if (next == AICORE_COMPLETION_LINK_UNPUBLISHED_V1 || next < AICORE_COMPLETION_INBOX_EMPTY_V1) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context
            );
            return false;
        }
        if (!aicore_resolve_completion_v1(
                graph, sidecar_base, context, run_control, task_id, wake_stats, completion_stats, trace_enabled
            )) {
            return false;
        }
        ++batch_count;
        task_id = next;
    }
    if (completion_stats != nullptr) completion_stats->resolve_count += batch_count;
    aicore_gm_fetch_add_v0(run_control->resolved_task_count, batch_count);
    if (made_progress != nullptr) *made_progress = true;
    return true;
}
