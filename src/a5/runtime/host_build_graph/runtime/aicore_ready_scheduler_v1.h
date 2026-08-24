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

enum class AicoreRouteResultV1 : uint64_t {
    READY = 0,
    READY_TO_ENQUEUE = 1,
    WAITING = 2,
    COMPLETED = 3,
    ERROR = 4,
};

struct AicoreWakeStatsV1 {
    uint64_t fanin_state_load_count{0};
    uint64_t wake_register_count{0};
    uint64_t wake_cas_retry_count{0};
    uint64_t wake_closed_retry_count{0};
    uint64_t wake_migrate_count{0};
    uint64_t wake_close_count{0};
};

struct AicoreReadyStatsV1 {
    uint64_t enqueue_count{0};
    uint64_t batch_count{0};
    uint64_t pop_count{0};
    uint64_t steal_count{0};
    uint64_t cas_retry_count{0};
    uint64_t link_wait_count{0};
    uint64_t link_wait_max{0};
};

struct AicoreFreeSlotStatsV1 {
    uint64_t claim_count{0};
    uint64_t stale_count{0};
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

struct AicoreReadyBatchV1 {
    int64_t head{AICORE_INBOX_EMPTY_V1};
    int64_t tail{AICORE_INBOX_EMPTY_V1};
    uint64_t count{0};
};

struct AicoreReadyClaimV1 {
    int64_t task_id{AICORE_TASK_ID_INVALID_V1};
    uint64_t inbox_index{UINT64_MAX};
    AicoreReadySourceV1 source{AicoreReadySourceV1::LOCAL};
    uint64_t claim_start_cycles{0};
    uint64_t claim_end_cycles{0};
};

struct AicoreFreeSlotClaimV1 {
    uint64_t worker_id{UINT64_MAX};
    uint32_t slot_index{UINT32_MAX};
    uint32_t generation{0};
};

struct AicoreRefillTimingV1 {
    uint64_t ready_claim_cycles{0};
    uint64_t slot_fill_cycles{0};
    uint64_t free_advertise_cycles{0};
};

struct AicoreCompletionServiceTimingV1 {
    uint64_t scan_cycles{0};
    uint64_t consume_cycles{0};
    uint64_t resolve_cycles{0};
    uint64_t ready_publish_cycles{0};
    uint64_t refill_cycles{0};
    uint64_t finalize_cycles{0};
};

struct AicoreDispatchFillTimingV1 {
    uint64_t prepare_cycles{0};
    uint64_t materialize_cycles{0};
    uint64_t publish_cycles{0};
};

inline __host__ __aicore__ uint64_t aicore_scheduler_cycles_v1() {
#if defined(__CCE_AICORE__)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

inline __host__ __aicore__ uint32_t aicore_core_type_index_v1(int32_t core_type) {
    return core_type == static_cast<int32_t>(AicoreRootCoreTypeV0::AIC) ? 0U : 1U;
}

inline __host__ __aicore__ uint32_t aicore_metadata_core_type_index_v1(uint8_t subtask_slot) {
    return subtask_slot == 0 ? 0U : 1U;
}

inline __host__ __aicore__ uint8_t aicore_metadata_single_subtask_slot_v1(uint8_t active_mask) {
    if ((active_mask & 1U) != 0) return 0;
    if ((active_mask & 2U) != 0) return 1;
    return 2;
}

inline __host__ __aicore__ uint64_t
aicore_completion_id_v1(__gm__ const AicoreWorkerContextV1 *context, uint64_t local_completion_index) {
    return local_completion_index * context->runtime_worker_count + context->worker_index;
}

inline __host__ __aicore__ uint64_t aicore_completion_inbox_index_v1(
    __gm__ const AicoreWorkerContextV1 *context, uint64_t resolver_count, uint64_t local_completion_index
) {
    return resolver_count == 0 ? 0 : aicore_completion_id_v1(context, local_completion_index) % resolver_count;
}

inline __host__ __aicore__ __gm__ AicoreTaskControlV1 *
aicore_task_control_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, int64_t task_id) {
    return aicore_sidecar_at_v1<AicoreTaskControlV1>(
        sidecar_base, context->task_controls_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskControlV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreTaskMetadataV1 *
aicore_task_metadata_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, int64_t task_id) {
    return aicore_sidecar_at_v1<AicoreTaskMetadataV1>(
        sidecar_base, context->task_metadata_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskMetadataV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreCompletionInboxV1 *aicore_completion_inbox_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t inbox_index
) {
    return aicore_sidecar_at_v1<AicoreCompletionInboxV1>(
        sidecar_base, context->completion_inboxes_offset + inbox_index * sizeof(AicoreCompletionInboxV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreReadyInboxV1 *aicore_ready_inbox_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint32_t core_type_index,
    uint64_t inbox_index
) {
    uint64_t linear = static_cast<uint64_t>(core_type_index) * AICORE_WORKER_CAPACITY_V1 + inbox_index;
    return aicore_sidecar_at_v1<AicoreReadyInboxV1>(
        sidecar_base, context->ready_inboxes_offset + linear * sizeof(AicoreReadyInboxV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreReadyDirectoryV1 *
aicore_ready_directory_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context) {
    return aicore_sidecar_at_v1<AicoreReadyDirectoryV1>(sidecar_base, context->ready_directory_offset);
}

inline __host__ __aicore__ __gm__ AicoreFreeSlotDirectoryV1 *
aicore_free_slot_directory_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context) {
    return aicore_sidecar_at_v1<AicoreFreeSlotDirectoryV1>(sidecar_base, context->free_slot_directory_offset);
}

inline __host__ __aicore__ __gm__ AicoreGangCoordinatorV1 *
aicore_gang_coordinator_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context) {
    return aicore_sidecar_at_v1<AicoreGangCoordinatorV1>(sidecar_base, context->gang_coordinator_offset);
}

inline __aicore__ void aicore_publish_gang_ready_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, __gm__ AicoreTaskControlV1 *control,
    uint8_t metadata_flags
) {
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, context);
    aicore_gm_fetch_or_v0(coordinator->ready_priority_bits, aicore_task_priority_bit_v1(metadata_flags));
}

inline __host__ __aicore__ __gm__ AicoreTaskClaimBindingV1 *
aicore_claim_binding_at_v1(__gm__ void *sidecar_base, __gm__ const AicoreRunControlV1 *run_control, int64_t task_id) {
    return aicore_sidecar_at_v1<AicoreTaskClaimBindingV1>(
        sidecar_base,
        run_control->claim_bindings_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskClaimBindingV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreWorkerContextV1 *aicore_worker_context_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t worker_id
) {
    return aicore_sidecar_at_v1<AicoreWorkerContextV1>(
        sidecar_base, context->worker_contexts_offset + worker_id * sizeof(AicoreWorkerContextV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreDispatchSlotV1 *aicore_dispatch_slot_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t worker_id, uint32_t slot
) {
    return aicore_sidecar_at_v1<AicoreDispatchSlotV1>(
        sidecar_base,
        context->dispatch_slots_offset +
            (worker_id * AICORE_PENDING_SLOT_COUNT_V1 + static_cast<uint64_t>(slot)) * sizeof(AicoreDispatchSlotV1)
    );
}

inline __host__ __aicore__ uint64_t
aicore_dispatch_publication_v1(uint32_t generation, AicoreDispatchPublicationV1 state) {
    return (static_cast<uint64_t>(generation) << 8) | static_cast<uint64_t>(state);
}

inline __host__ __aicore__ uint32_t aicore_dispatch_generation_v1(uint64_t publication) {
    return static_cast<uint32_t>(publication >> 8);
}

inline __host__ __aicore__ AicoreDispatchPublicationV1 aicore_dispatch_state_v1(uint64_t publication) {
    return static_cast<AicoreDispatchPublicationV1>(publication & UINT64_C(0xff));
}

inline __aicore__ void aicore_record_scheduler_error_v1(
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreRootStatusV0 status,
    const AicoreReadonlyGraphV0 *graph = nullptr, __gm__ const AicoreWorkerContextV1 *context = nullptr,
    uint64_t error_site = 0
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
    aicore_gm_store_v0(run_control->error_reserved[0], error_site);
    aicore_gm_publish_v0(run_control->scheduler_error, static_cast<uint64_t>(status));
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

inline __aicore__ AicoreRouteResultV1 aicore_route_task_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats,
    bool validate_current_state = true
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (validate_current_state) {
        int64_t state = aicore_gm_query_v0(control->state);
        if (state == static_cast<int64_t>(AicoreTaskStateV1::DONE)) return AicoreRouteResultV1::COMPLETED;
        if (state == static_cast<int64_t>(AicoreTaskStateV1::READY)) return AicoreRouteResultV1::READY;
        if (state != static_cast<int64_t>(AicoreTaskStateV1::BLOCKED)) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(2)
            );
            return AicoreRouteResultV1::ERROR;
        }
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
        __gm__ AicoreTaskControlV1 *producer_control = aicore_task_control_at_v1(sidecar_base, context, producer);
        if (aicore_gm_query_v0(producer_control->state) == static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
            ++next_fanin;
            continue;
        }
        while (true) {
            int64_t observed = aicore_gm_query_v0(producer_control->wake_list_head);
            if (observed == AICORE_WAKE_LIST_CLOSED_V1) {
                if (stats != nullptr) ++stats->wake_closed_retry_count;
                ++next_fanin;
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
                ++next_fanin;
                break;
            }
        }
    }
    aicore_publish_waiter_metadata_v1(
        control, AICORE_TASK_ID_INVALID_V1, fanin_count, static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1)
    );
    return AicoreRouteResultV1::READY_TO_ENQUEUE;
}

// No task can execute while the bootstrap barrier is closed. Executable
// producers therefore have open wake lists, and non-executable producers are
// the inline-completed tasks initialized by the host. The barrier makes it
// safe to publish the new head before publishing the waiter's link.
inline __aicore__ AicoreRouteResultV1 aicore_bootstrap_route_task_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
    if (fanin_count < 0) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_FANIN_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }

    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    for (int32_t next_fanin = 0; next_fanin < fanin_count; ++next_fanin) {
        int32_t producer = aicore_graph_fanin_id_v0(graph, task_id, next_fanin);
        if (producer < 0 || producer >= task_id) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_FANIN_ID, &graph, context
            );
            return AicoreRouteResultV1::ERROR;
        }
        __gm__ AicoreTaskMetadataV1 *producer_metadata = aicore_task_metadata_at_v1(sidecar_base, context, producer);
        if (!aicore_task_is_executable_v1(producer_metadata->flags)) continue;

        __gm__ AicoreTaskControlV1 *producer_control = aicore_task_control_at_v1(sidecar_base, context, producer);
        int64_t previous = aicore_gm_exchange_v0(producer_control->wake_list_head, task_id);
        if (previous < AICORE_WAKE_LIST_OPEN_V1) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(60)
            );
            return AicoreRouteResultV1::ERROR;
        }
        control->next_waiter = previous;
        control->next_fanin_index = next_fanin;
        control->waiting_producer = producer;
        aicore_writeback_cache_line_v0(&control->next_waiter);
        if (stats != nullptr) ++stats->wake_register_count;
        return AicoreRouteResultV1::WAITING;
    }

    control->next_fanin_index = fanin_count;
    control->waiting_producer = static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1);
    return AicoreRouteResultV1::READY_TO_ENQUEUE;
}

inline __aicore__ bool aicore_bootstrap_ready_batch_append_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, int64_t task_id, AicoreReadyBatchV1 *batch,
    AicoreReadyStatsV1 *stats, bool trace_enabled = false
) {
    if (batch == nullptr || task_id < 0 || static_cast<uint64_t>(task_id) >= context->graph_task_count) return false;
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    if (batch->head == AICORE_INBOX_EMPTY_V1) {
        control->next_waiter = AICORE_INBOX_EMPTY_V1;
        batch->tail = task_id;
    } else {
        control->next_waiter = batch->head;
    }
    aicore_writeback_cache_line_v0(&control->next_waiter);
    if (trace_enabled) {
        __gm__ AicoreTaskTraceCellV1 *cells =
            aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
        cells[task_id].ready_transition_cycles = aicore_scheduler_cycles_v1();
        aicore_writeback_cache_line_v0(&cells[task_id].ready_transition_cycles);
    }
    batch->head = task_id;
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool aicore_bootstrap_ready_batch_publish_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint32_t core_type_index, uint64_t inbox_index,
    AicoreReadyBatchV1 *batch, AicoreReadyStatsV1 *stats, uint64_t *ready_types
) {
    if (batch == nullptr || batch->head == AICORE_INBOX_EMPTY_V1) return true;
    if (core_type_index >= AICORE_CORE_TYPE_COUNT_V1 || inbox_index >= AICORE_WORKER_CAPACITY_V1 || batch->tail < 0 ||
        ready_types == nullptr)
        return false;
    __gm__ AicoreReadyInboxV1 *inbox = aicore_ready_inbox_at_v1(sidecar_base, context, core_type_index, inbox_index);
    aicore_gm_store_v0(inbox->head, batch->head);
    *ready_types |= UINT64_C(1) << core_type_index;
    if (stats != nullptr) ++stats->batch_count;
    *batch = {};
    return true;
}

inline __aicore__ void aicore_bootstrap_ready_directory_publish_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint64_t resolver_count
) {
    __gm__ AicoreReadyDirectoryV1 *directory = aicore_ready_directory_at_v1(sidecar_base, context);
    for (uint64_t inbox_index = 0; inbox_index < resolver_count; inbox_index += 8)
        aicore_invalidate_cache_line_v0(&directory->bootstrap_ready_types[inbox_index]);
    aicore_cache_barrier_v0();
    uint32_t shard_count = static_cast<uint32_t>(
        (resolver_count + AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1 - 1) /
        AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1
    );
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        for (uint32_t shard = 0; shard < shard_count; ++shard) {
            uint64_t bits = 0;
            uint64_t shard_begin = static_cast<uint64_t>(shard) * AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
            uint64_t shard_end = shard_begin + AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
            if (shard_end > resolver_count) shard_end = resolver_count;
            for (uint64_t inbox_index = shard_begin; inbox_index < shard_end; ++inbox_index) {
                uint64_t ready_types = directory->bootstrap_ready_types[inbox_index];
                if ((ready_types & (UINT64_C(1) << type)) != 0) bits |= UINT64_C(1) << (inbox_index - shard_begin);
            }
            directory->core_types[type][shard].bits = bits;
            aicore_publish_cache_line_v0(&directory->core_types[type][shard]);
        }
    }
}

inline __aicore__ bool aicore_ready_batch_append_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, int64_t task_id, AicoreReadyBatchV1 *batch,
    AicoreReadyStatsV1 *stats, bool trace_enabled = false
) {
    if (batch == nullptr || task_id < 0 || static_cast<uint64_t>(task_id) >= context->graph_task_count) return false;
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    if (batch->head == AICORE_INBOX_EMPTY_V1) {
        control->next_waiter = AICORE_INBOX_LINK_UNPUBLISHED_V1;
        batch->tail = task_id;
    } else {
        control->next_waiter = batch->head;
    }
    aicore_publish_cache_line_v0(&control->next_waiter);
    if (trace_enabled) {
        __gm__ AicoreTaskTraceCellV1 *cells =
            aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
        __gm__ AicoreTaskTraceCellV1 *trace = &cells[task_id];
        aicore_observe_cache_line_v0(&trace->ready_transition_cycles);
        trace->ready_transition_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&trace->ready_transition_cycles);
    }
    batch->head = task_id;
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ void aicore_ready_directory_set_v1(
    __gm__ AicoreReadyDirectoryV1 *directory, uint32_t core_type_index, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    uint64_t bit = UINT64_C(1) << (inbox_index % AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1);
    aicore_gm_fetch_or_v0(directory->core_types[core_type_index][shard].bits, bit);
}

inline __aicore__ void aicore_ready_directory_clear_and_recheck_v1(
    __gm__ AicoreReadyDirectoryV1 *directory, __gm__ AicoreReadyInboxV1 *inbox, uint32_t core_type_index,
    uint64_t inbox_index
) {
    uint64_t shard = inbox_index / AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    uint64_t bit = UINT64_C(1) << (inbox_index % AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1);
    aicore_gm_fetch_and_v0(directory->core_types[core_type_index][shard].bits, ~bit);
    if (aicore_gm_query_v0(inbox->head) != AICORE_INBOX_EMPTY_V1)
        aicore_gm_fetch_or_v0(directory->core_types[core_type_index][shard].bits, bit);
}

inline __aicore__ bool aicore_ready_batch_push_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint32_t core_type_index, uint64_t inbox_index,
    AicoreReadyBatchV1 *batch, AicoreReadyStatsV1 *stats
) {
    if (batch == nullptr || batch->head == AICORE_INBOX_EMPTY_V1) return true;
    if (core_type_index >= AICORE_CORE_TYPE_COUNT_V1 || inbox_index >= AICORE_WORKER_CAPACITY_V1 || batch->tail < 0)
        return false;
    aicore_cache_barrier_v0();
    __gm__ AicoreReadyInboxV1 *inbox = aicore_ready_inbox_at_v1(sidecar_base, context, core_type_index, inbox_index);
    __gm__ AicoreTaskControlV1 *tail = aicore_task_control_at_v1(sidecar_base, context, batch->tail);
    int64_t previous = aicore_gm_query_v0(inbox->head);
    while (true) {
        if (previous < AICORE_INBOX_EMPTY_V1) return false;
        // Keep the link on the cache-coherent DCCI path. This line also carries
        // trace fields, so mixing raw-GM link updates with later whole-line
        // writeback could restore a stale UNPUBLISHED value.
        aicore_observe_cache_line_v0(&tail->next_waiter);
        tail->next_waiter = previous;
        aicore_publish_cache_line_v0(&tail->next_waiter);
        int64_t actual = aicore_gm_compare_exchange_v0(inbox->head, previous, batch->head);
        if (actual == previous) break;
        previous = actual;
        if (stats != nullptr) ++stats->cas_retry_count;
    }
    if (previous == AICORE_INBOX_EMPTY_V1) {
        aicore_ready_directory_set_v1(
            aicore_ready_directory_at_v1(sidecar_base, context), core_type_index, inbox_index
        );
    }
    if (stats != nullptr) ++stats->batch_count;
    *batch = {};
    return true;
}

inline __aicore__ bool aicore_ready_pop_from_inbox_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint32_t core_type_index, uint64_t inbox_index, int64_t *task_id,
    AicoreReadyStatsV1 *stats
) {
    if (task_id == nullptr) return false;
    *task_id = AICORE_TASK_ID_INVALID_V1;
    __gm__ AicoreReadyInboxV1 *inbox = aicore_ready_inbox_at_v1(sidecar_base, context, core_type_index, inbox_index);
    __gm__ AicoreReadyDirectoryV1 *directory = aicore_ready_directory_at_v1(sidecar_base, context);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        int64_t head = aicore_gm_query_v0(inbox->head);
        if (head == AICORE_INBOX_EMPTY_V1) {
            aicore_ready_directory_clear_and_recheck_v1(directory, inbox, core_type_index, inbox_index);
            return true;
        }
        if (head < 0 || static_cast<uint64_t>(head) >= graph.task_count) {
            aicore_record_scheduler_error_v1(
                run_control, head, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context, UINT64_C(40)
            );
            return false;
        }
        __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, head);
        int64_t next = aicore_observe_next_waiter_v1(control);
        uint64_t waits = 0;
        bool head_changed = false;
        while (next == AICORE_INBOX_LINK_UNPUBLISHED_V1 && waits < UINT64_C(1048576)) {
            if (aicore_gm_query_v0(inbox->head) != head) {
                head_changed = true;
                break;
            }
            ++waits;
            next = aicore_observe_next_waiter_v1(control);
        }
        if (stats != nullptr) {
            stats->link_wait_count += waits;
            if (waits > stats->link_wait_max) stats->link_wait_max = waits;
        }
        if (head_changed) {
            if (stats != nullptr) ++stats->cas_retry_count;
            continue;
        }
        if (next < AICORE_INBOX_EMPTY_V1) {
            aicore_record_scheduler_error_v1(
                run_control, head, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(41)
            );
            return false;
        }
        int64_t actual = aicore_gm_compare_exchange_v0(inbox->head, head, next);
        if (actual != head) {
            if (stats != nullptr) ++stats->cas_retry_count;
            continue;
        }
        if (next == AICORE_INBOX_EMPTY_V1)
            aicore_ready_directory_clear_and_recheck_v1(directory, inbox, core_type_index, inbox_index);
        if (stats != nullptr) ++stats->pop_count;
        *task_id = head;
        return true;
    }
    return true;
}

inline __aicore__ uint64_t aicore_load_ready_directory_shard_v1(
    __gm__ AicoreReadyDirectoryV1 *directory, uint64_t resolver_count, uint32_t core_type_index, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    uint64_t shard_begin = shard * AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    uint64_t shard_end = shard_begin + AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    if (shard_end > resolver_count) shard_end = resolver_count;
    uint64_t valid_bits = shard_end > shard_begin ? (UINT64_C(1) << (shard_end - shard_begin)) - 1 : 0;
    return aicore_gm_query_v0(directory->core_types[core_type_index][shard].bits) & valid_bits;
}

static __attribute__((noinline)) __aicore__ bool aicore_steal_ready_from_shard_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint32_t core_type_index, uint64_t shard_begin, uint64_t shard_end,
    uint64_t start, uint64_t *victim_cursor, uint64_t bits, AicoreReadyStatsV1 *stats, AicoreReadyClaimV1 *claim,
    bool trace_enabled
) {
    int64_t task_id = AICORE_TASK_ID_INVALID_V1;
    bits &= ~(UINT64_C(1) << (context->inbox_index - shard_begin));
    for (uint32_t pass = 0; pass < 2; ++pass) {
        uint64_t range_begin = pass == 0 ? start : shard_begin;
        uint64_t range_end = pass == 0 ? shard_end : start;
        if (range_begin == range_end) continue;
        uint32_t lower_bit = static_cast<uint32_t>(range_begin - shard_begin);
        uint32_t upper_bit = static_cast<uint32_t>(range_end - shard_begin);
        uint64_t candidates = bits & (((UINT64_C(1) << upper_bit) - 1) & ~((UINT64_C(1) << lower_bit) - 1));
        while (candidates != 0) {
            uint32_t bit_index = static_cast<uint32_t>(__builtin_ctzll(candidates));
            candidates &= candidates - 1;
            uint64_t victim = shard_begin + bit_index;
            *victim_cursor = victim + 1 == shard_end ? shard_begin : victim + 1;
            if (!aicore_ready_pop_from_inbox_v1(
                    graph, sidecar_base, context, run_control, core_type_index, victim, &task_id, stats
                ))
                return false;
            if (task_id >= 0) {
                claim->task_id = task_id;
                claim->inbox_index = victim;
                claim->source = AicoreReadySourceV1::STOLEN;
                claim->claim_end_cycles = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
                if (stats != nullptr) ++stats->steal_count;
                return true;
            }
        }
    }
    return true;
}

inline __aicore__ bool aicore_claim_ready_for_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint32_t core_type_index, uint64_t *victim_cursor,
    AicoreReadyStatsV1 *stats, AicoreReadyClaimV1 *claim, bool trace_enabled = false
) {
    if (victim_cursor == nullptr || claim == nullptr || resolver_count == 0) return false;
    *claim = {};
    claim->claim_start_cycles = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
    int64_t task_id = AICORE_TASK_ID_INVALID_V1;
    if (!aicore_ready_pop_from_inbox_v1(
            graph, sidecar_base, context, run_control, core_type_index, context->inbox_index, &task_id, stats
        ))
        return false;
    if (task_id >= 0) {
        claim->task_id = task_id;
        claim->inbox_index = context->inbox_index;
        claim->claim_end_cycles = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        return true;
    }

    __gm__ AicoreReadyDirectoryV1 *directory = aicore_ready_directory_at_v1(sidecar_base, context);
    uint64_t shard_begin = context->inbox_index / AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1 *
                           AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    uint64_t shard_end = shard_begin + AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
    if (shard_end > resolver_count) shard_end = resolver_count;
    uint64_t start = *victim_cursor;
    if (start < shard_begin || start >= shard_end) start = shard_begin;
    uint64_t bits =
        aicore_load_ready_directory_shard_v1(directory, resolver_count, core_type_index, context->inbox_index);
    if (bits != 0 && !aicore_steal_ready_from_shard_v1(
                         graph, sidecar_base, context, run_control, core_type_index, shard_begin, shard_end, start,
                         victim_cursor, bits, stats, claim, trace_enabled
                     ))
        return false;
    if (claim->task_id >= 0) return true;
    *victim_cursor = start + 1 == shard_end ? shard_begin : start + 1;
    claim->claim_end_cycles = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
    return true;
}

inline __aicore__ bool aicore_ready_directory_nonempty_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint64_t resolver_count, uint32_t core_type_index
) {
    __gm__ AicoreReadyDirectoryV1 *directory = aicore_ready_directory_at_v1(sidecar_base, context);
    return aicore_load_ready_directory_shard_v1(directory, resolver_count, core_type_index, context->inbox_index) != 0;
}

inline __aicore__ void aicore_advertise_free_slot_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint32_t core_type_index, uint64_t worker_id,
    uint32_t slot_index
) {
    uint64_t linear = worker_id * AICORE_PENDING_SLOT_COUNT_V1 + slot_index;
    __gm__ AicoreFreeSlotDirectoryV1 *directory = aicore_free_slot_directory_at_v1(sidecar_base, context);
    aicore_gm_fetch_or_v0(directory->words[core_type_index][linear / 64], UINT64_C(1) << (linear % 64));
}

inline __aicore__ bool aicore_load_free_slot_directory_masks_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, uint32_t core_type_index, uint64_t capacity,
    uint64_t *masks
) {
    if (context == nullptr || masks == nullptr || core_type_index >= AICORE_CORE_TYPE_COUNT_V1 ||
        capacity > AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1)
        return false;
    for (uint32_t word = 0; word < AICORE_FREE_SLOT_DIRECTORY_WORD_COUNT_V1; ++word)
        masks[word] = 0;
    if (capacity == 0) return false;

    __gm__ AicoreFreeSlotDirectoryV1 *directory = aicore_free_slot_directory_at_v1(sidecar_base, context);
    uint32_t word_count = static_cast<uint32_t>((capacity + 63) / 64);
    uint64_t combined = 0;
    for (uint32_t word = 0; word < word_count; ++word) {
        masks[word] = aicore_gm_query_v0(directory->words[core_type_index][word]);
        if (word + 1 == word_count && capacity % 64 != 0) masks[word] &= (UINT64_C(1) << (capacity % 64)) - 1;
        combined |= masks[word];
    }
    return combined != 0;
}

inline __aicore__ bool aicore_try_claim_free_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint32_t core_type_index, uint64_t *cursor, AicoreFreeSlotStatsV1 *stats,
    AicoreFreeSlotClaimV1 *claim
) {
    if (context == nullptr || cursor == nullptr || claim == nullptr || core_type_index >= AICORE_CORE_TYPE_COUNT_V1)
        return false;
    *claim = {};
    uint64_t capacity = context->runtime_worker_count * AICORE_PENDING_SLOT_COUNT_V1;
    if (capacity > AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1) return false;
    uint64_t masks[AICORE_FREE_SLOT_DIRECTORY_WORD_COUNT_V1]{};
    if (!aicore_load_free_slot_directory_masks_v1(sidecar_base, context, core_type_index, capacity, masks)) return true;

    __gm__ AicoreFreeSlotDirectoryV1 *directory = aicore_free_slot_directory_at_v1(sidecar_base, context);
    uint64_t start = *cursor % capacity;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        uint64_t range_begin = pass == 0 ? start : 0;
        uint64_t range_end = pass == 0 ? capacity : start;
        if (range_begin >= range_end) continue;
        uint32_t first_word = static_cast<uint32_t>(range_begin / 64);
        uint32_t last_word = static_cast<uint32_t>((range_end - 1) / 64);
        for (uint32_t word = first_word; word <= last_word; ++word) {
            uint64_t candidates = masks[word];
            if (word == first_word && range_begin % 64 != 0) candidates &= ~((UINT64_C(1) << (range_begin % 64)) - 1);
            if (word == last_word && range_end % 64 != 0) candidates &= (UINT64_C(1) << (range_end % 64)) - 1;
            while (candidates != 0) {
                uint32_t bit_index = static_cast<uint32_t>(__builtin_ctzll(candidates));
                candidates &= candidates - 1;
                uint64_t linear = static_cast<uint64_t>(word) * 64 + bit_index;
                uint64_t bit = UINT64_C(1) << bit_index;
                while (true) {
                    uint64_t observed = aicore_gm_query_v0(directory->words[core_type_index][word]);
                    if ((observed & bit) == 0) break;
                    uint64_t actual = aicore_gm_compare_exchange_v0(
                        directory->words[core_type_index][word], observed, observed & ~bit
                    );
                    if (actual != observed) continue;
                    uint64_t worker_id = linear / AICORE_PENDING_SLOT_COUNT_V1;
                    uint32_t slot_index = static_cast<uint32_t>(linear % AICORE_PENDING_SLOT_COUNT_V1);
                    __gm__ AicoreWorkerContextV1 *target =
                        aicore_worker_context_at_v1(sidecar_base, context, worker_id);
                    aicore_observe_cache_line_v0(target);
                    if (target->active == 0 || aicore_core_type_index_v1(target->core_type) != core_type_index) {
                        aicore_record_scheduler_error_v1(
                            run_control, AICORE_TASK_ID_INVALID_V1, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph,
                            context, UINT64_C(42)
                        );
                        return false;
                    }
                    __gm__ AicoreDispatchSlotV1 *slot =
                        aicore_dispatch_slot_at_v1(sidecar_base, context, worker_id, slot_index);
                    uint64_t publication = aicore_gm_query_v0(slot->publication);
                    if (aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::FREE ||
                        aicore_gm_compare_exchange_v0(
                            slot->publication, publication,
                            aicore_dispatch_publication_v1(
                                aicore_dispatch_generation_v1(publication), AicoreDispatchPublicationV1::FILLING
                            )
                        ) != publication) {
                        if (stats != nullptr) ++stats->stale_count;
                        break;
                    }
                    aicore_observe_cache_line_v0(slot);
                    if (slot->task_id != AICORE_TASK_ID_INVALID_V1 ||
                        slot->generation != aicore_dispatch_generation_v1(publication)) {
                        aicore_record_scheduler_error_v1(
                            run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context,
                            UINT64_C(43)
                        );
                        return false;
                    }
                    claim->worker_id = worker_id;
                    claim->slot_index = slot_index;
                    claim->generation = slot->generation;
                    *cursor = (linear + 1) % capacity;
                    if (stats != nullptr) ++stats->claim_count;
                    return true;
                }
            }
        }
    }
    *cursor = (start + 1) % capacity;
    return true;
}

inline __aicore__ void aicore_initialize_free_slot_v1(__gm__ AicoreDispatchSlotV1 *slot) {
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = AICORE_TASK_ID_INVALID_V1;
    slot->generation = generation;
    aicore_publish_cache_line_v0(slot);
    aicore_gm_publish_v0(
        slot->publication, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FREE)
    );
}

inline __aicore__ bool aicore_claim_private_free_slot_v1(__gm__ AicoreDispatchSlotV1 *slot, uint32_t generation) {
    uint64_t free = aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FREE);
    return aicore_gm_compare_exchange_v0(
               slot->publication, free, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FILLING)
           ) == free;
}

inline __aicore__ bool aicore_observe_claim_binding_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreRunControlV1 *run_control, int64_t task_id,
    AicoreTaskClaimBindingV1 *binding
) {
    if (binding == nullptr) return false;
    __gm__ AicoreTaskClaimBindingV1 *source = aicore_claim_binding_at_v1(sidecar_base, run_control, task_id);
    aicore_observe_cache_line_v0(source);
    binding->task_id = source->task_id;
    binding->callable_address = source->callable_address;
    binding->dispatch_payload_offset = source->dispatch_payload_offset;
    binding->owner_worker_id = source->owner_worker_id;
    binding->kernel_id = source->kernel_id;
    binding->subtask_slot = source->subtask_slot;
    binding->pending_slot = source->pending_slot;
    binding->dispatch_generation = source->dispatch_generation;
    binding->ready_claim_start_cycles = source->ready_claim_start_cycles;
    binding->ready_claim_end_cycles = source->ready_claim_end_cycles;
    binding->ready_claim_resolver_worker_id = source->ready_claim_resolver_worker_id;
    return binding->task_id == task_id && binding->pending_slot < AICORE_PENDING_SLOT_COUNT_V1 &&
           binding->dispatch_payload_offset != 0 && binding->callable_address != 0;
}

inline __aicore__ bool aicore_fill_dispatch_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, const AicoreFreeSlotClaimV1 &slot_claim,
    const AicoreReadyClaimV1 &ready_claim, bool trace_enabled = false, AicoreDispatchFillTimingV1 *timing = nullptr
) {
    if (ready_claim.task_id < 0 || static_cast<uint64_t>(ready_claim.task_id) >= graph.task_count ||
        slot_claim.worker_id >= resolver->runtime_worker_count || slot_claim.slot_index >= AICORE_PENDING_SLOT_COUNT_V1)
        return false;
    const bool record_timeline = timing != nullptr;
    uint64_t operation_start = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    __gm__ AicoreTaskMetadataV1 *metadata_source =
        aicore_task_metadata_at_v1(sidecar_base, resolver, ready_claim.task_id);
    aicore_observe_cache_line_v0(metadata_source);
    AicoreTaskMetadataV1 metadata{};
    metadata.kernel_ids[0] = metadata_source->kernel_ids[0];
    metadata.kernel_ids[1] = metadata_source->kernel_ids[1];
    metadata.kernel_ids[2] = metadata_source->kernel_ids[2];
    metadata.active_mask = metadata_source->active_mask;
    metadata.flags = metadata_source->flags;
    metadata.logical_block_num = metadata_source->logical_block_num;
    metadata.total_required_subtasks = metadata_source->total_required_subtasks;
    const uint8_t subtask_slot = aicore_metadata_single_subtask_slot_v1(metadata.active_mask);
    const uint16_t kernel_id = metadata.kernel_ids[subtask_slot];
    __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, slot_claim.worker_id);
    aicore_observe_cache_line_v0(target);
    if (!aicore_task_is_executable_v1(metadata.flags) || aicore_task_is_gang_v1(metadata.flags) ||
        aicore_metadata_core_type_index_v1(subtask_slot) != aicore_core_type_index_v1(target->core_type)) {
        aicore_record_scheduler_error_v1(
            run_control, ready_claim.task_id, AicoreRootStatusV0::UNSUPPORTED_SHAPE, &graph, resolver, UINT64_C(44)
        );
        return false;
    }
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, slot_claim.worker_id, slot_claim.slot_index);
    uint32_t generation = slot_claim.generation + 1;
    if (generation == 0) generation = 1;
    __gm__ uint64_t *callable_addresses =
        aicore_sidecar_at_v1<uint64_t>(sidecar_base, resolver->callable_addresses_offset);
    aicore_observe_cache_line_v0(&callable_addresses[kernel_id]);
    uint64_t callable_address = callable_addresses[kernel_id];
    if (callable_address == 0) {
        aicore_record_scheduler_error_v1(
            run_control, ready_claim.task_id, AicoreRootStatusV0::INVALID_CALLABLE, &graph, resolver, UINT64_C(45)
        );
        return false;
    }

    slot->task_id = ready_claim.task_id;
    slot->ready_inbox_index = ready_claim.inbox_index;
    slot->claim_start_cycles = ready_claim.claim_start_cycles;
    slot->claim_end_cycles = ready_claim.claim_end_cycles;
    slot->claim_worker_id = resolver->worker_index;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = aicore_task_has_fanin_v1(metadata.flags) ? 1 : 0;
    slot->ready_source = static_cast<uint8_t>(ready_claim.source);
    slot->pending_slot = static_cast<uint8_t>(slot_claim.slot_index);
    slot->generation = generation;
    slot->block_idx = 0;
    slot->block_num = 1;
    slot->cohort_generation = 0;
    slot->cohort_index = UINT8_MAX;
    slot->gang = 0;
    aicore_writeback_cache_line_v0(slot);

    AicoreTaskClaimBindingV1 binding{};
    binding.task_id = ready_claim.task_id;
    binding.callable_address = callable_address;
    binding.dispatch_payload_offset =
        target->dispatch_payload_offset + static_cast<uint64_t>(slot_claim.slot_index) * sizeof(PTO2DispatchPayload);
    binding.owner_worker_id = slot_claim.worker_id;
    binding.kernel_id = kernel_id;
    binding.subtask_slot = subtask_slot;
    binding.pending_slot = static_cast<uint8_t>(slot_claim.slot_index);
    binding.dispatch_generation = generation;
    binding.ready_claim_start_cycles = ready_claim.claim_start_cycles;
    binding.ready_claim_end_cycles = ready_claim.claim_end_cycles;
    binding.ready_claim_resolver_worker_id = resolver->worker_index;
    __gm__ AicoreTaskClaimBindingV1 *binding_destination =
        aicore_claim_binding_at_v1(sidecar_base, run_control, ready_claim.task_id);
    binding_destination->task_id = binding.task_id;
    binding_destination->callable_address = binding.callable_address;
    binding_destination->dispatch_payload_offset = binding.dispatch_payload_offset;
    binding_destination->owner_worker_id = binding.owner_worker_id;
    binding_destination->kernel_id = binding.kernel_id;
    binding_destination->subtask_slot = binding.subtask_slot;
    binding_destination->pending_slot = binding.pending_slot;
    binding_destination->dispatch_generation = binding.dispatch_generation;
    binding_destination->ready_claim_start_cycles = binding.ready_claim_start_cycles;
    binding_destination->ready_claim_end_cycles = binding.ready_claim_end_cycles;
    binding_destination->ready_claim_resolver_worker_id = binding.ready_claim_resolver_worker_id;
    aicore_writeback_cache_line_v0(binding_destination);

    uint64_t operation_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (timing != nullptr) timing->prepare_cycles += operation_end - operation_start;

    AicoreTaskInfoV0 task{
        ready_claim.task_id,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV,
    };
    __gm__ PTO2DispatchPayload *payload =
        aicore_sidecar_at_v1<PTO2DispatchPayload>(sidecar_base, binding.dispatch_payload_offset);
    AicoreRootStatusV0 status = aicore_materialize_task_payload_resolved_v0(graph, task, callable_address, payload);
    if (status != AicoreRootStatusV0::OK) {
        aicore_record_scheduler_error_v1(run_control, ready_claim.task_id, status, &graph, resolver, UINT64_C(46));
        return false;
    }
    uint64_t materialize_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (timing != nullptr) timing->materialize_cycles += materialize_end - operation_end;
    aicore_publish_dispatch_payload_v1(payload);
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, resolver, ready_claim.task_id);
    if (trace_enabled) {
        aicore_observe_cache_line_v0(&control->next_waiter);
        control->ready_publish_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&control->next_waiter);
    }
    uint64_t publish_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (timing != nullptr) timing->publish_cycles += publish_end - materialize_end;
    aicore_gm_publish_v0(
        slot->publication, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::READY)
    );
    return true;
}

inline __aicore__ bool aicore_release_completed_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreTaskClaimBindingV1 *binding,
    AicoreFreeSlotClaimV1 *slot_claim
) {
    if (!aicore_observe_claim_binding_v1(sidecar_base, run_control, task_id, binding)) return false;
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, binding->owner_worker_id, binding->pending_slot);
    aicore_observe_cache_line_v0(slot);
    uint64_t ready = aicore_dispatch_publication_v1(binding->dispatch_generation, AicoreDispatchPublicationV1::READY);
    if (slot->task_id != task_id || slot->generation != binding->dispatch_generation ||
        aicore_gm_query_v0(slot->publication) != ready) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(47)
        );
        return false;
    }
    slot->task_id = AICORE_TASK_ID_INVALID_V1;
    aicore_publish_cache_line_v0(slot);
    aicore_gm_publish_v0(
        slot->publication,
        aicore_dispatch_publication_v1(binding->dispatch_generation, AicoreDispatchPublicationV1::FREE)
    );
    slot_claim->worker_id = binding->owner_worker_id;
    slot_claim->slot_index = binding->pending_slot;
    slot_claim->generation = binding->dispatch_generation;
    return true;
}

inline __aicore__ bool aicore_enqueue_completion_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, int64_t task_id, uint64_t local_completion_index,
    AicoreCompletionStatsV1 *stats, bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count || resolver_count == 0) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(48)
        );
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    aicore_observe_cache_line_v0(&control->next_waiter);
    if (trace_enabled) {
        control->completion_enqueue_cycles = aicore_scheduler_cycles_v1();
    }
    uint64_t inbox_index = aicore_completion_inbox_index_v1(context, resolver_count, local_completion_index);
    __gm__ AicoreCompletionInboxV1 *inbox = aicore_completion_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t previous = aicore_gm_query_v0(inbox->head);
    while (true) {
        if (previous < AICORE_INBOX_EMPTY_V1) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(49)
            );
            return false;
        }
        control->inbox_next = previous;
        aicore_publish_cache_line_v0(&control->next_waiter);
        int64_t actual = aicore_gm_compare_exchange_v0(inbox->head, previous, task_id);
        if (actual == previous) break;
        previous = actual;
    }
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool aicore_resolve_completion_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *wake_stats,
    AicoreReadyStatsV1 *ready_stats, AicoreCompletionStatsV1 *completion_stats, bool trace_enabled = false,
    bool validate_done_state = true, uint64_t *ready_publish_cycles = nullptr
) {
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (validate_done_state && aicore_gm_query_v0(control->state) != static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(61)
        );
        return false;
    }
    uint64_t resolve_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
    if (trace_enabled) {
        aicore_observe_cache_line_v0(&control->next_waiter);
        control->completion_resolve_start_cycles = resolve_start;
        control->resolver_worker_id = context->worker_index;
        if (control->completion_enqueue_cycles != 0 && resolve_start >= control->completion_enqueue_cycles) {
            uint64_t lag = resolve_start - control->completion_enqueue_cycles;
            completion_stats->completion_lag_cycles += lag;
            if (lag > completion_stats->completion_lag_max_cycles) completion_stats->completion_lag_max_cycles = lag;
        }
    }
    int64_t waiter = aicore_gm_exchange_v0(control->wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    if (waiter == AICORE_WAKE_LIST_CLOSED_V1) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(62)
        );
        return false;
    }
    if (wake_stats != nullptr) ++wake_stats->wake_close_count;
    AicoreReadyBatchV1 batches[AICORE_CORE_TYPE_COUNT_V1]{};
    while (waiter >= 0) {
        if (static_cast<uint64_t>(waiter) >= graph.task_count) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context, UINT64_C(63)
            );
            return false;
        }
        __gm__ AicoreTaskControlV1 *waiter_control = aicore_task_control_at_v1(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v1(waiter_control);
        if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        AicoreRouteResultV1 route = aicore_route_task_v1(graph, sidecar_base, context, run_control, waiter, wake_stats);
        if (route == AicoreRouteResultV1::ERROR) return false;
        if (route == AicoreRouteResultV1::READY_TO_ENQUEUE) {
            __gm__ AicoreTaskMetadataV1 *metadata = aicore_task_metadata_at_v1(sidecar_base, context, waiter);
            aicore_observe_cache_line_v0(metadata);
            if (!aicore_task_is_executable_v1(metadata->flags)) {
                aicore_record_scheduler_error_v1(
                    run_control, waiter, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(64)
                );
                return false;
            }
            if (aicore_task_is_gang_v1(metadata->flags)) {
                aicore_publish_gang_ready_v1(sidecar_base, context, waiter_control, metadata->flags);
            } else if (!aicore_ready_batch_append_v1(
                           sidecar_base, context, waiter,
                           &batches[aicore_metadata_core_type_index_v1(
                               aicore_metadata_single_subtask_slot_v1(metadata->active_mask)
                           )],
                           ready_stats, trace_enabled
                       )) {
                aicore_record_scheduler_error_v1(
                    run_control, waiter, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(65)
                );
                return false;
            }
        }
        waiter = next;
    }
    uint64_t ready_publish_start = ready_publish_cycles == nullptr ? 0 : aicore_scheduler_cycles_v1();
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        if (!aicore_ready_batch_push_v1(
                sidecar_base, context, type, context->inbox_index, &batches[type], ready_stats
            )) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(66)
            );
            return false;
        }
    }
    if (ready_publish_cycles != nullptr) *ready_publish_cycles += aicore_scheduler_cycles_v1() - ready_publish_start;
    if (trace_enabled) {
        control->completion_resolve_end_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&control->next_waiter);
    }
    return true;
}

inline __aicore__ bool aicore_refill_private_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, const AicoreFreeSlotClaimV1 &slot_claim,
    uint64_t *ready_victim_cursors, AicoreReadyStatsV1 *ready_stats, bool trace_enabled,
    AicoreRefillTimingV1 *timing = nullptr
) {
    __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, slot_claim.worker_id);
    aicore_observe_cache_line_v0(target);
    uint32_t type = aicore_core_type_index_v1(target->core_type);
    AicoreReadyClaimV1 ready{};
    uint64_t operation_start = timing == nullptr ? 0 : aicore_scheduler_cycles_v1();
    if (!aicore_claim_ready_for_slot_v1(
            graph, sidecar_base, resolver, run_control, resolver_count, type, &ready_victim_cursors[type], ready_stats,
            &ready, trace_enabled
        ))
        return false;
    if (timing != nullptr) {
        uint64_t operation_end = aicore_scheduler_cycles_v1();
        timing->ready_claim_cycles += operation_end - operation_start;
        operation_start = operation_end;
    }
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, slot_claim.worker_id, slot_claim.slot_index);
    if (ready.task_id >= 0) {
        if (!aicore_claim_private_free_slot_v1(slot, slot_claim.generation)) return false;
        bool filled =
            aicore_fill_dispatch_slot_v1(graph, sidecar_base, resolver, run_control, slot_claim, ready, trace_enabled);
        if (timing != nullptr) timing->slot_fill_cycles += aicore_scheduler_cycles_v1() - operation_start;
        return filled;
    }
    aicore_advertise_free_slot_v1(sidecar_base, resolver, type, slot_claim.worker_id, slot_claim.slot_index);
    if (timing != nullptr) timing->free_advertise_cycles += aicore_scheduler_cycles_v1() - operation_start;
    return true;
}

inline __aicore__ bool aicore_service_completion_inboxes_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint64_t *completion_victim_cursor,
    uint64_t *ready_victim_cursors, AicoreWakeStatsV1 *wake_stats, AicoreReadyStatsV1 *ready_stats,
    AicoreCompletionStatsV1 *completion_stats, bool *made_progress, bool trace_enabled = false
) {
    if (made_progress != nullptr) *made_progress = false;
    if (context->core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIV)) return true;
    if (resolver_count == 0 || completion_victim_cursor == nullptr || ready_victim_cursors == nullptr ||
        context->inbox_index >= resolver_count)
        return true;
    uint64_t inbox_index = context->inbox_index;
    __gm__ AicoreCompletionInboxV1 *inbox = aicore_completion_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t task_id = AICORE_INBOX_EMPTY_V1;
    if (aicore_gm_query_v0(inbox->head) != AICORE_INBOX_EMPTY_V1)
        task_id = aicore_gm_exchange_v0(inbox->head, AICORE_INBOX_EMPTY_V1);
    if (task_id == AICORE_INBOX_EMPTY_V1 && resolver_count > 1) {
        uint64_t victim = *completion_victim_cursor % resolver_count;
        *completion_victim_cursor = (victim + 1) % resolver_count;
        if (victim == inbox_index) victim = *completion_victim_cursor;
        inbox = aicore_completion_inbox_at_v1(sidecar_base, context, victim);
        if (aicore_gm_query_v0(inbox->head) != AICORE_INBOX_EMPTY_V1)
            task_id = aicore_gm_exchange_v0(inbox->head, AICORE_INBOX_EMPTY_V1);
        if (task_id != AICORE_INBOX_EMPTY_V1 && completion_stats != nullptr) ++completion_stats->steal_count;
    }
    if (task_id == AICORE_INBOX_EMPTY_V1) return true;
    if (completion_stats != nullptr) ++completion_stats->batch_count;
    uint64_t count = 0;
    while (task_id >= 0) {
        if (static_cast<uint64_t>(task_id) >= graph.task_count) return false;
        __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
        aicore_observe_cache_line_v0(&control->next_waiter);
        int64_t next = control->inbox_next;
        uint64_t waits = 0;
        while (next == AICORE_INBOX_LINK_UNPUBLISHED_V1 && waits < UINT64_C(1048576)) {
            ++waits;
            aicore_observe_cache_line_v0(&control->next_waiter);
            next = control->inbox_next;
        }
        if (completion_stats != nullptr) {
            completion_stats->link_wait_count += waits;
            if (waits > completion_stats->link_wait_max) completion_stats->link_wait_max = waits;
        }
        if (next < AICORE_INBOX_EMPTY_V1) return false;
        AicoreTaskClaimBindingV1 binding{};
        AicoreFreeSlotClaimV1 slot_claim{};
        if (!aicore_release_completed_slot_v1(
                graph, sidecar_base, context, run_control, task_id, &binding, &slot_claim
            ) ||
            !aicore_resolve_completion_v1(
                graph, sidecar_base, context, run_control, task_id, wake_stats, ready_stats, completion_stats,
                trace_enabled
            ) ||
            !aicore_refill_private_slot_v1(
                graph, sidecar_base, context, run_control, resolver_count, slot_claim, ready_victim_cursors,
                ready_stats, trace_enabled
            ))
            return false;
        ++count;
        task_id = next;
    }
    if (completion_stats != nullptr) completion_stats->resolve_count += count;
    aicore_gm_fetch_add_v0(run_control->resolved_task_count, count);
    if (made_progress != nullptr) *made_progress = true;
    return true;
}

inline __aicore__ bool aicore_dispatch_one_overflow_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint32_t core_type_index,
    uint64_t *ready_victim_cursor, uint64_t *free_slot_cursor, AicoreReadyStatsV1 *ready_stats,
    AicoreFreeSlotStatsV1 *free_stats, bool *made_progress, bool trace_enabled = false
) {
    if (made_progress != nullptr) *made_progress = false;
    if (!aicore_ready_directory_nonempty_v1(sidecar_base, resolver, resolver_count, core_type_index)) return true;
    AicoreFreeSlotClaimV1 slot_claim{};
    if (!aicore_try_claim_free_slot_v1(
            graph, sidecar_base, resolver, run_control, core_type_index, free_slot_cursor, free_stats, &slot_claim
        ))
        return false;
    if (slot_claim.worker_id == UINT64_MAX) return true;
    AicoreReadyClaimV1 ready{};
    if (!aicore_claim_ready_for_slot_v1(
            graph, sidecar_base, resolver, run_control, resolver_count, core_type_index, ready_victim_cursor,
            ready_stats, &ready, trace_enabled
        ))
        return false;
    if (ready.task_id < 0) {
        __gm__ AicoreDispatchSlotV1 *slot =
            aicore_dispatch_slot_at_v1(sidecar_base, resolver, slot_claim.worker_id, slot_claim.slot_index);
        aicore_gm_publish_v0(
            slot->publication, aicore_dispatch_publication_v1(slot_claim.generation, AicoreDispatchPublicationV1::FREE)
        );
        aicore_advertise_free_slot_v1(
            sidecar_base, resolver, core_type_index, slot_claim.worker_id, slot_claim.slot_index
        );
        return true;
    }
    if (!aicore_fill_dispatch_slot_v1(graph, sidecar_base, resolver, run_control, slot_claim, ready, trace_enabled))
        return false;
    if (made_progress != nullptr) *made_progress = true;
    return true;
}
