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
    READY_TO_PUBLISH = 1,
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

struct AicoreRootPrepareStatsV1 {
    uint64_t enqueue_count{0};
    uint64_t batch_count{0};
    uint64_t resolve_count{0};
    uint64_t steal_count{0};
    uint64_t link_wait_count{0};
    uint64_t link_wait_max{0};
    uint64_t prepare_cycles{0};
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
    int32_t next_fanin_index;
    int32_t waiting_producer;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t has_fanin;
    uint8_t payload_needs_observe;
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
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        UINT16_MAX,
        UINT8_MAX,
        0,
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
        sidecar_base, context->task_controls_offset + static_cast<uint64_t>(task_id) * sizeof(AicoreTaskControlV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreCompletionInboxV1 *aicore_completion_inbox_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t inbox_index
) {
    return aicore_sidecar_at_v1<AicoreCompletionInboxV1>(
        sidecar_base, context->completion_inboxes_offset + inbox_index * sizeof(AicoreCompletionInboxV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreRootPrepareInboxV1 *aicore_root_prepare_inbox_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t inbox_index
) {
    return aicore_sidecar_at_v1<AicoreRootPrepareInboxV1>(
        sidecar_base, context->root_prepare_inboxes_offset + inbox_index * sizeof(AicoreRootPrepareInboxV1)
    );
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

inline __host__ __aicore__ __gm__ AicoreClaimPrefetchV1 *aicore_claim_prefetch_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t worker_id
) {
    return aicore_sidecar_at_v1<AicoreClaimPrefetchV1>(
        sidecar_base, context->claim_prefetches_offset + worker_id * sizeof(AicoreClaimPrefetchV1)
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

inline __aicore__ void aicore_retire_dispatch_slot_v1(__gm__ AicoreDispatchSlotV1 *slot) {
    aicore_observe_cache_line_v0(slot);
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = AICORE_TASK_ID_INVALID_V1;
    slot->generation = generation;
    aicore_publish_cache_line_v0(slot);
    aicore_gm_publish_v0(
        slot->publication, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::RETIRED)
    );
}

inline __host__ __aicore__ AicoreTaskClaimBindingV1 aicore_make_claim_binding_v1(
    __gm__ const AicoreWorkerContextV1 *context, int64_t task_id, uint16_t kernel_id, uint8_t subtask_slot,
    uint8_t pending_slot, uint64_t callable_address, uint32_t dispatch_generation = 0
) {
    AicoreTaskClaimBindingV1 binding{};
    binding.task_id = task_id;
    binding.callable_address = callable_address;
    binding.dispatch_payload_offset =
        context->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(PTO2DispatchPayload);
    binding.owner_worker_id = context->worker_index;
    binding.kernel_id = kernel_id;
    binding.subtask_slot = subtask_slot;
    binding.pending_slot = pending_slot;
    binding.dispatch_generation = dispatch_generation;
    return binding;
}

inline __aicore__ void aicore_write_claim_binding_v1(
    __gm__ void *sidecar_base, __gm__ AicoreRunControlV1 *run_control, const AicoreTaskClaimBindingV1 &binding
) {
    __gm__ AicoreTaskClaimBindingV1 *destination =
        aicore_claim_binding_at_v1(sidecar_base, run_control, binding.task_id);
    destination->callable_address = binding.callable_address;
    destination->dispatch_payload_offset = binding.dispatch_payload_offset;
    destination->owner_worker_id = binding.owner_worker_id;
    destination->kernel_id = binding.kernel_id;
    destination->subtask_slot = binding.subtask_slot;
    destination->pending_slot = binding.pending_slot;
    destination->dispatch_generation = binding.dispatch_generation;
    destination->root_prepare_start_cycles = 0;
    destination->root_prepare_end_cycles = 0;
    destination->root_prepare_resolver_worker_id = UINT64_MAX;
    destination->task_id = binding.task_id;
    aicore_writeback_cache_line_v0(destination);
}

inline __aicore__ void aicore_publish_claim_binding_v1(
    __gm__ void *sidecar_base, __gm__ AicoreRunControlV1 *run_control, const AicoreTaskClaimBindingV1 &binding
) {
    aicore_write_claim_binding_v1(sidecar_base, run_control, binding);
    aicore_cache_barrier_v0();
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
    binding->root_prepare_start_cycles = source->root_prepare_start_cycles;
    binding->root_prepare_end_cycles = source->root_prepare_end_cycles;
    binding->root_prepare_resolver_worker_id = source->root_prepare_resolver_worker_id;
    return binding->task_id == task_id && binding->pending_slot < AICORE_PENDING_SLOT_COUNT_V1 &&
           binding->dispatch_payload_offset != 0 && binding->callable_address != 0;
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

inline __host__ AicoreRootStatusV0 aicore_make_task_ticket_v1(
    const AicoreReadonlyGraphV0 &graph, int64_t task_id, AicoreRootCoreTypeV0 expected_core_type, int32_t max_func_id,
    AicoreTaskTicketV1 *ticket
) {
    if (ticket == nullptr) return AicoreRootStatusV0::INVALID_ARGUMENTS;
    AicoreTaskInfoV0 task{};
    AicoreRootStatusV0 status = aicore_classify_task_v0(graph, task_id, &task);
    if (status != AicoreRootStatusV0::OK) return status;
    if (task.core_type != expected_core_type || task.kernel_id >= max_func_id ||
        task.kernel_id > static_cast<int32_t>(UINT16_MAX)) {
        return AicoreRootStatusV0::UNSUPPORTED_SHAPE;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    *ticket = aicore_task_ticket_make_v1(
        static_cast<uint32_t>(task_id), static_cast<uint16_t>(task.kernel_id), static_cast<uint8_t>(task.subtask_slot),
        *reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0) != 0
    );
    return AicoreRootStatusV0::OK;
}

inline __host__ __aicore__ void aicore_pending_initialize_v1(
    const AicoreTaskTicketV1 &ticket, uint64_t stream_index, AicoreClaimKindV1 claim_kind, uint64_t claim_start_cycles,
    uint64_t claim_end_cycles, AicorePendingSlotV1 *slot
) {
    *slot = {
        static_cast<int64_t>(aicore_task_ticket_task_id_v1(ticket)),
        0,
        static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1),
        aicore_task_ticket_kernel_id_v1(ticket),
        aicore_task_ticket_subtask_slot_v1(ticket),
        static_cast<uint8_t>(aicore_task_ticket_has_fanin_v1(ticket) ? 1 : 0),
        0,
        stream_index,
        claim_start_cycles,
        claim_end_cycles,
        0,
        claim_kind,
    };
}

inline __host__ __aicore__ void
aicore_load_task_ticket_v1(__gm__ const AicoreTaskTicketV1 *source, AicoreTaskTicketV1 *ticket) {
    ticket->task_id = source->task_id;
    ticket->kernel_id = source->kernel_id;
    ticket->subtask_slot = source->subtask_slot;
    ticket->flags = source->flags;
    ticket->reserved = 0;
}

inline __aicore__ AicoreRouteResultV1 aicore_route_task_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats, bool trace_enabled = false,
    bool validate_current_state = true
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return AicoreRouteResultV1::ERROR;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (validate_current_state) {
        int64_t current_state = aicore_gm_load_v0(control->state);
        if (current_state == static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
            return AicoreRouteResultV1::COMPLETED;
        }
        if (current_state == static_cast<int64_t>(AicoreTaskStateV1::READY)) {
            return AicoreRouteResultV1::READY;
        }
        if (current_state != static_cast<int64_t>(AicoreTaskStateV1::BLOCKED)) {
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
        if (aicore_gm_load_v0(producer_control->state) == static_cast<int64_t>(AicoreTaskStateV1::DONE)) {
            ++next_fanin;
            continue;
        }

        while (true) {
            int64_t observed = aicore_gm_load_v0(producer_control->wake_list_head);
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
    return AicoreRouteResultV1::READY_TO_PUBLISH;
}

inline __aicore__ bool aicore_materialize_claim_payload_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, const AicoreTaskClaimBindingV1 &binding, bool publish_payload
) {
    if (binding.task_id < 0 || static_cast<uint64_t>(binding.task_id) >= graph.task_count ||
        binding.dispatch_payload_offset == 0) {
        aicore_record_scheduler_error_v1(
            run_control, binding.task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(31)
        );
        return false;
    }
    if (binding.pending_slot >= AICORE_PENDING_SLOT_COUNT_V1) {
        aicore_record_scheduler_error_v1(
            run_control, binding.task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(32)
        );
        return false;
    }
    AicoreTaskInfoV0 task{
        binding.task_id,
        static_cast<int32_t>(binding.kernel_id),
        static_cast<int32_t>(binding.subtask_slot),
        binding.subtask_slot == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV,
    };
    __gm__ PTO2DispatchPayload *payload =
        aicore_sidecar_at_v1<PTO2DispatchPayload>(sidecar_base, binding.dispatch_payload_offset);
    AicoreRootStatusV0 status =
        aicore_materialize_task_payload_resolved_v0(graph, task, binding.callable_address, payload);
    if (status != AicoreRootStatusV0::OK) {
        aicore_record_scheduler_error_v1(run_control, binding.task_id, status, &graph, context, UINT64_C(4));
        return false;
    }
    if (publish_payload) aicore_publish_dispatch_payload_v1(payload);
    return true;
}

inline __aicore__ bool aicore_finalize_ready_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, const AicoreTaskClaimBindingV1 &binding, bool publish_payload,
    bool trace_enabled = false
) {
    if (!aicore_materialize_claim_payload_v1(graph, sidecar_base, context, run_control, binding, publish_payload)) {
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, binding.task_id);
    if (trace_enabled) {
        control->ready_publish_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&control->next_waiter);
    }
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    if (binding.dispatch_generation != 0) {
        __gm__ AicoreDispatchSlotV1 *slot =
            aicore_dispatch_slot_at_v1(sidecar_base, context, binding.owner_worker_id, binding.pending_slot);
        aicore_observe_cache_line_v0(slot);
        if (slot->task_id != binding.task_id || slot->generation != binding.dispatch_generation) {
            aicore_record_scheduler_error_v1(
                run_control, binding.task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(5)
            );
            return false;
        }
        aicore_gm_publish_v0(
            slot->publication,
            aicore_dispatch_publication_v1(binding.dispatch_generation, AicoreDispatchPublicationV1::READY)
        );
    }
    return true;
}

inline __aicore__ AicorePendingStateV1 aicore_pending_state_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, AicorePendingSlotV1 *slot
) {
    if (slot == nullptr || slot->task_id < 0) return AicorePendingStateV1::EMPTY;
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, slot->task_id);
    int64_t state = aicore_gm_load_v0(control->state);
    if (state == static_cast<int64_t>(AicoreTaskStateV1::BLOCKED) ||
        state == static_cast<int64_t>(AicoreTaskStateV1::PREPARING)) {
        return AicorePendingStateV1::BLOCKED;
    }
    if (state == static_cast<int64_t>(AicoreTaskStateV1::READY)) return AicorePendingStateV1::READY;
    return AicorePendingStateV1::ERROR;
}

inline __aicore__ bool aicore_refresh_pending_debug_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, AicorePendingSlotV1 *slot
) {
    if (slot == nullptr || slot->task_id < 0 || slot->has_fanin == 0) return false;
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, slot->task_id);
    int32_t previous_fanin = slot->next_fanin_index;
    int32_t previous_producer = slot->waiting_producer;
    aicore_observe_cache_line_v0(&control->next_waiter);
    slot->next_fanin_index = control->next_fanin_index;
    slot->waiting_producer = control->waiting_producer;
    return previous_fanin != slot->next_fanin_index || previous_producer != slot->waiting_producer;
}

inline __aicore__ bool aicore_claim_ticket_v1(
    __gm__ void *sidecar_base, __gm__ AicoreTaskStreamV1 *stream, uint64_t *stream_index, AicoreTaskTicketV1 *ticket
) {
    if (stream_index == nullptr || ticket == nullptr) return false;
    uint64_t index = aicore_gm_fetch_add_v0(stream->next_index, UINT64_C(1));
    *stream_index = index;
    if (index >= stream->task_count) {
        ticket->task_id = UINT64_MAX;
        return false;
    }
    __gm__ AicoreTaskTicketV1 *tickets = aicore_sidecar_at_v1<AicoreTaskTicketV1>(sidecar_base, stream->tickets_offset);
    aicore_load_task_ticket_v1(&tickets[index], ticket);
    return true;
}

inline __aicore__ bool aicore_fill_claim_prefetch_v1(
    __gm__ void *sidecar_base, __gm__ AicoreTaskStreamV1 *stream, __gm__ AicoreClaimPrefetchV1 *prefetch,
    uint64_t claim_worker_id, bool *filled, bool *exhausted, uint64_t *ticket_claim_count = nullptr,
    uint64_t *ticket_exhaustion_count = nullptr, uint64_t *claim_cycles = nullptr
) {
    if (filled != nullptr) *filled = false;
    if (exhausted != nullptr) *exhausted = false;
    if (stream == nullptr || prefetch == nullptr) return false;
    const uint64_t empty = static_cast<uint64_t>(AicoreClaimPrefetchStateV1::EMPTY);
    const uint64_t filling = static_cast<uint64_t>(AicoreClaimPrefetchStateV1::FILLING);
    if (aicore_gm_compare_exchange_v0(prefetch->state, empty, filling) != empty) return true;

    uint64_t claim_start = aicore_scheduler_cycles_v1();
    uint64_t stream_index = 0;
    AicoreTaskTicketV1 ticket{};
    bool valid = aicore_claim_ticket_v1(sidecar_base, stream, &stream_index, &ticket);
    uint64_t claim_end = aicore_scheduler_cycles_v1();
    if (claim_cycles != nullptr) *claim_cycles += claim_end - claim_start;
    if (valid && ticket_claim_count != nullptr) ++*ticket_claim_count;
    if (!valid && ticket_exhaustion_count != nullptr) ++*ticket_exhaustion_count;
    prefetch->ticket_task_id = ticket.task_id;
    prefetch->ticket_kernel_id = ticket.kernel_id;
    prefetch->ticket_subtask_slot = ticket.subtask_slot;
    prefetch->ticket_flags = ticket.flags;
    prefetch->ticket_reserved = 0;
    prefetch->stream_index = stream_index;
    prefetch->claim_start_cycles = claim_start;
    prefetch->claim_end_cycles = claim_end;
    prefetch->claim_worker_id = claim_worker_id;
    aicore_publish_cache_line_v0(prefetch);
    aicore_gm_publish_v0(
        prefetch->state,
        static_cast<uint64_t>(valid ? AicoreClaimPrefetchStateV1::VALID : AicoreClaimPrefetchStateV1::EXHAUSTED)
    );
    if (filled != nullptr) *filled = valid;
    if (exhausted != nullptr) *exhausted = !valid;
    return true;
}

inline __aicore__ bool aicore_take_claim_prefetch_v1(
    __gm__ AicoreClaimPrefetchV1 *prefetch, AicoreTaskTicketV1 *ticket, uint64_t *stream_index,
    uint64_t *claim_start_cycles, uint64_t *claim_end_cycles, uint64_t *claim_worker_id
) {
    if (prefetch == nullptr || ticket == nullptr || stream_index == nullptr || claim_start_cycles == nullptr ||
        claim_end_cycles == nullptr || claim_worker_id == nullptr) {
        return false;
    }
    const uint64_t valid = static_cast<uint64_t>(AicoreClaimPrefetchStateV1::VALID);
    const uint64_t consuming = static_cast<uint64_t>(AicoreClaimPrefetchStateV1::CONSUMING);
    if (aicore_gm_compare_exchange_v0(prefetch->state, valid, consuming) != valid) return false;
    aicore_observe_cache_line_v0(prefetch);
    ticket->task_id = prefetch->ticket_task_id;
    ticket->kernel_id = prefetch->ticket_kernel_id;
    ticket->subtask_slot = prefetch->ticket_subtask_slot;
    ticket->flags = prefetch->ticket_flags;
    ticket->reserved = 0;
    *stream_index = prefetch->stream_index;
    *claim_start_cycles = prefetch->claim_start_cycles;
    *claim_end_cycles = prefetch->claim_end_cycles;
    *claim_worker_id = prefetch->claim_worker_id;
    aicore_gm_publish_v0(prefetch->state, static_cast<uint64_t>(AicoreClaimPrefetchStateV1::EMPTY));
    return true;
}

inline __aicore__ __gm__ AicoreTaskStreamV1 *aicore_worker_stream_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, __gm__ const AicoreWorkerContextV1 *worker
) {
    return aicore_sidecar_at_v1<AicoreTaskStreamV1>(
        sidecar_base, worker->core_type == static_cast<int32_t>(AicoreRootCoreTypeV0::AIC) ?
                          context->aic_stream_offset :
                          context->aiv_stream_offset
    );
}

inline __aicore__ bool aicore_prepare_dispatch_binding_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, __gm__ AicoreWorkerContextV1 *target, uint64_t target_worker_index,
    uint32_t pending_slot, const AicoreTaskTicketV1 &ticket, uint64_t stream_index, AicoreClaimKindV1 claim_kind,
    uint64_t claim_start_cycles, uint64_t claim_end_cycles, uint64_t claim_worker_id, AicoreWakeStatsV1 *wake_stats,
    AicoreTaskClaimBindingV1 *binding, AicoreRouteResultV1 *route, bool trace_enabled = false,
    bool slot_already_observed = false
) {
    if (target == nullptr || binding == nullptr || route == nullptr || pending_slot >= AICORE_PENDING_SLOT_COUNT_V1) {
        return false;
    }
    const int64_t task_id = static_cast<int64_t>(aicore_task_ticket_task_id_v1(ticket));
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, resolver);
        return false;
    }
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, target_worker_index, pending_slot);
    if (!slot_already_observed) aicore_observe_cache_line_v0(slot);
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    const uint16_t kernel_id = aicore_task_ticket_kernel_id_v1(ticket);
    const uint8_t subtask_slot = aicore_task_ticket_subtask_slot_v1(ticket);
    __gm__ uint64_t *callable_addresses =
        aicore_sidecar_at_v1<uint64_t>(sidecar_base, resolver->callable_addresses_offset);
    aicore_observe_cache_line_v0(&callable_addresses[kernel_id]);
    const uint64_t callable_address = callable_addresses[kernel_id];
    if (callable_address == 0) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(1)
        );
        return false;
    }

    slot->task_id = task_id;
    slot->stream_index = stream_index;
    slot->claim_start_cycles = claim_start_cycles;
    slot->claim_end_cycles = claim_end_cycles;
    slot->pending_wait_start_cycles = 0;
    slot->claim_worker_id = claim_worker_id;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = aicore_task_ticket_has_fanin_v1(ticket) ? 1 : 0;
    slot->claim_kind = static_cast<uint8_t>(claim_kind);
    slot->pending_slot = static_cast<uint8_t>(pending_slot);
    slot->generation = generation;
    aicore_writeback_cache_line_v0(slot);

    binding->task_id = task_id;
    binding->callable_address = callable_address;
    binding->dispatch_payload_offset =
        target->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(PTO2DispatchPayload);
    binding->owner_worker_id = target_worker_index;
    binding->kernel_id = kernel_id;
    binding->subtask_slot = subtask_slot;
    binding->pending_slot = static_cast<uint8_t>(pending_slot);
    binding->dispatch_generation = generation;
    binding->root_prepare_start_cycles = 0;
    binding->root_prepare_end_cycles = 0;
    binding->root_prepare_resolver_worker_id = 0;
    aicore_write_claim_binding_v1(sidecar_base, run_control, *binding);
    __gm__ AicoreTaskTraceCellV1 *trace = nullptr;
    if (trace_enabled) {
        __gm__ AicoreTaskTraceCellV1 *trace_cells =
            aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, resolver->trace_cells_offset);
        trace = &trace_cells[task_id];
        trace->claim_kind = static_cast<uint64_t>(claim_kind);
        trace->stream_index = stream_index;
        trace->claim_start_cycles = claim_start_cycles;
        trace->claim_end_cycles = claim_end_cycles;
        trace->claim_worker_id = claim_worker_id;
        aicore_writeback_cache_line_v0(trace);
    }
    aicore_cache_barrier_v0();
    if (trace != nullptr) trace->initialize_end_cycles = aicore_scheduler_cycles_v1();

    if (slot->has_fanin == 0) {
        *route = AicoreRouteResultV1::READY_TO_PUBLISH;
    } else {
        *route =
            aicore_route_task_v1(graph, sidecar_base, resolver, run_control, task_id, wake_stats, trace_enabled, false);
    }
    if (*route == AicoreRouteResultV1::WAITING) {
        slot->pending_wait_start_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(slot);
    }
    if (trace_enabled) {
        __gm__ AicoreTaskTraceCellV1 *trace_cells =
            aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, resolver->trace_cells_offset);
        trace_cells[task_id].route_end_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&trace_cells[task_id].initialize_end_cycles);
    }
    return *route != AicoreRouteResultV1::ERROR && *route != AicoreRouteResultV1::COMPLETED;
}

inline __aicore__ bool aicore_publish_prepared_dispatch_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, const AicoreTaskClaimBindingV1 &binding, bool root_task,
    AicoreRootPrepareStatsV1 *root_stats, bool trace_enabled = false
) {
    uint64_t prepare_start = root_task ? aicore_scheduler_cycles_v1() : 0;
    if (!aicore_finalize_ready_v1(graph, sidecar_base, resolver, run_control, binding, true, trace_enabled))
        return false;
    if (!root_task) return true;
    uint64_t prepare_end = aicore_scheduler_cycles_v1();
    if (root_stats != nullptr) {
        ++root_stats->resolve_count;
        root_stats->prepare_cycles += prepare_end - prepare_start;
    }
    if (trace_enabled) {
        __gm__ AicoreTaskClaimBindingV1 *published =
            aicore_claim_binding_at_v1(sidecar_base, run_control, binding.task_id);
        published->root_prepare_start_cycles = prepare_start;
        published->root_prepare_end_cycles = prepare_end;
        published->root_prepare_resolver_worker_id = resolver->worker_index;
        aicore_publish_cache_line_v0(published);
    }
    return true;
}

inline __aicore__ bool aicore_refill_completed_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, int64_t completed_task_id, AicoreWakeStatsV1 *wake_stats,
    AicoreRootPrepareStatsV1 *root_stats, bool trace_enabled = false, uint64_t *ticket_claim_count = nullptr,
    uint64_t *ticket_exhaustion_count = nullptr, uint64_t *claim_cycles = nullptr,
    uint64_t refill_start_cycles = 0
) {
    __gm__ AicoreTaskTraceCellV1 *refill_trace = nullptr;
    if (trace_enabled && completed_task_id >= 0 && static_cast<uint64_t>(completed_task_id) < graph.task_count) {
        __gm__ AicoreTaskTraceCellV1 *trace_cells =
            aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, resolver->trace_cells_offset);
        refill_trace = &trace_cells[completed_task_id];
        refill_trace->refill_start_cycles =
            refill_start_cycles != 0 ? refill_start_cycles : aicore_scheduler_cycles_v1();
        refill_trace->refill_resolver_worker_id = resolver->worker_index;
        refill_trace->refill_task_id = UINT64_MAX;
        refill_trace->refill_reserved = 0;
    }
    AicoreTaskClaimBindingV1 completed_binding{};
    if (!aicore_observe_claim_binding_v1(sidecar_base, run_control, completed_task_id, &completed_binding))
        return false;
    __gm__ AicoreWorkerContextV1 *target =
        aicore_worker_context_at_v1(sidecar_base, resolver, completed_binding.owner_worker_id);
    aicore_observe_cache_line_v0(target);
    aicore_observe_cache_line_v0(&target->sidecar_base_address);
    aicore_observe_cache_line_v0(&target->pending_task_id[0]);
    aicore_observe_cache_line_v0(&target->dispatch_slots_offset);
    __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(
        sidecar_base, resolver, completed_binding.owner_worker_id, completed_binding.pending_slot
    );
    aicore_observe_cache_line_v0(slot);
    if (completed_binding.dispatch_generation == 0 || slot->generation != completed_binding.dispatch_generation ||
        slot->task_id != completed_task_id) {
        aicore_record_scheduler_error_v1(
            run_control, completed_task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver
        );
        return false;
    }
    if (refill_trace != nullptr) refill_trace->refill_observe_end_cycles = aicore_scheduler_cycles_v1();

    __gm__ AicoreTaskStreamV1 *stream = aicore_worker_stream_v1(sidecar_base, resolver, target);
    __gm__ AicoreClaimPrefetchV1 *prefetch =
        aicore_claim_prefetch_at_v1(sidecar_base, resolver, completed_binding.owner_worker_id);
    AicoreTaskTicketV1 ticket{};
    uint64_t stream_index = 0;
    uint64_t claim_start = 0;
    uint64_t claim_end = 0;
    uint64_t claim_worker_id = resolver->worker_index;
    AicoreClaimKindV1 claim_kind = AicoreClaimKindV1::PREFETCH;
    bool valid =
        aicore_take_claim_prefetch_v1(prefetch, &ticket, &stream_index, &claim_start, &claim_end, &claim_worker_id);
    if (!valid) {
        claim_kind = AicoreClaimKindV1::REFILL_FALLBACK;
        claim_start = aicore_scheduler_cycles_v1();
        valid = aicore_claim_ticket_v1(sidecar_base, stream, &stream_index, &ticket);
        claim_end = aicore_scheduler_cycles_v1();
        if (claim_cycles != nullptr) *claim_cycles += claim_end - claim_start;
        if (valid && ticket_claim_count != nullptr) ++*ticket_claim_count;
        if (!valid && ticket_exhaustion_count != nullptr) ++*ticket_exhaustion_count;
    }
    if (refill_trace != nullptr) refill_trace->refill_prefetch_take_end_cycles = aicore_scheduler_cycles_v1();
    if (!valid) {
        aicore_retire_dispatch_slot_v1(slot);
        if (refill_trace != nullptr) {
            const uint64_t refill_end = aicore_scheduler_cycles_v1();
            refill_trace->refill_bind_route_end_cycles = refill_end;
            refill_trace->refill_payload_publish_end_cycles = refill_end;
            refill_trace->refill_next_prefetch_end_cycles = refill_end;
            refill_trace->refill_end_cycles = refill_end;
            aicore_publish_cache_line_v0(&refill_trace->refill_start_cycles);
        }
        return true;
    }
    if (refill_trace != nullptr) refill_trace->refill_task_id = ticket.task_id;

    AicoreTaskClaimBindingV1 binding{};
    AicoreRouteResultV1 route = AicoreRouteResultV1::ERROR;
    if (!aicore_prepare_dispatch_binding_v1(
            graph, sidecar_base, resolver, run_control, target, completed_binding.owner_worker_id,
            completed_binding.pending_slot, ticket, stream_index, claim_kind, claim_start, claim_end, claim_worker_id,
            wake_stats, &binding, &route, trace_enabled, true
        )) {
        return false;
    }
    if (refill_trace != nullptr) {
        refill_trace->refill_bind_route_end_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&refill_trace->completion_inbox_probe_start_cycles);
    }
    if (route == AicoreRouteResultV1::READY_TO_PUBLISH &&
        !aicore_publish_prepared_dispatch_v1(
            graph, sidecar_base, resolver, run_control, binding, !aicore_task_ticket_has_fanin_v1(ticket), root_stats,
            trace_enabled
        )) {
        return false;
    }
    if (refill_trace != nullptr) refill_trace->refill_payload_publish_end_cycles = aicore_scheduler_cycles_v1();
    bool prefetch_filled = false;
    bool prefetch_exhausted = false;
    if (!aicore_fill_claim_prefetch_v1(
            sidecar_base, stream, prefetch, resolver->worker_index, &prefetch_filled, &prefetch_exhausted,
            ticket_claim_count, ticket_exhaustion_count, claim_cycles
        )) {
        return false;
    }
    if (refill_trace != nullptr) {
        refill_trace->refill_next_prefetch_end_cycles = aicore_scheduler_cycles_v1();
        refill_trace->refill_end_cycles = aicore_scheduler_cycles_v1();
        aicore_publish_cache_line_v0(&refill_trace->refill_start_cycles);
    }
    (void)prefetch_filled;
    (void)prefetch_exhausted;
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
            aicore_record_scheduler_error_v1(run_control, waiter, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
            return false;
        }
        __gm__ AicoreTaskControlV1 *waiter_control = aicore_task_control_at_v1(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v1(waiter_control);
        if (stats != nullptr) ++stats->wake_migrate_count;
        AicoreRouteResultV1 route = aicore_route_task_v1(graph, sidecar_base, context, run_control, waiter, stats);
        if (route == AicoreRouteResultV1::ERROR) {
            return false;
        }
        if (route == AicoreRouteResultV1::READY_TO_PUBLISH) {
            AicoreTaskClaimBindingV1 binding{};
            if (!aicore_observe_claim_binding_v1(sidecar_base, run_control, waiter, &binding) ||
                !aicore_finalize_ready_v1(graph, sidecar_base, context, run_control, binding, true)) {
                return false;
            }
        }
        waiter = next;
    }
    return true;
}

inline __aicore__ bool aicore_enqueue_completion_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, int64_t task_id, AicoreCompletionStatsV1 *stats,
    bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return false;
    }
    if (resolver_count == 0) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    if (trace_enabled) control->completion_enqueue_cycles = aicore_scheduler_cycles_v1();
    uint64_t inbox_index = static_cast<uint64_t>(task_id) % resolver_count;
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
            aicore_record_scheduler_error_v1(run_control, waiter, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
            return false;
        }
        __gm__ AicoreTaskControlV1 *waiter_control = aicore_task_control_at_v1(sidecar_base, context, waiter);
        int64_t next = aicore_observe_next_waiter_v1(waiter_control);
        if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        AicoreRouteResultV1 route =
            aicore_route_task_v1(graph, sidecar_base, context, run_control, waiter, wake_stats, trace_enabled);
        if (route == AicoreRouteResultV1::ERROR) {
            return false;
        }
        if (route == AicoreRouteResultV1::READY_TO_PUBLISH) {
            AicoreTaskClaimBindingV1 binding{};
            if (!aicore_observe_claim_binding_v1(sidecar_base, run_control, waiter, &binding) ||
                !aicore_finalize_ready_v1(graph, sidecar_base, context, run_control, binding, true, trace_enabled)) {
                return false;
            }
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
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint64_t *victim_cursor,
    AicoreWakeStatsV1 *wake_stats, AicoreCompletionStatsV1 *completion_stats, bool *made_progress,
    bool trace_enabled = false, AicoreRootPrepareStatsV1 *root_stats = nullptr, uint64_t *ticket_claim_count = nullptr,
    uint64_t *ticket_exhaustion_count = nullptr, uint64_t *claim_cycles = nullptr
) {
    if (made_progress != nullptr) *made_progress = false;
    if (context->core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIV)) return true;
    if (resolver_count == 0 || victim_cursor == nullptr || context->inbox_index >= resolver_count) return true;
    uint64_t inbox_index = context->inbox_index;
    __gm__ AicoreCompletionInboxV1 *inbox = aicore_completion_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t task_id = AICORE_COMPLETION_INBOX_EMPTY_V1;
    uint64_t completion_inbox_probe_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
    uint64_t completion_inbox_detach_start = 0;
    uint64_t completion_inbox_detach_end = 0;
    uint64_t completion_inbox_index = inbox_index;
    bool completion_inbox_stolen = false;
    if (aicore_gm_load_v0(inbox->head) != AICORE_COMPLETION_INBOX_EMPTY_V1) {
        completion_inbox_detach_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        task_id = aicore_gm_exchange_v0(inbox->head, AICORE_COMPLETION_INBOX_EMPTY_V1);
        completion_inbox_detach_end = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
    }
    if (task_id == AICORE_COMPLETION_INBOX_EMPTY_V1 && resolver_count > 1) {
        uint64_t victim = *victim_cursor % resolver_count;
        *victim_cursor = (victim + 1) % resolver_count;
        if (victim == inbox_index) {
            victim = *victim_cursor;
            *victim_cursor = (victim + 1) % resolver_count;
        }
        inbox = aicore_completion_inbox_at_v1(sidecar_base, context, victim);
        completion_inbox_probe_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        completion_inbox_index = victim;
        if (aicore_gm_load_v0(inbox->head) != AICORE_COMPLETION_INBOX_EMPTY_V1) {
            completion_inbox_detach_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
            task_id = aicore_gm_exchange_v0(inbox->head, AICORE_COMPLETION_INBOX_EMPTY_V1);
            completion_inbox_detach_end = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        }
        if (task_id != AICORE_COMPLETION_INBOX_EMPTY_V1 && completion_stats != nullptr) {
            ++completion_stats->steal_count;
        }
        completion_inbox_stolen = task_id != AICORE_COMPLETION_INBOX_EMPTY_V1;
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
        const uint64_t completion_node_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        __gm__ AicoreTaskTraceCellV1 *completion_trace = nullptr;
        if (trace_enabled) {
            __gm__ AicoreTaskTraceCellV1 *trace_cells =
                aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
            completion_trace = &trace_cells[task_id];
            completion_trace->completion_inbox_probe_start_cycles = completion_inbox_probe_start;
            completion_trace->completion_inbox_detach_start_cycles = completion_inbox_detach_start;
            completion_trace->completion_inbox_detach_end_cycles = completion_inbox_detach_end;
            completion_trace->completion_node_start_cycles = completion_node_start;
            completion_trace->completion_inbox_index = completion_inbox_index;
            completion_trace->completion_inbox_stolen = completion_inbox_stolen ? 1 : 0;
            completion_trace->refill_resolver_worker_id = context->worker_index;
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
        const uint64_t refill_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        AicoreTaskClaimBindingV1 completed_binding{};
        if (aicore_observe_claim_binding_v1(sidecar_base, run_control, task_id, &completed_binding) &&
            completed_binding.dispatch_generation != 0 &&
            !aicore_refill_completed_slot_v1(
                graph, sidecar_base, context, run_control, task_id, wake_stats, root_stats, trace_enabled,
                ticket_claim_count, ticket_exhaustion_count, claim_cycles, refill_start
            )) {
            return false;
        }
        if (completion_trace != nullptr) {
            aicore_publish_cache_line_v0(&completion_trace->completion_inbox_probe_start_cycles);
        }
        if (!aicore_resolve_completion_v1(
                graph, sidecar_base, context, run_control, task_id, wake_stats, completion_stats, trace_enabled
            )) {
            return false;
        }
        ++batch_count;
        task_id = next;
        completion_inbox_probe_start = 0;
        completion_inbox_detach_start = 0;
        completion_inbox_detach_end = 0;
        completion_inbox_stolen = false;
    }
    if (completion_stats != nullptr) completion_stats->resolve_count += batch_count;
    aicore_gm_fetch_add_v0(run_control->resolved_task_count, batch_count);
    if (made_progress != nullptr) *made_progress = true;
    return true;
}

inline __aicore__ bool aicore_enqueue_root_prepare_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, int64_t task_id, AicoreRootPrepareStatsV1 *stats
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count || resolver_count == 0) {
        aicore_record_scheduler_error_v1(
            run_control, task_id,
            resolver_count == 0 ? AicoreRootStatusV0::INVALID_ARGUMENTS : AicoreRootStatusV0::INVALID_TASK_ID, &graph,
            context
        );
        return false;
    }
    __gm__ uint8_t *payload = aicore_graph_payload_v0(graph, task_id);
    if (*reinterpret_cast<__gm__ int32_t *>(payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0) != 0) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (aicore_gm_load_v0(control->state) != static_cast<int64_t>(AicoreTaskStateV1::BLOCKED) ||
        aicore_gm_load_v0(control->completion_next) != AICORE_COMPLETION_LINK_UNPUBLISHED_V1) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }

    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::PREPARING));
    uint64_t inbox_index = static_cast<uint64_t>(task_id) % resolver_count;
    __gm__ AicoreRootPrepareInboxV1 *inbox = aicore_root_prepare_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t previous = aicore_gm_exchange_v0(inbox->head, task_id);
    control->completion_next = previous;
    aicore_publish_cache_line_v0(&control->next_waiter);
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool aicore_resolve_root_prepare_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, int64_t task_id, AicoreRootPrepareStatsV1 *stats, bool trace_enabled = false
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_TASK_ID, &graph, context);
        return false;
    }
    __gm__ uint8_t *graph_payload = aicore_graph_payload_v0(graph, task_id);
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, task_id);
    if (*reinterpret_cast<__gm__ int32_t *>(graph_payload + AICORE_GRAPH_FANIN_COUNT_OFFSET_V0) != 0 ||
        aicore_gm_load_v0(control->state) != static_cast<int64_t>(AicoreTaskStateV1::PREPARING)) {
        aicore_record_scheduler_error_v1(run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context);
        return false;
    }

    uint64_t prepare_start = aicore_scheduler_cycles_v1();
    AicoreTaskClaimBindingV1 binding{};
    if (!aicore_observe_claim_binding_v1(sidecar_base, run_control, task_id, &binding) ||
        !aicore_materialize_claim_payload_v1(graph, sidecar_base, context, run_control, binding, true)) {
        return false;
    }
    uint64_t prepare_end = aicore_scheduler_cycles_v1();
    if (stats != nullptr && prepare_end >= prepare_start) stats->prepare_cycles += prepare_end - prepare_start;
    if (trace_enabled) {
        __gm__ AicoreTaskClaimBindingV1 *published = aicore_claim_binding_at_v1(sidecar_base, run_control, task_id);
        published->root_prepare_start_cycles = prepare_start;
        published->root_prepare_end_cycles = prepare_end;
        published->root_prepare_resolver_worker_id = context->worker_index;
        aicore_publish_cache_line_v0(published);
        control->ready_publish_cycles = aicore_scheduler_cycles_v1();
    }

    control->completion_next = AICORE_COMPLETION_LINK_UNPUBLISHED_V1;
    aicore_publish_cache_line_v0(&control->next_waiter);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    return true;
}

inline __aicore__ bool aicore_service_root_prepare_inboxes_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint64_t *victim_cursor,
    AicoreRootPrepareStatsV1 *stats, bool *made_progress, bool trace_enabled = false
) {
    if (made_progress != nullptr) *made_progress = false;
    if (context->core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIV)) return true;
    if (resolver_count == 0 || victim_cursor == nullptr || context->inbox_index >= resolver_count) return true;

    uint64_t inbox_index = context->inbox_index;
    __gm__ AicoreRootPrepareInboxV1 *inbox = aicore_root_prepare_inbox_at_v1(sidecar_base, context, inbox_index);
    int64_t task_id = AICORE_ROOT_PREPARE_INBOX_EMPTY_V1;
    if (aicore_gm_load_v0(inbox->head) != AICORE_ROOT_PREPARE_INBOX_EMPTY_V1) {
        task_id = aicore_gm_exchange_v0(inbox->head, AICORE_ROOT_PREPARE_INBOX_EMPTY_V1);
    }
    if (task_id == AICORE_ROOT_PREPARE_INBOX_EMPTY_V1 && resolver_count > 1) {
        uint64_t victim = *victim_cursor % resolver_count;
        *victim_cursor = (victim + 1) % resolver_count;
        if (victim == inbox_index) {
            victim = *victim_cursor;
            *victim_cursor = (victim + 1) % resolver_count;
        }
        inbox = aicore_root_prepare_inbox_at_v1(sidecar_base, context, victim);
        if (aicore_gm_load_v0(inbox->head) != AICORE_ROOT_PREPARE_INBOX_EMPTY_V1) {
            task_id = aicore_gm_exchange_v0(inbox->head, AICORE_ROOT_PREPARE_INBOX_EMPTY_V1);
        }
        if (task_id != AICORE_ROOT_PREPARE_INBOX_EMPTY_V1 && stats != nullptr) ++stats->steal_count;
    }
    if (task_id == AICORE_ROOT_PREPARE_INBOX_EMPTY_V1) return true;
    if (stats != nullptr) ++stats->batch_count;

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
        if (stats != nullptr) stats->link_wait_count += waits;
        if (stats != nullptr && waits > stats->link_wait_max) stats->link_wait_max = waits;
        if (next == AICORE_COMPLETION_LINK_UNPUBLISHED_V1 || next < AICORE_ROOT_PREPARE_INBOX_EMPTY_V1) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context
            );
            return false;
        }
        if (!aicore_resolve_root_prepare_v1(graph, sidecar_base, context, run_control, task_id, stats, trace_enabled)) {
            return false;
        }
        ++batch_count;
        task_id = next;
    }
    if (stats != nullptr) stats->resolve_count += batch_count;
    if (made_progress != nullptr) *made_progress = true;
    return true;
}
