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
#include "aicore_task_profiling_v1.h"
#include "aicore_ticket_scheduler_v1.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 128;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

static_assert(RUNTIME_MAX_FUNC_ID <= static_cast<int32_t>(UINT16_MAX) + 1, "pending kernel id cache is too narrow");
static_assert(AICORE_CALLABLE_CAPACITY_V1 == RUNTIME_MAX_FUNC_ID, "AICore final offload layout mismatch");

struct AicoreWorkerStatsV1 {
    uint64_t seeded_task_count{0};
    uint64_t ticket_claim_count{0};
    uint64_t ticket_exhaustion_count{0};
    uint64_t executed_task_count{0};
    uint64_t pending_peak{0};
    uint64_t task_state_poll_count{0};
    AicoreWakeStatsV1 wake{};
    AicoreCompletionStatsV1 completion{};
    AicoreRootPrepareStatsV1 root_prepare{};
    uint64_t idle_iteration_count{0};
    uint64_t backoff_cycles{0};
    uint64_t claim_cycles{0};
    uint64_t dependency_wait_cycles{0};
    uint64_t payload_cycles{0};
    uint64_t kernel_cycles{0};
    uint64_t completion_enqueue_cycles{0};
    uint64_t drain_cycles{0};
    uint64_t drain_start_cycles{0};
    uint64_t drain_end_cycles{0};
    uint64_t executor_drain_publish_start_cycles{0};
    uint64_t executor_drain_publish_end_cycles{0};
    uint64_t exit_wait_start_cycles{0};
    uint64_t exit_observed_cycles{0};
    uint64_t final_stats_publish_start_cycles{0};
    uint64_t final_stats_publish_end_cycles{0};
    uint64_t exit_ack_publish_cycles{0};
};

__aicore__ __attribute__((always_inline)) void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((always_inline)) void local_backoff(uint32_t iterations) {
    for (volatile uint32_t spin = 0; spin < iterations; ++spin) {}
}

__aicore__ __attribute__((always_inline)) void
publish_worker_stats(__gm__ AicoreWorkerContextV1 *context, const AicoreWorkerStatsV1 &stats) {
    context->seeded_task_count = stats.seeded_task_count;
    context->ticket_claim_count = stats.ticket_claim_count;
    context->ticket_exhaustion_count = stats.ticket_exhaustion_count;
    context->executed_task_count = stats.executed_task_count;
    context->pending_peak = stats.pending_peak;
    context->task_state_poll_count = stats.task_state_poll_count;
    context->fanin_state_load_count = stats.wake.fanin_state_load_count;
    context->wake_register_count = stats.wake.wake_register_count;
    context->idle_iteration_count = stats.idle_iteration_count;
    context->backoff_cycles = stats.backoff_cycles;
    context->claim_cycles = stats.claim_cycles;
    context->dependency_wait_cycles = stats.dependency_wait_cycles;
    context->payload_cycles = stats.payload_cycles;
    context->kernel_cycles = stats.kernel_cycles;
    context->completion_enqueue_cycles = stats.completion_enqueue_cycles;
    context->drain_cycles = stats.drain_cycles;
    aicore_publish_cache_line_v0(&context->seeded_task_count);
    context->drain_start_cycles = stats.drain_start_cycles;
    context->drain_end_cycles = stats.drain_end_cycles;
    context->wake_cas_retry_count = stats.wake.wake_cas_retry_count;
    context->wake_closed_retry_count = stats.wake.wake_closed_retry_count;
    context->wake_migrate_count = stats.wake.wake_migrate_count;
    context->wake_close_count = stats.wake.wake_close_count;
    context->completion_enqueue_count = stats.completion.enqueue_count;
    context->completion_batch_count = stats.completion.batch_count;
    context->completion_resolve_count = stats.completion.resolve_count;
    context->completion_steal_count = stats.completion.steal_count;
    context->completion_link_wait_count = stats.completion.link_wait_count;
    context->completion_link_wait_max = stats.completion.link_wait_max;
    context->completion_lag_cycles = stats.completion.completion_lag_cycles;
    context->completion_lag_max_cycles = stats.completion.completion_lag_max_cycles;
    context->ready_to_kernel_cycles = stats.completion.ready_to_kernel_cycles;
    context->ready_to_kernel_max_cycles = stats.completion.ready_to_kernel_max_cycles;
    aicore_publish_cache_line_v0(&context->drain_start_cycles);
    context->executor_drain_publish_start_cycles = stats.executor_drain_publish_start_cycles;
    context->executor_drain_publish_end_cycles = stats.executor_drain_publish_end_cycles;
    context->exit_wait_start_cycles = stats.exit_wait_start_cycles;
    context->exit_observed_cycles = stats.exit_observed_cycles;
    context->final_stats_publish_start_cycles = stats.final_stats_publish_start_cycles;
    context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
    context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
    context->root_prepare_enqueue_count = stats.root_prepare.enqueue_count;
    context->root_prepare_batch_count = stats.root_prepare.batch_count;
    context->root_prepare_resolve_count = stats.root_prepare.resolve_count;
    context->root_prepare_steal_count = stats.root_prepare.steal_count;
    context->root_prepare_link_wait_count = stats.root_prepare.link_wait_count;
    context->root_prepare_link_wait_max = stats.root_prepare.link_wait_max;
    context->root_prepare_cycles = stats.root_prepare.prepare_cycles;
    aicore_publish_cache_line_v0(&context->executor_drain_publish_start_cycles);
    aicore_publish_cache_line_v0(&context->root_prepare_batch_count);
}

__aicore__ __attribute__((always_inline)) void commit_task_trace(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, const AicorePendingSlotV1 &pending,
    uint64_t pending_wait_end, uint64_t ready_scan_start, uint64_t ready_observe, uint64_t payload_start,
    uint64_t kernel_start, uint64_t kernel_end, uint64_t completion_end, uint64_t completion_bookkeeping_end,
    uint64_t previous_trace_commit_end, uint64_t payload_base_load_start, uint64_t payload_cache_control_end,
    uint64_t payload_cache_invalidate_end, uint64_t payload_observe_end, uint64_t inter_task_poll_cycles,
    uint64_t inter_task_publication_poll_cycles, uint64_t inter_task_backoff_cycles
) {
    __gm__ AicoreTaskTraceCellV1 *trace_cells =
        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
    __gm__ AicoreTaskTraceCellV1 *trace = &trace_cells[pending.task_id];
    aicore_observe_cache_line_v0(&trace->completion_bookkeeping_end_cycles);
    trace->valid = 0;
    trace->claim_kind = static_cast<uint64_t>(pending.claim_kind);
    trace->worker_id = context->worker_index;
    trace->core_type = static_cast<uint64_t>(context->core_type);
    trace->task_id = static_cast<uint64_t>(pending.task_id);
    trace->stream_index = pending.stream_index;
    trace->claim_start_cycles = pending.claim_start_cycles;
    trace->claim_end_cycles = pending.claim_end_cycles;
    trace->pending_wait_start_cycles = pending.pending_wait_start_cycles;
    trace->pending_wait_end_cycles = pending_wait_end;
    trace->payload_start_cycles = payload_start;
    trace->kernel_start_cycles = kernel_start;
    trace->kernel_end_cycles = kernel_end;
    trace->completion_end_cycles = completion_end;
    trace->ready_scan_start_cycles = ready_scan_start;
    trace->ready_observe_cycles = ready_observe;
    trace->completion_bookkeeping_end_cycles = completion_bookkeeping_end;
    trace->payload_base_load_start_cycles = payload_base_load_start;
    trace->inter_task_publication_poll_cycles = inter_task_publication_poll_cycles;
    trace->previous_trace_commit_end_cycles = previous_trace_commit_end;
    trace->payload_cache_control_end_cycles = payload_cache_control_end;
    trace->payload_cache_invalidate_end_cycles = payload_cache_invalidate_end;
    trace->payload_observe_end_cycles = payload_observe_end;
    trace->inter_task_poll_cycles = inter_task_poll_cycles;
    trace->inter_task_backoff_cycles = inter_task_backoff_cycles;
    aicore_publish_cache_line_v0(&trace->completion_bookkeeping_end_cycles);
    aicore_publish_cache_line_v0(&trace->aicore_entry_cycles);
    aicore_publish_cache_line_v0(&trace->pending_wait_start_cycles);
    trace->valid = 1;
    aicore_publish_cache_line_v0(trace);
}

__aicore__ bool bootstrap_dispatch_worker(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, __gm__ AicoreWorkerContextV1 *target, uint64_t target_worker_index,
    AicoreWorkerStatsV1 *stats, bool trace_enabled
) {
    __gm__ AicoreTaskStreamV1 *stream = aicore_worker_stream_v1(sidecar_base, resolver, target);
    const bool executor_worker = static_cast<uint64_t>(target->type_rank) < stream->initial_ticket_count;
    if (!executor_worker) {
        for (uint32_t slot_index = 0; slot_index < AICORE_PENDING_SLOT_COUNT_V1; ++slot_index) {
            aicore_retire_dispatch_slot_v1(
                aicore_dispatch_slot_at_v1(sidecar_base, resolver, target_worker_index, slot_index)
            );
        }
        aicore_gm_publish_v0(target->bootstrap_done, UINT64_C(1));
        return true;
    }

    __gm__ AicoreTaskTicketV1 *tickets = aicore_sidecar_at_v1<AicoreTaskTicketV1>(sidecar_base, stream->tickets_offset);
    AicoreTaskTicketV1 seed_ticket{};
    AicoreTaskClaimBindingV1 seed_binding{};
    AicoreRouteResultV1 seed_route = AicoreRouteResultV1::ERROR;
    uint64_t seed_time = aicore_scheduler_cycles_v1();
    aicore_load_task_ticket_v1(&tickets[target->type_rank], &seed_ticket);
    if (!aicore_prepare_dispatch_binding_v1(
            graph, sidecar_base, resolver, run_control, target, target_worker_index, 0, seed_ticket,
            static_cast<uint64_t>(target->type_rank), AicoreClaimKindV1::SEED, seed_time, seed_time,
            resolver->worker_index, &stats->wake, &seed_binding, &seed_route, trace_enabled
        )) {
        return false;
    }
    ++stats->seeded_task_count;
    if (seed_route == AicoreRouteResultV1::READY_TO_PUBLISH &&
        !aicore_publish_prepared_dispatch_v1(
            graph, sidecar_base, resolver, run_control, seed_binding, !aicore_task_ticket_has_fanin_v1(seed_ticket),
            &stats->root_prepare, trace_enabled
        )) {
        return false;
    }

    AicoreTaskTicketV1 second_ticket{};
    uint64_t stream_index = 0;
    uint64_t claim_start = aicore_scheduler_cycles_v1();
    bool second_valid = aicore_claim_ticket_v1(sidecar_base, stream, &stream_index, &second_ticket);
    uint64_t claim_end = aicore_scheduler_cycles_v1();
    stats->claim_cycles += claim_end - claim_start;
    if (second_valid) {
        AicoreTaskClaimBindingV1 second_binding{};
        AicoreRouteResultV1 second_route = AicoreRouteResultV1::ERROR;
        if (!aicore_prepare_dispatch_binding_v1(
                graph, sidecar_base, resolver, run_control, target, target_worker_index, 1, second_ticket, stream_index,
                AicoreClaimKindV1::TICKET, claim_start, claim_end, resolver->worker_index, &stats->wake, &second_binding,
                &second_route, trace_enabled
            )) {
            return false;
        }
        ++stats->ticket_claim_count;
        if (second_route == AicoreRouteResultV1::READY_TO_PUBLISH &&
            !aicore_publish_prepared_dispatch_v1(
                graph, sidecar_base, resolver, run_control, second_binding,
                !aicore_task_ticket_has_fanin_v1(second_ticket), &stats->root_prepare, trace_enabled
            )) {
            return false;
        }
    } else {
        ++stats->ticket_exhaustion_count;
        aicore_retire_dispatch_slot_v1(aicore_dispatch_slot_at_v1(sidecar_base, resolver, target_worker_index, 1));
    }

    __gm__ AicoreClaimPrefetchV1 *prefetch = aicore_claim_prefetch_at_v1(sidecar_base, resolver, target_worker_index);
    bool prefetch_filled = false;
    bool prefetch_exhausted = false;
    if (second_valid) {
        if (!aicore_fill_claim_prefetch_v1(
                sidecar_base, stream, prefetch, resolver->worker_index, &prefetch_filled, &prefetch_exhausted,
                &stats->ticket_claim_count, &stats->ticket_exhaustion_count, &stats->claim_cycles
            )) {
            return false;
        }
    } else {
        aicore_gm_publish_v0(prefetch->state, static_cast<uint64_t>(AicoreClaimPrefetchStateV1::EXHAUSTED));
        prefetch_exhausted = true;
    }
    (void)prefetch_filled;
    (void)prefetch_exhausted;

    aicore_gm_publish_v0(target->bootstrap_done, UINT64_C(1));
    return true;
}

__aicore__ bool bootstrap_owned_workers(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, AicoreWorkerStatsV1 *stats, bool trace_enabled
) {
    if (resolver_count == 0 || resolver->inbox_index >= resolver_count) return false;
    for (uint64_t worker_id = 0; worker_id < resolver->runtime_worker_count; ++worker_id) {
        if (worker_id % resolver_count != resolver->inbox_index) continue;
        __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
        aicore_observe_cache_line_v0(target);
        aicore_observe_cache_line_v0(&target->sidecar_base_address);
        aicore_observe_cache_line_v0(&target->pending_task_id[0]);
        aicore_observe_cache_line_v0(&target->dispatch_slots_offset);
        if (target->active == 0) continue;
        if (!bootstrap_dispatch_worker(
                graph, sidecar_base, resolver, run_control, target, worker_id, stats, trace_enabled
            )) {
            return false;
        }
    }
    return true;
}

__aicore__ bool run_offloaded_dispatch_loop(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, AicoreTaskProfilingStateV1 *task_profiling, AicoreWorkerStatsV1 *stats,
    bool trace_enabled, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles,
    uint64_t register_release_cycles, uint64_t descriptor_cache_observed_cycles
) {
    const uint64_t resolver_count = aicore_gm_load_v0(run_control->aiv_active_worker_count);
    const bool resolver_worker = context->core_type == static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    if (resolver_count == 0) {
        aicore_record_scheduler_error_v1(
            run_control, AICORE_TASK_ID_INVALID_V1, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context
        );
        return false;
    }
    if (resolver_worker &&
        !bootstrap_owned_workers(graph, sidecar_base, context, run_control, resolver_count, stats, trace_enabled)) {
        return false;
    }

    uint64_t seen_publication[AICORE_PENDING_SLOT_COUNT_V1]{};
    bool retired[AICORE_PENDING_SLOT_COUNT_V1]{};
    uint64_t victim_cursor = resolver_worker ? (context->inbox_index + 1) % resolver_count : 0;
    uint64_t previous_trace_commit_end = 0;
    uint64_t inter_task_poll_cycles = 0;
    uint64_t inter_task_publication_poll_cycles = 0;
    uint64_t inter_task_backoff_cycles = 0;
    uint32_t scan_start = 0;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    bool common_profile_recorded = false;
    bool executor_drained = false;
    bool bootstrap_observed = false;

    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (trace_enabled) stats->exit_observed_cycles = get_sys_cnt_aicore();
            break;
        }
        if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
            scheduler_error_poll_count = 0;
            if (aicore_gm_load_v0(run_control->scheduler_error) != 0) return false;
        }

        bool completion_progress = false;
        if (resolver_worker &&
            !aicore_service_completion_inboxes_v1(
                graph, sidecar_base, context, run_control, resolver_count, &victim_cursor, &stats->wake,
                &stats->completion, &completion_progress, trace_enabled, &stats->root_prepare,
                &stats->ticket_claim_count, &stats->ticket_exhaustion_count, &stats->claim_cycles
            )) {
            return false;
        }
        if (completion_progress) {
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }

        if (!bootstrap_observed) {
            if (aicore_gm_load_v0(context->bootstrap_done) == 0) {
                ++stats->idle_iteration_count;
                uint64_t backoff_start = get_sys_cnt_aicore();
                local_backoff(backoff_iterations);
                stats->backoff_cycles += get_sys_cnt_aicore() - backoff_start;
                if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
                continue;
            }
            bootstrap_observed = true;
        }

        int32_t ready_slot = -1;
        uint64_t ready_publication = 0;
        uint64_t ready_scan_start = get_sys_cnt_aicore();
        uint64_t scan_publication_poll_cycles = 0;
        for (uint32_t offset = 0; offset < AICORE_PENDING_SLOT_COUNT_V1; ++offset) {
            uint32_t slot_index = (scan_start + offset) % AICORE_PENDING_SLOT_COUNT_V1;
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, slot_index);
            uint64_t slot_poll_start = trace_enabled ? get_sys_cnt_aicore() : 0;
            uint64_t publication = aicore_gm_load_v0(slot->publication);
            ++stats->task_state_poll_count;
            if (publication == seen_publication[slot_index]) {
                if (trace_enabled) scan_publication_poll_cycles += get_sys_cnt_aicore() - slot_poll_start;
                continue;
            }
            AicoreDispatchPublicationV1 state = aicore_dispatch_state_v1(publication);
            if (state == AicoreDispatchPublicationV1::RETIRED) {
                seen_publication[slot_index] = publication;
                retired[slot_index] = true;
                if (trace_enabled) scan_publication_poll_cycles += get_sys_cnt_aicore() - slot_poll_start;
                continue;
            }
            if (state == AicoreDispatchPublicationV1::READY) {
                ready_slot = static_cast<int32_t>(slot_index);
                ready_publication = publication;
                if (trace_enabled) scan_publication_poll_cycles += get_sys_cnt_aicore() - slot_poll_start;
                break;
            }
            if (trace_enabled) scan_publication_poll_cycles += get_sys_cnt_aicore() - slot_poll_start;
        }

        if (ready_slot >= 0) {
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, slot_index);
            aicore_observe_cache_line_v0(slot);
            if (slot->pending_slot != slot_index || slot->task_id < 0 ||
                static_cast<uint64_t>(slot->task_id) >= graph.task_count ||
                slot->generation != aicore_dispatch_generation_v1(ready_publication)) {
                aicore_record_scheduler_error_v1(
                    run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context
                );
                return false;
            }
            seen_publication[slot_index] = ready_publication;
            retired[slot_index] = false;

            AicorePendingSlotV1 pending{};
            aicore_pending_clear_v1(&pending);
            pending.task_id = slot->task_id;
            pending.kernel_id = slot->kernel_id;
            pending.subtask_slot = slot->subtask_slot;
            pending.has_fanin = slot->has_fanin;
            pending.payload_needs_observe = 1;
            pending.stream_index = slot->stream_index;
            pending.claim_start_cycles = slot->claim_start_cycles;
            pending.claim_end_cycles = slot->claim_end_cycles;
            pending.pending_wait_start_cycles = slot->pending_wait_start_cycles;
            pending.claim_kind = static_cast<AicoreClaimKindV1>(slot->claim_kind);

            uint64_t ready_observe = get_sys_cnt_aicore();
            uint64_t pending_wait_end = pending.pending_wait_start_cycles == 0 ? 0 : ready_observe;
            if (pending_wait_end != 0) {
                stats->dependency_wait_cycles += pending_wait_end - pending.pending_wait_start_cycles;
            }
            uint64_t payload_base_load_start = get_sys_cnt_aicore();
            __gm__ PTO2DispatchPayload *dispatch_payload = aicore_sidecar_at_v1<PTO2DispatchPayload>(
                sidecar_base,
                context->dispatch_payload_offset + static_cast<uint64_t>(slot_index) * sizeof(PTO2DispatchPayload)
            );
            uint64_t payload_start = get_sys_cnt_aicore();
            aicore_observe_dispatch_payload_control_v1(dispatch_payload);
            uint64_t payload_cache_control_end = trace_enabled ? get_sys_cnt_aicore() : 0;
            aicore_observe_dispatch_payload_arguments_v1(dispatch_payload);
            uint64_t payload_cache_invalidate_end = trace_enabled ? get_sys_cnt_aicore() : 0;
            aicore_observe_dispatch_payload_barrier_v1();
            uint64_t payload_observe_end = trace_enabled ? get_sys_cnt_aicore() : 0;
            OUT_OF_ORDER_STORE_BARRIER();

            __gm__ ChipSwimlaneAicoreTaskRecord *profile_record =
                common_profile_recorded ? nullptr : aicore_task_profiling_reserve_v1(task_profiling);
            uint64_t kernel_start = get_sys_cnt_aicore();
            if (trace_enabled) {
                __gm__ AicoreTaskControlV1 *task_control =
                    aicore_task_control_at_v1(sidecar_base, context, pending.task_id);
                aicore_observe_cache_line_v0(&task_control->next_waiter);
                if (task_control->ready_publish_cycles != 0 && kernel_start >= task_control->ready_publish_cycles) {
                    uint64_t lag = kernel_start - task_control->ready_publish_cycles;
                    stats->completion.ready_to_kernel_cycles += lag;
                    if (lag > stats->completion.ready_to_kernel_max_cycles) {
                        stats->completion.ready_to_kernel_max_cycles = lag;
                    }
                }
            }
            execute_task(dispatch_payload);
            uint64_t kernel_end = get_sys_cnt_aicore();
            // Completion transfers this slot back to the Scheduler. Release every cache line before publishing
            // that handoff so a refill can never race a stale Executor write-back.
            aicore_publish_dispatch_payload_v1(dispatch_payload);
            uint64_t completion_start = get_sys_cnt_aicore();
            if (!aicore_enqueue_completion_v1(
                    graph, sidecar_base, context, run_control, resolver_count, pending.task_id, &stats->completion,
                    trace_enabled
                )) {
                return false;
            }
            uint64_t completion_end = get_sys_cnt_aicore();
            aicore_task_profiling_commit_v1(
                profile_record, static_cast<uint64_t>(pending.task_id), pending.claim_end_cycles, kernel_start,
                kernel_end
            );
            if (profile_record != nullptr) common_profile_recorded = true;
            ++stats->executed_task_count;
            stats->payload_cycles += kernel_start - payload_start;
            stats->kernel_cycles += kernel_end - kernel_start;
            stats->completion_enqueue_cycles += completion_end - completion_start;
            if (stats->pending_peak < AICORE_PENDING_SLOT_COUNT_V1) stats->pending_peak = AICORE_PENDING_SLOT_COUNT_V1;
            scan_start = (slot_index + 1) % AICORE_PENDING_SLOT_COUNT_V1;
            backoff_iterations = kInitialBackoffIterations;

            if (trace_enabled) {
                if (pending.claim_kind == AicoreClaimKindV1::SEED) {
                    __gm__ AicoreTaskTraceCellV1 *trace_cells =
                        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
                    __gm__ AicoreTaskTraceCellV1 *trace = &trace_cells[pending.task_id];
                    trace->aicore_entry_cycles = aicore_entry_cycles;
                    trace->handshake_publish_cycles = handshake_publish_cycles;
                    trace->register_release_cycles = register_release_cycles;
                    trace->descriptor_cache_observed_cycles = descriptor_cache_observed_cycles;
                }
                uint64_t bookkeeping_end = get_sys_cnt_aicore();
                commit_task_trace(
                    sidecar_base, context, pending, pending_wait_end, ready_scan_start, ready_observe, payload_start,
                    kernel_start, kernel_end, completion_end, bookkeeping_end, previous_trace_commit_end,
                    payload_base_load_start, payload_cache_control_end, payload_cache_invalidate_end,
                    payload_observe_end, inter_task_poll_cycles, inter_task_publication_poll_cycles,
                    inter_task_backoff_cycles
                );
                previous_trace_commit_end = get_sys_cnt_aicore();
                inter_task_poll_cycles = 0;
                inter_task_publication_poll_cycles = 0;
                inter_task_backoff_cycles = 0;
            }
            continue;
        }

        uint64_t failed_scan_end = trace_enabled ? get_sys_cnt_aicore() : ready_scan_start;
        inter_task_poll_cycles += failed_scan_end - ready_scan_start;
        inter_task_publication_poll_cycles += scan_publication_poll_cycles;

        if (retired[0] && retired[1] && !executor_drained) {
            if (aicore_gm_load_v0(run_control->scheduler_error) != 0) return false;
            if (trace_enabled) stats->executor_drain_publish_start_cycles = get_sys_cnt_aicore();
            aicore_gm_fetch_add_v0(run_control->executed_task_count, stats->executed_task_count);
            aicore_gm_fetch_add_v0(run_control->executor_drained_worker_count, UINT64_C(1));
            if (trace_enabled) {
                stats->executor_drain_publish_end_cycles = get_sys_cnt_aicore();
                stats->exit_wait_start_cycles = stats->executor_drain_publish_end_cycles;
            }
            stats->drain_start_cycles = get_sys_cnt_aicore();
            stats->drain_end_cycles = get_sys_cnt_aicore();
            stats->drain_cycles = stats->drain_end_cycles - stats->drain_start_cycles;
            executor_drained = true;
        }

        ++stats->idle_iteration_count;
        uint64_t backoff_start = get_sys_cnt_aicore();
        local_backoff(backoff_iterations);
        uint64_t backoff_end = get_sys_cnt_aicore();
        stats->backoff_cycles += backoff_end - backoff_start;
        if (trace_enabled) inter_task_backoff_cycles += backoff_end - backoff_start;
        if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
    }
    return true;
}

}  // namespace

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    const uint64_t aicore_entry_cycles = get_aicore_entry_cycles();
    const bool chip_swimlane_enabled =
        SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[block_idx]);
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    const uint64_t handshake_publish_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

    uint32_t startup_signal = 0;
    while (startup_signal != AICPU_IDLE_TASK_ID && startup_signal != AICORE_EXIT_SIGNAL) {
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }
    const uint64_t register_release_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    aicore_observe_cache_line_v0(my_hank);
    __gm__ AicoreWorkerContextV1 *worker_context = reinterpret_cast<__gm__ AicoreWorkerContextV1 *>(my_hank->task);
    aicore_observe_cache_line_v0(worker_context);
    aicore_observe_cache_line_v0(&worker_context->sidecar_base_address);
    aicore_observe_cache_line_v0(&worker_context->pending_task_id[0]);
    aicore_observe_cache_line_v0(&worker_context->dispatch_slots_offset);
    worker_context->worker_index = static_cast<uint64_t>(block_idx);
    aicore_publish_cache_line_v0(&worker_context->sidecar_base_address);
    AicoreTaskProfilingStateV1 task_profiling{};
    aicore_task_profiling_init_v1(
        &task_profiling, chip_swimlane_enabled, chip_swimlane_enabled ? get_chip_swimlane_aicore_head() : nullptr
    );

    __gm__ void *sidecar_base = reinterpret_cast<__gm__ void *>(worker_context->sidecar_base_address);
    __gm__ AicoreRunControlV1 *run_control =
        aicore_sidecar_at_v1<AicoreRunControlV1>(sidecar_base, worker_context->run_control_offset);
    aicore_observe_cache_line_v0(run_control);
    aicore_observe_cache_line_v0(&run_control->expected_root_prepare_count);
    AicoreReadonlyGraphV0 graph{
        worker_context->graph_descriptors_address,
        worker_context->graph_payloads_address,
        worker_context->graph_task_count,
        worker_context->task_window_mask,
    };
    AicoreWorkerStatsV1 stats{};
    if (worker_context->active != 0) {
        aicore_observe_data_cache_v0(reinterpret_cast<__gm__ void *>(graph.descriptors_address));
        const uint64_t descriptor_cache_observed_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
        (void)run_offloaded_dispatch_loop(
            graph, sidecar_base, worker_context, run_control, &task_profiling, &stats, chip_swimlane_enabled,
            aicore_entry_cycles, handshake_publish_cycles, register_release_cycles, descriptor_cache_observed_cycles
        );
    }

    if (chip_swimlane_enabled && stats.exit_wait_start_cycles == 0) {
        stats.exit_wait_start_cycles = get_sys_cnt_aicore();
    }
    while (stats.exit_observed_cycles == 0 &&
           static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL)
        SPIN_WAIT_HINT();
    if (chip_swimlane_enabled && stats.exit_observed_cycles == 0) stats.exit_observed_cycles = get_sys_cnt_aicore();

    if (chip_swimlane_enabled) stats.final_stats_publish_start_cycles = get_sys_cnt_aicore();
    publish_worker_stats(worker_context, stats);
    if (chip_swimlane_enabled) {
        stats.final_stats_publish_end_cycles = get_sys_cnt_aicore();
        stats.exit_ack_publish_cycles = get_sys_cnt_aicore();
        worker_context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
        worker_context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
        aicore_publish_cache_line_v0(&worker_context->executor_drain_publish_start_cycles);
    }
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
}
