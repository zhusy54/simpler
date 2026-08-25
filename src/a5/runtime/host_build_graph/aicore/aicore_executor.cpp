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
// Cluster-local normal and gang scheduling share one device-side protocol.
#include "aicore_gang_scheduler_v1.h"
#include "aicore_task_profiling_v1.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 128;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

static_assert(AICORE_CALLABLE_CAPACITY_V1 == RUNTIME_MAX_FUNC_ID, "AICore ready scheduler layout mismatch");

struct AicoreWorkerStatsV1 {
    uint64_t bootstrap_task_count{0};
    AicoreReadyStatsV1 ready{};
    AicoreFreeSlotStatsV1 free_slot{};
    uint64_t executed_task_count{0};
    uint64_t task_state_poll_count{0};
    AicoreWakeStatsV1 wake{};
    AicoreCompletionStatsV1 completion{};
    uint64_t idle_iteration_count{0};
    uint64_t backoff_cycles{0};
    uint64_t payload_cycles{0};
    uint64_t kernel_cycles{0};
    uint64_t completion_enqueue_cycles{0};
    uint64_t bootstrap_start_cycles{0};
    uint64_t bootstrap_scan_end_cycles{0};
    uint64_t bootstrap_end_cycles{0};
    uint64_t target_bootstrap_start_cycles{0};
    uint64_t target_bootstrap_end_cycles{0};
    uint64_t bootstrap_target_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t bootstrap_ready_claim_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t bootstrap_slot_fill_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t bootstrap_free_advertise_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t drain_start_cycles{0};
    uint64_t drain_end_cycles{0};
    uint64_t exit_wait_start_cycles{0};
    uint64_t exit_observed_cycles{0};
    uint64_t final_stats_publish_start_cycles{0};
    uint64_t final_stats_publish_end_cycles{0};
    uint64_t exit_ack_publish_cycles{0};
};

struct AicoreInterTaskTimingV1 {
    uint64_t completion_service_cycles{0};
    uint64_t dispatch_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    AicoreCompletionServiceTimingV1 completion{};
    uint64_t gang_service_cycles{0};
    AicoreNormalDispatchTimingV1 dispatch{};
    uint64_t ready_poll_cycles{0};
    uint64_t backoff_cycles{0};

    __aicore__ void reset() { *this = {}; }
};

__aicore__ __attribute__((always_inline)) void publish_scheduler_tail_trace(
    __gm__ AicoreWorkerContextV1 *context, uint64_t start_cycles, uint64_t end_cycles,
    const AicoreInterTaskTimingV1 &timing
) {
    __gm__ AicoreSchedulerTailTraceV1 *trace = &context->scheduler_tail_trace;
    trace->start_cycles = start_cycles;
    trace->end_cycles = end_cycles;
    trace->completion_scan_cycles = timing.completion.scan_cycles;
    trace->completion_consume_cycles = timing.completion.consume_cycles;
    trace->completion_resolve_cycles = timing.completion.resolve_cycles;
    trace->completion_ready_publish_cycles = timing.completion.ready_publish_cycles;
    trace->completion_refill_cycles = timing.completion.refill_cycles;
    trace->completion_finalize_cycles = timing.completion.finalize_cycles;
    trace->gang_service_cycles = timing.gang_service_cycles;
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        trace->dispatch_probe_cycles[type] = timing.dispatch.probe_cycles[type];
        trace->dispatch_claim_cycles[type] = timing.dispatch.claim_cycles[type];
        trace->dispatch_prepare_cycles[type] = timing.dispatch.prepare_cycles[type];
        trace->dispatch_materialize_cycles[type] = timing.dispatch.materialize_cycles[type];
        trace->dispatch_publish_cycles[type] = timing.dispatch.publish_cycles[type];
    }
    trace->ready_poll_cycles = timing.ready_poll_cycles;
    trace->backoff_cycles = timing.backoff_cycles;
    aicore_publish_cache_line_v0(&trace->start_cycles);
    aicore_publish_cache_line_v0(&trace->dispatch_materialize_cycles[0]);
    aicore_gm_publish_v0(trace->valid, UINT64_C(1));
}

struct AicoreExecutionRecordV1 {
    int64_t task_id{AICORE_TASK_ID_INVALID_V1};
    uint64_t claim_worker_id{0};
    uint64_t claim_start_cycles{0};
    uint64_t claim_end_cycles{0};
    AicoreReadySourceV1 ready_source{AicoreReadySourceV1::LOCAL};
};

__aicore__ __attribute__((always_inline)) void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((always_inline)) bool should_commit_scheduler_trace(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, __gm__ AicoreDispatchSlotV1 *slot
) {
    if (slot->gang == 0) return true;
    if (slot->block_idx != 0) return false;
    __gm__ AicoreTaskMetadataV1 *metadata = aicore_task_metadata_at_v1(sidecar_base, context, slot->task_id);
    aicore_observe_cache_line_v0(metadata);
    return slot->subtask_slot == aicore_metadata_single_subtask_slot_v1(metadata->active_mask);
}

__aicore__ __attribute__((always_inline)) void local_backoff(uint32_t iterations) {
    for (volatile uint32_t spin = 0; spin < iterations; ++spin) {}
}

__aicore__ __attribute__((always_inline)) void
publish_worker_stats(__gm__ AicoreWorkerContextV1 *context, const AicoreWorkerStatsV1 &stats) {
    context->bootstrap_task_count = stats.bootstrap_task_count;
    context->ready_enqueue_count = stats.ready.enqueue_count;
    context->ready_batch_count = stats.ready.batch_count;
    context->ready_pop_count = stats.ready.pop_count;
    context->ready_steal_count = stats.ready.steal_count;
    context->ready_cas_retry_count = stats.ready.cas_retry_count;
    context->ready_link_wait_count = stats.ready.link_wait_count;
    context->ready_link_wait_max = stats.ready.link_wait_max;
    context->free_slot_claim_count = stats.free_slot.claim_count;
    context->free_slot_stale_count = stats.free_slot.stale_count;
    context->executed_task_count = stats.executed_task_count;
    context->task_state_poll_count = stats.task_state_poll_count;
    context->fanin_state_load_count = stats.wake.fanin_state_load_count;
    context->wake_register_count = stats.wake.wake_register_count;
    context->idle_iteration_count = stats.idle_iteration_count;
    context->backoff_cycles = stats.backoff_cycles;
    aicore_publish_cache_line_v0(&context->bootstrap_task_count);

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
    context->payload_cycles = stats.payload_cycles;
    context->kernel_cycles = stats.kernel_cycles;
    aicore_publish_cache_line_v0(&context->wake_cas_retry_count);

    context->completion_enqueue_cycles = stats.completion_enqueue_cycles;
    context->bootstrap_start_cycles = stats.bootstrap_start_cycles;
    context->bootstrap_end_cycles = stats.bootstrap_end_cycles;
    context->drain_start_cycles = stats.drain_start_cycles;
    context->drain_end_cycles = stats.drain_end_cycles;
    context->exit_wait_start_cycles = stats.exit_wait_start_cycles;
    context->exit_observed_cycles = stats.exit_observed_cycles;
    context->final_stats_publish_start_cycles = stats.final_stats_publish_start_cycles;
    context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
    context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
    context->bootstrap_slot_fill_aic_cycles = stats.bootstrap_slot_fill_cycles[0];
    context->bootstrap_slot_fill_aiv_cycles = stats.bootstrap_slot_fill_cycles[1];
    context->bootstrap_free_advertise_aic_cycles = stats.bootstrap_free_advertise_cycles[0];
    context->bootstrap_free_advertise_aiv_cycles = stats.bootstrap_free_advertise_cycles[1];
    aicore_publish_cache_line_v0(&context->completion_enqueue_cycles);
    aicore_publish_cache_line_v0(&context->bootstrap_slot_fill_aic_cycles);

    aicore_gm_publish_v0(context->bootstrap_scan_end_cycles, stats.bootstrap_scan_end_cycles);
    aicore_gm_publish_v0(context->target_bootstrap_start_cycles, stats.target_bootstrap_start_cycles);
    aicore_gm_publish_v0(context->target_bootstrap_end_cycles, stats.target_bootstrap_end_cycles);
    aicore_gm_publish_v0(context->bootstrap_target_aic_cycles, stats.bootstrap_target_cycles[0]);
    aicore_gm_publish_v0(context->bootstrap_target_aiv_cycles, stats.bootstrap_target_cycles[1]);
    aicore_gm_publish_v0(context->bootstrap_ready_claim_aic_cycles, stats.bootstrap_ready_claim_cycles[0]);
    aicore_gm_publish_v0(context->bootstrap_ready_claim_aiv_cycles, stats.bootstrap_ready_claim_cycles[1]);
}

__aicore__ __attribute__((always_inline)) void commit_task_trace(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, const AicoreExecutionRecordV1 &record,
    uint64_t ready_scan_start, uint64_t ready_observe, uint64_t kernel_start, uint64_t kernel_end,
    uint64_t completion_end, uint64_t bookkeeping_end, uint64_t previous_trace_commit_end, uint64_t aicore_entry_cycles,
    uint64_t handshake_publish_cycles, uint64_t register_release_cycles, uint64_t descriptor_cache_observed_cycles,
    uint64_t completion_id, uint64_t completion_inbox_index, const AicoreInterTaskTimingV1 &inter_task_timing
) {
    __gm__ AicoreTaskTraceCellV1 *cells =
        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
    __gm__ AicoreTaskTraceCellV1 *trace = &cells[record.task_id];
    trace->ready_source = static_cast<uint64_t>(record.ready_source);
    trace->worker_id = context->worker_index;
    trace->task_id = static_cast<uint64_t>(record.task_id);
    trace->claim_worker_id = record.claim_worker_id;
    trace->claim_start_cycles = record.claim_start_cycles;
    trace->claim_end_cycles = record.claim_end_cycles;
    trace->previous_trace_commit_end_cycles = previous_trace_commit_end;
    trace->kernel_start_cycles = kernel_start;
    trace->kernel_end_cycles = kernel_end;
    trace->completion_end_cycles = completion_end;
    trace->ready_scan_start_cycles = ready_scan_start;
    trace->ready_observe_cycles = ready_observe;
    trace->completion_bookkeeping_end_cycles = bookkeeping_end;
    trace->completion_id = completion_id;
    trace->completion_inbox_index = completion_inbox_index;
    aicore_observe_cache_line_v0(&trace->ready_transition_cycles);
    trace->inter_task_completion_service_cycles = inter_task_timing.completion_service_cycles;
    trace->inter_task_dispatch_aic_cycles = inter_task_timing.dispatch_cycles[0];
    trace->inter_task_dispatch_aiv_cycles = inter_task_timing.dispatch_cycles[1];
    trace->inter_task_ready_poll_cycles = inter_task_timing.ready_poll_cycles;
    trace->inter_task_backoff_cycles = inter_task_timing.backoff_cycles;
    trace->inter_task_completion_scan_cycles = inter_task_timing.completion.scan_cycles;
    trace->inter_task_completion_consume_cycles = inter_task_timing.completion.consume_cycles;
    trace->inter_task_completion_resolve_cycles = inter_task_timing.completion.resolve_cycles;
    trace->inter_task_completion_ready_publish_cycles = inter_task_timing.completion.ready_publish_cycles;
    trace->inter_task_completion_refill_cycles = inter_task_timing.completion.refill_cycles;
    trace->inter_task_completion_finalize_cycles = inter_task_timing.completion.finalize_cycles;
    trace->inter_task_gang_service_cycles = inter_task_timing.gang_service_cycles;
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        trace->inter_task_dispatch_probe_cycles[type] = inter_task_timing.dispatch.probe_cycles[type];
        trace->inter_task_dispatch_claim_cycles[type] = inter_task_timing.dispatch.claim_cycles[type];
        trace->inter_task_dispatch_prepare_cycles[type] = inter_task_timing.dispatch.prepare_cycles[type];
        trace->inter_task_dispatch_materialize_cycles[type] = inter_task_timing.dispatch.materialize_cycles[type];
        trace->inter_task_dispatch_publish_cycles[type] = inter_task_timing.dispatch.publish_cycles[type];
    }
    if (previous_trace_commit_end == 0) {
        trace->aicore_entry_cycles = aicore_entry_cycles;
        trace->handshake_publish_cycles = handshake_publish_cycles;
        trace->register_release_cycles = register_release_cycles;
        trace->descriptor_cache_observed_cycles = descriptor_cache_observed_cycles;
        aicore_publish_cache_line_v0(&trace->register_release_cycles);
    }
    aicore_publish_cache_line_v0(&trace->kernel_start_cycles);
    aicore_publish_cache_line_v0(&trace->ready_transition_cycles);
    trace->valid = 1;
    aicore_publish_cache_line_v0(trace);
}

__aicore__ bool bootstrap_ready_graph(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, AicoreWorkerStatsV1 *stats, bool trace_enabled,
    AicoreDeferredAivQueueV1 *deferred_aiv, AicoreReadyOwnerStateV1 *ready_owner
) {
    if (resolver_count == 0 || resolver->inbox_index >= resolver_count || ready_owner == nullptr) return false;
    if (trace_enabled) stats->bootstrap_start_cycles = aicore_scheduler_cycles_v1();
    AicoreReadyBatchV1 batches[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t tasks_per_resolver = graph.task_count / resolver_count;
    uint64_t remainder = graph.task_count % resolver_count;
    uint64_t task_begin = resolver->inbox_index * tasks_per_resolver +
                          (resolver->inbox_index < remainder ? resolver->inbox_index : remainder);
    uint64_t task_end = task_begin + tasks_per_resolver + (resolver->inbox_index < remainder ? 1 : 0);
    for (uint64_t task_id = task_begin; task_id < task_end; ++task_id) {
        __gm__ AicoreTaskMetadataV1 *metadata =
            aicore_task_metadata_at_v1(sidecar_base, resolver, static_cast<int64_t>(task_id));
        if (!aicore_task_is_executable_v1(metadata->flags)) continue;
        AicoreRouteResultV1 route =
            aicore_task_has_fanin_v1(metadata->flags) ?
                aicore_bootstrap_route_task_v1(
                    graph, sidecar_base, resolver, run_control, static_cast<int64_t>(task_id), &stats->wake
                ) :
                AicoreRouteResultV1::READY_TO_ENQUEUE;
        if (route == AicoreRouteResultV1::ERROR) return false;
        if (route == AicoreRouteResultV1::READY_TO_ENQUEUE) {
            if (aicore_task_is_gang_v1(metadata->flags)) {
                __gm__ AicoreTaskControlV1 *control =
                    aicore_task_control_at_v1(sidecar_base, resolver, static_cast<int64_t>(task_id));
                aicore_publish_gang_ready_v1(sidecar_base, resolver, control, metadata->flags);
            } else if (!aicore_bootstrap_ready_batch_append_v1(
                           sidecar_base, resolver, static_cast<int64_t>(task_id),
                           &batches[aicore_metadata_core_type_index_v1(
                               aicore_metadata_single_subtask_slot_v1(metadata->active_mask)
                           )],
                           &stats->ready, trace_enabled
                       )) {
                return false;
            }
        }
        ++stats->bootstrap_task_count;
    }
    aicore_cache_barrier_v0();
    uint64_t ready_types = 0;
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        if (!aicore_bootstrap_ready_batch_publish_v1(
                sidecar_base, resolver, type, resolver->inbox_index, &batches[type], &stats->ready, &ready_types
            ))
            return false;
        ready_owner->queues[type].advertised = (ready_types & (UINT64_C(1) << type)) != 0;
    }
    __gm__ AicoreReadyDirectoryV1 *ready_directory = aicore_ready_directory_at_v1(sidecar_base, resolver);
    aicore_gm_store_v0(ready_directory->bootstrap_ready_types[resolver->inbox_index], ready_types);
    if (trace_enabled) stats->bootstrap_scan_end_cycles = aicore_scheduler_cycles_v1();

    uint64_t arrived = aicore_gm_fetch_add_v0(run_control->bootstrap_scan_arrived_count, UINT64_C(1)) + 1;
    if (arrived == resolver_count) {
        aicore_bootstrap_ready_directory_publish_v1(sidecar_base, resolver, resolver_count);
        aicore_gm_publish_v0(run_control->bootstrap_scan_complete, UINT64_C(1));
    } else {
        uint32_t barrier_backoff = kInitialBackoffIterations;
        while (aicore_gm_query_v0(run_control->bootstrap_scan_complete) == 0) {
            if (aicore_gm_query_v0(run_control->scheduler_error) != 0) return false;
            local_backoff(barrier_backoff);
            if (barrier_backoff < kMaximumBackoffIterations) barrier_backoff <<= 1;
        }
    }
    if (trace_enabled) stats->target_bootstrap_start_cycles = aicore_scheduler_cycles_v1();

    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
        uint64_t target_start = trace_enabled ? aicore_scheduler_cycles_v1() : 0;
        __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
        aicore_observe_cache_line_v0(target);
        aicore_observe_cache_line_v0(&target->task_metadata_offset);
        if (target->active == 0) continue;
        uint32_t type = aicore_core_type_index_v1(target->core_type);
        for (uint32_t slot_index = 0; slot_index < AICORE_PENDING_SLOT_COUNT_V1; ++slot_index) {
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, slot_index);
            aicore_observe_cache_line_v0(slot);
            aicore_initialize_free_slot_v1(slot);
        }
        aicore_gm_publish_v0(target->bootstrap_done, UINT64_C(1));
        if (trace_enabled) stats->bootstrap_target_cycles[type] += aicore_scheduler_cycles_v1() - target_start;
    }
    if (trace_enabled) stats->target_bootstrap_end_cycles = aicore_scheduler_cycles_v1();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed. Gang service runs first to preserve SPMD > Mix > normal
    // priority; normal fill is suppressed while a gang dispatch is pending.
    uint64_t ready_victim_cursors[AICORE_CORE_TYPE_COUNT_V1]{
        (resolver->inbox_index + 1) % resolver_count,
        (resolver->inbox_index + 1) % resolver_count,
    };
    (void)aicore_service_gang_scheduler_v1(
        graph, sidecar_base, resolver, run_control, &stats->wake, &stats->ready, &stats->completion, ready_owner
    );
    (void)aicore_fill_cluster_normal_slots_v1(
        graph, sidecar_base, resolver, run_control, ready_victim_cursors, &stats->ready, trace_enabled, 0, nullptr,
        deferred_aiv, ready_owner
    );

    // This completion publication is observed by AICPU before it emits the
    // one and only DMB release. Resolvers do not wait on another barrier.
    arrived = aicore_gm_fetch_add_v0(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == resolver_count) aicore_gm_publish_v0(run_control->bootstrap_complete, UINT64_C(1));
    if (trace_enabled) stats->bootstrap_end_cycles = aicore_scheduler_cycles_v1();
    return true;
}

__aicore__ bool run_ready_dispatch_loop(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, AicoreTaskProfilingStateV1 *task_profiling, AicoreWorkerStatsV1 *stats,
    bool trace_enabled, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles,
    uint64_t register_release_cycles, uint64_t descriptor_cache_observed_cycles, AicoreDeferredAivQueueV1 *deferred_aiv,
    AicoreReadyOwnerStateV1 *ready_owner
) {
    uint64_t resolver_count = aicore_gm_query_v0(run_control->resolver_count);
    bool resolver_worker = context->is_resolver != 0;
    if (resolver_count == 0) return false;
    uint64_t ready_victim_cursors[AICORE_CORE_TYPE_COUNT_V1]{
        resolver_worker ? (context->inbox_index + 1) % resolver_count : 0,
        resolver_worker ? (context->inbox_index + 1) % resolver_count : 0,
    };
    uint64_t seen_publication[AICORE_PENDING_SLOT_COUNT_V1]{};
    uint64_t previous_trace_commit_end = 0;
    uint64_t inter_task_start_cycles = register_release_cycles;
    uint32_t scan_start = 0;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    AicoreInterTaskTimingV1 inter_task_timing{};
    // The common collector keeps one anchor per core. Full per-task timing is
    // stored in the task-indexed HBG sidecar and merged by host validation.
    bool common_profile_recorded = false;
    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (trace_enabled) {
                stats->exit_observed_cycles = get_sys_cnt_aicore();
                publish_scheduler_tail_trace(
                    context, inter_task_start_cycles, stats->exit_observed_cycles, inter_task_timing
                );
            }
            break;
        }
        if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
            scheduler_error_poll_count = 0;
            if (aicore_gm_query_v0(run_control->scheduler_error) != 0) return false;
        }

        bool scheduler_progress = false;
        if (resolver_worker && !aicore_ready_owner_maintain_v1(sidecar_base, context, ready_owner)) {
            aicore_record_scheduler_error_v1(
                run_control, AICORE_TASK_ID_INVALID_V1, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context,
                UINT64_C(79)
            );
            return false;
        }
        uint32_t preferred_ready_slot = UINT32_MAX;
        if (resolver_worker && deferred_aiv != nullptr && deferred_aiv->count != 0) {
            const uint32_t deferred_before = deferred_aiv->count;
            if (!aicore_drain_deferred_aiv_to_peer_v1(
                    graph, sidecar_base, context, run_control, deferred_aiv, &stats->wake, &stats->ready,
                    &stats->completion, trace_enabled, nullptr, nullptr, ready_owner
                ))
                return false;
            scheduler_progress = deferred_aiv->count != deferred_before;
            if (deferred_aiv->count != 0 &&
                !aicore_publish_deferred_aiv_to_resolver_v1(
                    graph, sidecar_base, context, run_control, deferred_aiv, trace_enabled, &preferred_ready_slot
                ))
                return false;
        }
        if (resolver_worker && preferred_ready_slot == UINT32_MAX) {
            uint64_t operation_start = trace_enabled ? get_sys_cnt_aicore() : 0;
            uint64_t direct_refilled_slot_mask = 0;
            AicoreCompletionServiceTimingV1 completion_timing{};
            const bool completion_progress = aicore_service_cluster_completions_v1(
                graph, sidecar_base, context, run_control, &stats->wake, &stats->ready, &stats->completion,
                ready_victim_cursors, trace_enabled, &direct_refilled_slot_mask,
                trace_enabled ? &completion_timing : nullptr, ready_owner
            );
            if (trace_enabled) {
                uint64_t completion_total = get_sys_cnt_aicore() - operation_start;
                uint64_t completion_detail = completion_timing.consume_cycles + completion_timing.resolve_cycles +
                                             completion_timing.ready_publish_cycles + completion_timing.refill_cycles +
                                             completion_timing.finalize_cycles;
                completion_timing.scan_cycles +=
                    completion_total > completion_detail ? completion_total - completion_detail : 0;
                inter_task_timing.completion_service_cycles += completion_total;
                inter_task_timing.completion.scan_cycles += completion_timing.scan_cycles;
                inter_task_timing.completion.consume_cycles += completion_timing.consume_cycles;
                inter_task_timing.completion.resolve_cycles += completion_timing.resolve_cycles;
                inter_task_timing.completion.ready_publish_cycles += completion_timing.ready_publish_cycles;
                inter_task_timing.completion.refill_cycles += completion_timing.refill_cycles;
                inter_task_timing.completion.finalize_cycles += completion_timing.finalize_cycles;
            }
            scheduler_progress = completion_progress;
            operation_start = trace_enabled ? get_sys_cnt_aicore() : 0;
            scheduler_progress = aicore_service_gang_scheduler_v1(
                                     graph, sidecar_base, context, run_control, &stats->wake, &stats->ready,
                                     &stats->completion, ready_owner
                                 ) ||
                                 scheduler_progress;
            if (trace_enabled) inter_task_timing.gang_service_cycles += get_sys_cnt_aicore() - operation_start;
            operation_start = trace_enabled ? get_sys_cnt_aicore() : 0;
            AicoreNormalDispatchTimingV1 dispatch_timing{};
            scheduler_progress =
                aicore_fill_cluster_normal_slots_v1(
                    graph, sidecar_base, context, run_control, ready_victim_cursors, &stats->ready, trace_enabled,
                    direct_refilled_slot_mask, trace_enabled ? &dispatch_timing : nullptr, deferred_aiv, ready_owner
                ) ||
                scheduler_progress;
            if (trace_enabled) {
                inter_task_timing.dispatch_cycles[0] += get_sys_cnt_aicore() - operation_start;
                for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
                    inter_task_timing.dispatch.probe_cycles[type] += dispatch_timing.probe_cycles[type];
                    inter_task_timing.dispatch.claim_cycles[type] += dispatch_timing.claim_cycles[type];
                    inter_task_timing.dispatch.prepare_cycles[type] += dispatch_timing.prepare_cycles[type];
                    inter_task_timing.dispatch.materialize_cycles[type] += dispatch_timing.materialize_cycles[type];
                    inter_task_timing.dispatch.publish_cycles[type] += dispatch_timing.publish_cycles[type];
                }
            }
            if (deferred_aiv != nullptr && deferred_aiv->count != 0) {
                const uint32_t deferred_before = deferred_aiv->count;
                if (!aicore_drain_deferred_aiv_to_peer_v1(
                        graph, sidecar_base, context, run_control, deferred_aiv, &stats->wake, &stats->ready,
                        &stats->completion, trace_enabled, nullptr, nullptr, ready_owner
                    ))
                    return false;
                scheduler_progress = scheduler_progress || deferred_aiv->count != deferred_before;
                if (deferred_aiv->count != 0 &&
                    !aicore_publish_deferred_aiv_to_resolver_v1(
                        graph, sidecar_base, context, run_control, deferred_aiv, trace_enabled, &preferred_ready_slot
                    ))
                    return false;
            }
        }

        int32_t ready_slot = -1;
        uint64_t ready_publication = 0;
        uint64_t ready_scan_start = trace_enabled ? get_sys_cnt_aicore() : 0;
        if (preferred_ready_slot != UINT32_MAX) {
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, preferred_ready_slot);
            const uint64_t publication = aicore_gm_query_v0(slot->publication);
            ++stats->task_state_poll_count;
            if (publication == seen_publication[preferred_ready_slot] ||
                aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::READY) {
                aicore_record_scheduler_error_v1(
                    run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(78)
                );
                return false;
            }
            ready_slot = static_cast<int32_t>(preferred_ready_slot);
            ready_publication = publication;
        } else {
            for (uint32_t offset = 0; offset < AICORE_PENDING_SLOT_COUNT_V1; ++offset) {
                uint32_t slot_index = (scan_start + offset) % AICORE_PENDING_SLOT_COUNT_V1;
                __gm__ AicoreDispatchSlotV1 *slot =
                    aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, slot_index);
                uint64_t publication = aicore_gm_query_v0(slot->publication);
                ++stats->task_state_poll_count;
                if (publication != seen_publication[slot_index] &&
                    aicore_dispatch_state_v1(publication) == AicoreDispatchPublicationV1::READY) {
                    ready_slot = static_cast<int32_t>(slot_index);
                    ready_publication = publication;
                    break;
                }
            }
        }
        uint64_t ready_poll_end = trace_enabled ? get_sys_cnt_aicore() : 0;

        if (ready_slot >= 0) {
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, slot_index);
            __gm__ PTO2DispatchPayload *payload = aicore_sidecar_at_v1<PTO2DispatchPayload>(
                sidecar_base,
                context->dispatch_payload_offset + static_cast<uint64_t>(slot_index) * sizeof(PTO2DispatchPayload)
            );
            uint64_t ready_observe = get_sys_cnt_aicore();
            aicore_invalidate_cache_line_v0(slot);
            aicore_observe_dispatch_payload_control_v1(payload);
            aicore_observe_dispatch_payload_arguments_v1(payload);
            aicore_observe_dispatch_payload_barrier_v1();
            if (slot->pending_slot != slot_index || slot->task_id < 0 ||
                static_cast<uint64_t>(slot->task_id) >= graph.task_count ||
                slot->generation != aicore_dispatch_generation_v1(ready_publication)) {
                aicore_record_scheduler_error_v1(
                    run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context, UINT64_C(50)
                );
                return false;
            }
            seen_publication[slot_index] = ready_publication;
            AicoreExecutionRecordV1 record{
                slot->task_id,
                slot->claim_worker_id,
                slot->claim_start_cycles,
                slot->claim_end_cycles,
                static_cast<AicoreReadySourceV1>(slot->ready_source),
            };
            const bool commit_scheduler_trace =
                trace_enabled && should_commit_scheduler_trace(sidecar_base, context, slot);
            OUT_OF_ORDER_STORE_BARRIER();
            __gm__ ChipSwimlaneAicoreTaskRecord *profile_record =
                common_profile_recorded ? nullptr : aicore_task_profiling_reserve_v1(task_profiling);
            uint64_t kernel_start = get_sys_cnt_aicore();
            if (trace_enabled) {
                __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, context, record.task_id);
                aicore_observe_cache_line_v0(&control->next_waiter);
                if (control->ready_publish_cycles != 0 && kernel_start >= control->ready_publish_cycles) {
                    uint64_t lag = kernel_start - control->ready_publish_cycles;
                    stats->completion.ready_to_kernel_cycles += lag;
                    if (lag > stats->completion.ready_to_kernel_max_cycles)
                        stats->completion.ready_to_kernel_max_cycles = lag;
                }
            }
            execute_task(payload);
            uint64_t kernel_end = get_sys_cnt_aicore();
            aicore_publish_dispatch_payload_v1(payload);
            uint64_t completion_start = get_sys_cnt_aicore();
            uint64_t local_completion_index = stats->completion.enqueue_count;
            uint64_t completion_id = aicore_completion_id_v1(context, local_completion_index);
            uint64_t completion_inbox_index = context->resolver_index;
            __gm__ AicoreCompletionInboxV1 *completion_line =
                aicore_completion_inbox_at_v1(sidecar_base, context, context->worker_index);
            aicore_gm_store_v0(completion_line->completed_generations[slot_index], slot->generation);
            ++stats->completion.enqueue_count;
            uint64_t completion_end = get_sys_cnt_aicore();
            aicore_task_profiling_commit_v1(
                profile_record, static_cast<uint64_t>(record.task_id), record.claim_end_cycles, kernel_start, kernel_end
            );
            if (profile_record != nullptr) common_profile_recorded = true;
            ++stats->executed_task_count;
            stats->payload_cycles += kernel_start - ready_observe;
            stats->kernel_cycles += kernel_end - kernel_start;
            stats->completion_enqueue_cycles += completion_end - completion_start;
            scan_start = (slot_index + 1) % AICORE_PENDING_SLOT_COUNT_V1;
            backoff_iterations = kInitialBackoffIterations;
            if (commit_scheduler_trace) {
                uint64_t bookkeeping_end = get_sys_cnt_aicore();
                commit_task_trace(
                    sidecar_base, context, record, ready_scan_start, ready_observe, kernel_start, kernel_end,
                    completion_end, bookkeeping_end, previous_trace_commit_end, aicore_entry_cycles,
                    handshake_publish_cycles, register_release_cycles, descriptor_cache_observed_cycles, completion_id,
                    completion_inbox_index, inter_task_timing
                );
                previous_trace_commit_end = get_sys_cnt_aicore();
                inter_task_start_cycles = previous_trace_commit_end;
            } else if (trace_enabled) {
                inter_task_start_cycles = get_sys_cnt_aicore();
            }
            inter_task_timing.reset();
            continue;
        }

        if (trace_enabled) inter_task_timing.ready_poll_cycles += ready_poll_end - ready_scan_start;

        if (scheduler_progress) {
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }
        ++stats->idle_iteration_count;
        uint64_t backoff_start = get_sys_cnt_aicore();
        local_backoff(backoff_iterations);
        uint64_t backoff_end = get_sys_cnt_aicore();
        stats->backoff_cycles += backoff_end - backoff_start;
        if (trace_enabled) inter_task_timing.backoff_cycles += backoff_end - backoff_start;
        if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
    }
    return true;
}

}  // namespace

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    uint64_t aicore_entry_cycles = get_aicore_entry_cycles();
    bool trace_enabled = SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->workers[block_idx]);
    handshake->physical_core_id = get_physical_core_id();
    handshake->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    handshake->aicore_done = block_idx + 1;
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    uint64_t handshake_publish_cycles = trace_enabled ? get_sys_cnt_aicore() : 0;

    // AICPU publishes the fully configured context through GM. This lets
    // AICore perform all pre-kernel work without consuming a DMB launch.
    uint32_t startup_signal = 0;
    while (handshake->task == 0 && startup_signal != AICORE_EXIT_SIGNAL) {
        aicore_observe_cache_line_v0(handshake);
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }

    __gm__ AicoreWorkerContextV1 *context = reinterpret_cast<__gm__ AicoreWorkerContextV1 *>(handshake->task);
    aicore_observe_cache_line_v0(context);
    aicore_observe_cache_line_v0(&context->task_metadata_offset);
    aicore_observe_cache_line_v0(&context->gang_coordinator_offset);
    context->worker_index = static_cast<uint64_t>(block_idx);
    aicore_publish_cache_line_v0(&context->sidecar_base_address);
    AicoreTaskProfilingStateV1 task_profiling{};
    aicore_task_profiling_init_v1(
        &task_profiling, trace_enabled, trace_enabled ? get_chip_swimlane_aicore_head() : nullptr
    );

    __gm__ void *sidecar_base = reinterpret_cast<__gm__ void *>(context->sidecar_base_address);
    __gm__ AicoreRunControlV1 *run_control =
        aicore_sidecar_at_v1<AicoreRunControlV1>(sidecar_base, context->run_control_offset);
    aicore_observe_cache_line_v0(run_control);
    AicoreReadonlyGraphV0 graph{
        context->graph_descriptors_address,
        context->graph_payloads_address,
        context->graph_task_count,
        context->task_window_mask,
    };
    AicoreWorkerStatsV1 stats{};
    AicoreDeferredAivQueueV1 deferred_aiv{};
    AicoreReadyOwnerStateV1 ready_owner{};
    uint64_t descriptor_cache_observed_cycles = 0;
    if (context->active != 0) {
        aicore_observe_data_cache_v0(reinterpret_cast<__gm__ void *>(graph.descriptors_address));
        descriptor_cache_observed_cycles = trace_enabled ? get_sys_cnt_aicore() : 0;
        if (context->is_resolver != 0 &&
            !bootstrap_ready_graph(
                graph, sidecar_base, context, run_control, aicore_gm_query_v0(run_control->resolver_count), &stats,
                trace_enabled, &deferred_aiv, &ready_owner
            )) {
            aicore_record_scheduler_error_v1(
                run_control, AICORE_TASK_ID_INVALID_V1, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, context,
                UINT64_C(90)
            );
        }
    }

    // Sole AICore-wide DMB: after this point a prepared slot may execute.
    startup_signal = 0;
    while (startup_signal != AICPU_IDLE_TASK_ID && startup_signal != AICORE_EXIT_SIGNAL) {
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        if (trace_enabled) stats.exit_observed_cycles = get_sys_cnt_aicore();
    } else {
        uint64_t register_release_cycles = trace_enabled ? get_sys_cnt_aicore() : 0;
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        if (context->active != 0) {
            (void)run_ready_dispatch_loop(
                graph, sidecar_base, context, run_control, &task_profiling, &stats, trace_enabled, aicore_entry_cycles,
                handshake_publish_cycles, register_release_cycles, descriptor_cache_observed_cycles, &deferred_aiv,
                &ready_owner
            );
        }
    }

    if (trace_enabled && stats.exit_wait_start_cycles == 0) stats.exit_wait_start_cycles = get_sys_cnt_aicore();
    while (stats.exit_observed_cycles == 0 &&
           static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL)
        SPIN_WAIT_HINT();
    if (trace_enabled && stats.exit_observed_cycles == 0) stats.exit_observed_cycles = get_sys_cnt_aicore();
    if (trace_enabled) stats.final_stats_publish_start_cycles = get_sys_cnt_aicore();
    publish_worker_stats(context, stats);
    if (trace_enabled) {
        stats.final_stats_publish_end_cycles = get_sys_cnt_aicore();
        stats.exit_ack_publish_cycles = get_sys_cnt_aicore();
        context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
        context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
        aicore_publish_cache_line_v0(&context->completion_enqueue_cycles);
    }
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
}
