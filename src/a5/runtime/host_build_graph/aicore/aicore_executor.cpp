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
#include "aicore_ready_scheduler_v1.h"
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
    uint64_t ready_poll_cycles{0};
    uint64_t backoff_cycles{0};

    __aicore__ void reset() { *this = {}; }
};

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
    __gm__ AicoreRunControlV1 *run_control, uint64_t resolver_count, uint64_t *ready_victim_cursors,
    AicoreWorkerStatsV1 *stats, bool trace_enabled
) {
    if (resolver_count == 0 || resolver->inbox_index >= resolver_count) return false;
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
        if (route == AicoreRouteResultV1::READY_TO_ENQUEUE &&
            !aicore_bootstrap_ready_batch_append_v1(
                sidecar_base, resolver, static_cast<int64_t>(task_id),
                &batches[aicore_metadata_core_type_index_v1(metadata->subtask_slot)], &stats->ready, trace_enabled
            ))
            return false;
        ++stats->bootstrap_task_count;
    }
    aicore_cache_barrier_v0();
    uint64_t ready_types = 0;
    for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
        if (!aicore_bootstrap_ready_batch_publish_v1(
                sidecar_base, resolver, type, resolver->inbox_index, &batches[type], &stats->ready, &ready_types
            ))
            return false;
    }
    __gm__ AicoreReadyDirectoryV1 *ready_directory = aicore_ready_directory_at_v1(sidecar_base, resolver);
    aicore_gm_store_v0(ready_directory->bootstrap_ready_types[resolver->inbox_index], ready_types);
    if (trace_enabled) stats->bootstrap_scan_end_cycles = aicore_scheduler_cycles_v1();

    uint64_t arrived = aicore_gm_fetch_add_v0(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == resolver_count) {
        aicore_bootstrap_ready_directory_publish_v1(sidecar_base, resolver, resolver_count);
        aicore_gm_publish_v0(run_control->bootstrap_complete, UINT64_C(1));
    } else {
        uint32_t barrier_backoff = kInitialBackoffIterations;
        while (aicore_gm_load_v0(run_control->bootstrap_complete) == 0) {
            if (aicore_gm_load_v0(run_control->scheduler_error) != 0) return false;
            local_backoff(barrier_backoff);
            if (barrier_backoff < kMaximumBackoffIterations) barrier_backoff <<= 1;
        }
    }
    if (trace_enabled) stats->bootstrap_end_cycles = aicore_scheduler_cycles_v1();
    if (trace_enabled) stats->target_bootstrap_start_cycles = aicore_scheduler_cycles_v1();

    for (uint64_t worker_id = resolver->inbox_index; worker_id < resolver->runtime_worker_count;
         worker_id += resolver_count) {
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
            AicoreFreeSlotClaimV1 slot_claim{worker_id, slot_index, slot->generation};
            AicoreRefillTimingV1 refill_timing{};
            if (!aicore_refill_private_slot_v1(
                    graph, sidecar_base, resolver, run_control, resolver_count, slot_claim, ready_victim_cursors,
                    &stats->ready, trace_enabled, trace_enabled ? &refill_timing : nullptr
                ))
                return false;
            stats->bootstrap_ready_claim_cycles[type] += refill_timing.ready_claim_cycles;
            stats->bootstrap_slot_fill_cycles[type] += refill_timing.slot_fill_cycles;
            stats->bootstrap_free_advertise_cycles[type] += refill_timing.free_advertise_cycles;
        }
        aicore_gm_publish_v0(target->bootstrap_done, UINT64_C(1));
        if (trace_enabled) stats->bootstrap_target_cycles[type] += aicore_scheduler_cycles_v1() - target_start;
    }
    if (trace_enabled) stats->target_bootstrap_end_cycles = aicore_scheduler_cycles_v1();
    return true;
}

__aicore__ bool run_ready_dispatch_loop(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context,
    __gm__ AicoreRunControlV1 *run_control, AicoreTaskProfilingStateV1 *task_profiling, AicoreWorkerStatsV1 *stats,
    bool trace_enabled, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles,
    uint64_t register_release_cycles, uint64_t descriptor_cache_observed_cycles
) {
    uint64_t resolver_count = aicore_gm_load_v0(run_control->aiv_active_worker_count);
    bool resolver_worker = context->core_type == static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    if (resolver_count == 0) return false;
    uint64_t completion_victim_cursor = resolver_worker ? (context->inbox_index + 1) % resolver_count : 0;
    uint64_t ready_victim_cursors[AICORE_CORE_TYPE_COUNT_V1]{
        resolver_worker ? (context->inbox_index + 1) % resolver_count : 0,
        resolver_worker ? (context->inbox_index + 1) % resolver_count : 0,
    };
    uint64_t free_slot_cursors[AICORE_CORE_TYPE_COUNT_V1]{
        context->worker_index * AICORE_PENDING_SLOT_COUNT_V1,
        context->worker_index * AICORE_PENDING_SLOT_COUNT_V1,
    };
    if (resolver_worker &&
        !bootstrap_ready_graph(
            graph, sidecar_base, context, run_control, resolver_count, ready_victim_cursors, stats, trace_enabled
        ))
        return false;

    uint64_t seen_publication[AICORE_PENDING_SLOT_COUNT_V1]{};
    uint64_t previous_trace_commit_end = 0;
    uint32_t scan_start = 0;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    AicoreInterTaskTimingV1 inter_task_timing{};
    // The common collector keeps one anchor per core. Full per-task timing is
    // stored in the task-indexed HBG sidecar and merged by host validation.
    bool common_profile_recorded = false;
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

        bool scheduler_progress = false;
        if (resolver_worker) {
            bool completion_progress = false;
            uint64_t operation_start = trace_enabled ? get_sys_cnt_aicore() : 0;
            if (!aicore_service_completion_inboxes_v1(
                    graph, sidecar_base, context, run_control, resolver_count, &completion_victim_cursor,
                    ready_victim_cursors, &stats->wake, &stats->ready, &stats->completion, &completion_progress,
                    trace_enabled
                ))
                return false;
            if (trace_enabled) inter_task_timing.completion_service_cycles += get_sys_cnt_aicore() - operation_start;
            scheduler_progress = completion_progress;
            for (uint32_t type = 0; type < AICORE_CORE_TYPE_COUNT_V1; ++type) {
                bool dispatch_progress = false;
                operation_start = trace_enabled ? get_sys_cnt_aicore() : 0;
                if (!aicore_dispatch_one_overflow_v1(
                        graph, sidecar_base, context, run_control, resolver_count, type, &ready_victim_cursors[type],
                        &free_slot_cursors[type], &stats->ready, &stats->free_slot, &dispatch_progress, trace_enabled
                    ))
                    return false;
                if (trace_enabled) inter_task_timing.dispatch_cycles[type] += get_sys_cnt_aicore() - operation_start;
                scheduler_progress = scheduler_progress || dispatch_progress;
            }
        }

        if (!bootstrap_observed) {
            if (aicore_gm_load_v0(context->bootstrap_done) == 0) {
                ++stats->idle_iteration_count;
                local_backoff(backoff_iterations);
                if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
                continue;
            }
            bootstrap_observed = true;
        }

        int32_t ready_slot = -1;
        uint64_t ready_publication = 0;
        uint64_t ready_scan_start = trace_enabled ? get_sys_cnt_aicore() : 0;
        for (uint32_t offset = 0; offset < AICORE_PENDING_SLOT_COUNT_V1; ++offset) {
            uint32_t slot_index = (scan_start + offset) % AICORE_PENDING_SLOT_COUNT_V1;
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, context, context->worker_index, slot_index);
            uint64_t publication = aicore_gm_load_v0(slot->publication);
            ++stats->task_state_poll_count;
            if (publication != seen_publication[slot_index] &&
                aicore_dispatch_state_v1(publication) == AicoreDispatchPublicationV1::READY) {
                ready_slot = static_cast<int32_t>(slot_index);
                ready_publication = publication;
                break;
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
            uint64_t completion_inbox_index =
                aicore_completion_inbox_index_v1(context, resolver_count, local_completion_index);
            if (!aicore_enqueue_completion_v1(
                    graph, sidecar_base, context, run_control, resolver_count, record.task_id, local_completion_index,
                    &stats->completion, trace_enabled
                ))
                return false;
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
            if (trace_enabled) {
                uint64_t bookkeeping_end = get_sys_cnt_aicore();
                commit_task_trace(
                    sidecar_base, context, record, ready_scan_start, ready_observe, kernel_start, kernel_end,
                    completion_end, bookkeeping_end, previous_trace_commit_end, aicore_entry_cycles,
                    handshake_publish_cycles, register_release_cycles, descriptor_cache_observed_cycles, completion_id,
                    completion_inbox_index, inter_task_timing
                );
                previous_trace_commit_end = get_sys_cnt_aicore();
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

    uint32_t startup_signal = 0;
    while (startup_signal != AICPU_IDLE_TASK_ID && startup_signal != AICORE_EXIT_SIGNAL) {
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        return;
    }
    uint64_t register_release_cycles = trace_enabled ? get_sys_cnt_aicore() : 0;
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    aicore_observe_cache_line_v0(handshake);
    __gm__ AicoreWorkerContextV1 *context = reinterpret_cast<__gm__ AicoreWorkerContextV1 *>(handshake->task);
    aicore_observe_cache_line_v0(context);
    aicore_observe_cache_line_v0(&context->task_metadata_offset);
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
    if (context->active != 0) {
        aicore_observe_data_cache_v0(reinterpret_cast<__gm__ void *>(graph.descriptors_address));
        uint64_t descriptor_cache_observed_cycles = trace_enabled ? get_sys_cnt_aicore() : 0;
        (void)run_ready_dispatch_loop(
            graph, sidecar_base, context, run_control, &task_profiling, &stats, trace_enabled, aicore_entry_cycles,
            handshake_publish_cycles, register_release_cycles, descriptor_cache_observed_cycles
        );
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
