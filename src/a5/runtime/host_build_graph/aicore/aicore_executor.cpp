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
#include "aicore_worker_debug_policy_v1.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 1024;
constexpr uint32_t kSchedulerErrorPollInterval = 64;
constexpr uint64_t kCompletionFairnessInterval = 4;

static_assert(RUNTIME_MAX_FUNC_ID <= static_cast<int32_t>(UINT16_MAX) + 1, "pending kernel id cache is too narrow");

struct AicoreWorkerStatsV1 {
    uint64_t seeded_task_count{0};
    uint64_t ticket_claim_count{0};
    uint64_t ticket_exhaustion_count{0};
    uint64_t executed_task_count{0};
    uint64_t pending_peak{0};
    uint64_t task_state_poll_count{0};
    AicoreWakeStatsV1 wake{};
    AicoreCompletionStatsV1 completion{};
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

__aicore__ __attribute__((always_inline)) void publish_worker_config(__gm__ AicoreWorkerContextV1 *context) {
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

__aicore__ __attribute__((always_inline)) void publish_worker_debug(
    __gm__ AicoreWorkerContextV1 *context, const AicorePendingSlotV1 pending[AICORE_PENDING_SLOT_COUNT_V1],
    bool cursor_exhausted, uint64_t lifecycle_state, AicoreWorkerDebugEventV1 event
) {
    if (!aicore_worker_debug_requires_publish_v1(event)) return;
    for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot) {
        context->pending_task_id[slot] = pending[slot].task_id;
        context->pending_next_fanin[slot] = pending[slot].next_fanin_index;
        context->pending_waiting_producer[slot] = pending[slot].waiting_producer;
    }
    context->cursor_exhausted = cursor_exhausted ? 1 : 0;
    context->lifecycle_state = lifecycle_state;
    aicore_publish_cache_line_v0(&context->pending_task_id[0]);
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
}

__aicore__ __attribute__((always_inline)) void commit_task_trace(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, const AicorePendingSlotV1 &pending,
    uint64_t pending_wait_end, uint64_t ready_scan_start, uint64_t ready_observe, uint64_t payload_start,
    uint64_t kernel_start, uint64_t kernel_end, uint64_t completion_end, uint64_t completion_bookkeeping_end,
    uint64_t completion_service_start, uint64_t completion_service_end, uint64_t previous_trace_commit_end
) {
    __gm__ AicoreTaskTraceCellV1 *trace_cells =
        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
    __gm__ AicoreTaskTraceCellV1 *trace = &trace_cells[pending.task_id];
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
    trace->completion_service_start_cycles = completion_service_start;
    trace->completion_service_end_cycles = completion_service_end;
    trace->previous_trace_commit_end_cycles = previous_trace_commit_end;
    aicore_publish_cache_line_v0(&trace->completion_bookkeeping_end_cycles);
    aicore_publish_cache_line_v0(&trace->pending_wait_start_cycles);
    trace->valid = 1;
    aicore_publish_cache_line_v0(trace);
}

__aicore__ __attribute__((always_inline)) void record_task_route_end(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, int64_t task_id
) {
    __gm__ AicoreTaskTraceCellV1 *trace_cells =
        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
    trace_cells[task_id].route_end_cycles = get_sys_cnt_aicore();
}

__aicore__ __attribute__((always_inline)) void record_task_initialize_end(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *context, int64_t task_id
) {
    __gm__ AicoreTaskTraceCellV1 *trace_cells =
        aicore_sidecar_at_v1<AicoreTaskTraceCellV1>(sidecar_base, context->trace_cells_offset);
    trace_cells[task_id].initialize_end_cycles = get_sys_cnt_aicore();
}

}  // namespace

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
    __gm__ AicoreWorkerContextV1 *worker_context = reinterpret_cast<__gm__ AicoreWorkerContextV1 *>(my_hank->task);
    aicore_observe_cache_line_v0(worker_context);
    worker_context->physical_core_id = static_cast<int32_t>(my_hank->physical_core_id);
    worker_context->core_type = static_cast<int32_t>(core_type);
    publish_worker_config(worker_context);

    const bool chip_swimlane_enabled =
        SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    AicoreTaskProfilingStateV1 task_profiling{};
    aicore_task_profiling_init_v1(
        &task_profiling, chip_swimlane_enabled, chip_swimlane_enabled ? get_chip_swimlane_aicore_head() : nullptr
    );

    __gm__ void *sidecar_base = reinterpret_cast<__gm__ void *>(worker_context->sidecar_base_address);
    __gm__ AicoreRunControlV1 *run_control =
        aicore_sidecar_at_v1<AicoreRunControlV1>(sidecar_base, worker_context->run_control_offset);
    AicoreReadonlyGraphV0 graph{
        worker_context->graph_descriptors_address,
        worker_context->graph_payloads_address,
        worker_context->graph_task_count,
        worker_context->task_window_mask,
    };
    aicore_gm_fetch_add_v0(run_control->attached_count, UINT64_C(1));

    while (aicore_gm_load_v0(run_control->startup_phase) == static_cast<uint64_t>(AicoreRunPhaseV1::ATTACH) &&
           aicore_gm_load_v0(run_control->exit_requested) == 0) {
        SPIN_WAIT_HINT();
    }

    aicore_observe_cache_line_v0(worker_context);
    if (worker_context->active != 0 && aicore_gm_load_v0(run_control->exit_requested) == 0) {
        __gm__ AicoreTaskStreamV1 *stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(
            sidecar_base,
            core_type == CoreType::AIC ? worker_context->aic_stream_offset : worker_context->aiv_stream_offset
        );
        aicore_observe_cache_line_v0(stream);
        aicore_observe_data_cache_v0(reinterpret_cast<__gm__ void *>(graph.descriptors_address));

        AicorePendingSlotV1 pending[AICORE_PENDING_SLOT_COUNT_V1];
        bool payload_slot_used[AICORE_PENDING_SLOT_COUNT_V1]{};
        for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot)
            aicore_pending_clear_v1(&pending[slot]);
        AicoreWorkerStatsV1 stats{};
        const uint64_t resolver_count = aicore_gm_load_v0(run_control->aiv_active_worker_count);
        bool common_profile_recorded = false;
        uint32_t scan_start = 0;
        uint32_t backoff_iterations = kInitialBackoffIterations;
        uint32_t scheduler_error_poll_count = 0;
        const bool resolver_worker = core_type == CoreType::AIV;
        const bool executor_worker = static_cast<uint64_t>(worker_context->type_rank) < stream->initial_ticket_count;
        bool cursor_exhausted = !executor_worker;
        bool executor_drained = false;
        uint64_t victim_cursor =
            resolver_worker ? (worker_context->inbox_index + 1) % resolver_count : 0;
        uint64_t completions_since_service = 0;
        uint64_t previous_trace_commit_end = 0;

        if (executor_worker) {
            uint64_t seed_time = get_sys_cnt_aicore();
            __gm__ AicoreTaskTicketV1 *tickets =
                aicore_sidecar_at_v1<AicoreTaskTicketV1>(sidecar_base, stream->tickets_offset);
            AicoreTaskTicketV1 seed_ticket{};
            aicore_load_task_ticket_v1(&tickets[worker_context->type_rank], &seed_ticket);
            int64_t seed_task_id = static_cast<int64_t>(aicore_task_ticket_task_id_v1(seed_ticket));
            aicore_pending_initialize_v1(
                seed_ticket, static_cast<uint64_t>(worker_context->type_rank), AicoreClaimKindV1::SEED, seed_time,
                seed_time, &pending[0]
            );
            pending[0].claim_end_cycles = get_sys_cnt_aicore();
            AicoreTaskClaimBindingV1 seed_binding = aicore_make_claim_binding_v1(
                worker_context, seed_task_id, pending[0].kernel_id, pending[0].subtask_slot, 0,
                runtime->func_id_to_addr_[pending[0].kernel_id]
            );
            const bool seed_local_ready = pending[0].fanin_count == 0;
            if (!seed_local_ready) aicore_publish_claim_binding_v1(sidecar_base, run_control, seed_binding);
            payload_slot_used[0] = true;
            if (chip_swimlane_enabled) record_task_initialize_end(sidecar_base, worker_context, seed_task_id);
            AicoreRouteResultV1 route = seed_local_ready
                                            ? AicoreRouteResultV1::READY
                                            : aicore_route_task_v1(
                                                  graph, sidecar_base, worker_context, run_control, seed_task_id,
                                                  &stats.wake, chip_swimlane_enabled, false
                                              );
            if (route == AicoreRouteResultV1::ERROR || route == AicoreRouteResultV1::COMPLETED) {
                if (route == AicoreRouteResultV1::COMPLETED) {
                    aicore_record_scheduler_error_v1(
                        run_control, seed_task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, worker_context
                    );
                }
            } else {
                if (route == AicoreRouteResultV1::READY_TO_PUBLISH &&
                    !aicore_finalize_ready_v1(
                        graph, sidecar_base, worker_context, run_control, seed_binding, false, chip_swimlane_enabled
                    )) {
                    route = AicoreRouteResultV1::ERROR;
                }
                if (route == AicoreRouteResultV1::WAITING) {
                    (void)aicore_pending_state_v1(sidecar_base, worker_context, &pending[0]);
                    pending[0].pending_wait_start_cycles = get_sys_cnt_aicore();
                    pending[0].payload_needs_observe = 1;
                }
                if (chip_swimlane_enabled) record_task_route_end(sidecar_base, worker_context, seed_task_id);
                stats.seeded_task_count = 1;
                stats.pending_peak = 1;
            }
        }

        while (true) {
            if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
                scheduler_error_poll_count = 0;
                if (aicore_gm_load_v0(run_control->scheduler_error) != 0) break;
            }
            int32_t ready_slot = -1;
            uint32_t occupied = 0;
            uint64_t readiness_start = get_sys_cnt_aicore();
            for (uint32_t offset = 0; offset < AICORE_PENDING_SLOT_COUNT_V1; ++offset) {
                uint32_t slot = (scan_start + offset) % AICORE_PENDING_SLOT_COUNT_V1;
                if (pending[slot].task_id < 0) continue;
                ++occupied;
                ++stats.task_state_poll_count;
                int32_t previous_waiting = pending[slot].waiting_producer;
                AicorePendingStateV1 state = aicore_pending_state_v1(sidecar_base, worker_context, &pending[slot]);
                if (state == AicorePendingStateV1::ERROR) {
                    aicore_record_scheduler_error_v1(
                        run_control, pending[slot].task_id, AicoreRootStatusV0::INVALID_FANIN_ID, &graph, worker_context
                    );
                    ready_slot = -2;
                    break;
                }
                if (state == AicorePendingStateV1::READY) {
                    ready_slot = static_cast<int32_t>(slot);
                    break;
                }
                if (state == AicorePendingStateV1::BLOCKED) {
                    if (pending[slot].pending_wait_start_cycles == 0) {
                        pending[slot].pending_wait_start_cycles = readiness_start;
                    }
                    if (previous_waiting != pending[slot].waiting_producer) {
                        publish_worker_debug(
                            worker_context, pending, cursor_exhausted, 1,
                            AicoreWorkerDebugEventV1::WAITING_PRODUCER_CHANGED
                        );
                    }
                }
            }
            if (ready_slot == -2) break;
            if (ready_slot >= 0) {
                AicorePendingSlotV1 &task_pending = pending[ready_slot];
                uint64_t ready_observe = get_sys_cnt_aicore();
                uint64_t pending_wait_end = task_pending.pending_wait_start_cycles == 0 ? 0 : ready_observe;
                if (pending_wait_end != 0) {
                    stats.dependency_wait_cycles += pending_wait_end - task_pending.pending_wait_start_cycles;
                }
                uint64_t payload_start = get_sys_cnt_aicore();
                __gm__ PTO2DispatchPayload *dispatch_payload =
                    aicore_sidecar_at_v1<PTO2DispatchPayload>(
                        sidecar_base, worker_context->dispatch_payload_offset +
                                          static_cast<uint64_t>(ready_slot) * sizeof(PTO2DispatchPayload)
                    );
                if (task_pending.fanin_count == 0) {
                    AicoreTaskClaimBindingV1 binding = aicore_make_claim_binding_v1(
                        worker_context, task_pending.task_id, task_pending.kernel_id, task_pending.subtask_slot,
                        static_cast<uint8_t>(ready_slot), runtime->func_id_to_addr_[task_pending.kernel_id]
                    );
                    if (!aicore_materialize_claim_payload_v1(
                            graph, sidecar_base, worker_context, run_control, binding, false
                        )) {
                        break;
                    }
                } else if (task_pending.payload_needs_observe != 0) {
                    aicore_observe_dispatch_payload_v1(dispatch_payload);
                }

                OUT_OF_ORDER_STORE_BARRIER();
                __gm__ ChipSwimlaneAicoreTaskRecord *profile_record =
                    common_profile_recorded ? nullptr : aicore_task_profiling_reserve_v1(&task_profiling);
                uint64_t kernel_start = get_sys_cnt_aicore();
                if (chip_swimlane_enabled) {
                    __gm__ AicoreTaskControlV1 *task_control =
                        aicore_task_control_at_v1(sidecar_base, worker_context, task_pending.task_id);
                    aicore_observe_cache_line_v0(&task_control->next_waiter);
                    if (task_control->ready_publish_cycles != 0 && kernel_start >= task_control->ready_publish_cycles) {
                        uint64_t lag = kernel_start - task_control->ready_publish_cycles;
                        stats.completion.ready_to_kernel_cycles += lag;
                        if (lag > stats.completion.ready_to_kernel_max_cycles) {
                            stats.completion.ready_to_kernel_max_cycles = lag;
                        }
                    }
                }
                execute_task(dispatch_payload);
                uint64_t kernel_end = get_sys_cnt_aicore();
                uint64_t completion_start = get_sys_cnt_aicore();
                bool enqueue_ok = aicore_enqueue_completion_v1(
                    graph, sidecar_base, worker_context, run_control, resolver_count, task_pending.task_id,
                    &stats.completion, chip_swimlane_enabled
                );
                uint64_t completion_end = get_sys_cnt_aicore();
                if (!enqueue_ok) break;
                AicorePendingSlotV1 completed_trace_pending{};
                if (chip_swimlane_enabled) completed_trace_pending = task_pending;
                aicore_task_profiling_commit_v1(
                    profile_record, static_cast<uint64_t>(task_pending.task_id), task_pending.claim_end_cycles,
                    kernel_start, kernel_end
                );
                if (profile_record != nullptr) common_profile_recorded = true;

                ++stats.executed_task_count;
                stats.payload_cycles += kernel_start - payload_start;
                stats.kernel_cycles += kernel_end - kernel_start;
                stats.completion_enqueue_cycles += completion_end - completion_start;
                aicore_pending_clear_v1(&task_pending);
                scan_start = (static_cast<uint32_t>(ready_slot) + 1) % AICORE_PENDING_SLOT_COUNT_V1;
                backoff_iterations = kInitialBackoffIterations;
                bool service_completion =
                    resolver_worker && ++completions_since_service == kCompletionFairnessInterval;
                uint64_t completion_bookkeeping_end = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
                uint64_t completion_service_start = 0;
                uint64_t completion_service_end = 0;
                bool service_ok = true;
                if (service_completion) {
                    completion_service_start = completion_bookkeeping_end;
                    bool resolved = false;
                    service_ok = aicore_service_completion_inboxes_v1(
                        graph, sidecar_base, worker_context, run_control, resolver_count, &victim_cursor, &stats.wake,
                        &stats.completion, &resolved, chip_swimlane_enabled
                    );
                    completion_service_end = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
                    completions_since_service = 0;
                }
                if (chip_swimlane_enabled) {
                    commit_task_trace(
                        sidecar_base, worker_context, completed_trace_pending, pending_wait_end, readiness_start,
                        ready_observe, payload_start, kernel_start, kernel_end, completion_end,
                        completion_bookkeeping_end, completion_service_start, completion_service_end,
                        previous_trace_commit_end
                    );
                    previous_trace_commit_end = get_sys_cnt_aicore();
                }
                if (!service_ok) break;
                continue;
            }

            int32_t free_slot = -1;
            for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot) {
                if (pending[slot].task_id < 0) {
                    free_slot = static_cast<int32_t>(slot);
                    break;
                }
            }
            if (free_slot >= 0 && !cursor_exhausted) {
                uint64_t claim_start = get_sys_cnt_aicore();
                uint64_t stream_index = 0;
                AicoreTaskTicketV1 ticket{};
                bool valid = aicore_claim_ticket_v1(sidecar_base, stream, &stream_index, &ticket);
                uint64_t claim_end = get_sys_cnt_aicore();
                stats.claim_cycles += claim_end - claim_start;
                if (valid) {
                    int64_t task_id = static_cast<int64_t>(aicore_task_ticket_task_id_v1(ticket));
                    aicore_pending_initialize_v1(
                        ticket, stream_index, AicoreClaimKindV1::TICKET, claim_start, claim_end, &pending[free_slot]
                    );
                    __gm__ PTO2DispatchPayload *slot_payload =
                        aicore_sidecar_at_v1<PTO2DispatchPayload>(
                            sidecar_base, worker_context->dispatch_payload_offset +
                                              static_cast<uint64_t>(free_slot) * sizeof(PTO2DispatchPayload)
                        );
                    if (payload_slot_used[free_slot]) aicore_publish_dispatch_payload_v1(slot_payload);
                    AicoreTaskClaimBindingV1 binding = aicore_make_claim_binding_v1(
                        worker_context, task_id, pending[free_slot].kernel_id, pending[free_slot].subtask_slot,
                        static_cast<uint8_t>(free_slot), runtime->func_id_to_addr_[pending[free_slot].kernel_id]
                    );
                    const bool local_ready = pending[free_slot].fanin_count == 0;
                    if (!local_ready) aicore_publish_claim_binding_v1(sidecar_base, run_control, binding);
                    payload_slot_used[free_slot] = true;
                    if (chip_swimlane_enabled) record_task_initialize_end(sidecar_base, worker_context, task_id);
                    AicoreRouteResultV1 route = local_ready
                                                    ? AicoreRouteResultV1::READY
                                                    : aicore_route_task_v1(
                                                          graph, sidecar_base, worker_context, run_control, task_id,
                                                          &stats.wake, chip_swimlane_enabled, false
                                                      );
                    if (route == AicoreRouteResultV1::ERROR || route == AicoreRouteResultV1::COMPLETED) {
                        if (route == AicoreRouteResultV1::COMPLETED) {
                            aicore_record_scheduler_error_v1(
                                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, worker_context
                            );
                        }
                        break;
                    }
                    if (route == AicoreRouteResultV1::READY_TO_PUBLISH &&
                        !aicore_finalize_ready_v1(
                            graph, sidecar_base, worker_context, run_control, binding, false,
                            chip_swimlane_enabled
                        )) {
                        break;
                    }
                    if (route == AicoreRouteResultV1::WAITING) {
                        (void)aicore_pending_state_v1(sidecar_base, worker_context, &pending[free_slot]);
                        pending[free_slot].pending_wait_start_cycles = get_sys_cnt_aicore();
                        pending[free_slot].payload_needs_observe = 1;
                    }
                    if (chip_swimlane_enabled) record_task_route_end(sidecar_base, worker_context, task_id);
                    ++stats.ticket_claim_count;
                    uint64_t new_occupancy = occupied + 1;
                    if (new_occupancy > stats.pending_peak) stats.pending_peak = new_occupancy;
                    backoff_iterations = kInitialBackoffIterations;
                    continue;
                }
                cursor_exhausted = true;
                ++stats.ticket_exhaustion_count;
                publish_worker_debug(
                    worker_context, pending, cursor_exhausted, 1, AicoreWorkerDebugEventV1::CURSOR_EXHAUSTED
                );
            }

            bool pending_empty = true;
            for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot)
                pending_empty = pending_empty && pending[slot].task_id < 0;
            bool completion_progress = false;
            if (resolver_worker) {
                if (!aicore_service_completion_inboxes_v1(
                        graph, sidecar_base, worker_context, run_control, resolver_count, &victim_cursor, &stats.wake,
                        &stats.completion, &completion_progress, chip_swimlane_enabled
                    )) {
                    break;
                }
            }
            if (completion_progress) {
                backoff_iterations = kInitialBackoffIterations;
                continue;
            }
            if (cursor_exhausted && pending_empty && !executor_drained) {
                if (aicore_gm_load_v0(run_control->scheduler_error) != 0) break;
                aicore_gm_fetch_add_v0(run_control->executed_task_count, stats.executed_task_count);
                aicore_gm_fetch_add_v0(run_control->executor_drained_worker_count, UINT64_C(1));
                executor_drained = true;
            }
            const uint64_t executable_task_count =
                run_control->expected_task_count - run_control->inline_completed_count;
            if (executor_drained &&
                aicore_gm_load_v0(run_control->executor_drained_worker_count) == run_control->active_worker_count &&
                aicore_gm_load_v0(run_control->resolved_task_count) == executable_task_count) {
                uint64_t drain_start = get_sys_cnt_aicore();
                publish_worker_debug(worker_context, pending, cursor_exhausted, 2, AicoreWorkerDebugEventV1::DRAINING);
                stats.drain_start_cycles = drain_start;
                stats.drain_end_cycles = get_sys_cnt_aicore();
                stats.drain_cycles = stats.drain_end_cycles - drain_start;
                publish_worker_stats(worker_context, stats);
                aicore_gm_fetch_add_v0(run_control->drained_worker_count, UINT64_C(1));
                break;
            }

            ++stats.idle_iteration_count;
            uint64_t backoff_start = get_sys_cnt_aicore();
            local_backoff(backoff_iterations);
            stats.backoff_cycles += get_sys_cnt_aicore() - backoff_start;
            if (backoff_iterations < kMaximumBackoffIterations) {
                backoff_iterations <<= 1;
                if (backoff_iterations > kMaximumBackoffIterations) backoff_iterations = kMaximumBackoffIterations;
            }
        }
    }

    while (aicore_gm_load_v0(run_control->exit_requested) == 0) {
        local_backoff(kMaximumBackoffIterations);
    }
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
    OUT_OF_ORDER_STORE_BARRIER();
    aicore_gm_fetch_add_v0(run_control->finished_count, UINT64_C(1));
}
