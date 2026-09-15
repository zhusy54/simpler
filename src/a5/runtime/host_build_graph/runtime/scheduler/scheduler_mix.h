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

#include "scheduler_dispatch.h"

struct SchedulerMixTaskMetadata {
    uint16_t kernel_ids[3]{};
    uint8_t active_mask{0};
    uint8_t flags{0};
    uint16_t logical_block_num{0};
    uint16_t trace_index_base{0};
    int32_t timing_slot{-1};
};
static_assert(sizeof(SchedulerMixTaskMetadata) == 16, "Mix metadata cache must stay compact");

struct SchedulerMixDispatchPlan {
    SchedulerMixTaskMetadata metadata{};
    SchedulerResolvedPayloadSource payload_source{};
    uint64_t callable_addresses[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint8_t physical_lanes[PLATFORM_CORES_PER_BLOCKDIM]{0, 1, 2};
    bool predicate_passes{true};
};

struct SchedulerPendingMixDispatch {
    SchedulerReadyClaim ready{};
    SchedulerMixTaskMetadata metadata{};
};
static_assert(sizeof(SchedulerPendingMixDispatch) == 56, "Pending Mix state must stay compact");

struct SchedulerMixState {
    SchedulerPendingMixDispatch pending{};
    SchedulerMixTracker trackers[SCHEDULER_MIX_TRACKER_COUNT]{};
    uint64_t victim_cursor{0};
    uint32_t next_generation{1};
    bool probe_requested{true};
};

inline __aicore__ void
scheduler_mix_state_init(SchedulerMixState *state, uint64_t scheduler_index, uint64_t scheduler_count) {
    if (state == nullptr) return;
    *state = SchedulerMixState{};
    state->victim_cursor = scheduler_count == 0 ? 0 : (scheduler_index + 1) % scheduler_count;
    state->next_generation = 1;
    state->probe_requested = true;
}

inline __aicore__ uint32_t scheduler_mix_popcount(uint8_t mask) {
    return static_cast<uint32_t>(__builtin_popcount(static_cast<uint32_t>(mask & 7U)));
}

inline __aicore__ int32_t scheduler_mix_free_tracker(const SchedulerMixState *state) {
    if (state == nullptr) return -1;
    for (uint32_t index = 0; index < SCHEDULER_MIX_TRACKER_COUNT; ++index) {
        if (state->trackers[index].task_id == SCHEDULER_TASK_ID_INVALID) return static_cast<int32_t>(index);
    }
    return -1;
}

inline __aicore__ int32_t scheduler_mix_find_free_pending_slot(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint64_t worker_id
) {
    for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
        __gm__ SchedulerDispatchSlot *slot =
            scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
        if (scheduler_dispatch_state(scheduler_gm_query(slot->publication)) == SchedulerDispatchSlotState::FREE)
            return static_cast<int32_t>(pending_slot);
    }
    return -1;
}

inline __aicore__ uint64_t scheduler_mix_pending_skip_mask(const SchedulerMixState *state) {
    if (state == nullptr || state->pending.ready.task_id < 0) return 0;
    uint8_t physical_mask = state->pending.metadata.active_mask;
    if ((physical_mask & 1U) != 0 && scheduler_mix_popcount(physical_mask & 6U) == 1) physical_mask |= 6U;
    uint64_t result = 0;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((physical_mask & (1U << lane)) == 0) continue;
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            result |= UINT64_C(1) << (lane * SCHEDULER_PENDING_SLOT_COUNT + slot);
    }
    return result;
}

inline __aicore__ bool scheduler_mix_validate_claim(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerReadyClaim &ready, SchedulerPendingMixDispatch *pending
) {
    if (ready.task_id < 0 || static_cast<uint64_t>(ready.task_id) >= graph.task_count || pending == nullptr)
        return false;
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, scheduler, ready.task_id);
    scheduler_observe_cache_line(metadata);
    if (!scheduler_task_is_executable(metadata->flags) || !scheduler_task_is_mix(metadata->flags) ||
        scheduler_task_is_spmd(metadata->flags) || scheduler_task_requires_sync_start(metadata->flags) ||
        scheduler_task_is_inline(metadata->flags) || metadata->logical_block_num != 1 ||
        scheduler_mix_popcount(metadata->active_mask) < 2) {
        scheduler_record_error(
            run_control, ready.task_id, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            SchedulerErrorSite::MIX_DISPATCH_INVALID_SHAPE
        );
        return false;
    }
    const uint8_t active_mask = metadata->active_mask & 7U;
    const uint8_t active_aiv_mask = active_mask & 6U;
    const bool single_aiv = (active_mask & 1U) != 0 && scheduler_mix_popcount(active_aiv_mask) == 1;
    bool active_aiv_candidate = false;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        bool validate_lane = (active_mask & (1U << lane)) != 0;
        if (single_aiv && lane > 0) validate_lane = true;
        if (!validate_lane) continue;
        const uint64_t worker_id = scheduler->cluster_worker_ids[lane];
        if (worker_id >= scheduler->runtime_worker_count) {
            scheduler_record_error(
                run_control, ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::MIX_DISPATCH_INVALID_TOPOLOGY
            );
            return false;
        }
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        scheduler_observe_cache_line(target);
        const int32_t expected_core_type =
            lane == 0 ? static_cast<int32_t>(CoreType::AIC) : static_cast<int32_t>(CoreType::AIV);
        if (target->core_type != expected_core_type || (target->active == 0 && !(single_aiv && lane > 0))) {
            scheduler_record_error(
                run_control, ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::MIX_DISPATCH_INVALID_TOPOLOGY
            );
            return false;
        }
        if (single_aiv && lane > 0 && target->active != 0) active_aiv_candidate = true;
    }
    if (single_aiv && !active_aiv_candidate) {
        scheduler_record_error(
            run_control, ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::MIX_DISPATCH_INVALID_TOPOLOGY
        );
        return false;
    }
    pending->ready = ready;
    pending->metadata.kernel_ids[0] = metadata->kernel_ids[0];
    pending->metadata.kernel_ids[1] = metadata->kernel_ids[1];
    pending->metadata.kernel_ids[2] = metadata->kernel_ids[2];
    pending->metadata.active_mask = active_mask;
    pending->metadata.flags = metadata->flags;
    pending->metadata.logical_block_num = metadata->logical_block_num;
    pending->metadata.trace_index_base = metadata->trace_index_base;
    pending->metadata.timing_slot = metadata->timing_slot;
    return true;
}

inline __aicore__ bool scheduler_mix_prepare_plan(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerPendingMixDispatch &pending, SchedulerMixDispatchPlan *plan
) {
    if (plan == nullptr) return false;
    SchedulerMixDispatchPlan prepared{};
    prepared.metadata = pending.metadata;

    const uint8_t active_aiv_mask = prepared.metadata.active_mask & 6U;
    if ((prepared.metadata.active_mask & 1U) != 0 && scheduler_mix_popcount(active_aiv_mask) == 1) {
        const uint8_t logical_aiv = (active_aiv_mask & 2U) != 0 ? 1 : 2;
        prepared.physical_lanes[logical_aiv] = scheduler->cluster_worker_ids[2] == scheduler->worker_index ? 1 : 2;
    }
    SchedulerGraphResult status =
        scheduler_resolve_task_payload_source(graph, pending.ready.task_id, &prepared.payload_source);
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(
            run_control, pending.ready.task_id, status, &graph, scheduler,
            SchedulerErrorSite::MIX_DISPATCH_MATERIALIZE_FAILED
        );
        return false;
    }
    __gm__ uint64_t *callable_table =
        scheduler_state_at<uint64_t>(scheduler_state_base, scheduler->callable_addresses_offset);
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((prepared.metadata.active_mask & (1U << lane)) == 0) continue;
        if (!scheduler_lookup_callable_address(
                callable_table, prepared.metadata.kernel_ids[lane], &prepared.callable_addresses[lane]
            )) {
            scheduler_record_error(
                run_control, pending.ready.task_id, SchedulerGraphResult::INVALID_CALLABLE, &graph, scheduler,
                SchedulerErrorSite::MIX_DISPATCH_INVALID_CALLABLE
            );
            return false;
        }
    }
    if (scheduler_task_has_predicate(prepared.metadata.flags)) {
        const SchedulerPredicateResult predicate = scheduler_evaluate_task_predicate(graph, pending.ready.task_id);
        if (predicate == SchedulerPredicateResult::MALFORMED) {
            scheduler_record_error(
                run_control, pending.ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::MIX_DISPATCH_MATERIALIZE_FAILED
            );
            return false;
        }
        prepared.predicate_passes = predicate != SchedulerPredicateResult::FAIL;
    }
    *plan = prepared;
    return true;
}

inline __aicore__ bool scheduler_mix_prepare_lane(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerReadyClaim &ready, const SchedulerMixDispatchPlan &plan,
    uint8_t subtask_slot, uint32_t pending_slot, uint32_t tracker_index, uint32_t tracker_generation
) {
    const uint8_t physical_lane = plan.physical_lanes[subtask_slot];
    const uint64_t worker_id = scheduler->cluster_worker_ids[physical_lane];
    const uint16_t kernel_id = plan.metadata.kernel_ids[subtask_slot];
    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
    const uint64_t publication = scheduler_gm_query(slot->publication);
    if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::FREE) return false;
    uint32_t generation = scheduler_dispatch_generation(publication) + 1;
    if (generation == 0) generation = 1;
    slot->task_id = ready.task_id;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = scheduler_task_has_fanin(plan.metadata.flags) ? 1 : 0;
    slot->pending_slot = static_cast<uint8_t>(pending_slot);
    slot->block_num = 1;
    slot->generation = generation;
    slot->block_idx = 0;
    slot->cohort_generation = tracker_generation;
    slot->cohort_index = static_cast<uint8_t>(tracker_index);
    slot->gang = 1;
    scheduler_writeback_cache_line(slot);

    SchedulerTaskInfo task{
        ready.task_id,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? CoreType::AIC : CoreType::AIV,
    };
    __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
        scheduler_state_base,
        target->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(DispatchPayload)
    );
    SchedulerDispatchPayloadDirtyMask dirty_mask = 0;
    SchedulerGraphResult status = scheduler_materialize_task_payload_from_source(
        task, plan.callable_addresses[subtask_slot], plan.payload_source, payload, &dirty_mask, 0, 1
    );
    if (status == SchedulerGraphResult::OK && !plan.predicate_passes) payload->function_bin_addr = 0;
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(
            run_control, ready.task_id, status, &graph, scheduler, SchedulerErrorSite::MIX_DISPATCH_MATERIALIZE_FAILED
        );
        return false;
    }
    if (subtask_slot != 0) payload->global_context.sub_block_id = physical_lane == 2 ? 1 : 0;
    scheduler_publish_dispatch_payload(payload, dirty_mask);
    return true;
}

inline __aicore__ bool scheduler_mix_claim_free_slots(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, SchedulerMixDispatchPlan *plan,
    int32_t *pending_slots
) {
    const uint8_t active_mask = plan->metadata.active_mask;
    const uint8_t active_aiv_mask = active_mask & 6U;
    const bool single_aiv = (active_mask & 1U) != 0 && scheduler_mix_popcount(active_aiv_mask) == 1;
    const uint8_t logical_aiv = (active_aiv_mask & 2U) != 0 ? 1 : 2;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((active_mask & (1U << lane)) == 0 || (single_aiv && lane == logical_aiv)) continue;
        const uint8_t physical_lane = plan->physical_lanes[lane];
        pending_slots[lane] = scheduler_mix_find_free_pending_slot(
            scheduler_state_base, scheduler, scheduler->cluster_worker_ids[physical_lane]
        );
        if (pending_slots[lane] < 0) return false;
    }
    if (!single_aiv) return true;

    const uint8_t preferred_lane = plan->physical_lanes[logical_aiv];
    __gm__ SchedulerWorkerContext *preferred =
        scheduler_worker_context_at(scheduler_state_base, scheduler, scheduler->cluster_worker_ids[preferred_lane]);
    pending_slots[logical_aiv] = -1;
    if (preferred->active != 0)
        pending_slots[logical_aiv] = scheduler_mix_find_free_pending_slot(
            scheduler_state_base, scheduler, scheduler->cluster_worker_ids[preferred_lane]
        );
    if (pending_slots[logical_aiv] >= 0) return true;
    const uint8_t fallback_lane = preferred_lane == 1 ? 2 : 1;
    __gm__ SchedulerWorkerContext *fallback =
        scheduler_worker_context_at(scheduler_state_base, scheduler, scheduler->cluster_worker_ids[fallback_lane]);
    if (fallback->active != 0)
        pending_slots[logical_aiv] = scheduler_mix_find_free_pending_slot(
            scheduler_state_base, scheduler, scheduler->cluster_worker_ids[fallback_lane]
        );
    if (pending_slots[logical_aiv] < 0) return false;
    plan->physical_lanes[logical_aiv] = fallback_lane;
    return true;
}

inline __aicore__ bool scheduler_mix_dispatch_pending(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerMixState *state, uint64_t profiling_level, bool *dispatched
) {
    if (dispatched != nullptr) *dispatched = false;
    if (state == nullptr || state->pending.ready.task_id < 0) return true;
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t dispatch_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
    const int32_t tracker_index = scheduler_mix_free_tracker(state);
    if (tracker_index < 0) return true;

    SchedulerMixDispatchPlan plan{};
    if (!scheduler_mix_prepare_plan(graph, scheduler_state_base, scheduler, run_control, state->pending, &plan))
        return false;
    int32_t pending_slots[PLATFORM_CORES_PER_BLOCKDIM] = {-1, -1, -1};
    if (!scheduler_mix_claim_free_slots(scheduler_state_base, scheduler, &plan, pending_slots)) return true;

    uint32_t tracker_generation = state->next_generation++;
    if (tracker_generation == 0) tracker_generation = state->next_generation++;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((plan.metadata.active_mask & (1U << lane)) == 0) continue;
        if (!scheduler_mix_prepare_lane(
                graph, scheduler_state_base, scheduler, run_control, state->pending.ready, plan, lane,
                static_cast<uint32_t>(pending_slots[lane]), static_cast<uint32_t>(tracker_index), tracker_generation
            ))
            return false;
    }

    SchedulerMixTracker &tracker = state->trackers[static_cast<uint32_t>(tracker_index)];
    tracker.task_id = state->pending.ready.task_id;
    tracker.generation = tracker_generation;
    tracker.active_mask = plan.metadata.active_mask;
    tracker.completed_mask = 0;

    if (task_timing_enabled) {
        for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if ((plan.metadata.active_mask & (1U << lane)) == 0) continue;
            __gm__ SchedulerTaskTrace *trace = scheduler_task_trace_at(
                scheduler_state_base, scheduler, plan.metadata.trace_index_base, plan.metadata.active_mask, lane
            );
            trace->worker_id = scheduler->cluster_worker_ids[plan.physical_lanes[lane]];
            trace->task_id = static_cast<uint64_t>(state->pending.ready.task_id);
            if (phase_timing_enabled) {
                trace->ready_source = static_cast<uint64_t>(state->pending.ready.source);
                trace->publication_mode = static_cast<uint64_t>(state->pending.ready.publication_mode);
                trace->state_probe_scheduler_worker_id = scheduler->worker_index;
                trace->state_probe_start_cycles = state->pending.ready.state_probe_start_cycles;
                trace->state_probe_end_cycles = state->pending.ready.state_probe_end_cycles;
                trace->dispatch_start_cycles = dispatch_start_cycles;
                trace->dispatch_scheduler_worker_id = scheduler->worker_index;
                trace->dispatch_loop_iter = scheduler->profiling_loop_iter;
            }
            if (schedule_timing_enabled) trace->dispatch_end_cycles = scheduler_cycles();
            scheduler_publish_cache_line(trace);
            if (schedule_timing_enabled) scheduler_publish_cache_line(&trace->dispatch_start_cycles);
        }
    }

#if defined(__CCE_AICORE__)
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif

    if (phase_timing_enabled) {
        __gm__ SchedulerTaskControl *control =
            scheduler_task_control_at(scheduler_state_base, scheduler, state->pending.ready.task_id);
        scheduler_observe_cache_line(&control->next_waiter);
        control->ready_publish_cycles = scheduler_cycles();
        scheduler_publish_cache_line(&control->next_waiter);
    }
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((plan.metadata.active_mask & (1U << lane)) == 0) continue;
        __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
            scheduler_state_base, scheduler, scheduler->cluster_worker_ids[plan.physical_lanes[lane]],
            static_cast<uint32_t>(pending_slots[lane])
        );
        scheduler_gm_publish(
            slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
        );
    }
    state->pending.ready.task_id = SCHEDULER_TASK_ID_INVALID;
    state->probe_requested = true;
    if (dispatched != nullptr) *dispatched = true;
    return true;
}

// One call handles a bounded scheduling event: dispatch an existing pending Mix,
// then claim/dispatch more Mix work while local tracker capacity is available.
// A miss clears probe_requested, so the hot loop does not poll an empty Mix
// directory again until bootstrap or a completion event explicitly rearms it.
static __attribute__((noinline)) __aicore__ bool scheduler_service_mix_event(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerMixState *state, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, __gm__ SchedulerReadyOwnerState *owner_state, bool *progress
) {
    if (progress != nullptr) *progress = false;
    if (state == nullptr || owner_state == nullptr) return false;
    for (uint32_t attempt = 0; attempt < SCHEDULER_MIX_TRACKER_COUNT + 1; ++attempt) {
        bool dispatched = false;
        if (!scheduler_mix_dispatch_pending(
                graph, scheduler_state_base, scheduler, run_control, state, profiling_level, &dispatched
            ))
            return false;
        if (dispatched && progress != nullptr) *progress = true;
        if (state->pending.ready.task_id >= 0 || scheduler_mix_free_tracker(state) < 0 || !state->probe_requested)
            return true;

        SchedulerReadyClaim ready{};
        const uint64_t probe_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->scheduler_count,
                SCHEDULER_MIX_READY_QUEUE, &state->victim_cursor, ready_stats, &ready, owner_state
            ))
            return false;
        if (ready.task_id < 0) {
            state->probe_requested = false;
            return true;
        }
        ready.state_probe_start_cycles = probe_start;
        ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        if (!scheduler_mix_validate_claim(graph, scheduler_state_base, scheduler, run_control, ready, &state->pending))
            return false;
        if (progress != nullptr) *progress = true;
    }
    return true;
}
