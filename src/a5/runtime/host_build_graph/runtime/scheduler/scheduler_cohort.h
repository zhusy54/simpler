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

#include "scheduler_ready.h"

enum class SchedulerCohortTokenPhase : uint32_t {
    DRAIN = 0,
    STAGE = 1,
    DISPATCH = 2,
    COMPLETION = 3,
};

inline __aicore__ __gm__ SchedulerCohort *scheduler_cohort_record_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint32_t cohort_index
) {
    return scheduler_state_at<SchedulerCohort>(
        scheduler_state_base,
        context->cohort_cohorts_offset + static_cast<uint64_t>(cohort_index) * sizeof(SchedulerCohort)
    );
}

inline __aicore__ __gm__ SchedulerCohortParticipant *scheduler_cohort_participant_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint32_t cohort_index,
    uint64_t scheduler_index
) {
    const uint64_t linear = static_cast<uint64_t>(cohort_index) * SCHEDULER_CLUSTER_CAPACITY + scheduler_index;
    return scheduler_state_at<SchedulerCohortParticipant>(
        scheduler_state_base, context->cohort_participants_offset + linear * sizeof(SchedulerCohortParticipant)
    );
}

inline __aicore__ __gm__ SchedulerCohortCommand *scheduler_cohort_command_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint64_t scheduler_index
) {
    return scheduler_state_at<SchedulerCohortCommand>(
        scheduler_state_base, context->cohort_commands_offset + scheduler_index * sizeof(SchedulerCohortCommand)
    );
}

inline __aicore__ void scheduler_cohort_publish_command(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint64_t scheduler_index,
    uint32_t cohort_index, uint64_t generation, SchedulerCohortState state
) {
    __gm__ SchedulerCohortCommand *command =
        scheduler_cohort_command_at(scheduler_state_base, scheduler, scheduler_index);
    command->generation[cohort_index] = generation;
    command->state[cohort_index] = static_cast<uint64_t>(state);
    scheduler_publish_cache_line(command);
}

inline __aicore__ bool scheduler_cohort_forward_command(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerCohortParticipant *participant, uint32_t cohort_index, uint64_t generation,
    SchedulerCohortState state
) {
    if (participant->forwarded_generation == generation && participant->forwarded_state == static_cast<uint32_t>(state))
        return false;
    const uint64_t children[2] = {scheduler->scheduler_index * 2 + 1, scheduler->scheduler_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant->participant_count) continue;
        scheduler_cohort_publish_command(
            scheduler_state_base, scheduler, children[child_slot], cohort_index, generation, state
        );
    }
    participant->forwarded_generation = generation;
    participant->forwarded_state = static_cast<uint32_t>(state);
    scheduler_publish_cache_line(participant);
    return true;
}

inline __aicore__ uint32_t scheduler_cohort_popcount(uint32_t mask) {
    return static_cast<uint32_t>(__builtin_popcount(mask));
}

inline __aicore__ __gm__ volatile uint64_t *
scheduler_cohort_local_token(__gm__ SchedulerCohortParticipant *participant, SchedulerCohortTokenPhase phase) {
    if (phase == SchedulerCohortTokenPhase::DRAIN) return &participant->drain_local_token;
    if (phase == SchedulerCohortTokenPhase::STAGE) return &participant->stage_local_token;
    if (phase == SchedulerCohortTokenPhase::DISPATCH) return &participant->dispatch_local_token;
    return &participant->completion_local_token;
}

inline __aicore__ __gm__ volatile uint64_t *
scheduler_cohort_subtree_token(__gm__ SchedulerCohortParticipant *participant, SchedulerCohortTokenPhase phase) {
    if (phase == SchedulerCohortTokenPhase::DRAIN) return &participant->drain_subtree_token;
    if (phase == SchedulerCohortTokenPhase::STAGE) return &participant->stage_subtree_token;
    if (phase == SchedulerCohortTokenPhase::DISPATCH) return &participant->dispatch_subtree_token;
    return &participant->completion_subtree_token;
}

inline __aicore__ bool scheduler_cohort_publish_local_token(
    __gm__ SchedulerCohortParticipant *participant, SchedulerCohortTokenPhase phase, uint64_t generation
) {
    __gm__ volatile uint64_t *token = scheduler_cohort_local_token(participant, phase);
    if (*token == generation) return false;
    *token = generation;
    scheduler_publish_cache_line(token);
    return true;
}

inline __aicore__ bool scheduler_cohort_update_subtree_token(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t cohort_index,
    uint64_t participant_count, uint64_t generation, SchedulerCohortTokenPhase phase
) {
    const uint64_t scheduler_index = scheduler->scheduler_index;
    if (scheduler_index >= participant_count) return false;
    __gm__ SchedulerCohortParticipant *participant =
        scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, scheduler_index);
    __gm__ volatile uint64_t *local_token = scheduler_cohort_local_token(participant, phase);
    __gm__ volatile uint64_t *subtree_token = scheduler_cohort_subtree_token(participant, phase);
    if (*subtree_token == generation) return true;
    scheduler_observe_cache_line(local_token);
    if (*local_token != generation) return false;
    const uint64_t children[2] = {scheduler_index * 2 + 1, scheduler_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant_count) continue;
        __gm__ SchedulerCohortParticipant *child =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, children[child_slot]);
        __gm__ volatile uint64_t *child_token = scheduler_cohort_subtree_token(child, phase);
        scheduler_observe_cache_line(child_token);
        if (*child_token != generation) return false;
    }
    *subtree_token = generation;
    scheduler_publish_cache_line(subtree_token);
    return true;
}

inline __aicore__ bool scheduler_cohort_root_token_ready(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t cohort_index,
    uint64_t generation, SchedulerCohortTokenPhase phase
) {
    __gm__ SchedulerCohortParticipant *root =
        scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, 0);
    __gm__ volatile uint64_t *token = scheduler_cohort_subtree_token(root, phase);
    scheduler_observe_cache_line(token);
    return *token == generation;
}

inline __aicore__ uint64_t scheduler_cohort_retire_token(uint64_t generation) { return ~generation; }

inline __aicore__ void scheduler_cohort_publish_retire(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerCohortParticipant *participant, uint32_t cohort_index, uint64_t generation
) {
    const uint64_t retire_token = scheduler_cohort_retire_token(generation);
    scheduler_cohort_publish_local_token(participant, SchedulerCohortTokenPhase::COMPLETION, retire_token);
    (void)scheduler_cohort_update_subtree_token(
        scheduler_state_base, scheduler, cohort_index, participant->participant_count, retire_token,
        SchedulerCohortTokenPhase::COMPLETION
    );
}

inline __aicore__ bool scheduler_fill_explicit_dispatch_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t worker_id, uint32_t pending_slot, int64_t task_id,
    uint8_t subtask_slot, uint32_t block_idx, uint32_t block_num, uint32_t cohort_index, uint32_t cohort_generation,
    SchedulerDispatchSlotState publication
) {
    if (worker_id >= scheduler->runtime_worker_count || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT || task_id < 0 ||
        static_cast<uint64_t>(task_id) >= graph.task_count || subtask_slot >= 3 || block_num == 0 ||
        block_idx >= block_num) {
        return false;
    }
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, scheduler, task_id);
    scheduler_observe_cache_line(metadata);
    if ((metadata->active_mask & (1U << subtask_slot)) == 0) return false;
    const uint16_t kernel_id = metadata->kernel_ids[subtask_slot];
    __gm__ uint64_t *callable_addresses =
        scheduler_state_at<uint64_t>(scheduler_state_base, scheduler->callable_addresses_offset);
    uint64_t callable_address = 0;
    if (!scheduler_lookup_callable_address(callable_addresses, kernel_id, &callable_address)) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::INVALID_CALLABLE, &graph, scheduler,
            SchedulerErrorSite::COHORT_INVALID_CALLABLE
        );
        return false;
    }

    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    scheduler_observe_cache_line(target);
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
    scheduler_observe_cache_line(slot);
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = task_id;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = scheduler_task_has_fanin(metadata->flags) ? 1 : 0;
    slot->pending_slot = static_cast<uint8_t>(pending_slot);
    slot->block_num = static_cast<uint16_t>(block_num);
    slot->generation = generation;
    slot->block_idx = block_idx;
    slot->cohort_generation = cohort_generation;
    slot->cohort_index = static_cast<uint8_t>(cohort_index);
    scheduler_writeback_cache_line(slot);

    SchedulerTaskInfo task{
        task_id,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? CoreType::AIC : CoreType::AIV,
    };
    __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
        scheduler_state_base,
        target->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(DispatchPayload)
    );
    SchedulerGraphResult status = scheduler_materialize_task_payload_resolved(
        graph, task, callable_address, payload, static_cast<int32_t>(block_idx), static_cast<int32_t>(block_num)
    );
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(
            run_control, task_id, status, &graph, scheduler, SchedulerErrorSite::COHORT_MATERIALIZE_FAILED
        );
        return false;
    }
    scheduler_publish_dispatch_payload(payload);
    scheduler_gm_store(slot->publication, scheduler_dispatch_publication(generation, publication));
    return true;
}

inline __aicore__ int32_t scheduler_cohort_find_free_pending_slot(
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

inline __aicore__ bool scheduler_cohort_fill_single(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerCohortParticipant *participant, uint32_t cohort_index,
    uint64_t worker_id, uint8_t subtask_slot, uint32_t block_idx, SchedulerDispatchSlotState publication
) {
    const int32_t pending_slot = scheduler_cohort_find_free_pending_slot(scheduler_state_base, scheduler, worker_id);
    if (pending_slot < 0) return false;
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, static_cast<uint32_t>(pending_slot));
    const uint32_t generation = scheduler_dispatch_generation(scheduler_gm_query(slot->publication));
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FILLING)
    );
    if (!scheduler_fill_explicit_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, worker_id, static_cast<uint32_t>(pending_slot),
            participant->task_id, subtask_slot, block_idx, participant->logical_block_num, cohort_index,
            static_cast<uint32_t>(participant->config_generation), publication
        )) {
        return false;
    }
    ++participant->local_published_subtasks;
    return true;
}

inline __aicore__ bool scheduler_cohort_fill_mix_block(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerCohortParticipant *participant, uint32_t cohort_index,
    uint32_t block_idx, SchedulerDispatchSlotState publication
) {
    int32_t pending_slots[3] = {-1, -1, -1};
    // A MIX block only requires one free slot on each active lane. Slot indices
    // are lane-local and need not match; this avoids artificial head-of-line
    // blocking when complementary slots are free across the Cluster.
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        const uint64_t worker_id = scheduler->cluster_worker_ids[subtask_slot];
        pending_slots[subtask_slot] =
            scheduler_cohort_find_free_pending_slot(scheduler_state_base, scheduler, worker_id);
        if (pending_slots[subtask_slot] < 0) return false;
    }
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        const uint64_t worker_id = scheduler->cluster_worker_ids[subtask_slot];
        __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
            scheduler_state_base, scheduler, worker_id, static_cast<uint32_t>(pending_slots[subtask_slot])
        );
        const uint32_t generation = scheduler_dispatch_generation(scheduler_gm_query(slot->publication));
        scheduler_gm_store(
            slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FILLING)
        );
    }
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        if (!scheduler_fill_explicit_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->cluster_worker_ids[subtask_slot],
                static_cast<uint32_t>(pending_slots[subtask_slot]), participant->task_id, subtask_slot, block_idx,
                participant->logical_block_num, cohort_index, static_cast<uint32_t>(participant->config_generation),
                SchedulerDispatchSlotState::GATED
            )) {
            return false;
        }
    }
    if (publication == SchedulerDispatchSlotState::READY) {
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, scheduler, scheduler->cluster_worker_ids[subtask_slot],
                static_cast<uint32_t>(pending_slots[subtask_slot])
            );
            scheduler_gm_store(
                slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
            );
        }
    }
    participant->local_published_subtasks += scheduler_cohort_popcount(participant->active_mask);
    return true;
}

inline __aicore__ bool scheduler_cohort_local_slots_drained(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t active_mask
) {
    const bool single_aiv = scheduler_cohort_popcount(active_mask) == 1 && (active_mask & 6U) != 0;
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const bool used =
            cluster_lane == 0 ? (active_mask & 1U) != 0 : (single_aiv || (active_mask & (1U << cluster_lane)) != 0);
        if (!used) continue;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, scheduler, scheduler->cluster_worker_ids[cluster_lane], pending_slot
            );
            if (scheduler_dispatch_state(scheduler_gm_query(slot->publication)) != SchedulerDispatchSlotState::FREE)
                return false;
        }
    }
    return true;
}

inline __aicore__ void scheduler_cohort_fill_participant(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerCohortParticipant *participant, uint32_t cohort_index,
    SchedulerDispatchSlotState publication
) {
    const uint32_t active_mask = participant->active_mask;
    if (scheduler_cohort_popcount(active_mask) > 1) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!scheduler_cohort_fill_mix_block(
                    graph, scheduler_state_base, scheduler, run_control, participant, cohort_index, block_idx,
                    publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else if ((active_mask & 1U) != 0) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!scheduler_cohort_fill_single(
                    graph, scheduler_state_base, scheduler, run_control, participant, cohort_index,
                    scheduler->cluster_worker_ids[0], 0, block_idx, publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else {
        const uint8_t subtask_slot = (active_mask & 2U) != 0 ? 1 : 2;
        for (uint32_t lane = 0; lane < 2; ++lane) {
            while (participant->next_block[lane] < participant->logical_block_num) {
                const uint32_t block_idx = participant->next_block[lane];
                if (!scheduler_cohort_fill_single(
                        graph, scheduler_state_base, scheduler, run_control, participant, cohort_index,
                        scheduler->cluster_worker_ids[lane + 1], subtask_slot, block_idx, publication
                    ))
                    break;
                participant->next_block[lane] += participant->block_stride;
            }
        }
    }
    scheduler_publish_cache_line(participant);
}

inline __aicore__ void scheduler_cohort_release_local_slots(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t cohort_index,
    uint64_t generation
) {
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
                scheduler_state_base, scheduler, scheduler->cluster_worker_ids[cluster_lane], pending_slot
            );
            const uint64_t publication = scheduler_gm_query(slot->publication);
            if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::GATED) continue;
            scheduler_observe_cache_line(slot);
            if (slot->cohort_index != cohort_index || slot->cohort_generation != generation) continue;
            scheduler_gm_store(
                slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
            );
        }
    }
}

inline __aicore__ uint32_t
scheduler_cohort_assigned_blocks(uint32_t first, uint32_t stride, uint32_t logical_block_num) {
    if (first >= logical_block_num) return 0;
    return 1 + (logical_block_num - 1 - first) / stride;
}

inline __aicore__ int64_t scheduler_cohort_select_sync_ready_task(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerCohortCoordinator *coordinator
) {
    if (scheduler->graph_task_count == 0) return SCHEDULER_TASK_ID_INVALID;
    const uint64_t start = coordinator->scan_cursor % scheduler->graph_task_count;
    for (uint64_t offset = 0; offset < scheduler->graph_task_count; ++offset) {
        const uint64_t task_index = (start + offset) % scheduler->graph_task_count;
        __gm__ SchedulerTaskMetadata *metadata =
            scheduler_task_metadata_at(scheduler_state_base, scheduler, static_cast<int64_t>(task_index));
        scheduler_observe_cache_line(metadata);
        if (!scheduler_task_requires_sync_start(metadata->flags)) continue;
        __gm__ SchedulerTaskControl *control =
            scheduler_task_control_at(scheduler_state_base, scheduler, static_cast<int64_t>(task_index));
        if (scheduler_gm_query(control->state) == static_cast<int64_t>(SchedulerTaskState::READY)) {
            coordinator->scan_cursor = task_index + 1;
            return static_cast<int64_t>(task_index);
        }
    }
    return SCHEDULER_TASK_ID_INVALID;
}

inline __aicore__ uint32_t
scheduler_cohort_claim_record(__gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler) {
    for (uint32_t index = 0; index < SCHEDULER_COHORT_COUNT; ++index) {
        __gm__ SchedulerCohort *cohort = scheduler_cohort_record_at(scheduler_state_base, scheduler, index);
        if (scheduler_gm_compare_exchange(
                cohort->state, static_cast<uint64_t>(SchedulerCohortState::FREE),
                static_cast<uint64_t>(SchedulerCohortState::CLAIMING)
            ) == static_cast<uint64_t>(SchedulerCohortState::FREE))
            return index;
    }
    return UINT32_MAX;
}

inline __aicore__ bool scheduler_cohort_claim_regular_ready(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerReadyOwnerState *owner_state,
    SchedulerReadyStats *ready_stats, SchedulerReadyClaim *claim
) {
    const uint32_t queues[] = {
        static_cast<uint32_t>(SchedulerReadyQueueIndex::REGULAR_MIX),
        static_cast<uint32_t>(SchedulerReadyQueueIndex::REGULAR_SPMD_AIC),
        static_cast<uint32_t>(SchedulerReadyQueueIndex::REGULAR_SPMD_AIV),
    };
    for (uint32_t queue : queues) {
        if (!scheduler_ready_owner_maintain_type(scheduler_state_base, scheduler, queue, owner_state)) return false;
        for (uint64_t offset = 0; offset < scheduler->scheduler_count; ++offset) {
            const uint64_t victim = (scheduler->scheduler_index + offset) % scheduler->scheduler_count;
            int64_t task_id = SCHEDULER_TASK_ID_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph, scheduler_state_base, scheduler, run_control, queue, victim, &task_id, ready_stats
                ))
                return false;
            if (task_id < 0) continue;
            claim->task_id = task_id;
            claim->inbox_index = victim;
            claim->source = offset == 0 ? SchedulerReadySource::LOCAL : SchedulerReadySource::STOLEN;
            if (offset != 0 && ready_stats != nullptr) ++ready_stats->steal_count;
            return true;
        }
    }
    *claim = {};
    return true;
}

inline __aicore__ void scheduler_cohort_initialize_participants(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t cohort_index, int64_t task_id,
    uint64_t generation, uint32_t active_mask, uint32_t block_num, uint32_t base_scheduler, SchedulerCohortPolicy policy
) {
    const uint32_t scheduler_count = static_cast<uint32_t>(scheduler->scheduler_count);
    const bool single_aiv = scheduler_cohort_popcount(active_mask) == 1 && (active_mask & 6U) != 0;
    for (uint32_t physical = 0; physical < scheduler_count; ++physical) {
        const uint32_t rank = (physical + scheduler_count - base_scheduler) % scheduler_count;
        __gm__ SchedulerCohortParticipant *participant =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, physical);
        participant->task_id = task_id;
        participant->active_mask = active_mask;
        participant->logical_block_num = block_num;
        participant->local_published_subtasks = 0;
        participant->local_completed_subtasks = 0;
        participant->forwarded_state = UINT32_MAX;
        participant->forwarded_generation = 0;
        participant->drain_local_token = 0;
        participant->drain_subtree_token = 0;
        participant->stage_local_token = 0;
        participant->stage_subtree_token = 0;
        participant->dispatch_local_token = 0;
        participant->dispatch_subtree_token = 0;
        participant->completion_local_token = 0;
        participant->completion_subtree_token = 0;
        participant->participant_count = scheduler_count;
        if (single_aiv) {
            participant->block_stride = scheduler_count * 2;
            participant->next_block[0] = rank * 2;
            participant->next_block[1] = rank * 2 + 1;
            participant->local_expected_subtasks =
                policy == SchedulerCohortPolicy::SYNC_START ?
                    scheduler_cohort_assigned_blocks(rank * 2, scheduler_count * 2, block_num) +
                        scheduler_cohort_assigned_blocks(rank * 2 + 1, scheduler_count * 2, block_num) :
                    0;
        } else {
            participant->block_stride = scheduler_count;
            participant->next_block[0] = rank;
            participant->next_block[1] = UINT32_MAX;
            participant->local_expected_subtasks =
                policy == SchedulerCohortPolicy::SYNC_START ?
                    scheduler_cohort_assigned_blocks(rank, scheduler_count, block_num) *
                        scheduler_cohort_popcount(active_mask) :
                    0;
        }
        scheduler_publish_cache_line(&participant->next_block[0]);
        scheduler_publish_cache_line(participant);
        scheduler_publish_cache_line(&participant->drain_local_token);
        // config_generation is the publication token for the participant line.
        scheduler_gm_publish(participant->config_generation, generation);
    }
}

inline __aicore__ bool scheduler_cohort_initialize_record(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, int64_t task_id, SchedulerCohortPolicy policy
) {
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, scheduler, task_id);
    scheduler_observe_cache_line(metadata);
    const uint32_t active_mask = metadata->active_mask;
    const uint32_t block_num = metadata->logical_block_num;
    const bool single_aiv = scheduler_cohort_popcount(active_mask) == 1 && (active_mask & 6U) != 0;
    const uint32_t capacity = single_aiv ? static_cast<uint32_t>(scheduler->scheduler_count * 2) :
                                           static_cast<uint32_t>(scheduler->scheduler_count);
    __gm__ SchedulerCohortCoordinator *coordinator = scheduler_cohort_coordinator_at(scheduler_state_base, scheduler);
    if (policy == SchedulerCohortPolicy::SYNC_START && block_num > capacity) {
        scheduler_gm_fetch_add(coordinator->capacity_reject_count, UINT64_C(1));
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            SchedulerErrorSite::COHORT_UNSUPPORTED_SHAPE
        );
        scheduler_gm_store(
            scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index)->state,
            static_cast<uint64_t>(SchedulerCohortState::FREE)
        );
        return false;
    }
    uint64_t generation = scheduler_gm_fetch_add(coordinator->next_generation, UINT64_C(1)) + 1;
    if (generation == 0) generation = scheduler_gm_fetch_add(coordinator->next_generation, UINT64_C(1)) + 1;
    const uint32_t base_scheduler =
        policy == SchedulerCohortPolicy::SYNC_START ?
            static_cast<uint32_t>(coordinator->next_sync_scheduler % scheduler->scheduler_count) :
            static_cast<uint32_t>(scheduler->scheduler_index);
    if (policy == SchedulerCohortPolicy::SYNC_START) {
        const uint32_t used = single_aiv ? (block_num + 1) / 2 : block_num;
        coordinator->next_sync_scheduler = (base_scheduler + used) % scheduler->scheduler_count;
    }
    scheduler_cohort_initialize_participants(
        scheduler_state_base, scheduler, cohort_index, task_id, generation, active_mask, block_num, base_scheduler,
        policy
    );
    __gm__ SchedulerCohort *cohort = scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index);
    cohort->task_id = task_id;
    cohort->generation = generation;
    cohort->priority_bit = scheduler_task_priority_bit(metadata->flags);
    cohort->active_mask = active_mask;
    cohort->logical_block_num = block_num;
    cohort->participant_count = scheduler->scheduler_count;
    cohort->local_stride = single_aiv ? scheduler->scheduler_count * 2 : scheduler->scheduler_count;
    cohort->policy = static_cast<uint64_t>(policy);
    cohort->owner_scheduler = scheduler->scheduler_index;
    cohort->base_scheduler = base_scheduler;
    cohort->participant_bitmap = 0;
    cohort->completion_bitmap = 0;
    cohort->admitted_cycles = scheduler_cycles();
    cohort->dispatch_complete_cycles = 0;
    cohort->completion_cycles = 0;
    scheduler_publish_cache_line(cohort);
    scheduler_publish_cache_line(&cohort->policy);
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DISPATCHING));
    const SchedulerCohortState initial = policy == SchedulerCohortPolicy::SYNC_START ?
                                             SchedulerCohortState::DRAINING :
                                             SchedulerCohortState::DISPATCHING;
    scheduler_gm_publish(cohort->state, static_cast<uint64_t>(initial));
    scheduler_gm_fetch_add(coordinator->admitted_count, UINT64_C(1));
    if (policy == SchedulerCohortPolicy::SYNC_START) scheduler_gm_fetch_add(coordinator->sync_drain_count, UINT64_C(1));
    else if (scheduler_task_is_mix(metadata->flags))
        scheduler_gm_fetch_add(coordinator->mix_dispatch_count, UINT64_C(1));
    else scheduler_gm_fetch_add(coordinator->spmd_dispatch_count, UINT64_C(1));
    return true;
}

inline __aicore__ bool scheduler_cohort_admit_sync(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control
) {
    if (scheduler->scheduler_index != 0) return false;
    __gm__ SchedulerCohortCoordinator *coordinator = scheduler_cohort_coordinator_at(scheduler_state_base, scheduler);
    if (scheduler_gm_query(coordinator->active_dispatch_cohort) != UINT64_MAX ||
        scheduler_gm_query(coordinator->sync_ready_bits) == 0)
        return false;
    const uint32_t cohort_index = scheduler_cohort_claim_record(scheduler_state_base, scheduler);
    if (cohort_index == UINT32_MAX) return false;
    const int64_t task_id = scheduler_cohort_select_sync_ready_task(scheduler_state_base, scheduler, coordinator);
    if (task_id < 0) {
        scheduler_gm_store(
            scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index)->state,
            static_cast<uint64_t>(SchedulerCohortState::FREE)
        );
        scheduler_gm_store(coordinator->sync_ready_bits, UINT64_C(0));
        if (scheduler_cohort_select_sync_ready_task(scheduler_state_base, scheduler, coordinator) >= 0)
            scheduler_gm_store(coordinator->sync_ready_bits, UINT64_C(1));
        return false;
    }
    if (!scheduler_cohort_initialize_record(
            graph, scheduler_state_base, scheduler, run_control, cohort_index, task_id,
            SchedulerCohortPolicy::SYNC_START
        ))
        return false;
    scheduler_gm_publish(coordinator->active_dispatch_cohort, cohort_index);
    __gm__ SchedulerCohort *cohort = scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index);
    scheduler_cohort_publish_command(
        scheduler_state_base, scheduler, 0, cohort_index, cohort->generation, SchedulerCohortState::DRAINING
    );
    return true;
}

inline __aicore__ bool scheduler_cohort_admit_regular(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerReadyStats *ready_stats,
    __gm__ SchedulerReadyOwnerState *owner_state
) {
    __gm__ SchedulerCohortCoordinator *coordinator = scheduler_cohort_coordinator_at(scheduler_state_base, scheduler);
    if (scheduler_gm_query(coordinator->active_dispatch_cohort) != UINT64_MAX ||
        scheduler_gm_query(coordinator->sync_ready_bits) != 0)
        return false;
    const uint32_t cohort_index = scheduler_cohort_claim_record(scheduler_state_base, scheduler);
    if (cohort_index == UINT32_MAX) return false;
    SchedulerReadyClaim ready{};
    if (!scheduler_cohort_claim_regular_ready(
            graph, scheduler_state_base, scheduler, run_control, owner_state, ready_stats, &ready
        )) {
        scheduler_gm_store(
            scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index)->state,
            static_cast<uint64_t>(SchedulerCohortState::FREE)
        );
        return false;
    }
    if (ready.task_id < 0) {
        scheduler_gm_store(
            scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index)->state,
            static_cast<uint64_t>(SchedulerCohortState::FREE)
        );
        return false;
    }
    return scheduler_cohort_initialize_record(
        graph, scheduler_state_base, scheduler, run_control, cohort_index, ready.task_id, SchedulerCohortPolicy::REGULAR
    );
}

inline constexpr uint32_t SCHEDULER_COHORT_BLOCK_CLAIM_BUSY = UINT32_C(1) << 31;

inline __aicore__ __gm__ volatile uint64_t *
scheduler_cohort_claim_pair(__gm__ SchedulerCohortParticipant *participant) {
    return reinterpret_cast<__gm__ volatile uint64_t *>(const_cast<__gm__ uint32_t *>(participant->next_block));
}

inline __aicore__ bool
scheduler_cohort_try_claim_block(__gm__ SchedulerCohortParticipant *participant, uint32_t lane, uint32_t *block_idx) {
    if (lane >= 2 || block_idx == nullptr) return false;
    __gm__ volatile uint64_t *pair = scheduler_cohort_claim_pair(participant);
    for (uint32_t attempt = 0; attempt < 8; ++attempt) {
        const uint64_t observed = scheduler_gm_query(*pair);
        const uint32_t current = static_cast<uint32_t>(observed >> (lane * 32));
        if (current >= participant->logical_block_num || (current & SCHEDULER_COHORT_BLOCK_CLAIM_BUSY) != 0)
            return false;
        const uint64_t lane_mask = UINT64_C(0xffffffff) << (lane * 32);
        const uint64_t desired = (observed & ~lane_mask) |
                                 (static_cast<uint64_t>(current | SCHEDULER_COHORT_BLOCK_CLAIM_BUSY) << (lane * 32));
        if (scheduler_gm_compare_exchange(*pair, observed, desired) == observed) {
            *block_idx = current;
            return true;
        }
    }
    return false;
}

inline __aicore__ void scheduler_cohort_finish_block_claim(
    __gm__ SchedulerCohortParticipant *participant, uint32_t lane, uint32_t block_idx, bool advance
) {
    __gm__ volatile uint64_t *pair = scheduler_cohort_claim_pair(participant);
    const uint32_t replacement = advance ? block_idx + participant->block_stride : block_idx;
    while (true) {
        const uint64_t observed = scheduler_gm_query(*pair);
        const uint64_t lane_mask = UINT64_C(0xffffffff) << (lane * 32);
        const uint64_t desired = (observed & ~lane_mask) | (static_cast<uint64_t>(replacement) << (lane * 32));
        if (scheduler_gm_compare_exchange(*pair, observed, desired) == observed) return;
    }
}

inline __aicore__ bool scheduler_cohort_regular_claims_exhausted(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler, uint32_t cohort_index,
    uint32_t logical_block_num
) {
    for (uint32_t physical = 0; physical < scheduler->scheduler_count; ++physical) {
        __gm__ SchedulerCohortParticipant *source =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, physical);
        const uint64_t claims = scheduler_gm_query(*scheduler_cohort_claim_pair(source));
        for (uint32_t lane = 0; lane < 2; ++lane) {
            const uint32_t next = static_cast<uint32_t>(claims >> (lane * 32));
            if (next == UINT32_MAX) continue;
            if ((next & SCHEDULER_COHORT_BLOCK_CLAIM_BUSY) != 0 || next < logical_block_num) return false;
        }
    }
    return true;
}

inline __aicore__ bool scheduler_cohort_dispatch_one_regular_block(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, __gm__ SchedulerCohort *cohort
) {
    __gm__ SchedulerCohortParticipant *local =
        scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, scheduler->scheduler_index);
    scheduler_observe_cache_line(local);
    if (local->config_generation != cohort->generation) return false;
    const bool single_aiv = scheduler_cohort_popcount(local->active_mask) == 1 && (local->active_mask & 6U) != 0;
    for (uint32_t victim_offset = 0; victim_offset < scheduler->scheduler_count; ++victim_offset) {
        const uint32_t victim =
            static_cast<uint32_t>((scheduler->scheduler_index + victim_offset) % scheduler->scheduler_count);
        __gm__ SchedulerCohortParticipant *source =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, victim);
        scheduler_observe_cache_line(source);
        if (source->config_generation != cohort->generation) continue;
        const uint32_t lane_count = single_aiv ? 2 : 1;
        for (uint32_t lane = 0; lane < lane_count; ++lane) {
            uint32_t block_idx = 0;
            if (!scheduler_cohort_try_claim_block(source, lane, &block_idx)) continue;
            const uint32_t before = local->local_published_subtasks;
            bool filled = false;
            if (scheduler_cohort_popcount(local->active_mask) > 1) {
                filled = scheduler_cohort_fill_mix_block(
                    graph, scheduler_state_base, scheduler, run_control, local, cohort_index, block_idx,
                    SchedulerDispatchSlotState::READY
                );
            } else if ((local->active_mask & 1U) != 0) {
                filled = scheduler_cohort_fill_single(
                    graph, scheduler_state_base, scheduler, run_control, local, cohort_index,
                    scheduler->cluster_worker_ids[0], 0, block_idx, SchedulerDispatchSlotState::READY
                );
            } else {
                const uint8_t subtask_slot = (local->active_mask & 2U) != 0 ? 1 : 2;
                filled = scheduler_cohort_fill_single(
                    graph, scheduler_state_base, scheduler, run_control, local, cohort_index,
                    scheduler->cluster_worker_ids[lane + 1], subtask_slot, block_idx, SchedulerDispatchSlotState::READY
                );
            }
            if (filled) {
                local->local_expected_subtasks += local->local_published_subtasks - before;
                scheduler_publish_cache_line(local);
                scheduler_gm_fetch_or(cohort->participant_bitmap, UINT64_C(1) << scheduler->scheduler_index);
            }
            scheduler_cohort_finish_block_claim(source, lane, block_idx, filled);
            return filled;
        }
    }
    return false;
}

inline __aicore__ uint32_t scheduler_cohort_resource_mask(__gm__ SchedulerCohort *cohort) {
    const uint32_t active_mask = static_cast<uint32_t>(cohort->active_mask);
    if (scheduler_cohort_popcount(active_mask) > 1) return active_mask & 7U;
    return (active_mask & 1U) != 0 ? 1U : 6U;
}

inline __aicore__ bool scheduler_cohort_service_regular(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, __gm__ SchedulerCohort *cohort,
    SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats,
    __gm__ SchedulerReadyOwnerState *owner_state, uint64_t profiling_level
) {
    bool progress = false;
    auto state = static_cast<SchedulerCohortState>(scheduler_gm_query(cohort->state));
    if (state == SchedulerCohortState::DISPATCHING) {
        progress = scheduler_cohort_dispatch_one_regular_block(
            graph, scheduler_state_base, scheduler, run_control, cohort_index, cohort
        );
        if (scheduler->scheduler_index == cohort->owner_scheduler &&
            scheduler_cohort_regular_claims_exhausted(
                scheduler_state_base, scheduler, cohort_index, static_cast<uint32_t>(cohort->logical_block_num)
            )) {
            scheduler_gm_store(cohort->dispatch_complete_cycles, scheduler_cycles());
            scheduler_cohort_priority_release(
                scheduler_cohort_coordinator_at(scheduler_state_base, scheduler), scheduler_cohort_resource_mask(cohort)
            );
            scheduler_gm_publish(cohort->state, static_cast<uint64_t>(SchedulerCohortState::EXECUTING));
            state = SchedulerCohortState::EXECUTING;
            progress = true;
        }
    }
    if (state == SchedulerCohortState::EXECUTING) {
        __gm__ SchedulerCohortParticipant *local =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, scheduler->scheduler_index);
        scheduler_observe_cache_line(local);
        const uint64_t bit = UINT64_C(1) << scheduler->scheduler_index;
        if (local->config_generation == cohort->generation && local->local_expected_subtasks != 0 &&
            local->local_completed_subtasks == local->local_expected_subtasks &&
            (scheduler_gm_query(cohort->completion_bitmap) & bit) == 0) {
            scheduler_gm_fetch_or(cohort->completion_bitmap, bit);
            progress = true;
        }
        if (scheduler->scheduler_index == cohort->owner_scheduler) {
            const uint64_t participants = scheduler_gm_query(cohort->participant_bitmap);
            const uint64_t completed = scheduler_gm_query(cohort->completion_bitmap);
            if (participants != 0 && (completed & participants) == participants) {
                __gm__ SchedulerTaskControl *control =
                    scheduler_task_control_at(scheduler_state_base, scheduler, cohort->task_id);
                scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
                if (!scheduler_resolve_completion(
                        graph, scheduler_state_base, scheduler, run_control, cohort->task_id, wake_stats, ready_stats,
                        completion_stats, owner_state, profiling_level, false
                    ))
                    return false;
                scheduler_gm_store(cohort->completion_cycles, scheduler_cycles());
                cohort->task_id = SCHEDULER_TASK_ID_INVALID;
                scheduler_gm_publish(cohort->state, static_cast<uint64_t>(SchedulerCohortState::FREE));
                scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
                progress = true;
            }
        }
    }
    return progress;
}

inline __aicore__ bool scheduler_cohort_service_participant(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, uint64_t cohort_generation,
    SchedulerCohortState state
) {
    __gm__ SchedulerCohortParticipant *participant =
        scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, scheduler->scheduler_index);
    scheduler_observe_cache_line(participant);
    if (participant->config_generation != cohort_generation ||
        scheduler->scheduler_index >= participant->participant_count)
        return false;
    // RETIRING is owner-local bookkeeping. Participants must continue to see
    // the published FREE command before they acknowledge retirement.
    bool progress = state != SchedulerCohortState::RETIRING &&
                    scheduler_cohort_forward_command(
                        scheduler_state_base, scheduler, participant, cohort_index, cohort_generation, state
                    );
    if (state == SchedulerCohortState::DRAINING) {
        if (scheduler_cohort_local_slots_drained(scheduler_state_base, scheduler, participant->active_mask)) {
            progress = scheduler_cohort_publish_local_token(
                           participant, SchedulerCohortTokenPhase::DRAIN, cohort_generation
                       ) ||
                       progress;
        }
        (void)scheduler_cohort_update_subtree_token(
            scheduler_state_base, scheduler, cohort_index, participant->participant_count, cohort_generation,
            SchedulerCohortTokenPhase::DRAIN
        );
    } else if (state == SchedulerCohortState::STAGING) {
        const uint32_t before = participant->local_published_subtasks;
        scheduler_cohort_fill_participant(
            graph, scheduler_state_base, scheduler, run_control, participant, cohort_index,
            SchedulerDispatchSlotState::GATED
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            scheduler_cohort_publish_local_token(participant, SchedulerCohortTokenPhase::STAGE, cohort_generation);
        (void)scheduler_cohort_update_subtree_token(
            scheduler_state_base, scheduler, cohort_index, participant->participant_count, cohort_generation,
            SchedulerCohortTokenPhase::STAGE
        );
    } else if (state == SchedulerCohortState::RELEASING) {
        scheduler_cohort_release_local_slots(scheduler_state_base, scheduler, cohort_index, cohort_generation);
        progress =
            scheduler_cohort_publish_local_token(participant, SchedulerCohortTokenPhase::DISPATCH, cohort_generation) ||
            progress;
        (void)scheduler_cohort_update_subtree_token(
            scheduler_state_base, scheduler, cohort_index, participant->participant_count, cohort_generation,
            SchedulerCohortTokenPhase::DISPATCH
        );
    } else if (state == SchedulerCohortState::DISPATCHING) {
        const uint32_t before = participant->local_published_subtasks;
        scheduler_cohort_fill_participant(
            graph, scheduler_state_base, scheduler, run_control, participant, cohort_index,
            SchedulerDispatchSlotState::READY
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            scheduler_cohort_publish_local_token(participant, SchedulerCohortTokenPhase::DISPATCH, cohort_generation);
        (void)scheduler_cohort_update_subtree_token(
            scheduler_state_base, scheduler, cohort_index, participant->participant_count, cohort_generation,
            SchedulerCohortTokenPhase::DISPATCH
        );
    } else if (state == SchedulerCohortState::RETIRING) {
        scheduler_cohort_publish_retire(scheduler_state_base, scheduler, participant, cohort_index, cohort_generation);
    }
    if (state != SchedulerCohortState::RETIRING &&
        participant->local_completed_subtasks == participant->local_expected_subtasks) {
        scheduler_cohort_publish_local_token(participant, SchedulerCohortTokenPhase::COMPLETION, cohort_generation);
    }
    if (state != SchedulerCohortState::RETIRING) {
        (void)scheduler_cohort_update_subtree_token(
            scheduler_state_base, scheduler, cohort_index, participant->participant_count, cohort_generation,
            SchedulerCohortTokenPhase::COMPLETION
        );
    }
    return progress;
}

inline __aicore__ bool scheduler_cohort_service_owner(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cohort_index, __gm__ SchedulerCohort *cohort,
    SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats,
    __gm__ SchedulerReadyOwnerState *owner_state = nullptr, uint64_t profiling_level = 0
) {
    if (scheduler->scheduler_index != cohort->owner_scheduler || cohort->participant_count == 0 ||
        cohort->policy != static_cast<uint64_t>(SchedulerCohortPolicy::SYNC_START))
        return false;
    const uint64_t generation = cohort->generation;
    auto state = static_cast<SchedulerCohortState>(cohort->state);
    bool progress = false;
    if (state == SchedulerCohortState::DRAINING &&
        scheduler_cohort_root_token_ready(
            scheduler_state_base, scheduler, cohort_index, generation, SchedulerCohortTokenPhase::DRAIN
        )) {
        cohort->state = static_cast<uint64_t>(SchedulerCohortState::STAGING);
        scheduler_publish_cache_line(cohort);
        scheduler_cohort_publish_command(
            scheduler_state_base, scheduler, 0, cohort_index, generation, SchedulerCohortState::STAGING
        );
        progress = true;
    } else if (state == SchedulerCohortState::STAGING &&
               scheduler_cohort_root_token_ready(
                   scheduler_state_base, scheduler, cohort_index, generation, SchedulerCohortTokenPhase::STAGE
               )) {
        cohort->state = static_cast<uint64_t>(SchedulerCohortState::RELEASING);
        scheduler_publish_cache_line(cohort);
        scheduler_cohort_publish_command(
            scheduler_state_base, scheduler, 0, cohort_index, generation, SchedulerCohortState::RELEASING
        );
        progress = true;
    } else if ((state == SchedulerCohortState::RELEASING || state == SchedulerCohortState::DISPATCHING) &&
               scheduler_cohort_root_token_ready(
                   scheduler_state_base, scheduler, cohort_index, generation, SchedulerCohortTokenPhase::DISPATCH
               )) {
        scheduler_gm_store(cohort->dispatch_complete_cycles, scheduler_cycles());
        cohort->state = static_cast<uint64_t>(SchedulerCohortState::EXECUTING);
        scheduler_publish_cache_line(cohort);
        __gm__ SchedulerCohortCoordinator *coordinator =
            scheduler_cohort_coordinator_at(scheduler_state_base, scheduler);
        coordinator->active_dispatch_cohort = UINT64_MAX;
        scheduler_cohort_priority_release(coordinator, scheduler_cohort_resource_mask(cohort));
        scheduler_publish_cache_line(&coordinator->active_dispatch_cohort);
        scheduler_cohort_publish_command(
            scheduler_state_base, scheduler, 0, cohort_index, generation, SchedulerCohortState::EXECUTING
        );
        progress = true;
    }
    state = static_cast<SchedulerCohortState>(cohort->state);
    if (state == SchedulerCohortState::EXECUTING &&
        scheduler_cohort_root_token_ready(
            scheduler_state_base, scheduler, cohort_index, generation, SchedulerCohortTokenPhase::COMPLETION
        )) {
        __gm__ SchedulerTaskControl *control =
            scheduler_task_control_at(scheduler_state_base, scheduler, cohort->task_id);
        scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph, scheduler_state_base, scheduler, run_control, cohort->task_id, wake_stats, ready_stats,
                completion_stats, owner_state, profiling_level, false
            )) {
            return false;
        }
        scheduler_gm_store(cohort->completion_cycles, scheduler_cycles());
        cohort->state = static_cast<uint64_t>(SchedulerCohortState::RETIRING);
        scheduler_publish_cache_line(cohort);
        scheduler_cohort_publish_command(
            scheduler_state_base, scheduler, 0, cohort_index, generation, SchedulerCohortState::FREE
        );
        __gm__ SchedulerCohortParticipant *root =
            scheduler_cohort_participant_at(scheduler_state_base, scheduler, cohort_index, 0);
        scheduler_observe_cache_line(root);
        (void)scheduler_cohort_forward_command(
            scheduler_state_base, scheduler, root, cohort_index, generation, SchedulerCohortState::FREE
        );
        scheduler_cohort_publish_retire(scheduler_state_base, scheduler, root, cohort_index, generation);
        progress = true;
    }
    state = static_cast<SchedulerCohortState>(cohort->state);
    if (state == SchedulerCohortState::RETIRING &&
        scheduler_cohort_root_token_ready(
            scheduler_state_base, scheduler, cohort_index, scheduler_cohort_retire_token(generation),
            SchedulerCohortTokenPhase::COMPLETION
        )) {
        cohort->task_id = SCHEDULER_TASK_ID_INVALID;
        cohort->state = static_cast<uint64_t>(SchedulerCohortState::FREE);
        scheduler_publish_cache_line(cohort);
        // A cohort task is globally resolved only after every Scheduler has observed
        // the FREE command. Otherwise AICPU may stop the AICore schedulers while
        // the tree still contains a retiring generation, making the cohort unsafe
        // to reuse and failing final scheduler validation.
        scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
        progress = true;
    }
    return progress;
}

inline __aicore__ bool scheduler_service_cohort(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, __gm__ SchedulerReadyOwnerState *owner_state = nullptr,
    uint64_t profiling_level = 0
) {
    if (scheduler->is_scheduler == 0) return false;
    __gm__ SchedulerCohortCoordinator *coordinator = scheduler_cohort_coordinator_at(scheduler_state_base, scheduler);
    if (coordinator->cohort_task_count == 0) return false;
    bool progress = false;
    const bool sync_owner = scheduler->scheduler_index == 0;
    __gm__ SchedulerCohortCommand *command =
        scheduler_cohort_command_at(scheduler_state_base, scheduler, scheduler->scheduler_index);
    if (!sync_owner) scheduler_observe_cache_line(command);
    for (uint32_t cohort_index = 0; cohort_index < SCHEDULER_COHORT_COUNT; ++cohort_index) {
        __gm__ SchedulerCohort *cohort = scheduler_cohort_record_at(scheduler_state_base, scheduler, cohort_index);
        scheduler_observe_cache_line(cohort);
        const auto record_state = static_cast<SchedulerCohortState>(cohort->state);
        if (record_state == SchedulerCohortState::CLAIMING) continue;
        if (record_state != SchedulerCohortState::FREE) scheduler_observe_cache_line(&cohort->policy);
        if (record_state != SchedulerCohortState::FREE &&
            cohort->policy == static_cast<uint64_t>(SchedulerCohortPolicy::REGULAR)) {
            progress = scheduler_cohort_service_regular(
                           graph, scheduler_state_base, scheduler, run_control, cohort_index, cohort, wake_stats,
                           ready_stats, completion_stats, owner_state, profiling_level
                       ) ||
                       progress;
            continue;
        }
        uint64_t generation = sync_owner ? cohort->generation : command->generation[cohort_index];
        auto state = static_cast<SchedulerCohortState>(sync_owner ? cohort->state : command->state[cohort_index]);
        if (state == SchedulerCohortState::FREE) {
            if (!sync_owner && generation != 0) {
                __gm__ SchedulerCohortParticipant *participant = scheduler_cohort_participant_at(
                    scheduler_state_base, scheduler, cohort_index, scheduler->scheduler_index
                );
                // This Scheduler necessarily observed its participant config in
                // an earlier active phase before the cohort could complete.
                // Keep that owner-local line hot after FREE instead of forcing
                // a DCCI invalidate on every idle scheduler iteration.
                if (participant->config_generation == generation &&
                    scheduler->scheduler_index < participant->participant_count) {
                    progress = scheduler_cohort_forward_command(
                                   scheduler_state_base, scheduler, participant, cohort_index, generation, state
                               ) ||
                               progress;
                    scheduler_cohort_publish_retire(
                        scheduler_state_base, scheduler, participant, cohort_index, generation
                    );
                }
            }
            continue;
        }
        progress = scheduler_cohort_service_participant(
                       graph, scheduler_state_base, scheduler, run_control, cohort_index, generation, state
                   ) ||
                   progress;
        if (sync_owner) {
            progress = scheduler_cohort_service_owner(
                           graph, scheduler_state_base, scheduler, run_control, cohort_index, cohort, wake_stats,
                           ready_stats, completion_stats, owner_state, profiling_level
                       ) ||
                       progress;
        }
    }
    if (sync_owner)
        progress = scheduler_cohort_admit_sync(graph, scheduler_state_base, scheduler, run_control) || progress;
    progress =
        scheduler_cohort_admit_regular(graph, scheduler_state_base, scheduler, run_control, ready_stats, owner_state) ||
        progress;
    return progress;
}
