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

#include "aicore_ready_scheduler_v1.h"

enum class AicoreGangTokenPhaseV1 : uint32_t {
    DRAIN = 0,
    STAGE = 1,
    DISPATCH = 2,
    COMPLETION = 3,
};

inline __host__ __aicore__ __gm__ AicoreGangCohortV1 *aicore_gang_cohort_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint32_t cohort_index
) {
    return aicore_sidecar_at_v1<AicoreGangCohortV1>(
        sidecar_base, context->gang_cohorts_offset + static_cast<uint64_t>(cohort_index) * sizeof(AicoreGangCohortV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreGangParticipantV1 *aicore_gang_participant_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint32_t cohort_index,
    uint64_t resolver_index
) {
    const uint64_t linear = static_cast<uint64_t>(cohort_index) * AICORE_CLUSTER_CAPACITY_V1 + resolver_index;
    return aicore_sidecar_at_v1<AicoreGangParticipantV1>(
        sidecar_base, context->gang_participants_offset + linear * sizeof(AicoreGangParticipantV1)
    );
}

inline __host__ __aicore__ __gm__ AicoreGangCommandV1 *aicore_gang_command_at_v1(
    __gm__ void *sidecar_base, __gm__ const AicoreWorkerContextV1 *context, uint64_t resolver_index
) {
    return aicore_sidecar_at_v1<AicoreGangCommandV1>(
        sidecar_base, context->gang_commands_offset + resolver_index * sizeof(AicoreGangCommandV1)
    );
}

inline __aicore__ void aicore_gang_publish_command_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint64_t resolver_index, uint32_t cohort_index,
    uint64_t generation, AicoreGangCohortStateV1 state
) {
    __gm__ AicoreGangCommandV1 *command = aicore_gang_command_at_v1(sidecar_base, resolver, resolver_index);
    command->generation[cohort_index] = generation;
    command->state[cohort_index] = static_cast<uint64_t>(state);
    aicore_publish_cache_line_v0(command);
}

inline __aicore__ bool aicore_gang_forward_command_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, __gm__ AicoreGangParticipantV1 *participant,
    uint32_t cohort_index, uint64_t generation, AicoreGangCohortStateV1 state
) {
    if (participant->forwarded_generation == generation && participant->forwarded_state == static_cast<uint32_t>(state))
        return false;
    const uint64_t children[2] = {resolver->resolver_index * 2 + 1, resolver->resolver_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant->participant_count) continue;
        aicore_gang_publish_command_v1(sidecar_base, resolver, children[child_slot], cohort_index, generation, state);
    }
    participant->forwarded_generation = generation;
    participant->forwarded_state = static_cast<uint32_t>(state);
    aicore_publish_cache_line_v0(participant);
    return true;
}

inline __host__ __aicore__ uint32_t aicore_gang_popcount_v1(uint32_t mask) {
    return static_cast<uint32_t>(__builtin_popcount(mask));
}

inline __aicore__ __gm__ volatile uint64_t *
aicore_gang_local_token_v1(__gm__ AicoreGangParticipantV1 *participant, AicoreGangTokenPhaseV1 phase) {
    if (phase == AicoreGangTokenPhaseV1::DRAIN) return &participant->drain_local_token;
    if (phase == AicoreGangTokenPhaseV1::STAGE) return &participant->stage_local_token;
    if (phase == AicoreGangTokenPhaseV1::DISPATCH) return &participant->dispatch_local_token;
    return &participant->completion_local_token;
}

inline __aicore__ __gm__ volatile uint64_t *
aicore_gang_subtree_token_v1(__gm__ AicoreGangParticipantV1 *participant, AicoreGangTokenPhaseV1 phase) {
    if (phase == AicoreGangTokenPhaseV1::DRAIN) return &participant->drain_subtree_token;
    if (phase == AicoreGangTokenPhaseV1::STAGE) return &participant->stage_subtree_token;
    if (phase == AicoreGangTokenPhaseV1::DISPATCH) return &participant->dispatch_subtree_token;
    return &participant->completion_subtree_token;
}

inline __aicore__ bool aicore_gang_publish_local_token_v1(
    __gm__ AicoreGangParticipantV1 *participant, AicoreGangTokenPhaseV1 phase, uint64_t generation
) {
    __gm__ volatile uint64_t *token = aicore_gang_local_token_v1(participant, phase);
    if (*token == generation) return false;
    *token = generation;
    aicore_publish_cache_line_v0(token);
    return true;
}

inline __aicore__ bool aicore_gang_update_subtree_token_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint32_t cohort_index,
    uint64_t participant_count, uint64_t generation, AicoreGangTokenPhaseV1 phase
) {
    const uint64_t resolver_index = resolver->resolver_index;
    if (resolver_index >= participant_count) return false;
    __gm__ AicoreGangParticipantV1 *participant =
        aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, resolver_index);
    __gm__ volatile uint64_t *local_token = aicore_gang_local_token_v1(participant, phase);
    __gm__ volatile uint64_t *subtree_token = aicore_gang_subtree_token_v1(participant, phase);
    if (*subtree_token == generation) return true;
    aicore_observe_cache_line_v0(local_token);
    if (*local_token != generation) return false;
    const uint64_t children[2] = {resolver_index * 2 + 1, resolver_index * 2 + 2};
    for (uint32_t child_slot = 0; child_slot < 2; ++child_slot) {
        if (children[child_slot] >= participant_count) continue;
        __gm__ AicoreGangParticipantV1 *child =
            aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, children[child_slot]);
        __gm__ volatile uint64_t *child_token = aicore_gang_subtree_token_v1(child, phase);
        aicore_observe_cache_line_v0(child_token);
        if (*child_token != generation) return false;
    }
    *subtree_token = generation;
    aicore_publish_cache_line_v0(subtree_token);
    return true;
}

inline __aicore__ bool aicore_gang_root_token_ready_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint32_t cohort_index, uint64_t generation,
    AicoreGangTokenPhaseV1 phase
) {
    __gm__ AicoreGangParticipantV1 *root = aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, 0);
    __gm__ volatile uint64_t *token = aicore_gang_subtree_token_v1(root, phase);
    aicore_observe_cache_line_v0(token);
    return *token == generation;
}

inline __host__ __aicore__ uint64_t aicore_gang_retire_token_v1(uint64_t generation) { return ~generation; }

inline __aicore__ void aicore_gang_publish_retire_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, __gm__ AicoreGangParticipantV1 *participant,
    uint32_t cohort_index, uint64_t generation
) {
    const uint64_t retire_token = aicore_gang_retire_token_v1(generation);
    aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::COMPLETION, retire_token);
    (void)aicore_gang_update_subtree_token_v1(
        sidecar_base, resolver, cohort_index, participant->participant_count, retire_token,
        AicoreGangTokenPhaseV1::COMPLETION
    );
}

inline __aicore__ bool aicore_fill_explicit_dispatch_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t worker_id, uint32_t pending_slot, int64_t task_id,
    uint8_t subtask_slot, uint32_t block_idx, uint32_t block_num, uint32_t cohort_index, uint32_t cohort_generation,
    AicoreDispatchPublicationV1 publication
) {
    if (worker_id >= resolver->runtime_worker_count || pending_slot >= AICORE_PENDING_SLOT_COUNT_V1 || task_id < 0 ||
        static_cast<uint64_t>(task_id) >= graph.task_count || subtask_slot >= 3 || block_num == 0 ||
        block_idx >= block_num) {
        return false;
    }
    __gm__ AicoreTaskMetadataV1 *metadata = aicore_task_metadata_at_v1(sidecar_base, resolver, task_id);
    aicore_observe_cache_line_v0(metadata);
    if ((metadata->active_mask & (1U << subtask_slot)) == 0) return false;
    const uint16_t kernel_id = metadata->kernel_ids[subtask_slot];
    __gm__ uint64_t *callable_addresses =
        aicore_sidecar_at_v1<uint64_t>(sidecar_base, resolver->callable_addresses_offset);
    aicore_observe_cache_line_v0(&callable_addresses[kernel_id]);
    const uint64_t callable_address = callable_addresses[kernel_id];
    if (kernel_id == UINT16_MAX || callable_address == 0) {
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::INVALID_CALLABLE, &graph, resolver, UINT64_C(70)
        );
        return false;
    }

    __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
    aicore_observe_cache_line_v0(target);
    __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, pending_slot);
    aicore_observe_cache_line_v0(slot);
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = task_id;
    slot->ready_inbox_index = resolver->resolver_index;
    slot->claim_start_cycles = 0;
    slot->claim_end_cycles = 0;
    slot->claim_worker_id = resolver->worker_index;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = aicore_task_has_fanin_v1(metadata->flags) ? 1 : 0;
    slot->ready_source = static_cast<uint8_t>(AicoreReadySourceV1::LOCAL);
    slot->pending_slot = static_cast<uint8_t>(pending_slot);
    slot->block_num = static_cast<uint16_t>(block_num);
    slot->generation = generation;
    slot->block_idx = block_idx;
    slot->cohort_generation = cohort_generation;
    slot->cohort_index = static_cast<uint8_t>(cohort_index);
    slot->gang = 1;
    aicore_writeback_cache_line_v0(slot);

    AicoreTaskInfoV0 task{
        task_id,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV,
    };
    __gm__ PTO2DispatchPayload *payload = aicore_sidecar_at_v1<PTO2DispatchPayload>(
        sidecar_base,
        target->dispatch_payload_offset + static_cast<uint64_t>(pending_slot) * sizeof(PTO2DispatchPayload)
    );
    AicoreRootStatusV0 status = aicore_materialize_task_payload_resolved_v0(
        graph, task, callable_address, payload, static_cast<int32_t>(block_idx), static_cast<int32_t>(block_num)
    );
    if (status != AicoreRootStatusV0::OK) {
        aicore_record_scheduler_error_v1(run_control, task_id, status, &graph, resolver, UINT64_C(71));
        return false;
    }
    aicore_publish_dispatch_payload_v1(payload);
    aicore_gm_store_v0(slot->publication, aicore_dispatch_publication_v1(generation, publication));
    return true;
}

inline __aicore__ int32_t aicore_gang_find_free_pending_slot_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint64_t worker_id
) {
    for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
        __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, pending_slot);
        if (aicore_dispatch_state_v1(aicore_gm_query_v0(slot->publication)) == AicoreDispatchPublicationV1::FREE)
            return static_cast<int32_t>(pending_slot);
    }
    return -1;
}

inline __aicore__ bool aicore_gang_fill_single_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, __gm__ AicoreGangParticipantV1 *participant, uint32_t cohort_index,
    uint64_t worker_id, uint8_t subtask_slot, uint32_t block_idx, AicoreDispatchPublicationV1 publication
) {
    const int32_t pending_slot = aicore_gang_find_free_pending_slot_v1(sidecar_base, resolver, worker_id);
    if (pending_slot < 0) return false;
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, static_cast<uint32_t>(pending_slot));
    const uint32_t generation = aicore_dispatch_generation_v1(aicore_gm_query_v0(slot->publication));
    aicore_gm_store_v0(
        slot->publication, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FILLING)
    );
    if (!aicore_fill_explicit_dispatch_slot_v1(
            graph, sidecar_base, resolver, run_control, worker_id, static_cast<uint32_t>(pending_slot),
            participant->task_id, subtask_slot, block_idx, participant->logical_block_num, cohort_index,
            static_cast<uint32_t>(participant->config_generation), publication
        )) {
        return false;
    }
    ++participant->local_published_subtasks;
    return true;
}

inline __aicore__ bool aicore_gang_fill_mix_block_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, __gm__ AicoreGangParticipantV1 *participant, uint32_t cohort_index,
    uint32_t block_idx, AicoreDispatchPublicationV1 publication
) {
    int32_t pending_slot = -1;
    for (uint32_t candidate = 0; candidate < AICORE_PENDING_SLOT_COUNT_V1; ++candidate) {
        bool all_free = true;
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
            const uint64_t worker_id = resolver->cluster_worker_ids[subtask_slot];
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, candidate);
            if (aicore_dispatch_state_v1(aicore_gm_query_v0(slot->publication)) != AicoreDispatchPublicationV1::FREE) {
                all_free = false;
                break;
            }
        }
        if (all_free) {
            pending_slot = static_cast<int32_t>(candidate);
            break;
        }
    }
    if (pending_slot < 0) return false;
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        const uint64_t worker_id = resolver->cluster_worker_ids[subtask_slot];
        __gm__ AicoreDispatchSlotV1 *slot =
            aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, static_cast<uint32_t>(pending_slot));
        const uint32_t generation = aicore_dispatch_generation_v1(aicore_gm_query_v0(slot->publication));
        aicore_gm_store_v0(
            slot->publication, aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FILLING)
        );
    }
    for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
        if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
        if (!aicore_fill_explicit_dispatch_slot_v1(
                graph, sidecar_base, resolver, run_control, resolver->cluster_worker_ids[subtask_slot],
                static_cast<uint32_t>(pending_slot), participant->task_id, subtask_slot, block_idx,
                participant->logical_block_num, cohort_index, static_cast<uint32_t>(participant->config_generation),
                AicoreDispatchPublicationV1::GATED
            )) {
            return false;
        }
    }
    if (publication == AicoreDispatchPublicationV1::READY) {
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((participant->active_mask & (1U << subtask_slot)) == 0) continue;
            __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(
                sidecar_base, resolver, resolver->cluster_worker_ids[subtask_slot], static_cast<uint32_t>(pending_slot)
            );
            aicore_gm_store_v0(
                slot->publication, aicore_dispatch_publication_v1(slot->generation, AicoreDispatchPublicationV1::READY)
            );
        }
    }
    participant->local_published_subtasks += aicore_gang_popcount_v1(participant->active_mask);
    return true;
}

inline __aicore__ bool aicore_gang_local_slots_drained_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint32_t active_mask
) {
    const bool single_aiv = aicore_gang_popcount_v1(active_mask) == 1 && (active_mask & 6U) != 0;
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const bool used =
            cluster_lane == 0 ? (active_mask & 1U) != 0 : (single_aiv || (active_mask & (1U << cluster_lane)) != 0);
        if (!used) continue;
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
            __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(
                sidecar_base, resolver, resolver->cluster_worker_ids[cluster_lane], pending_slot
            );
            if (aicore_dispatch_state_v1(aicore_gm_query_v0(slot->publication)) != AicoreDispatchPublicationV1::FREE)
                return false;
        }
    }
    return true;
}

inline __aicore__ void aicore_gang_fill_participant_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, __gm__ AicoreGangParticipantV1 *participant, uint32_t cohort_index,
    AicoreDispatchPublicationV1 publication
) {
    const uint32_t active_mask = participant->active_mask;
    if (aicore_gang_popcount_v1(active_mask) > 1) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!aicore_gang_fill_mix_block_v1(
                    graph, sidecar_base, resolver, run_control, participant, cohort_index, block_idx, publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else if ((active_mask & 1U) != 0) {
        while (participant->next_block[0] < participant->logical_block_num) {
            const uint32_t block_idx = participant->next_block[0];
            if (!aicore_gang_fill_single_v1(
                    graph, sidecar_base, resolver, run_control, participant, cohort_index,
                    resolver->cluster_worker_ids[0], 0, block_idx, publication
                ))
                break;
            participant->next_block[0] += participant->block_stride;
        }
    } else {
        const uint8_t subtask_slot = (active_mask & 2U) != 0 ? 1 : 2;
        for (uint32_t lane = 0; lane < 2; ++lane) {
            while (participant->next_block[lane] < participant->logical_block_num) {
                const uint32_t block_idx = participant->next_block[lane];
                if (!aicore_gang_fill_single_v1(
                        graph, sidecar_base, resolver, run_control, participant, cohort_index,
                        resolver->cluster_worker_ids[lane + 1], subtask_slot, block_idx, publication
                    ))
                    break;
                participant->next_block[lane] += participant->block_stride;
            }
        }
    }
    aicore_publish_cache_line_v0(participant);
}

inline __aicore__ void aicore_gang_release_local_slots_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint32_t cohort_index, uint64_t generation
) {
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
            __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(
                sidecar_base, resolver, resolver->cluster_worker_ids[cluster_lane], pending_slot
            );
            const uint64_t publication = aicore_gm_query_v0(slot->publication);
            if (aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::GATED) continue;
            aicore_observe_cache_line_v0(slot);
            if (slot->cohort_index != cohort_index || slot->cohort_generation != generation) continue;
            aicore_gm_store_v0(
                slot->publication, aicore_dispatch_publication_v1(slot->generation, AicoreDispatchPublicationV1::READY)
            );
        }
    }
}

inline __aicore__ int64_t aicore_gang_select_ready_task_v1(
    __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver, uint64_t priority_bit
) {
    for (uint64_t task_id = 0; task_id < resolver->graph_task_count; ++task_id) {
        __gm__ AicoreTaskMetadataV1 *metadata =
            aicore_task_metadata_at_v1(sidecar_base, resolver, static_cast<int64_t>(task_id));
        if (!aicore_task_is_gang_v1(metadata->flags) || aicore_task_priority_bit_v1(metadata->flags) != priority_bit)
            continue;
        __gm__ AicoreTaskControlV1 *control =
            aicore_task_control_at_v1(sidecar_base, resolver, static_cast<int64_t>(task_id));
        if (aicore_gm_query_v0(control->state) == static_cast<int64_t>(AicoreTaskStateV1::READY))
            return static_cast<int64_t>(task_id);
    }
    return AICORE_TASK_ID_INVALID_V1;
}

inline __aicore__ uint32_t aicore_gang_assigned_blocks_v1(uint32_t first, uint32_t stride, uint32_t logical_block_num) {
    if (first >= logical_block_num) return 0;
    return 1 + (logical_block_num - 1 - first) / stride;
}

inline __aicore__ bool aicore_gang_admit_one_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control
) {
    if (resolver->resolver_index != 0) return false;
    __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, resolver);
    aicore_observe_cache_line_v0(coordinator);
    aicore_observe_cache_line_v0(&coordinator->active_dispatch_cohort);
    if (coordinator->active_dispatch_cohort != UINT64_MAX) return false;
    uint64_t ready_bits = coordinator->ready_priority_bits;
    if (ready_bits == 0) return false;
    const uint64_t priority_bit = (ready_bits & 1U) != 0 ? 1U : ((ready_bits & 2U) != 0 ? 2U : 4U);
    uint32_t cohort_index = UINT32_MAX;
    for (uint32_t candidate = 0; candidate < AICORE_GANG_COHORT_COUNT_V1; ++candidate) {
        __gm__ AicoreGangCohortV1 *cohort = aicore_gang_cohort_at_v1(sidecar_base, resolver, candidate);
        aicore_observe_cache_line_v0(cohort);
        if (cohort->state == static_cast<uint64_t>(AicoreGangCohortStateV1::FREE)) {
            cohort_index = candidate;
            break;
        }
    }
    if (cohort_index == UINT32_MAX) return false;
    const int64_t task_id = aicore_gang_select_ready_task_v1(sidecar_base, resolver, priority_bit);
    if (task_id < 0) {
        aicore_gm_fetch_and_v0(coordinator->ready_priority_bits, ~priority_bit);
        if (aicore_gang_select_ready_task_v1(sidecar_base, resolver, priority_bit) >= 0)
            aicore_gm_fetch_or_v0(coordinator->ready_priority_bits, priority_bit);
        return false;
    }
    __gm__ AicoreTaskMetadataV1 *metadata = aicore_task_metadata_at_v1(sidecar_base, resolver, task_id);
    aicore_observe_cache_line_v0(metadata);
    const uint32_t active_mask = metadata->active_mask;
    const uint32_t block_num = metadata->logical_block_num;
    const bool mix = aicore_task_is_mix_v1(metadata->flags);
    const bool single_aiv = !mix && (active_mask & 6U) != 0;
    const uint32_t capacity = single_aiv ? static_cast<uint32_t>(resolver->resolver_count * 2) :
                                           static_cast<uint32_t>(resolver->resolver_count);
    if (aicore_task_requires_sync_start_v1(metadata->flags) && block_num > capacity) {
        ++coordinator->capacity_reject_count;
        aicore_publish_cache_line_v0(&coordinator->admitted_count);
        aicore_record_scheduler_error_v1(
            run_control, task_id, AicoreRootStatusV0::UNSUPPORTED_SHAPE, &graph, resolver, UINT64_C(72)
        );
        return false;
    }
    const uint32_t required_participants = single_aiv ? (block_num + 1) / 2 : block_num;
    const uint32_t participant_count = required_participants < resolver->resolver_count ?
                                           required_participants :
                                           static_cast<uint32_t>(resolver->resolver_count);
    uint64_t generation = ++coordinator->next_generation;
    if (generation == 0) generation = ++coordinator->next_generation;
    for (uint32_t participant_index = 0; participant_index < participant_count; ++participant_index) {
        __gm__ AicoreGangParticipantV1 *participant =
            aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, participant_index);
        participant->config_generation = generation;
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
        if (single_aiv) {
            participant->block_stride = static_cast<uint32_t>(resolver->resolver_count * 2);
            participant->next_block[0] = participant_index * 2;
            participant->next_block[1] = participant_index * 2 + 1;
            participant->local_expected_subtasks =
                aicore_gang_assigned_blocks_v1(participant->next_block[0], participant->block_stride, block_num) +
                aicore_gang_assigned_blocks_v1(participant->next_block[1], participant->block_stride, block_num);
        } else {
            participant->block_stride = static_cast<uint32_t>(resolver->resolver_count);
            participant->next_block[0] = participant_index;
            participant->next_block[1] = UINT32_MAX;
            participant->local_expected_subtasks =
                aicore_gang_assigned_blocks_v1(participant_index, participant->block_stride, block_num) *
                aicore_gang_popcount_v1(active_mask);
        }
        participant->participant_count = participant_count;
        aicore_publish_cache_line_v0(participant);
        aicore_publish_cache_line_v0(&participant->drain_local_token);
    }
    __gm__ AicoreGangCohortV1 *cohort = aicore_gang_cohort_at_v1(sidecar_base, resolver, cohort_index);
    cohort->task_id = task_id;
    cohort->generation = generation;
    cohort->priority_bit = priority_bit;
    cohort->active_mask = active_mask;
    cohort->logical_block_num = block_num;
    cohort->participant_count = participant_count;
    cohort->local_stride = single_aiv ? resolver->resolver_count * 2 : resolver->resolver_count;
    cohort->admitted_cycles = aicore_scheduler_cycles_v1();
    cohort->state = static_cast<uint64_t>(
        aicore_task_requires_sync_start_v1(metadata->flags) ? AicoreGangCohortStateV1::DRAINING :
                                                              AicoreGangCohortStateV1::DISPATCHING
    );
    aicore_publish_cache_line_v0(cohort);
    __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, resolver, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DISPATCHING));
    coordinator->active_dispatch_cohort = cohort_index;
    ++coordinator->admitted_count;
    if (priority_bit == 1) ++coordinator->sync_drain_count;
    else if (priority_bit == 2) ++coordinator->mix_dispatch_count;
    else ++coordinator->spmd_dispatch_count;
    aicore_publish_cache_line_v0(&coordinator->active_dispatch_cohort);
    aicore_publish_cache_line_v0(&coordinator->admitted_count);
    aicore_gang_publish_command_v1(
        sidecar_base, resolver, 0, cohort_index, generation,
        aicore_task_requires_sync_start_v1(metadata->flags) ? AicoreGangCohortStateV1::DRAINING :
                                                              AicoreGangCohortStateV1::DISPATCHING
    );
    return true;
}

inline __aicore__ bool aicore_gang_service_participant_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint32_t cohort_index, uint64_t cohort_generation,
    AicoreGangCohortStateV1 state
) {
    __gm__ AicoreGangParticipantV1 *participant =
        aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, resolver->resolver_index);
    aicore_observe_cache_line_v0(participant);
    if (participant->config_generation != cohort_generation ||
        resolver->resolver_index >= participant->participant_count)
        return false;
    bool progress =
        aicore_gang_forward_command_v1(sidecar_base, resolver, participant, cohort_index, cohort_generation, state);
    if (state == AicoreGangCohortStateV1::DRAINING) {
        if (aicore_gang_local_slots_drained_v1(sidecar_base, resolver, participant->active_mask)) {
            progress =
                aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::DRAIN, cohort_generation) ||
                progress;
        }
        (void)aicore_gang_update_subtree_token_v1(
            sidecar_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            AicoreGangTokenPhaseV1::DRAIN
        );
    } else if (state == AicoreGangCohortStateV1::STAGING) {
        const uint32_t before = participant->local_published_subtasks;
        aicore_gang_fill_participant_v1(
            graph, sidecar_base, resolver, run_control, participant, cohort_index, AicoreDispatchPublicationV1::GATED
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::STAGE, cohort_generation);
        (void)aicore_gang_update_subtree_token_v1(
            sidecar_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            AicoreGangTokenPhaseV1::STAGE
        );
    } else if (state == AicoreGangCohortStateV1::RELEASING) {
        aicore_gang_release_local_slots_v1(sidecar_base, resolver, cohort_index, cohort_generation);
        progress =
            aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::DISPATCH, cohort_generation) ||
            progress;
        (void)aicore_gang_update_subtree_token_v1(
            sidecar_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            AicoreGangTokenPhaseV1::DISPATCH
        );
    } else if (state == AicoreGangCohortStateV1::DISPATCHING) {
        const uint32_t before = participant->local_published_subtasks;
        aicore_gang_fill_participant_v1(
            graph, sidecar_base, resolver, run_control, participant, cohort_index, AicoreDispatchPublicationV1::READY
        );
        progress = participant->local_published_subtasks != before;
        if (participant->local_published_subtasks == participant->local_expected_subtasks)
            aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::DISPATCH, cohort_generation);
        (void)aicore_gang_update_subtree_token_v1(
            sidecar_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            AicoreGangTokenPhaseV1::DISPATCH
        );
    } else if (state == AicoreGangCohortStateV1::RETIRING) {
        aicore_gang_publish_retire_v1(sidecar_base, resolver, participant, cohort_index, cohort_generation);
    }
    if (state != AicoreGangCohortStateV1::RETIRING &&
        participant->local_completed_subtasks == participant->local_expected_subtasks &&
        participant->local_expected_subtasks != 0) {
        aicore_gang_publish_local_token_v1(participant, AicoreGangTokenPhaseV1::COMPLETION, cohort_generation);
    }
    if (state != AicoreGangCohortStateV1::RETIRING) {
        (void)aicore_gang_update_subtree_token_v1(
            sidecar_base, resolver, cohort_index, participant->participant_count, cohort_generation,
            AicoreGangTokenPhaseV1::COMPLETION
        );
    }
    return progress;
}

inline __aicore__ bool aicore_gang_service_owner_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint32_t cohort_index, __gm__ AicoreGangCohortV1 *cohort,
    AicoreWakeStatsV1 *wake_stats, AicoreReadyStatsV1 *ready_stats, AicoreCompletionStatsV1 *completion_stats,
    __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (resolver->resolver_index != 0 || cohort->participant_count == 0) return false;
    const uint64_t generation = cohort->generation;
    auto state = static_cast<AicoreGangCohortStateV1>(cohort->state);
    bool progress = false;
    if (state == AicoreGangCohortStateV1::DRAINING &&
        aicore_gang_root_token_ready_v1(
            sidecar_base, resolver, cohort_index, generation, AicoreGangTokenPhaseV1::DRAIN
        )) {
        cohort->drain_complete_cycles = aicore_scheduler_cycles_v1();
        cohort->state = static_cast<uint64_t>(AicoreGangCohortStateV1::STAGING);
        aicore_publish_cache_line_v0(cohort);
        aicore_gang_publish_command_v1(
            sidecar_base, resolver, 0, cohort_index, generation, AicoreGangCohortStateV1::STAGING
        );
        progress = true;
    } else if (
        state == AicoreGangCohortStateV1::STAGING &&
        aicore_gang_root_token_ready_v1(sidecar_base, resolver, cohort_index, generation, AicoreGangTokenPhaseV1::STAGE)
    ) {
        cohort->stage_complete_cycles = aicore_scheduler_cycles_v1();
        cohort->state = static_cast<uint64_t>(AicoreGangCohortStateV1::RELEASING);
        aicore_publish_cache_line_v0(cohort);
        aicore_gang_publish_command_v1(
            sidecar_base, resolver, 0, cohort_index, generation, AicoreGangCohortStateV1::RELEASING
        );
        progress = true;
    } else if (
        (state == AicoreGangCohortStateV1::RELEASING || state == AicoreGangCohortStateV1::DISPATCHING) &&
        aicore_gang_root_token_ready_v1(
            sidecar_base, resolver, cohort_index, generation, AicoreGangTokenPhaseV1::DISPATCH
        )
    ) {
        cohort->dispatch_complete_cycles = aicore_scheduler_cycles_v1();
        cohort->state = static_cast<uint64_t>(AicoreGangCohortStateV1::EXECUTING);
        aicore_publish_cache_line_v0(cohort);
        __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, resolver);
        coordinator->active_dispatch_cohort = UINT64_MAX;
        aicore_publish_cache_line_v0(&coordinator->active_dispatch_cohort);
        aicore_gang_publish_command_v1(
            sidecar_base, resolver, 0, cohort_index, generation, AicoreGangCohortStateV1::EXECUTING
        );
        progress = true;
    }
    state = static_cast<AicoreGangCohortStateV1>(cohort->state);
    if (state == AicoreGangCohortStateV1::EXECUTING &&
        aicore_gang_root_token_ready_v1(
            sidecar_base, resolver, cohort_index, generation, AicoreGangTokenPhaseV1::COMPLETION
        )) {
        __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, resolver, cohort->task_id);
        aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
        if (!aicore_resolve_completion_v1(
                graph, sidecar_base, resolver, run_control, cohort->task_id, wake_stats, ready_stats, completion_stats,
                false, false, nullptr, owner_state
            )) {
            return false;
        }
        if (completion_stats != nullptr) ++completion_stats->resolve_count;
        cohort->completion_cycles = aicore_scheduler_cycles_v1();
        cohort->state = static_cast<uint64_t>(AicoreGangCohortStateV1::RETIRING);
        aicore_publish_cache_line_v0(cohort);
        aicore_gang_publish_command_v1(
            sidecar_base, resolver, 0, cohort_index, generation, AicoreGangCohortStateV1::FREE
        );
        __gm__ AicoreGangParticipantV1 *root = aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, 0);
        aicore_observe_cache_line_v0(root);
        (void)aicore_gang_forward_command_v1(
            sidecar_base, resolver, root, cohort_index, generation, AicoreGangCohortStateV1::FREE
        );
        aicore_gang_publish_retire_v1(sidecar_base, resolver, root, cohort_index, generation);
        progress = true;
    }
    state = static_cast<AicoreGangCohortStateV1>(cohort->state);
    if (state == AicoreGangCohortStateV1::RETIRING &&
        aicore_gang_root_token_ready_v1(
            sidecar_base, resolver, cohort_index, aicore_gang_retire_token_v1(generation),
            AicoreGangTokenPhaseV1::COMPLETION
        )) {
        cohort->task_id = AICORE_TASK_ID_INVALID_V1;
        cohort->state = static_cast<uint64_t>(AicoreGangCohortStateV1::FREE);
        aicore_publish_cache_line_v0(cohort);
        // A gang task is globally resolved only after every Resolver has observed
        // the FREE command. Otherwise AICPU may stop the AICore schedulers while
        // the tree still contains a retiring generation, making the cohort unsafe
        // to reuse and failing final scheduler validation.
        aicore_gm_fetch_add_v0(run_control->resolved_task_count, UINT64_C(1));
        progress = true;
    }
    return progress;
}

inline __aicore__ bool aicore_service_gang_scheduler_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, AicoreWakeStatsV1 *wake_stats, AicoreReadyStatsV1 *ready_stats,
    AicoreCompletionStatsV1 *completion_stats, __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (resolver->is_resolver == 0) return false;
    __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, resolver);
    if (coordinator->gang_task_count == 0) return false;
    bool progress = false;
    const bool owner = resolver->resolver_index == 0;
    uint64_t command_generations[AICORE_GANG_COHORT_COUNT_V1]{};
    uint64_t command_states[AICORE_GANG_COHORT_COUNT_V1]{};
    if (!owner) {
        __gm__ AicoreGangCommandV1 *command =
            aicore_gang_command_at_v1(sidecar_base, resolver, resolver->resolver_index);
        aicore_observe_cache_line_v0(command);
        for (uint32_t cohort_index = 0; cohort_index < AICORE_GANG_COHORT_COUNT_V1; ++cohort_index) {
            command_generations[cohort_index] = command->generation[cohort_index];
            command_states[cohort_index] = command->state[cohort_index];
        }
    }
    for (uint32_t cohort_index = 0; cohort_index < AICORE_GANG_COHORT_COUNT_V1; ++cohort_index) {
        __gm__ AicoreGangCohortV1 *cohort = nullptr;
        uint64_t generation = command_generations[cohort_index];
        auto state = static_cast<AicoreGangCohortStateV1>(command_states[cohort_index]);
        if (owner) {
            cohort = aicore_gang_cohort_at_v1(sidecar_base, resolver, cohort_index);
            aicore_observe_cache_line_v0(cohort);
            generation = cohort->generation;
            state = static_cast<AicoreGangCohortStateV1>(cohort->state);
        }
        if (state == AicoreGangCohortStateV1::FREE) {
            if (!owner && generation != 0) {
                __gm__ AicoreGangParticipantV1 *participant =
                    aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, resolver->resolver_index);
                // This Resolver necessarily observed its participant config in
                // an earlier active phase before the cohort could complete.
                // Keep that owner-local line hot after FREE instead of forcing
                // a DCCI invalidate on every idle scheduler iteration.
                if (participant->config_generation == generation &&
                    resolver->resolver_index < participant->participant_count) {
                    progress = aicore_gang_forward_command_v1(
                                   sidecar_base, resolver, participant, cohort_index, generation, state
                               ) ||
                               progress;
                    aicore_gang_publish_retire_v1(sidecar_base, resolver, participant, cohort_index, generation);
                }
            }
            continue;
        }
        progress = aicore_gang_service_participant_v1(
                       graph, sidecar_base, resolver, run_control, cohort_index, generation, state
                   ) ||
                   progress;
        if (owner) {
            progress = aicore_gang_service_owner_v1(
                           graph, sidecar_base, resolver, run_control, cohort_index, cohort, wake_stats, ready_stats,
                           completion_stats, owner_state
                       ) ||
                       progress;
        }
    }
    if (owner) progress = aicore_gang_admit_one_v1(graph, sidecar_base, resolver, run_control) || progress;
    return progress;
}

struct AicoreDeferredAivDispatchV1 {
    AicoreReadyClaimV1 ready{};
    AicoreFreeSlotClaimV1 reserved_slot{};
};

struct AicoreDeferredAivQueueV1 {
    // Every entry owns one Resolver slot held in FILLING, so a peer miss can
    // always fall back to local execution without another capacity decision.
    AicoreDeferredAivDispatchV1 entries[AICORE_PENDING_SLOT_COUNT_V1]{};
    uint32_t count{0};
};

inline __host__ __aicore__ void aicore_deferred_aiv_pop_front_v1(AicoreDeferredAivQueueV1 *queue) {
    if (queue == nullptr || queue->count == 0) return;
    for (uint32_t index = 1; index < queue->count; ++index)
        queue->entries[index - 1] = queue->entries[index];
    --queue->count;
    queue->entries[queue->count] = {};
}

inline __aicore__ bool aicore_service_cluster_completion_slot_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint32_t cluster_lane, uint32_t pending_slot, uint32_t completed_generation,
    AicoreWakeStatsV1 *wake_stats, AicoreReadyStatsV1 *ready_stats, AicoreCompletionStatsV1 *completion_stats,
    uint64_t *ready_victim_cursors, bool trace_enabled, const AicoreReadyClaimV1 *replacement_ready,
    bool *direct_refilled, AicoreCompletionServiceTimingV1 *timing = nullptr,
    __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (direct_refilled != nullptr) *direct_refilled = false;
    if (cluster_lane >= 3 || pending_slot >= AICORE_PENDING_SLOT_COUNT_V1 || completed_generation == 0) return false;
    const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
    __gm__ AicoreCompletionInboxV1 *completion_line = aicore_completion_inbox_at_v1(sidecar_base, resolver, worker_id);
    __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, pending_slot);
    const uint64_t publication = aicore_gm_query_v0(slot->publication);
    if (aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::READY ||
        aicore_dispatch_generation_v1(publication) != completed_generation) {
        aicore_record_scheduler_error_v1(
            run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(74)
        );
        return false;
    }

    const bool record_timeline = timing != nullptr;
    uint64_t operation_start = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    aicore_observe_cache_line_v0(slot);
    const int64_t task_id = slot->task_id;
    const bool gang = slot->gang != 0;
    const uint8_t completed_subtask_slot = slot->subtask_slot;
    const uint32_t cohort_index = slot->cohort_index;
    const uint32_t cohort_generation = slot->cohort_generation;
    if (gang && replacement_ready != nullptr) return false;
    aicore_gm_store_v0(completion_line->completed_generations[pending_slot], UINT32_C(0));
    uint64_t operation_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (timing != nullptr) timing->consume_cycles += operation_end - operation_start;
    operation_start = operation_end;
    uint64_t ready_publish_cycles = 0;
    uint64_t refill_cycles = 0;
    uint64_t finalize_cycles = 0;
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    if (gang) {
        if (cohort_index >= AICORE_GANG_COHORT_COUNT_V1) return false;
        __gm__ AicoreGangParticipantV1 *participant =
            aicore_gang_participant_at_v1(sidecar_base, resolver, cohort_index, resolver->resolver_index);
        aicore_observe_cache_line_v0(participant);
        if (participant->config_generation != cohort_generation || participant->task_id != task_id ||
            participant->local_completed_subtasks >= participant->local_expected_subtasks) {
            aicore_record_scheduler_error_v1(
                run_control, task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(73)
            );
            return false;
        }
        ++participant->local_completed_subtasks;
        aicore_publish_cache_line_v0(participant);
    } else {
        __gm__ AicoreTaskControlV1 *control = aicore_task_control_at_v1(sidecar_base, resolver, task_id);
        aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
        if (!aicore_resolve_completion_v1(
                graph, sidecar_base, resolver, run_control, task_id, wake_stats, ready_stats, completion_stats, false,
                false, timing == nullptr ? nullptr : &ready_publish_cycles, owner_state
            ))
            return false;
        if (timing != nullptr) timing->ready_publish_cycles += ready_publish_cycles;
        if (completion_stats != nullptr) ++completion_stats->resolve_count;
        uint64_t resolved_count_start = timing == nullptr ? 0 : aicore_scheduler_cycles_v1();
        aicore_gm_fetch_add_v0(run_control->resolved_task_count, UINT64_C(1));
        if (timing != nullptr) {
            finalize_cycles = aicore_scheduler_cycles_v1() - resolved_count_start;
            timing->finalize_cycles += finalize_cycles;
        }
        refill_start_cycles = record_timeline ? aicore_scheduler_cycles_v1() : 0;
        AicoreReadyClaimV1 ready{};
        bool ready_available = replacement_ready != nullptr;
        if (ready_available) {
            ready = *replacement_ready;
        } else if (ready_victim_cursors != nullptr && worker_id != resolver->worker_index) {
            // A normal AIV task is never refilled directly onto the Resolver.
            // Its completed slot becomes capacity for late binding instead.
            __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, resolver);
            bool normal_fill_allowed = coordinator->gang_task_count == 0;
            if (!normal_fill_allowed) {
                aicore_observe_cache_line_v0(coordinator);
                aicore_observe_cache_line_v0(&coordinator->active_dispatch_cohort);
                normal_fill_allowed =
                    coordinator->ready_priority_bits == 0 && coordinator->active_dispatch_cohort == UINT64_MAX;
            }
            if (normal_fill_allowed) {
                const uint32_t core_type = aicore_metadata_core_type_index_v1(completed_subtask_slot);
                if (!aicore_claim_ready_for_slot_v1(
                        graph, sidecar_base, resolver, run_control, resolver->resolver_count, core_type,
                        &ready_victim_cursors[core_type], ready_stats, &ready, trace_enabled, owner_state
                    ))
                    return false;
                ready_available = ready.task_id >= 0;
            }
        }
        if (ready_available) {
            AicoreFreeSlotClaimV1 claim{worker_id, pending_slot, slot->generation};
            if (!aicore_fill_dispatch_slot_v1(graph, sidecar_base, resolver, run_control, claim, ready, trace_enabled))
                return false;
            refilled = true;
        }
        refill_end_cycles = record_timeline ? aicore_scheduler_cycles_v1() : 0;
        if (timing != nullptr) {
            refill_cycles = refill_end_cycles - refill_start_cycles;
            timing->refill_cycles += refill_cycles;
        }
    }
    operation_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (refill_start_cycles == 0) refill_start_cycles = operation_end;
    if (refill_end_cycles == 0) refill_end_cycles = operation_end;
    if (timing != nullptr) {
        uint64_t resolve_total = operation_end - operation_start;
        uint64_t excluded = ready_publish_cycles + refill_cycles + finalize_cycles;
        timing->resolve_cycles += resolve_total > excluded ? resolve_total - excluded : 0;
    }
    operation_start = operation_end;
    if (!refilled) {
        slot->task_id = AICORE_TASK_ID_INVALID_V1;
        aicore_writeback_cache_line_v0(slot);
        aicore_gm_store_v0(
            slot->publication, aicore_dispatch_publication_v1(slot->generation, AicoreDispatchPublicationV1::FREE)
        );
    }
    const uint64_t completion_end = record_timeline ? aicore_scheduler_cycles_v1() : 0;
    if (timing != nullptr) timing->finalize_cycles += completion_end - operation_start;
    if (direct_refilled != nullptr) *direct_refilled = refilled;
    return true;
}

inline __host__ __aicore__ uint32_t aicore_completion_catchup_mask_v1(uint32_t initial_completion_mask) {
    constexpr uint32_t all_pending_slots_mask = (1U << AICORE_PENDING_SLOT_COUNT_V1) - 1;
    return initial_completion_mask == 0 ? 0 : all_pending_slots_mask & ~initial_completion_mask;
}

inline __aicore__ bool aicore_service_cluster_completions_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, AicoreWakeStatsV1 *wake_stats, AicoreReadyStatsV1 *ready_stats,
    AicoreCompletionStatsV1 *completion_stats, uint64_t *ready_victim_cursors = nullptr, bool trace_enabled = false,
    uint64_t *direct_refilled_slot_mask = nullptr, AicoreCompletionServiceTimingV1 *timing = nullptr,
    __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (resolver->is_resolver == 0) return false;
    if (direct_refilled_slot_mask != nullptr) *direct_refilled_slot_mask = 0;
    bool progress = false;
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
        __gm__ AicoreCompletionInboxV1 *completion_line =
            aicore_completion_inbox_at_v1(sidecar_base, resolver, worker_id);
        uint64_t completed_generations = aicore_gm_query_u32_pair_v0(completion_line->completed_generations);
        uint32_t initial_completion_mask = 0;
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
            if (static_cast<uint32_t>(completed_generations >> (pending_slot * 32)) != 0)
                initial_completion_mask |= 1U << pending_slot;
        }
        uint32_t scan_mask = initial_completion_mask;
        // Completion processing can be much slower than the sibling kernel. Refresh the packed completion line once
        // for slots that were incomplete in the initial scan, excluding newly refilled slots from the catch-up pass.
        for (uint32_t scan_pass = 0; scan_pass < 2 && scan_mask != 0; ++scan_pass) {
            for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
                if ((scan_mask & (1U << pending_slot)) == 0) continue;
                const uint32_t completed_generation =
                    static_cast<uint32_t>(completed_generations >> (pending_slot * 32));
                if (completed_generation == 0) continue;
                bool direct_refilled = false;
                if (!aicore_service_cluster_completion_slot_v1(
                        graph, sidecar_base, resolver, run_control, cluster_lane, pending_slot, completed_generation,
                        wake_stats, ready_stats, completion_stats, ready_victim_cursors, trace_enabled, nullptr,
                        &direct_refilled, timing, owner_state
                    ))
                    return false;
                if (direct_refilled && direct_refilled_slot_mask != nullptr)
                    *direct_refilled_slot_mask |= UINT64_C(1)
                                                  << (cluster_lane * AICORE_PENDING_SLOT_COUNT_V1 + pending_slot);
                progress = true;
            }
            scan_mask = scan_pass == 0 ? aicore_completion_catchup_mask_v1(initial_completion_mask) : 0;
            if (scan_mask != 0)
                completed_generations = aicore_gm_query_u32_pair_v0(completion_line->completed_generations);
        }
    }
    return progress;
}

struct AicoreNormalDispatchTimingV1 {
    uint64_t probe_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t claim_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t prepare_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t materialize_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
    uint64_t publish_cycles[AICORE_CORE_TYPE_COUNT_V1]{};
};

inline __aicore__ uint64_t
aicore_normal_dispatch_detail_cycles_v1(const AicoreNormalDispatchTimingV1 &timing, uint32_t core_type) {
    return timing.claim_cycles[core_type] + timing.prepare_cycles[core_type] + timing.materialize_cycles[core_type] +
           timing.publish_cycles[core_type];
}

inline __aicore__ void aicore_finish_normal_dispatch_stage_v1(
    AicoreNormalDispatchTimingV1 *timing, uint32_t core_type, uint64_t stage_start, uint64_t detail_start
) {
    if (timing == nullptr) return;
    uint64_t stage_cycles = aicore_scheduler_cycles_v1() - stage_start;
    uint64_t detail_cycles = aicore_normal_dispatch_detail_cycles_v1(*timing, core_type) - detail_start;
    timing->probe_cycles[core_type] += stage_cycles > detail_cycles ? stage_cycles - detail_cycles : 0;
}

inline __host__ __aicore__ bool aicore_normal_aiv_worker_precedes_v1(
    uint32_t candidate_occupied_slots, bool candidate_is_resolver, uint32_t selected_occupied_slots,
    bool selected_is_resolver
) {
    if (candidate_is_resolver != selected_is_resolver) return !candidate_is_resolver;
    return candidate_occupied_slots < selected_occupied_slots;
}

inline __aicore__ bool aicore_fill_cluster_normal_slots_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, uint64_t *ready_victim_cursors, AicoreReadyStatsV1 *ready_stats,
    bool trace_enabled, uint64_t skip_slot_mask = 0, AicoreNormalDispatchTimingV1 *timing = nullptr,
    AicoreDeferredAivQueueV1 *deferred_aiv = nullptr, __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (resolver->is_resolver == 0) return false;
    const uint32_t aic_core_type = static_cast<uint32_t>(AicoreRootCoreTypeV0::AIC);
    uint64_t stage_start = timing == nullptr ? 0 : aicore_scheduler_cycles_v1();
    uint64_t detail_start = timing == nullptr ? 0 : aicore_normal_dispatch_detail_cycles_v1(*timing, aic_core_type);
    __gm__ AicoreGangCoordinatorV1 *coordinator = aicore_gang_coordinator_at_v1(sidecar_base, resolver);
    if (coordinator->gang_task_count != 0) {
        aicore_observe_cache_line_v0(coordinator);
        aicore_observe_cache_line_v0(&coordinator->active_dispatch_cohort);
        if (coordinator->ready_priority_bits != 0 || coordinator->active_dispatch_cohort != UINT64_MAX) {
            aicore_finish_normal_dispatch_stage_v1(timing, aic_core_type, stage_start, detail_start);
            return false;
        }
    }
    bool progress = false;

    // AIC has no peer lane in its Cluster, so preserve the existing slot order.
    if (aicore_ready_directory_nonempty_v1(sidecar_base, resolver, resolver->resolver_count, aic_core_type)) {
        bool aic_ready_available = true;
        for (uint32_t cluster_lane = 0; cluster_lane < 3 && aic_ready_available; ++cluster_lane) {
            const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
            __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
            if (target->active == 0 || target->core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIC)) continue;
            for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * AICORE_PENDING_SLOT_COUNT_V1 + pending_slot))) !=
                    0)
                    continue;
                __gm__ AicoreDispatchSlotV1 *slot =
                    aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, pending_slot);
                const uint64_t publication = aicore_gm_query_v0(slot->publication);
                if (aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::FREE) continue;
                AicoreReadyClaimV1 ready{};
                if (!aicore_claim_ready_for_slot_v1(
                        graph, sidecar_base, resolver, run_control, resolver->resolver_count, aic_core_type,
                        &ready_victim_cursors[aic_core_type], ready_stats, &ready, trace_enabled, owner_state
                    ))
                    return progress;
                if (timing != nullptr && ready.claim_end_cycles >= ready.claim_start_cycles)
                    timing->claim_cycles[aic_core_type] += ready.claim_end_cycles - ready.claim_start_cycles;
                if (ready.task_id < 0) {
                    aic_ready_available = false;
                    break;
                }
                AicoreFreeSlotClaimV1 claim{
                    worker_id,
                    pending_slot,
                    aicore_dispatch_generation_v1(publication),
                };
                aicore_gm_store_v0(
                    slot->publication,
                    aicore_dispatch_publication_v1(claim.generation, AicoreDispatchPublicationV1::FILLING)
                );
                AicoreDispatchFillTimingV1 fill_timing{};
                if (!aicore_fill_dispatch_slot_v1(
                        graph, sidecar_base, resolver, run_control, claim, ready, trace_enabled,
                        timing == nullptr ? nullptr : &fill_timing
                    ))
                    return false;
                if (timing != nullptr) {
                    timing->prepare_cycles[aic_core_type] += fill_timing.prepare_cycles;
                    timing->materialize_cycles[aic_core_type] += fill_timing.materialize_cycles;
                    timing->publish_cycles[aic_core_type] += fill_timing.publish_cycles;
                }
                progress = true;
            }
        }
    }
    aicore_finish_normal_dispatch_stage_v1(timing, aic_core_type, stage_start, detail_start);

    // A Resolver shares its AIV with Executor work. Exhaust the non-Resolver
    // peer's free slots first, then claim more work only against reserved
    // Resolver capacity. The caller decides the reserved work's owner after
    // the rest of this scheduling round completes.
    struct AivWorkerSlots {
        uint64_t worker_id{UINT64_MAX};
        uint64_t publications[AICORE_PENDING_SLOT_COUNT_V1]{};
        uint32_t free_mask{0};
        uint32_t occupied_slots{0};
        bool is_resolver{false};
    };
    const uint32_t aiv_core_type = static_cast<uint32_t>(AicoreRootCoreTypeV0::AIV);
    stage_start = timing == nullptr ? 0 : aicore_scheduler_cycles_v1();
    detail_start = timing == nullptr ? 0 : aicore_normal_dispatch_detail_cycles_v1(*timing, aiv_core_type);
    if (aicore_ready_directory_nonempty_v1(sidecar_base, resolver, resolver->resolver_count, aiv_core_type)) {
        AivWorkerSlots aiv_workers[2]{};
        uint32_t aiv_worker_count = 0;
        for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
            const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
            __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
            if (target->active == 0 || target->core_type != static_cast<int32_t>(AicoreRootCoreTypeV0::AIV)) continue;
            if (aiv_worker_count >= 2) return false;
            AivWorkerSlots &worker = aiv_workers[aiv_worker_count++];
            worker.worker_id = worker_id;
            worker.is_resolver = worker_id == resolver->worker_index;
            for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * AICORE_PENDING_SLOT_COUNT_V1 + pending_slot))) !=
                    0) {
                    ++worker.occupied_slots;
                    continue;
                }
                __gm__ AicoreDispatchSlotV1 *slot =
                    aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker_id, pending_slot);
                const uint64_t publication = aicore_gm_query_v0(slot->publication);
                worker.publications[pending_slot] = publication;
                if (aicore_dispatch_state_v1(publication) == AicoreDispatchPublicationV1::FREE)
                    worker.free_mask |= 1U << pending_slot;
                else ++worker.occupied_slots;
            }
        }
        for (uint32_t attempt = 0; attempt < aiv_worker_count * AICORE_PENDING_SLOT_COUNT_V1; ++attempt) {
            uint32_t selected = aiv_worker_count;
            for (uint32_t worker_index = 0; worker_index < aiv_worker_count; ++worker_index) {
                const AivWorkerSlots &candidate = aiv_workers[worker_index];
                if (candidate.free_mask == 0) continue;
                if (selected == aiv_worker_count ||
                    aicore_normal_aiv_worker_precedes_v1(
                        candidate.occupied_slots, candidate.is_resolver, aiv_workers[selected].occupied_slots,
                        aiv_workers[selected].is_resolver
                    ))
                    selected = worker_index;
            }
            if (selected == aiv_worker_count) break;
            AivWorkerSlots &worker = aiv_workers[selected];
            const uint32_t pending_slot = static_cast<uint32_t>(__builtin_ctz(worker.free_mask));
            worker.free_mask &= ~(1U << pending_slot);
            if (worker.is_resolver && (deferred_aiv == nullptr || deferred_aiv->count >= AICORE_PENDING_SLOT_COUNT_V1))
                break;
            const uint64_t publication = worker.publications[pending_slot];
            AicoreFreeSlotClaimV1 claim{
                worker.worker_id,
                pending_slot,
                aicore_dispatch_generation_v1(publication),
            };
            __gm__ AicoreDispatchSlotV1 *slot =
                aicore_dispatch_slot_at_v1(sidecar_base, resolver, worker.worker_id, pending_slot);
            aicore_gm_store_v0(
                slot->publication,
                aicore_dispatch_publication_v1(claim.generation, AicoreDispatchPublicationV1::FILLING)
            );
            AicoreReadyClaimV1 ready{};
            if (!aicore_claim_ready_for_slot_v1(
                    graph, sidecar_base, resolver, run_control, resolver->resolver_count, aiv_core_type,
                    &ready_victim_cursors[aiv_core_type], ready_stats, &ready, trace_enabled, owner_state
                )) {
                aicore_gm_store_v0(
                    slot->publication,
                    aicore_dispatch_publication_v1(claim.generation, AicoreDispatchPublicationV1::FREE)
                );
                return progress;
            }
            if (timing != nullptr && ready.claim_end_cycles >= ready.claim_start_cycles)
                timing->claim_cycles[aiv_core_type] += ready.claim_end_cycles - ready.claim_start_cycles;
            if (ready.task_id < 0) {
                aicore_gm_store_v0(
                    slot->publication,
                    aicore_dispatch_publication_v1(claim.generation, AicoreDispatchPublicationV1::FREE)
                );
                break;
            }
            if (worker.is_resolver) {
                deferred_aiv->entries[deferred_aiv->count++] = {ready, claim};
                ++worker.occupied_slots;
                progress = true;
                continue;
            }
            AicoreDispatchFillTimingV1 fill_timing{};
            if (!aicore_fill_dispatch_slot_v1(
                    graph, sidecar_base, resolver, run_control, claim, ready, trace_enabled,
                    timing == nullptr ? nullptr : &fill_timing
                ))
                return false;
            if (timing != nullptr) {
                timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
                timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
                timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
            }
            ++worker.occupied_slots;
            progress = true;
        }
    }
    aicore_finish_normal_dispatch_stage_v1(timing, aiv_core_type, stage_start, detail_start);
    return progress;
}

inline __aicore__ bool aicore_release_deferred_aiv_reservation_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, const AicoreFreeSlotClaimV1 &reservation
) {
    if (reservation.worker_id != resolver->worker_index || reservation.slot_index >= AICORE_PENDING_SLOT_COUNT_V1) {
        aicore_record_scheduler_error_v1(
            run_control, AICORE_TASK_ID_INVALID_V1, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver,
            UINT64_C(75)
        );
        return false;
    }
    __gm__ AicoreDispatchSlotV1 *slot =
        aicore_dispatch_slot_at_v1(sidecar_base, resolver, reservation.worker_id, reservation.slot_index);
    const uint64_t publication = aicore_gm_query_v0(slot->publication);
    if (aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::FILLING ||
        aicore_dispatch_generation_v1(publication) != reservation.generation ||
        slot->task_id != AICORE_TASK_ID_INVALID_V1) {
        aicore_record_scheduler_error_v1(
            run_control, slot->task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(76)
        );
        return false;
    }
    aicore_gm_publish_v0(
        slot->publication, aicore_dispatch_publication_v1(reservation.generation, AicoreDispatchPublicationV1::FREE)
    );
    return true;
}

inline __aicore__ int32_t
aicore_deferred_aiv_peer_lane_v1(__gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver) {
    for (uint32_t cluster_lane = 0; cluster_lane < 3; ++cluster_lane) {
        const uint64_t worker_id = resolver->cluster_worker_ids[cluster_lane];
        if (worker_id == resolver->worker_index) continue;
        __gm__ AicoreWorkerContextV1 *target = aicore_worker_context_at_v1(sidecar_base, resolver, worker_id);
        aicore_observe_cache_line_v0(target);
        if (target->active != 0 && target->core_type == static_cast<int32_t>(AicoreRootCoreTypeV0::AIV))
            return static_cast<int32_t>(cluster_lane);
    }
    return -1;
}

inline __aicore__ bool aicore_drain_deferred_aiv_to_peer_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, AicoreDeferredAivQueueV1 *queue, AicoreWakeStatsV1 *wake_stats,
    AicoreReadyStatsV1 *ready_stats, AicoreCompletionStatsV1 *completion_stats, bool trace_enabled,
    AicoreCompletionServiceTimingV1 *completion_timing = nullptr,
    AicoreNormalDispatchTimingV1 *dispatch_timing = nullptr,
    __gm__ AicoreReadyOwnerStateV1 *owner_state = nullptr
) {
    if (queue == nullptr || queue->count == 0) return true;
    const int32_t peer_lane = aicore_deferred_aiv_peer_lane_v1(sidecar_base, resolver);
    if (peer_lane < 0) return false;
    const uint64_t peer_worker_id = resolver->cluster_worker_ids[static_cast<uint32_t>(peer_lane)];
    __gm__ AicoreCompletionInboxV1 *completion_line =
        aicore_completion_inbox_at_v1(sidecar_base, resolver, peer_worker_id);
    const uint32_t aiv_core_type = static_cast<uint32_t>(AicoreRootCoreTypeV0::AIV);

    for (uint32_t pass = 0; pass < 2 && queue->count != 0; ++pass) {
        const uint64_t completed_generations =
            pass == 1 ? aicore_gm_query_u32_pair_v0(completion_line->completed_generations) : 0;
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1 && queue->count != 0;
             ++pending_slot) {
            __gm__ AicoreDispatchSlotV1 *peer_slot =
                aicore_dispatch_slot_at_v1(sidecar_base, resolver, peer_worker_id, pending_slot);
            const uint64_t publication = aicore_gm_query_v0(peer_slot->publication);
            const AicoreDispatchPublicationV1 state = aicore_dispatch_state_v1(publication);
            const uint32_t generation = aicore_dispatch_generation_v1(publication);
            bool refilled = false;
            if (pass == 0) {
                if (state != AicoreDispatchPublicationV1::FREE) continue;
                aicore_gm_store_v0(
                    peer_slot->publication,
                    aicore_dispatch_publication_v1(generation, AicoreDispatchPublicationV1::FILLING)
                );
                AicoreDispatchFillTimingV1 fill_timing{};
                if (!aicore_fill_dispatch_slot_v1(
                        graph, sidecar_base, resolver, run_control,
                        AicoreFreeSlotClaimV1{peer_worker_id, pending_slot, generation}, queue->entries[0].ready,
                        trace_enabled, dispatch_timing == nullptr ? nullptr : &fill_timing
                    ))
                    return false;
                if (dispatch_timing != nullptr) {
                    dispatch_timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
                    dispatch_timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
                    dispatch_timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
                }
                refilled = true;
            } else {
                if (state != AicoreDispatchPublicationV1::READY) continue;
                const uint32_t completed_generation =
                    static_cast<uint32_t>(completed_generations >> (pending_slot * 32));
                if (completed_generation != generation) continue;
                aicore_observe_cache_line_v0(peer_slot);
                if (peer_slot->gang != 0) continue;
                if (!aicore_service_cluster_completion_slot_v1(
                        graph, sidecar_base, resolver, run_control, static_cast<uint32_t>(peer_lane), pending_slot,
                        completed_generation, wake_stats, ready_stats, completion_stats, nullptr, trace_enabled,
                        &queue->entries[0].ready, &refilled, completion_timing, owner_state
                    ) ||
                    !refilled)
                    return false;
            }
            if (!aicore_release_deferred_aiv_reservation_v1(
                    graph, sidecar_base, resolver, run_control, queue->entries[0].reserved_slot
                ))
                return false;
            aicore_deferred_aiv_pop_front_v1(queue);
        }
    }
    return true;
}

inline __aicore__ bool aicore_publish_deferred_aiv_to_resolver_v1(
    const AicoreReadonlyGraphV0 &graph, __gm__ void *sidecar_base, __gm__ AicoreWorkerContextV1 *resolver,
    __gm__ AicoreRunControlV1 *run_control, AicoreDeferredAivQueueV1 *queue, bool trace_enabled,
    uint32_t *published_slot, AicoreNormalDispatchTimingV1 *timing = nullptr
) {
    if (published_slot != nullptr) *published_slot = UINT32_MAX;
    if (queue == nullptr || queue->count == 0) return true;
    const AicoreDeferredAivDispatchV1 &entry = queue->entries[0];
    __gm__ AicoreDispatchSlotV1 *slot = aicore_dispatch_slot_at_v1(
        sidecar_base, resolver, entry.reserved_slot.worker_id, entry.reserved_slot.slot_index
    );
    const uint64_t publication = aicore_gm_query_v0(slot->publication);
    if (entry.reserved_slot.worker_id != resolver->worker_index ||
        entry.reserved_slot.slot_index >= AICORE_PENDING_SLOT_COUNT_V1 ||
        aicore_dispatch_state_v1(publication) != AicoreDispatchPublicationV1::FILLING ||
        aicore_dispatch_generation_v1(publication) != entry.reserved_slot.generation) {
        aicore_record_scheduler_error_v1(
            run_control, entry.ready.task_id, AicoreRootStatusV0::INVALID_ARGUMENTS, &graph, resolver, UINT64_C(77)
        );
        return false;
    }
    const uint32_t aiv_core_type = static_cast<uint32_t>(AicoreRootCoreTypeV0::AIV);
    AicoreDispatchFillTimingV1 fill_timing{};
    if (!aicore_fill_dispatch_slot_v1(
            graph, sidecar_base, resolver, run_control, entry.reserved_slot, entry.ready, trace_enabled,
            timing == nullptr ? nullptr : &fill_timing
        ))
        return false;
    if (timing != nullptr) {
        timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
        timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
        timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
    }
    if (published_slot != nullptr) *published_slot = entry.reserved_slot.slot_index;
    aicore_deferred_aiv_pop_front_v1(queue);
    return true;
}
