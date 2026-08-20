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

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "aicore_gang_scheduler_v1.h"
#include "aicore_cluster_topology_v1.h"
#include "pto_runtime2_types.h"

namespace {

class SidecarBuffer {
public:
    explicit SidecarBuffer(const AicoreExecutionSidecarLayoutV1 &layout) :
        base_(std::aligned_alloc(AICORE_SIDECAR_ALIGNMENT_V1, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(aicore_sidecar_init_v1(base_, layout));
    }
    ~SidecarBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        task_count_(task_count) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        descriptors_ = std::make_unique<PTO2TaskDescriptor[]>(capacity_);
        payloads_ = std::make_unique<PTO2TaskPayload[]>(capacity_);
        for (size_t task = 0; task < capacity_; ++task) {
            descriptors_[task].task_id = PTO2TaskId::make(0, static_cast<int64_t>(task));
            for (int slot = 0; slot < 3; ++slot)
                descriptors_[task].kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t subtask_slot, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task, task_count_);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(AICORE_GRAPH_MAX_FANIN_V0));
        descriptors_[task].kernel_id[subtask_slot] = 1;
        payloads_[task].fanin_count = static_cast<int32_t>(fanins.size());
        std::copy(fanins.begin(), fanins.end(), payloads_[task].fanin_local_ids);
    }

    void mixed(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) != 0) descriptors_[task].kernel_id[subtask_slot] = 1;
        }
        payloads_[task].fanin_count = 0;
    }

    AicoreReadonlyGraphV0 graph() const {
        return {
            reinterpret_cast<uint64_t>(descriptors_.get()),
            reinterpret_cast<uint64_t>(payloads_.get()),
            task_count_,
            capacity_ - 1,
        };
    }

private:
    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<PTO2TaskDescriptor[]> descriptors_;
    std::unique_ptr<PTO2TaskPayload[]> payloads_;
};

struct FixtureStorage {
    explicit FixtureStorage(uint64_t task_count, uint64_t workers = 2) {
        EXPECT_TRUE(aicore_sidecar_plan_v1(task_count, task_count, 0, &layout));
        sidecar = std::make_unique<SidecarBuffer>(layout);
        run_control = aicore_sidecar_at_v1<AicoreRunControlV1>(sidecar->base(), layout.run_control_offset);
        contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(sidecar->base(), layout.worker_contexts_offset);
        run_control->claim_bindings_offset = layout.claim_bindings_offset;
        run_control->aiv_active_worker_count = workers;
        run_control->resolver_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            AicoreWorkerContextV1 &context = contexts[worker];
            context.core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
            context.active = 1;
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;
            context.completion_inboxes_offset = layout.completion_inboxes_offset;
            context.ready_inboxes_offset = layout.ready_inboxes_offset;
            context.ready_directory_offset = layout.ready_directory_offset;
            context.free_slot_directory_offset = layout.free_slot_directory_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.dispatch_slots_offset = layout.dispatch_slots_offset;
            context.callable_addresses_offset = layout.callable_addresses_offset;
            context.gang_coordinator_offset = layout.gang_coordinator_offset;
            context.gang_cohorts_offset = layout.gang_cohorts_offset;
            context.gang_participants_offset = layout.gang_participants_offset;
            context.gang_commands_offset = layout.gang_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * AICORE_PENDING_SLOT_COUNT_V1 * sizeof(PTO2DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.inbox_index = worker;
        }
        metadata = aicore_sidecar_at_v1<AicoreTaskMetadataV1>(sidecar->base(), layout.task_metadata_offset);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = 1;
            metadata[task].logical_block_num = 1;
            metadata[task].total_required_subtasks = 1;
            metadata[task].flags = AICORE_TASK_EXECUTABLE_V1;
        }
    }

    AicoreExecutionSidecarLayoutV1 layout{};
    std::unique_ptr<SidecarBuffer> sidecar;
    AicoreRunControlV1 *run_control{nullptr};
    AicoreWorkerContextV1 *contexts{nullptr};
    AicoreTaskMetadataV1 *metadata{nullptr};
};

TEST(AicoreReadySidecarV1, PlansAndInitializesReadyState) {
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(5, 3, 2, &layout));
    EXPECT_EQ(layout.total_size % AICORE_SIDECAR_ALIGNMENT_V1, 0u);
    EXPECT_EQ(layout.task_metadata_offset % alignof(AicoreTaskMetadataV1), 0u);
    EXPECT_EQ(layout.ready_inboxes_offset % alignof(AicoreReadyInboxV1), 0u);
    EXPECT_EQ(layout.ready_directory_offset % alignof(AicoreReadyDirectoryV1), 0u);
    EXPECT_EQ(layout.free_slot_directory_offset % alignof(AicoreFreeSlotDirectoryV1), 0u);

    SidecarBuffer storage(layout);
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.base(), layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(AicoreTaskStateV1::BLOCKED));
        EXPECT_EQ(controls[task].wake_list_head, AICORE_WAKE_LIST_OPEN_V1);
        EXPECT_EQ(controls[task].inbox_next, AICORE_INBOX_LINK_UNPUBLISHED_V1);
    }
    auto *ready = aicore_sidecar_at_v1<AicoreReadyInboxV1>(storage.base(), layout.ready_inboxes_offset);
    for (uint64_t inbox = 0; inbox < AICORE_CORE_TYPE_COUNT_V1 * AICORE_WORKER_CAPACITY_V1; ++inbox)
        EXPECT_EQ(ready[inbox].head, AICORE_INBOX_EMPTY_V1);
}

TEST(AicoreClusterTopologyV1, UsesMixedKernelLaunchCoordinate) {
    constexpr int32_t kClusters = 4;
    AicoreClusterCoordinateV1 coordinate{};
    ASSERT_TRUE(aicore_cluster_coordinate_from_worker_v1(2, true, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 0);
    ASSERT_TRUE(aicore_cluster_coordinate_from_worker_v1(9, false, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 2);
    EXPECT_FALSE(aicore_cluster_coordinate_from_worker_v1(12, false, kClusters, 2, &coordinate));
}

TEST(AicoreGangPriorityV1, SyncStartPrecedesMixAndSpmd) {
    EXPECT_EQ(aicore_task_priority_bit_v1(AICORE_TASK_SPMD_V1), 4u);
    EXPECT_EQ(aicore_task_priority_bit_v1(AICORE_TASK_MIX_V1), 2u);
    EXPECT_EQ(aicore_task_priority_bit_v1(AICORE_TASK_MIX_V1 | AICORE_TASK_SYNC_START_V1), 1u);

    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.mixed(1, 3);
    graph.executable(2, 1);
    auto *coordinator =
        aicore_sidecar_at_v1<AicoreGangCoordinatorV1>(storage.sidecar->base(), storage.layout.gang_coordinator_offset);
    coordinator->gang_task_count = 3;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 7;
    storage.run_control->resolver_count = 1;
    AicoreWorkerContextV1 &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    auto *controls =
        aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.sidecar->base(), storage.layout.task_controls_offset);
    for (uint32_t task = 0; task < 3; ++task)
        controls[task].state = static_cast<int64_t>(AicoreTaskStateV1::READY);
    storage.metadata[0].active_mask = 1;
    storage.metadata[0].logical_block_num = 4;
    storage.metadata[0].total_required_subtasks = 4;
    storage.metadata[0].flags = AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_SPMD_V1;
    storage.metadata[1].active_mask = 3;
    storage.metadata[1].logical_block_num = 1;
    storage.metadata[1].total_required_subtasks = 2;
    storage.metadata[1].flags = AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_MIX_V1;
    storage.metadata[2].active_mask = 2;
    storage.metadata[2].logical_block_num = 2;
    storage.metadata[2].total_required_subtasks = 2;
    storage.metadata[2].flags =
        AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_SPMD_V1 | AICORE_TASK_SYNC_START_V1;

    ASSERT_TRUE(aicore_gang_admit_one_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control
    ));
    auto *cohort =
        aicore_sidecar_at_v1<AicoreGangCohortV1>(storage.sidecar->base(), storage.layout.gang_cohorts_offset);
    EXPECT_EQ(cohort->task_id, 2);
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(AicoreGangCohortStateV1::DRAINING));
    EXPECT_EQ(controls[2].state, static_cast<int64_t>(AicoreTaskStateV1::DISPATCHING));
    EXPECT_FALSE(aicore_gang_admit_one_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control
    ));
}

TEST(AicoreSyncStartV1, RejectsImpossibleCohortBeforeDrain) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    AicoreWorkerContextV1 &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.run_control->resolver_count = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 3;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags =
        AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_SPMD_V1 | AICORE_TASK_SYNC_START_V1;
    auto *control = aicore_task_control_at_v1(storage.sidecar->base(), &resolver, 0);
    control->state = static_cast<int64_t>(AicoreTaskStateV1::READY);
    auto *coordinator = aicore_gang_coordinator_at_v1(storage.sidecar->base(), &resolver);
    coordinator->gang_task_count = 1;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 1;

    EXPECT_FALSE(aicore_gang_admit_one_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control
    ));
    EXPECT_EQ(coordinator->capacity_reject_count, 1u);
    EXPECT_EQ(
        storage.run_control->scheduler_error, static_cast<uint64_t>(AicoreRootStatusV0::UNSUPPORTED_SHAPE)
    );
    EXPECT_EQ(storage.run_control->error_reserved[0], 72u);
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
}

TEST(AicoreGangTokensV1, StaleGenerationCannotCompleteTree) {
    FixtureStorage storage(1, 3);
    AicoreWorkerContextV1 &resolver = storage.contexts[0];
    resolver.resolver_index = 0;
    resolver.resolver_count = 3;
    constexpr uint64_t kGeneration = 9;
    for (uint64_t participant_index = 0; participant_index < 3; ++participant_index) {
        auto *participant = aicore_gang_participant_at_v1(
            storage.sidecar->base(), &resolver, 0, participant_index
        );
        participant->dispatch_local_token = kGeneration;
    }
    auto *stale_child = aicore_gang_participant_at_v1(storage.sidecar->base(), &resolver, 0, 2);
    stale_child->dispatch_local_token = kGeneration - 1;
    resolver.resolver_index = 1;
    EXPECT_TRUE(aicore_gang_update_subtree_token_v1(
        storage.sidecar->base(), &resolver, 0, 3, kGeneration, AicoreGangTokenPhaseV1::DISPATCH
    ));
    resolver.resolver_index = 2;
    EXPECT_FALSE(aicore_gang_update_subtree_token_v1(
        storage.sidecar->base(), &resolver, 0, 3, kGeneration, AicoreGangTokenPhaseV1::DISPATCH
    ));
    resolver.resolver_index = 0;
    EXPECT_FALSE(aicore_gang_update_subtree_token_v1(
        storage.sidecar->base(), &resolver, 0, 3, kGeneration, AicoreGangTokenPhaseV1::DISPATCH
    ));
    stale_child->dispatch_local_token = kGeneration;
    resolver.resolver_index = 2;
    EXPECT_TRUE(aicore_gang_update_subtree_token_v1(
        storage.sidecar->base(), &resolver, 0, 3, kGeneration, AicoreGangTokenPhaseV1::DISPATCH
    ));
    resolver.resolver_index = 0;
    EXPECT_TRUE(aicore_gang_update_subtree_token_v1(
        storage.sidecar->base(), &resolver, 0, 3, kGeneration, AicoreGangTokenPhaseV1::DISPATCH
    ));
}

TEST(AicoreGangMixV1, ReservesWholeClusterOrNothing) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.mixed(0, 7);
    AicoreWorkerContextV1 &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIC);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot)
            aicore_initialize_free_slot_v1(
                aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, worker, pending_slot)
            );
    }
    storage.metadata[0].active_mask = 7;
    storage.metadata[0].kernel_ids[0] = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].kernel_ids[2] = 1;
    storage.metadata[0].logical_block_num = 1;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_MIX_V1;
    auto *callables =
        aicore_sidecar_at_v1<uint64_t>(storage.sidecar->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *participant = aicore_gang_participant_at_v1(storage.sidecar->base(), &resolver, 0, 0);
    participant->config_generation = 1;
    participant->task_id = 0;
    participant->active_mask = 7;
    participant->logical_block_num = 1;
    participant->local_expected_subtasks = 3;
    auto *blocked0 = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, 1, 0);
    auto *blocked1 = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, 2, 1);
    aicore_gm_store_v0(
        blocked0->publication,
        aicore_dispatch_publication_v1(blocked0->generation, AicoreDispatchPublicationV1::READY)
    );
    aicore_gm_store_v0(
        blocked1->publication,
        aicore_dispatch_publication_v1(blocked1->generation, AicoreDispatchPublicationV1::READY)
    );
    EXPECT_FALSE(aicore_gang_fill_mix_block_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, participant, 0, 0,
        AicoreDispatchPublicationV1::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 0u);
    aicore_gm_store_v0(
        blocked0->publication,
        aicore_dispatch_publication_v1(blocked0->generation, AicoreDispatchPublicationV1::FREE)
    );
    aicore_gm_store_v0(
        blocked1->publication,
        aicore_dispatch_publication_v1(blocked1->generation, AicoreDispatchPublicationV1::FREE)
    );
    ASSERT_TRUE(aicore_gang_fill_mix_block_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, participant, 0, 0,
        AicoreDispatchPublicationV1::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 3u);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        auto *slot = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, worker, 0);
        EXPECT_EQ(aicore_dispatch_state_v1(slot->publication), AicoreDispatchPublicationV1::GATED);
        EXPECT_EQ(slot->block_idx, 0u);
        EXPECT_EQ(slot->block_num, 1u);
    }
}

TEST(AicoreBootstrapV1, RegistersOnlyOnFirstExecutableProducer) {
    FixtureStorage storage(4, 2);
    GraphBuffer graph(4);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    graph.executable(3, 1, {2, 1});
    storage.metadata[1].active_mask = 2;
    storage.metadata[1].flags |= AICORE_TASK_HAS_FANIN_V1;
    storage.metadata[2].flags = 0;
    storage.metadata[3].active_mask = 2;
    storage.metadata[3].flags |= AICORE_TASK_HAS_FANIN_V1;
    auto *controls =
        aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.sidecar->base(), storage.layout.task_controls_offset);
    controls[2].state = static_cast<int64_t>(AicoreTaskStateV1::DONE);
    controls[2].wake_list_head = AICORE_WAKE_LIST_CLOSED_V1;

    AicoreWakeStatsV1 stats{};
    EXPECT_EQ(
        aicore_bootstrap_route_task_v1(
            graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 1, &stats
        ),
        AicoreRouteResultV1::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 1);
    EXPECT_EQ(controls[1].next_waiter, AICORE_WAKE_LIST_OPEN_V1);
    EXPECT_EQ(controls[1].waiting_producer, 0);

    EXPECT_EQ(
        aicore_bootstrap_route_task_v1(
            graph.graph(), storage.sidecar->base(), &storage.contexts[1], storage.run_control, 3, &stats
        ),
        AicoreRouteResultV1::WAITING
    );
    EXPECT_EQ(controls[1].wake_list_head, 3);
    EXPECT_EQ(controls[3].next_fanin_index, 1);
    EXPECT_EQ(controls[3].waiting_producer, 1);
    EXPECT_EQ(stats.wake_register_count, 2u);
    EXPECT_EQ(stats.fanin_state_load_count, 0u);
    EXPECT_EQ(stats.wake_cas_retry_count, 0u);
}

TEST(AicoreBootstrapV1, PublishesExclusiveInboxAndAggregatesDirectory) {
    FixtureStorage storage(2, 2);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].inbox_index = 1;
    storage.contexts[1].inbox_index = 0;
    AicoreReadyBatchV1 batch{};
    AicoreReadyStatsV1 stats{};
    ASSERT_TRUE(
        aicore_bootstrap_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[1], 0, &batch, &stats)
    );
    ASSERT_TRUE(
        aicore_bootstrap_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[1], 1, &batch, &stats)
    );
    aicore_cache_barrier_v0();
    uint64_t ready_types = 0;
    ASSERT_TRUE(aicore_bootstrap_ready_batch_publish_v1(
        storage.sidecar->base(), &storage.contexts[1], 0, 0, &batch, &stats, &ready_types
    ));
    auto *directory = aicore_ready_directory_at_v1(storage.sidecar->base(), &storage.contexts[1]);
    directory->bootstrap_ready_types[0] = ready_types;
    aicore_bootstrap_ready_directory_publish_v1(storage.sidecar->base(), &storage.contexts[1], 2);

    auto *controls =
        aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.sidecar->base(), storage.layout.task_controls_offset);
    EXPECT_EQ(controls[1].next_waiter, 0);
    EXPECT_EQ(controls[0].next_waiter, AICORE_INBOX_EMPTY_V1);
    EXPECT_EQ(aicore_ready_inbox_at_v1(storage.sidecar->base(), &storage.contexts[1], 0, 0)->head, 1);
    EXPECT_EQ(directory->words[0][0], 1u);
    EXPECT_EQ(directory->words[1][0], 0u);
    EXPECT_EQ(stats.enqueue_count, 2u);
    EXPECT_EQ(stats.batch_count, 1u);
}

TEST(AicoreReadyInboxV1, BatchPushAndPerTaskPopMaintainDirectory) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    AicoreReadyBatchV1 batch{};
    AicoreReadyStatsV1 stats{};
    for (uint64_t task = 0; task < kTasks; ++task)
        ASSERT_TRUE(aicore_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[0], task, &batch, &stats));
    ASSERT_TRUE(aicore_ready_batch_push_v1(storage.sidecar->base(), &storage.contexts[0], 0, 0, &batch, &stats));

    auto *directory =
        aicore_sidecar_at_v1<AicoreReadyDirectoryV1>(storage.sidecar->base(), storage.layout.ready_directory_offset);
    EXPECT_NE(directory->words[0][0] & 1, 0u);
    std::vector<bool> seen(kTasks, false);
    for (uint64_t index = 0; index < kTasks; ++index) {
        int64_t task = AICORE_TASK_ID_INVALID_V1;
        ASSERT_TRUE(aicore_ready_pop_from_inbox_v1(
            graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
        ));
        ASSERT_GE(task, 0);
        EXPECT_FALSE(seen[static_cast<size_t>(task)]);
        seen[static_cast<size_t>(task)] = true;
    }
    int64_t task = AICORE_TASK_ID_INVALID_V1;
    ASSERT_TRUE(aicore_ready_pop_from_inbox_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, AICORE_TASK_ID_INVALID_V1);
    EXPECT_EQ(directory->words[0][0] & 1, 0u);
    EXPECT_EQ(stats.pop_count, kTasks);
}

TEST(AicoreReadyInboxV1, StealsOnlyFromMarkedVictim) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    AicoreReadyBatchV1 batch{};
    AicoreReadyStatsV1 stats{};
    ASSERT_TRUE(aicore_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[1], 0, &batch, &stats));
    ASSERT_TRUE(aicore_ready_batch_push_v1(storage.sidecar->base(), &storage.contexts[1], 0, 1, &batch, &stats));

    uint64_t cursor = 1;
    AicoreReadyClaimV1 claim{};
    ASSERT_TRUE(aicore_claim_ready_for_slot_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 2, 0, &cursor, &stats, &claim
    ));
    EXPECT_EQ(claim.task_id, 0);
    EXPECT_EQ(claim.inbox_index, 1u);
    EXPECT_EQ(claim.source, AicoreReadySourceV1::STOLEN);
    EXPECT_EQ(stats.steal_count, 1u);
}

TEST(AicoreReadyInboxV1, ConcurrentConsumersNeverDuplicateTask) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kConsumerCount = 8;
    FixtureStorage storage(kTasks, kConsumerCount);
    GraphBuffer graph(kTasks);
    AicoreReadyBatchV1 batch{};
    for (uint64_t task = 0; task < kTasks; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(aicore_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[0], task, &batch, nullptr));
    }
    ASSERT_TRUE(aicore_ready_batch_push_v1(storage.sidecar->base(), &storage.contexts[0], 0, 0, &batch, nullptr));
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    auto consume = [&](uint64_t worker) {
        while (claimed.load(std::memory_order_relaxed) < kTasks) {
            int64_t task = AICORE_TASK_ID_INVALID_V1;
            if (!aicore_ready_pop_from_inbox_v1(
                    graph.graph(), storage.sidecar->base(), &storage.contexts[worker], storage.run_control, 0, 0, &task,
                    nullptr
                ))
                return;
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);
    for (uint64_t worker = 0; worker < kConsumerCount; ++worker)
        consumers.emplace_back(consume, worker);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(AicoreReadyWakeV1, WakeResolvePublishesConsumerToResolverLocalInbox) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= AICORE_TASK_HAS_FANIN_V1;
    AicoreWakeStatsV1 wake{};
    AicoreReadyStatsV1 ready{};
    AicoreCompletionStatsV1 completion{};
    EXPECT_EQ(
        aicore_route_task_v1(
            graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 1, &wake
        ),
        AicoreRouteResultV1::WAITING
    );
    auto *controls =
        aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.sidecar->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(AicoreTaskStateV1::DONE);
    ASSERT_TRUE(aicore_resolve_completion_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 0, &wake, &ready, &completion
    ));
    int64_t task = AICORE_TASK_ID_INVALID_V1;
    ASSERT_TRUE(aicore_ready_pop_from_inbox_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
    EXPECT_EQ(wake.wake_register_count, 1u);
    EXPECT_EQ(wake.wake_migrate_count, 1u);
}

TEST(AicoreReadyInboxV1, ReadyAndCompletionUseIndependentLinks) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    AicoreReadyBatchV1 batch{};
    ASSERT_TRUE(aicore_ready_batch_append_v1(storage.sidecar->base(), &storage.contexts[0], 0, &batch, nullptr));
    ASSERT_TRUE(aicore_ready_batch_push_v1(storage.sidecar->base(), &storage.contexts[0], 0, 0, &batch, nullptr));
    int64_t task = AICORE_TASK_ID_INVALID_V1;
    ASSERT_TRUE(aicore_ready_pop_from_inbox_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 0, 0, &task, nullptr
    ));
    auto *control = aicore_task_control_at_v1(storage.sidecar->base(), &storage.contexts[0], 0);
    ASSERT_EQ(control->next_waiter, AICORE_INBOX_EMPTY_V1);
    ASSERT_EQ(control->inbox_next, AICORE_INBOX_LINK_UNPUBLISHED_V1);
    ASSERT_TRUE(aicore_enqueue_completion_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 1, 0, 0, nullptr
    ));
    auto *completion = aicore_completion_inbox_at_v1(storage.sidecar->base(), &storage.contexts[0], 0);
    EXPECT_EQ(completion->head, 0);
    EXPECT_EQ(control->inbox_next, AICORE_INBOX_EMPTY_V1);
}

TEST(AicoreCompletionInboxV1, ConcurrentPublishAndDetachNeverExposeUnpublishedLink) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kProducerCount = 8;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    auto *controls =
        aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.sidecar->base(), storage.layout.task_controls_offset);
    auto *inbox = aicore_completion_inbox_at_v1(storage.sidecar->base(), &storage.contexts[0], 0);
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<bool> start{false};
    std::atomic<uint64_t> producers_done{0};
    std::atomic<uint64_t> consumed{0};
    std::atomic<bool> invalid_link{false};

    auto produce = [&](uint64_t producer) {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        for (uint64_t task = producer; task < kTasks; task += kProducerCount) {
            if (!aicore_enqueue_completion_v1(
                    graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 1,
                    static_cast<int64_t>(task), task, nullptr
                )) {
                invalid_link.store(true, std::memory_order_release);
                break;
            }
        }
        producers_done.fetch_add(1, std::memory_order_release);
    };
    auto consume = [&] {
        while (producers_done.load(std::memory_order_acquire) != kProducerCount ||
               aicore_gm_load_v0(inbox->head) != AICORE_INBOX_EMPTY_V1) {
            int64_t task = aicore_gm_exchange_v0(inbox->head, AICORE_INBOX_EMPTY_V1);
            if (task == AICORE_INBOX_EMPTY_V1) {
                std::this_thread::yield();
                continue;
            }
            while (task >= 0) {
                if (static_cast<uint64_t>(task) >= kTasks) {
                    invalid_link.store(true, std::memory_order_release);
                    break;
                }
                int64_t next = controls[task].inbox_next;
                if (next < AICORE_INBOX_EMPTY_V1) {
                    invalid_link.store(true, std::memory_order_release);
                    break;
                }
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                consumed.fetch_add(1, std::memory_order_relaxed);
                task = next;
            }
        }
    };

    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);
    for (uint64_t producer = 0; producer < kProducerCount; ++producer)
        producers.emplace_back(produce, producer);
    std::thread consumer(consume);
    start.store(true, std::memory_order_release);
    for (auto &producer : producers)
        producer.join();
    consumer.join();

    EXPECT_FALSE(invalid_link.load());
    EXPECT_EQ(consumed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(AicoreCompletionInboxV1, PerWorkerCompletionIdRotatesInitialAicWaveAcrossResolvers) {
    constexpr uint64_t kRuntimeWorkers = 84;
    constexpr uint64_t kResolvers = 56;
    constexpr uint64_t kAicWorkers = 28;
    std::vector<uint64_t> inbox_counts(kResolvers, 0);
    AicoreWorkerContextV1 context{};
    context.runtime_worker_count = kRuntimeWorkers;
    for (uint64_t worker = 0; worker < kAicWorkers; ++worker) {
        context.worker_index = worker;
        for (uint64_t local_completion = 0; local_completion < 2; ++local_completion) {
            uint64_t completion_id = aicore_completion_id_v1(&context, local_completion);
            EXPECT_EQ(completion_id, worker + local_completion * kRuntimeWorkers);
            ++inbox_counts[aicore_completion_inbox_index_v1(&context, kResolvers, local_completion)];
        }
    }
    for (uint64_t count : inbox_counts)
        EXPECT_EQ(count, 1u);
}

TEST(AicoreCompletionInboxV1, EnqueueUsesPerWorkerCompletionOrderInsteadOfTaskId) {
    FixtureStorage storage(2, 56);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].runtime_worker_count = 84;
    ASSERT_TRUE(aicore_enqueue_completion_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 56, 0, 0, nullptr
    ));
    ASSERT_TRUE(aicore_enqueue_completion_v1(
        graph.graph(), storage.sidecar->base(), &storage.contexts[0], storage.run_control, 56, 1, 1, nullptr
    ));
    EXPECT_EQ(aicore_completion_inbox_at_v1(storage.sidecar->base(), &storage.contexts[0], 0)->head, 0);
    EXPECT_EQ(aicore_completion_inbox_at_v1(storage.sidecar->base(), &storage.contexts[0], 28)->head, 1);
}

TEST(AicoreFreeSlotDirectoryV1, ClaimIsUniqueAndGenerationChecked) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    for (uint64_t worker = 0; worker < 2; ++worker) {
        for (uint32_t slot_index = 0; slot_index < AICORE_PENDING_SLOT_COUNT_V1; ++slot_index) {
            auto *slot = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &storage.contexts[0], worker, slot_index);
            aicore_initialize_free_slot_v1(slot);
            aicore_advertise_free_slot_v1(storage.sidecar->base(), &storage.contexts[0], 1, worker, slot_index);
        }
    }
    std::atomic<uint64_t> claimed_mask{0};
    auto claim = [&](uint64_t worker) {
        uint64_t cursor = worker * AICORE_PENDING_SLOT_COUNT_V1;
        for (uint32_t attempt = 0; attempt < AICORE_PENDING_SLOT_COUNT_V1; ++attempt) {
            AicoreFreeSlotClaimV1 result{};
            ASSERT_TRUE(aicore_try_claim_free_slot_v1(
                graph.graph(), storage.sidecar->base(), &storage.contexts[worker], storage.run_control, 1, &cursor,
                nullptr, &result
            ));
            ASSERT_NE(result.worker_id, UINT64_MAX);
            uint64_t linear = result.worker_id * AICORE_PENDING_SLOT_COUNT_V1 + result.slot_index;
            uint64_t old = claimed_mask.fetch_or(UINT64_C(1) << linear);
            EXPECT_EQ(old & (UINT64_C(1) << linear), 0u);
        }
    };
    std::thread first(claim, 0);
    std::thread second(claim, 1);
    first.join();
    second.join();
    EXPECT_EQ(claimed_mask.load(), UINT64_C(0xf));
}

TEST(AicoreClusterCompletionV1, SpscGenerationCompletesNormalTask) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    AicoreWorkerContextV1 &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    auto *slot = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, 0, 0);
    aicore_initialize_free_slot_v1(slot);
    slot->task_id = 0;
    slot->gang = 0;
    aicore_gm_store_v0(
        slot->publication, aicore_dispatch_publication_v1(slot->generation, AicoreDispatchPublicationV1::READY)
    );
    auto *completion_line = aicore_completion_inbox_at_v1(storage.sidecar->base(), &resolver, 0);
    completion_line->completed_generations[0] = slot->generation;
    auto *control = aicore_task_control_at_v1(storage.sidecar->base(), &resolver, 0);
    control->state = static_cast<int64_t>(AicoreTaskStateV1::READY);
    AicoreWakeStatsV1 wake_stats{};
    AicoreReadyStatsV1 ready_stats{};
    AicoreCompletionStatsV1 completion_stats{};
    ASSERT_TRUE(aicore_service_cluster_completions_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(completion_line->completed_generations[0], 0u);
    EXPECT_EQ(aicore_dispatch_state_v1(slot->publication), AicoreDispatchPublicationV1::FREE);
    EXPECT_EQ(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    EXPECT_EQ(control->wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(AicoreSyncStartV1, DrainsStagesAndReleasesBeforeCompletion) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    AicoreWorkerContextV1 &resolver = storage.contexts[1];
    resolver.is_resolver = 1;
    resolver.resolver_index = 0;
    resolver.resolver_count = 1;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIC);
    storage.run_control->resolver_count = 1;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < AICORE_PENDING_SLOT_COUNT_V1; ++pending_slot)
            aicore_initialize_free_slot_v1(
                aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, worker, pending_slot)
            );
    }
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 2;
    storage.metadata[0].total_required_subtasks = 2;
    storage.metadata[0].flags =
        AICORE_TASK_EXECUTABLE_V1 | AICORE_TASK_SPMD_V1 | AICORE_TASK_SYNC_START_V1;
    auto *callables =
        aicore_sidecar_at_v1<uint64_t>(storage.sidecar->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *control = aicore_task_control_at_v1(storage.sidecar->base(), &resolver, 0);
    control->state = static_cast<int64_t>(AicoreTaskStateV1::READY);
    auto *coordinator = aicore_gang_coordinator_at_v1(storage.sidecar->base(), &resolver);
    coordinator->gang_task_count = 1;
    coordinator->resolver_count = 1;
    coordinator->ready_priority_bits = 1;
    AicoreWakeStatsV1 wake_stats{};
    AicoreReadyStatsV1 ready_stats{};
    AicoreCompletionStatsV1 completion_stats{};
    ASSERT_TRUE(aicore_gang_admit_one_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control
    ));
    auto *cohort = aicore_gang_cohort_at_v1(storage.sidecar->base(), &resolver, 0);
    ASSERT_TRUE(aicore_service_gang_scheduler_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(AicoreGangCohortStateV1::STAGING));
    ASSERT_TRUE(aicore_service_gang_scheduler_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(AicoreGangCohortStateV1::RELEASING));
    ASSERT_TRUE(aicore_service_gang_scheduler_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(AicoreGangCohortStateV1::EXECUTING));
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
    for (uint32_t worker = 1; worker <= 2; ++worker) {
        auto *slot = aicore_dispatch_slot_at_v1(storage.sidecar->base(), &resolver, worker, 0);
        ASSERT_EQ(aicore_dispatch_state_v1(slot->publication), AicoreDispatchPublicationV1::READY);
        aicore_completion_inbox_at_v1(storage.sidecar->base(), &resolver, worker)
            ->completed_generations[0] = slot->generation;
    }
    ASSERT_TRUE(aicore_service_cluster_completions_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    ASSERT_TRUE(aicore_service_gang_scheduler_v1(
        graph.graph(), storage.sidecar->base(), &resolver, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(AicoreGangCohortStateV1::FREE));
    EXPECT_EQ(control->state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

}  // namespace
