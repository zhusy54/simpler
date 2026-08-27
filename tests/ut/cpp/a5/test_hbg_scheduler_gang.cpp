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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_topology.h"
#include "runtime_types.h"

namespace {

class SchedulerStateBuffer {
public:
    explicit SchedulerStateBuffer(const AicoreSchedulerLayout &layout) :
        base_(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout));
    }
    ~SchedulerStateBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        task_count_(task_count),
        image_(std::make_unique<GraphImage>()) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        if (capacity_ > kMaxTaskCount) throw std::invalid_argument("test graph exceeds GraphBuffer capacity");
        storage_ = image_->storage.data();
        fanins_ = image_->fanins.data();
        for (size_t task = 0; task < capacity_; ++task) {
            storage_[task].task.task_id = TaskId{static_cast<uint64_t>(task)};
            storage_[task].payload.bind_regions(
                nullptr, nullptr, fanins_ + task * static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN)
            );
            if (storage_[task].payload.fanin_data() == nullptr) {
                throw std::logic_error("test graph fanin region must share its contiguous image");
            }
            for (int slot = 0; slot < 3; ++slot)
                storage_[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t subtask_slot, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task, task_count_);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN));
        storage_[task].task.kernel_id[subtask_slot] = 1;
        storage_[task].payload.fanin_count = static_cast<int32_t>(fanins.size());
        ASSERT_TRUE(fanins.empty() || storage_[task].payload.fanin_data() != nullptr);
        std::copy(fanins.begin(), fanins.end(), storage_[task].payload.fanin_data());
    }

    void mixed(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) != 0) storage_[task].task.kernel_id[subtask_slot] = 1;
        }
        storage_[task].payload.fanin_count = 0;
    }

    SchedulerGraphView graph() const {
        return {
            reinterpret_cast<uint64_t>(storage_),
            0,
            task_count_,
            capacity_ - 1,
        };
    }

private:
    static constexpr size_t kMaxTaskCount = 8192;
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, kMaxTaskCount> storage{};
        std::array<int32_t, kMaxTaskCount * SCHEDULER_GRAPH_MAX_FANIN> fanins{};
    };

    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<GraphImage> image_;
    ChipTaskStorage *storage_{nullptr};
    int32_t *fanins_{nullptr};
};

struct FixtureStorage {
    explicit FixtureStorage(uint64_t task_count, uint64_t workers = 2) {
        EXPECT_TRUE(scheduler_plan_layout(task_count, task_count, 0, &layout));
        scheduler_state = std::make_unique<SchedulerStateBuffer>(layout);
        run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state->base(), layout.worker_contexts_offset);
        owner_states =
            scheduler_state_at<SchedulerReadyOwnerState>(scheduler_state->base(), layout.ready_owner_states_offset);
        run_control->aiv_active_worker_count = workers;
        run_control->scheduler_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            SchedulerWorkerContext &context = contexts[worker];
            context.core_type = static_cast<int32_t>(CoreType::AIV);
            context.active = 1;
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;
            context.completion_inboxes_offset = layout.completion_inboxes_offset;
            context.ready_inboxes_offset = layout.ready_inboxes_offset;
            context.ready_owner_states_offset = layout.ready_owner_states_offset;
            context.ready_directory_offset = layout.ready_directory_offset;
            context.trace_cells_offset = layout.trace_cells_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.dispatch_slots_offset = layout.dispatch_slots_offset;
            context.callable_addresses_offset = layout.callable_addresses_offset;
            context.gang_coordinator_offset = layout.gang_coordinator_offset;
            context.gang_cohorts_offset = layout.gang_cohorts_offset;
            context.gang_participants_offset = layout.gang_participants_offset;
            context.gang_commands_offset = layout.gang_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.inbox_index = worker;
            scheduler_ready_owner_init(&owner_states[worker]);
        }
        metadata = scheduler_state_at<SchedulerTaskMetadata>(scheduler_state->base(), layout.task_metadata_offset);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = 1;
            metadata[task].logical_block_num = 1;
            metadata[task].total_required_subtasks = 1;
            metadata[task].timing_slot = -1;
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
        }
    }

    AicoreSchedulerLayout layout{};
    std::unique_ptr<SchedulerStateBuffer> scheduler_state;
    SchedulerRunControl *run_control{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerReadyOwnerState *owner_states{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
};

TEST(SchedulerGangPriority, SyncStartPrecedesMixAndSpmd) {
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_SPMD), 4u);
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_MIX), 2u);
    EXPECT_EQ(scheduler_task_priority_bit(SCHEDULER_TASK_MIX | SCHEDULER_TASK_SYNC_START), 1u);

    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.mixed(1, 3);
    graph.executable(2, 1);
    auto *coordinator = scheduler_state_at<SchedulerGangCoordinator>(
        storage.scheduler_state->base(), storage.layout.gang_coordinator_offset
    );
    coordinator->gang_task_count = 3;
    coordinator->scheduler_count = 1;
    coordinator->ready_priority_bits = 7;
    storage.run_control->scheduler_count = 1;
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    for (uint32_t task = 0; task < 3; ++task)
        controls[task].state = static_cast<int64_t>(SchedulerTaskState::READY);
    storage.metadata[0].active_mask = 1;
    storage.metadata[0].logical_block_num = 4;
    storage.metadata[0].total_required_subtasks = 4;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    storage.metadata[1].active_mask = 3;
    storage.metadata[1].logical_block_num = 1;
    storage.metadata[1].total_required_subtasks = 2;
    storage.metadata[1].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
    storage.metadata[2].active_mask = 2;
    storage.metadata[2].logical_block_num = 2;
    storage.metadata[2].total_required_subtasks = 2;
    storage.metadata[2].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;

    ASSERT_TRUE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control)
    );
    auto *cohort =
        scheduler_state_at<SchedulerGangCohort>(storage.scheduler_state->base(), storage.layout.gang_cohorts_offset);
    EXPECT_EQ(cohort->task_id, 2);
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::DRAINING));
    EXPECT_EQ(controls[2].state, static_cast<int64_t>(SchedulerTaskState::DISPATCHING));
    EXPECT_FALSE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control)
    );
}

TEST(SchedulerSyncStart, RejectsImpossibleCohortBeforeDrain) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    storage.run_control->scheduler_count = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 3;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), &scheduler, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *coordinator = scheduler_gang_coordinator_at(storage.scheduler_state->base(), &scheduler);
    coordinator->gang_task_count = 1;
    coordinator->scheduler_count = 1;
    coordinator->ready_priority_bits = 1;

    EXPECT_FALSE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control)
    );
    EXPECT_EQ(coordinator->capacity_reject_count, 1u);
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::GANG_UNSUPPORTED_SHAPE));
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
}

TEST(SchedulerGangDispatch, AcceptsLastCallableAndRejectsCapacityBoundary) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, 0, 0);
    scheduler_initialize_free_slot(slot);
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);

    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(SCHEDULER_CALLABLE_CAPACITY - 1);
    callables[SCHEDULER_CALLABLE_CAPACITY - 1] = 0x1000;
    EXPECT_TRUE(scheduler_fill_explicit_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, 0, 0, 0, 0, 0, 1, 0, 1,
        SchedulerDispatchSlotState::GATED
    ));

    scheduler_initialize_free_slot(slot);
    storage.run_control->scheduler_error = 0;
    storage.run_control->error_site = 0;
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(SCHEDULER_CALLABLE_CAPACITY);
    EXPECT_FALSE(scheduler_fill_explicit_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, 0, 0, 0, 0, 0, 1, 0, 1,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::GANG_INVALID_CALLABLE));
}

TEST(SchedulerGangTokens, StaleGenerationCannotCompleteTree) {
    FixtureStorage storage(1, 3);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 3;
    constexpr uint64_t kGeneration = 9;
    for (uint64_t participant_index = 0; participant_index < 3; ++participant_index) {
        auto *participant =
            scheduler_gang_participant_at(storage.scheduler_state->base(), &scheduler, 0, participant_index);
        participant->dispatch_local_token = kGeneration;
    }
    auto *stale_child = scheduler_gang_participant_at(storage.scheduler_state->base(), &scheduler, 0, 2);
    stale_child->dispatch_local_token = kGeneration - 1;
    scheduler.scheduler_index = 1;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &scheduler, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    scheduler.scheduler_index = 2;
    EXPECT_FALSE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &scheduler, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    scheduler.scheduler_index = 0;
    EXPECT_FALSE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &scheduler, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    stale_child->dispatch_local_token = kGeneration;
    scheduler.scheduler_index = 2;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &scheduler, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
    scheduler.scheduler_index = 0;
    EXPECT_TRUE(scheduler_gang_update_subtree_token(
        storage.scheduler_state->base(), &scheduler, 0, 3, kGeneration, SchedulerGangTokenPhase::DISPATCH
    ));
}

TEST(SchedulerSyncStart, WaitsForEveryParticipantAcrossFullLifecycle) {
    FixtureStorage storage(1, 9);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    std::array<SchedulerWorkerContext *, 3> schedulers{};
    storage.run_control->scheduler_count = schedulers.size();
    for (uint32_t cluster = 0; cluster < schedulers.size(); ++cluster) {
        const uint32_t first_worker = cluster * 3;
        storage.contexts[first_worker].core_type = static_cast<int32_t>(CoreType::AIC);
        SchedulerWorkerContext &scheduler = storage.contexts[first_worker + 1];
        scheduler.is_scheduler = 1;
        scheduler.scheduler_index = cluster;
        scheduler.scheduler_count = schedulers.size();
        scheduler.inbox_index = cluster;
        scheduler.cluster_worker_ids[0] = first_worker;
        scheduler.cluster_worker_ids[1] = first_worker + 1;
        scheduler.cluster_worker_ids[2] = first_worker + 2;
        schedulers[cluster] = &scheduler;
        for (uint32_t worker = first_worker; worker < first_worker + 3; ++worker) {
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                scheduler_initialize_free_slot(
                    scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, worker, pending_slot)
                );
            }
        }
    }

    storage.metadata[0].active_mask = 1;
    storage.metadata[0].logical_block_num = 3;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), schedulers[0], 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *coordinator = scheduler_gang_coordinator_at(storage.scheduler_state->base(), schedulers[0]);
    coordinator->gang_task_count = 1;
    coordinator->scheduler_count = schedulers.size();
    coordinator->ready_priority_bits = 1;

    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    auto service = [&](uint32_t cluster) {
        return scheduler_service_gang(
            graph.graph(), storage.scheduler_state->base(), schedulers[cluster], storage.run_control, &wake_stats,
            &ready_stats, &completion_stats, &storage.owner_states[cluster]
        );
    };
    auto *cohort = scheduler_gang_cohort_at(storage.scheduler_state->base(), schedulers[0], 0);

    ASSERT_TRUE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), schedulers[0], storage.run_control)
    );
    ASSERT_TRUE(service(0));
    ASSERT_TRUE(service(1));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::DRAINING));
    ASSERT_TRUE(service(2));
    ASSERT_TRUE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::STAGING));

    ASSERT_TRUE(service(0));
    ASSERT_TRUE(service(1));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::STAGING));
    auto *unstaged = scheduler_dispatch_slot_at(storage.scheduler_state->base(), schedulers[2], 6, 0);
    EXPECT_EQ(scheduler_dispatch_state(unstaged->publication), SchedulerDispatchSlotState::FREE);
    ASSERT_TRUE(service(2));
    ASSERT_TRUE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::RELEASING));
    for (uint32_t cluster = 0; cluster < schedulers.size(); ++cluster) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), schedulers[cluster], cluster * 3, 0);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::GATED);
    }

    ASSERT_TRUE(service(0));
    ASSERT_TRUE(service(1));
    EXPECT_FALSE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::RELEASING));
    EXPECT_EQ(scheduler_dispatch_state(unstaged->publication), SchedulerDispatchSlotState::GATED);
    ASSERT_TRUE(service(2));
    ASSERT_TRUE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::EXECUTING));

    for (uint32_t cluster = 0; cluster < schedulers.size(); ++cluster) {
        const uint32_t worker = cluster * 3;
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), schedulers[cluster], worker, 0);
        ASSERT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
        scheduler_completion_inbox_at(storage.scheduler_state->base(), schedulers[cluster], worker)
            ->completed_generations[0] = slot->generation;
        ASSERT_TRUE(scheduler_service_cluster_completions(
            graph.graph(), storage.scheduler_state->base(), schedulers[cluster], storage.run_control, &wake_stats,
            &ready_stats, &completion_stats
        ));
    }

    ASSERT_TRUE(service(0));
    ASSERT_TRUE(service(1));
    ASSERT_TRUE(service(2));
    ASSERT_TRUE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::RETIRING));
    auto *child_command = scheduler_gang_command_at(storage.scheduler_state->base(), schedulers[0], 1);
    EXPECT_EQ(child_command->state[0], static_cast<uint64_t>(SchedulerGangCohortState::FREE));
    EXPECT_FALSE(service(0));
    EXPECT_EQ(child_command->state[0], static_cast<uint64_t>(SchedulerGangCohortState::FREE));
    ASSERT_TRUE(service(1));
    ASSERT_TRUE(service(2));
    ASSERT_TRUE(service(0));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::FREE));
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerGangMix, ReservesWholeClusterOrNothing) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.mixed(0, 7);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, worker, pending_slot)
            );
    }
    storage.metadata[0].active_mask = 7;
    storage.metadata[0].kernel_ids[0] = 1;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].kernel_ids[2] = 1;
    storage.metadata[0].logical_block_num = 1;
    storage.metadata[0].total_required_subtasks = 3;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *participant = scheduler_gang_participant_at(storage.scheduler_state->base(), &scheduler, 0, 0);
    participant->config_generation = 1;
    participant->task_id = 0;
    participant->active_mask = 7;
    participant->logical_block_num = 1;
    participant->local_expected_subtasks = 3;
    auto *blocked0 = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, 1, 0);
    auto *blocked1 = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, 2, 1);
    scheduler_gm_store(
        blocked0->publication, scheduler_dispatch_publication(blocked0->generation, SchedulerDispatchSlotState::READY)
    );
    scheduler_gm_store(
        blocked1->publication, scheduler_dispatch_publication(blocked1->generation, SchedulerDispatchSlotState::READY)
    );
    EXPECT_FALSE(scheduler_gang_fill_mix_block(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, participant, 0, 0,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 0u);
    scheduler_gm_store(
        blocked0->publication, scheduler_dispatch_publication(blocked0->generation, SchedulerDispatchSlotState::FREE)
    );
    scheduler_gm_store(
        blocked1->publication, scheduler_dispatch_publication(blocked1->generation, SchedulerDispatchSlotState::FREE)
    );
    ASSERT_TRUE(scheduler_gang_fill_mix_block(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, participant, 0, 0,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(participant->local_published_subtasks, 3u);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, worker, 0);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::GATED);
        EXPECT_EQ(slot->block_idx, 0u);
        EXPECT_EQ(slot->block_num, 1u);
    }
}

TEST(SchedulerSyncStart, DrainsStagesAndReleasesBeforeCompletion) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.run_control->scheduler_count = 1;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, worker, pending_slot)
            );
    }
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = 2;
    storage.metadata[0].logical_block_num = 2;
    storage.metadata[0].total_required_subtasks = 2;
    storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), &scheduler, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::READY);
    auto *coordinator = scheduler_gang_coordinator_at(storage.scheduler_state->base(), &scheduler);
    coordinator->gang_task_count = 1;
    coordinator->scheduler_count = 1;
    coordinator->ready_priority_bits = 1;
    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(
        scheduler_gang_admit_one(graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control)
    );
    auto *cohort = scheduler_gang_cohort_at(storage.scheduler_state->base(), &scheduler, 0);
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, &storage.owner_states[0]
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::STAGING));
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, &storage.owner_states[0]
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::RELEASING));
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, &storage.owner_states[0]
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::EXECUTING));
    EXPECT_EQ(coordinator->active_dispatch_cohort, UINT64_MAX);
    for (uint32_t worker = 1; worker <= 2; ++worker) {
        auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), &scheduler, worker, 0);
        ASSERT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
        scheduler_completion_inbox_at(storage.scheduler_state->base(), &scheduler, worker)->completed_generations[0] =
            slot->generation;
    }
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats
    ));
    auto *participant = scheduler_gang_participant_at(storage.scheduler_state->base(), &scheduler, 0, 0);
    EXPECT_EQ(participant->local_completed_subtasks, 2u);
    ASSERT_TRUE(scheduler_service_gang(
        graph.graph(), storage.scheduler_state->base(), &scheduler, storage.run_control, &wake_stats, &ready_stats,
        &completion_stats, &storage.owner_states[0]
    ));
    EXPECT_EQ(cohort->state, static_cast<uint64_t>(SchedulerGangCohortState::FREE));
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

}  // namespace
