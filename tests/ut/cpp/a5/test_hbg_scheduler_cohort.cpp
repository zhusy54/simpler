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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "runtime_types.h"
#include "scheduler/scheduler_dispatch.h"

namespace {

class StateBuffer {
public:
    explicit StateBuffer(const AicoreSchedulerLayout &layout) :
        base_(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout));
    }
    ~StateBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        task_count_(task_count),
        image_(std::make_unique<Image>()) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        if (capacity_ > kMaxTaskCount) throw std::invalid_argument("test graph is too large");
        for (size_t task = 0; task < capacity_; ++task) {
            image_->storage[task].task.task_id = TaskId{static_cast<uint64_t>(task)};
            image_->storage[task].payload.bind_regions(
                nullptr, nullptr, image_->fanins.data() + task * static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN)
            );
            for (int slot = 0; slot < 3; ++slot)
                image_->storage[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t slot = 0; slot < 3; ++slot) {
            if ((active_mask & (1U << slot)) != 0) image_->storage[task].task.kernel_id[slot] = 1;
        }
        image_->storage[task].payload.fanin_count = 0;
    }

    SchedulerGraphView graph() const {
        return {reinterpret_cast<uint64_t>(image_->storage.data()), 0, task_count_, capacity_ - 1};
    }

private:
    static constexpr size_t kMaxTaskCount = 64;
    struct alignas(64) Image {
        std::array<ChipTaskStorage, kMaxTaskCount> storage{};
        std::array<int32_t, kMaxTaskCount * SCHEDULER_GRAPH_MAX_FANIN> fanins{};
    };
    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<Image> image_;
};

struct Fixture {
    explicit Fixture(uint64_t task_count, uint64_t worker_count) {
        EXPECT_TRUE(scheduler_plan_layout(task_count, task_count, 0, &layout));
        state = std::make_unique<StateBuffer>(layout);
        run = scheduler_state_at<SchedulerRunControl>(state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(state->base(), layout.worker_contexts_offset);
        owners = scheduler_state_at<SchedulerReadyOwnerState>(state->base(), layout.ready_owner_states_offset);
        metadata = scheduler_state_at<SchedulerTaskMetadata>(state->base(), layout.task_metadata_offset);
        for (uint64_t worker = 0; worker < worker_count; ++worker) {
            auto &context = contexts[worker];
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
            context.cohort_coordinator_offset = layout.cohort_coordinator_offset;
            context.cohort_cohorts_offset = layout.cohort_cohorts_offset;
            context.cohort_participants_offset = layout.cohort_participants_offset;
            context.cohort_commands_offset = layout.cohort_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = worker_count;
            context.worker_index = worker;
        }
        auto *callables = scheduler_state_at<uint64_t>(state->base(), layout.callable_addresses_offset);
        callables[1] = 0x1000;
    }

    SchedulerWorkerContext *scheduler(uint32_t index, uint32_t count) {
        const uint32_t first = index * 3;
        contexts[first].core_type = static_cast<int32_t>(CoreType::AIC);
        contexts[first + 1].core_type = static_cast<int32_t>(CoreType::AIV);
        contexts[first + 2].core_type = static_cast<int32_t>(CoreType::AIV);
        auto *result = &contexts[first + 1];
        result->is_scheduler = 1;
        result->scheduler_index = index;
        result->scheduler_count = count;
        result->inbox_index = index;
        result->cluster_worker_ids[0] = first;
        result->cluster_worker_ids[1] = first + 1;
        result->cluster_worker_ids[2] = first + 2;
        run->scheduler_count = count;
        scheduler_ready_owner_init(&owners[index]);
        for (uint32_t worker = first; worker < first + 3; ++worker) {
            for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
                scheduler_initialize_free_slot(scheduler_dispatch_slot_at(state->base(), result, worker, slot));
        }
        return result;
    }

    void task(uint32_t id, uint8_t active_mask, uint16_t block_num, uint8_t flags) {
        metadata[id].active_mask = active_mask;
        metadata[id].logical_block_num = block_num;
        metadata[id].total_required_subtasks = block_num * static_cast<uint16_t>(__builtin_popcount(active_mask));
        metadata[id].flags = SCHEDULER_TASK_EXECUTABLE | flags;
        metadata[id].timing_slot = -1;
        for (uint8_t slot = 0; slot < 3; ++slot)
            metadata[id].kernel_ids[slot] = (active_mask & (1U << slot)) != 0 ? 1 : UINT16_MAX;
    }

    void enqueue_regular(SchedulerWorkerContext *owner, uint32_t task_id) {
        scheduler_publish_regular_cohort_priority(
            state->base(), owner, metadata[task_id].flags, metadata[task_id].active_mask
        );
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(scheduler_ready_batch_append(state->base(), owner, task_id, &batch, nullptr));
        ASSERT_TRUE(scheduler_ready_batch_push(
            state->base(), owner, scheduler_task_ready_queue(metadata[task_id].flags, metadata[task_id].active_mask),
            owner->inbox_index, &batch, nullptr, &owners[owner->inbox_index]
        ));
    }

    AicoreSchedulerLayout layout{};
    std::unique_ptr<StateBuffer> state;
    SchedulerRunControl *run{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerReadyOwnerState *owners{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
};

TEST(SchedulerCohortPriority, GatesOnlyRequiredResourceLanes) {
    Fixture fixture(1, 3);
    auto *scheduler = fixture.scheduler(0, 1);
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), scheduler);
    scheduler_cohort_priority_acquire(coordinator, 1U);
    EXPECT_TRUE(scheduler_cohort_lane_has_priority(coordinator, 0));
    EXPECT_FALSE(scheduler_cohort_lane_has_priority(coordinator, 1));
    scheduler_cohort_priority_release(coordinator, 1U);
    scheduler_cohort_priority_acquire(coordinator, 6U);
    EXPECT_FALSE(scheduler_cohort_lane_has_priority(coordinator, 0));
    EXPECT_TRUE(scheduler_cohort_lane_has_priority(coordinator, 1));
    EXPECT_TRUE(scheduler_cohort_lane_has_priority(coordinator, 2));
}

TEST(SchedulerCohortPriority, ReadyCohortPreventsNormalSlotPublication) {
    Fixture fixture(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    auto *scheduler = fixture.scheduler(0, 1);
    fixture.task(0, 1, 1, 0);
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(fixture.state->base(), scheduler, 0, &batch, nullptr));
    ASSERT_TRUE(scheduler_ready_batch_push(
        fixture.state->base(), scheduler, static_cast<uint32_t>(SchedulerReadyQueueIndex::NORMAL_AIC), 0, &batch,
        nullptr, &fixture.owners[0]
    ));
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), scheduler);
    scheduler_cohort_priority_acquire(coordinator, 1U);
    uint64_t cursors[SCHEDULER_READY_QUEUE_COUNT]{};
    bool failed = false;
    EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
        graph.graph(), fixture.state->base(), scheduler, fixture.run, cursors, nullptr, 0, 0, nullptr,
        &fixture.owners[0], &failed
    ));
    EXPECT_FALSE(failed);
    auto *slot = scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 0, 0);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(
        scheduler_ready_inbox_at(
            fixture.state->base(), scheduler, static_cast<uint32_t>(SchedulerReadyQueueIndex::NORMAL_AIC), 0
        )
            ->head,
        0
    );
}

TEST(SchedulerRegularSpmd, AnySchedulerCanAdmitAndStealLogicalBlocks) {
    Fixture fixture(1, 6);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    auto *scheduler0 = fixture.scheduler(0, 2);
    auto *scheduler1 = fixture.scheduler(1, 2);
    fixture.task(0, 1, 4, SCHEDULER_TASK_SPMD);
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), scheduler0);
    coordinator->cohort_task_count = 1;
    fixture.enqueue_regular(scheduler0, 0);
    ASSERT_TRUE(scheduler_cohort_admit_regular(
        graph.graph(), fixture.state->base(), scheduler1, fixture.run, nullptr, &fixture.owners[1]
    ));
    auto *cohort = scheduler_cohort_record_at(fixture.state->base(), scheduler1, 0);
    EXPECT_EQ(cohort->owner_scheduler, 1u);
    EXPECT_EQ(cohort->base_scheduler, 1u);

    auto *local = scheduler_cohort_participant_at(fixture.state->base(), scheduler0, 0, 0);
    auto *remote = scheduler_cohort_participant_at(fixture.state->base(), scheduler0, 0, 1);
    local->next_block[0] = UINT32_MAX;
    ASSERT_TRUE(scheduler_cohort_dispatch_one_regular_block(
        graph.graph(), fixture.state->base(), scheduler0, fixture.run, 0, cohort
    ));
    auto *slot = scheduler_dispatch_slot_at(fixture.state->base(), scheduler0, 0, 0);
    EXPECT_EQ(slot->block_idx, 0u);
    EXPECT_EQ(remote->next_block[0], 2u);
    EXPECT_EQ(local->local_expected_subtasks, 1u);
    EXPECT_EQ(cohort->participant_bitmap, 1u);
}

TEST(SchedulerRegularSpmd, TwoRecordsAllowConcurrentTasks) {
    Fixture fixture(2, 6);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    auto *scheduler0 = fixture.scheduler(0, 2);
    auto *scheduler1 = fixture.scheduler(1, 2);
    fixture.task(0, 1, 2, SCHEDULER_TASK_SPMD);
    fixture.task(1, 1, 2, SCHEDULER_TASK_SPMD);
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), scheduler0);
    coordinator->cohort_task_count = 2;
    fixture.enqueue_regular(scheduler0, 0);
    fixture.enqueue_regular(scheduler1, 1);
    ASSERT_TRUE(scheduler_cohort_admit_regular(
        graph.graph(), fixture.state->base(), scheduler0, fixture.run, nullptr, &fixture.owners[0]
    ));
    ASSERT_TRUE(scheduler_cohort_admit_regular(
        graph.graph(), fixture.state->base(), scheduler1, fixture.run, nullptr, &fixture.owners[1]
    ));
    for (uint32_t index = 0; index < SCHEDULER_COHORT_COUNT; ++index)
        EXPECT_NE(
            scheduler_cohort_record_at(fixture.state->base(), scheduler0, index)->state,
            static_cast<uint64_t>(SchedulerCohortState::FREE)
        );
}

TEST(SchedulerCohortMix, UsesIndependentPendingSlotIndicesAcrossLanes) {
    Fixture fixture(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 7);
    auto *scheduler = fixture.scheduler(0, 1);
    fixture.task(0, 7, 1, SCHEDULER_TASK_MIX);
    auto *participant = scheduler_cohort_participant_at(fixture.state->base(), scheduler, 0, 0);
    participant->config_generation = 1;
    participant->task_id = 0;
    participant->active_mask = 7;
    participant->logical_block_num = 1;
    auto *aiv0_slot0 = scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 1, 0);
    auto *aiv1_slot1 = scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 2, 1);
    scheduler_gm_store(
        aiv0_slot0->publication,
        scheduler_dispatch_publication(aiv0_slot0->generation, SchedulerDispatchSlotState::READY)
    );
    scheduler_gm_store(
        aiv1_slot1->publication,
        scheduler_dispatch_publication(aiv1_slot1->generation, SchedulerDispatchSlotState::READY)
    );
    ASSERT_TRUE(scheduler_cohort_fill_mix_block(
        graph.graph(), fixture.state->base(), scheduler, fixture.run, participant, 0, 0,
        SchedulerDispatchSlotState::GATED
    ));
    EXPECT_EQ(
        scheduler_dispatch_state(scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 0, 0)->publication),
        SchedulerDispatchSlotState::GATED
    );
    EXPECT_EQ(
        scheduler_dispatch_state(scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 1, 1)->publication),
        SchedulerDispatchSlotState::GATED
    );
    EXPECT_EQ(
        scheduler_dispatch_state(scheduler_dispatch_slot_at(fixture.state->base(), scheduler, 2, 0)->publication),
        SchedulerDispatchSlotState::GATED
    );
}

TEST(SchedulerSyncStart, RotatesLogicalBlockOwnership) {
    Fixture fixture(1, 9);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    std::array<SchedulerWorkerContext *, 3> schedulers{};
    for (uint32_t index = 0; index < schedulers.size(); ++index)
        schedulers[index] = fixture.scheduler(index, schedulers.size());
    fixture.task(0, 1, 2, SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START);
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), schedulers[0]);
    coordinator->cohort_task_count = 1;
    coordinator->next_sync_scheduler = 1;
    auto *control = scheduler_task_control_at(fixture.state->base(), schedulers[0], 0);
    scheduler_publish_cohort_ready(
        fixture.state->base(), schedulers[0], control, fixture.metadata[0].flags, fixture.metadata[0].active_mask
    );
    ASSERT_TRUE(scheduler_cohort_admit_sync(graph.graph(), fixture.state->base(), schedulers[0], fixture.run));
    auto *cohort = scheduler_cohort_record_at(fixture.state->base(), schedulers[0], 0);
    EXPECT_EQ(cohort->base_scheduler, 1u);
    EXPECT_EQ(coordinator->next_sync_scheduler, 0u);
    auto *participant0 = scheduler_cohort_participant_at(fixture.state->base(), schedulers[0], 0, 0);
    auto *participant1 = scheduler_cohort_participant_at(fixture.state->base(), schedulers[0], 0, 1);
    auto *participant2 = scheduler_cohort_participant_at(fixture.state->base(), schedulers[0], 0, 2);
    EXPECT_EQ(participant0->next_block[0], 2u);
    EXPECT_EQ(participant0->local_expected_subtasks, 0u);
    EXPECT_EQ(participant1->next_block[0], 0u);
    EXPECT_EQ(participant2->next_block[0], 1u);
}

TEST(SchedulerSyncStart, DrainsStagesAndReleases) {
    Fixture fixture(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 2);
    auto *scheduler = fixture.scheduler(0, 1);
    fixture.task(0, 2, 2, SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START);
    auto *coordinator = scheduler_cohort_coordinator_at(fixture.state->base(), scheduler);
    coordinator->cohort_task_count = 1;
    auto *control = scheduler_task_control_at(fixture.state->base(), scheduler, 0);
    scheduler_publish_cohort_ready(
        fixture.state->base(), scheduler, control, fixture.metadata[0].flags, fixture.metadata[0].active_mask
    );
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    ASSERT_TRUE(scheduler_cohort_admit_sync(graph.graph(), fixture.state->base(), scheduler, fixture.run));
    auto *cohort = scheduler_cohort_record_at(fixture.state->base(), scheduler, 0);
    for (SchedulerCohortState expected :
         {SchedulerCohortState::STAGING, SchedulerCohortState::RELEASING, SchedulerCohortState::EXECUTING}) {
        ASSERT_TRUE(scheduler_service_cohort(
            graph.graph(), fixture.state->base(), scheduler, fixture.run, &wake, &ready, &completion, &fixture.owners[0]
        ));
        EXPECT_EQ(cohort->state, static_cast<uint64_t>(expected));
    }
    EXPECT_EQ(coordinator->priority_lane_counts[1], 0u);
    EXPECT_EQ(coordinator->priority_lane_counts[2], 0u);
}

}  // namespace
