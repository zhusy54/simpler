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

#include <cstdint>
#include <cstdlib>

#include "scheduler/scheduler.h"

TEST(SchedulerNormalDispatchPriority, OrdersByRoleBeforeLoad) {
    EXPECT_TRUE(scheduler_normal_aiv_worker_precedes(0, false, 0, true));
    EXPECT_FALSE(scheduler_normal_aiv_worker_precedes(0, true, 0, false));
    EXPECT_TRUE(scheduler_normal_aiv_worker_precedes(1, false, 0, true));
    EXPECT_FALSE(scheduler_normal_aiv_worker_precedes(0, true, 1, false));
    EXPECT_TRUE(scheduler_normal_aiv_worker_precedes(0, false, 1, false));
    EXPECT_FALSE(scheduler_normal_aiv_worker_precedes(1, true, 0, true));
}

TEST(SchedulerNormalDispatchPriority, EmptyReadyDirectorySkipsAllSlotClaims) {
    SchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout));
    void *scheduler_state = std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size);
    ASSERT_NE(scheduler_state, nullptr);
    ASSERT_TRUE(scheduler_init_data_from_layout(scheduler_state, layout));

    auto *run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state, layout.run_control_offset);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < 3; ++worker) {
        contexts[worker].active = 1;
        contexts[worker].worker_index = worker;
        contexts[worker].core_type =
            worker == 0 ? static_cast<int32_t>(CoreType::AIC) : static_cast<int32_t>(CoreType::AIV);
        contexts[worker].worker_contexts_offset = layout.worker_contexts_offset;
        contexts[worker].dispatch_slots_offset = layout.dispatch_slots_offset;
        contexts[worker].ready_inboxes_offset = layout.ready_inboxes_offset;
        contexts[worker].ready_owner_states_offset = layout.ready_owner_states_offset;
        contexts[worker].ready_directory_offset = layout.ready_directory_offset;
        contexts[worker].gang_coordinator_offset = layout.gang_coordinator_offset;
        contexts[worker].runtime_worker_count = 3;
        contexts[worker].graph_task_count = 1;
    }
    SchedulerWorkerContext &resolver = contexts[1];
    resolver.is_resolver = 1;
    resolver.inbox_index = 0;
    resolver.resolver_count = 7;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    run_control->resolver_count = resolver.resolver_count;
    for (uint64_t worker = 0; worker < 3; ++worker) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
            scheduler_initialize_free_slot(scheduler_dispatch_slot_at(scheduler_state, &resolver, worker, slot));
        }
    }

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{1, 1};
    SchedulerReadyStats stats{};
    SchedulerGraphView graph{};
    graph.task_count = 1;
    EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
        graph, scheduler_state, &resolver, run_control, victim_cursors, &stats, false
    ));
    EXPECT_EQ(victim_cursors[0], 1u);
    EXPECT_EQ(victim_cursors[1], 1u);
    std::free(scheduler_state);
}
