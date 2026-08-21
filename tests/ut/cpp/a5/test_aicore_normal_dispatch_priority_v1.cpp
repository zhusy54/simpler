/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of this repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "aicore_gang_scheduler_v1.h"

TEST(AicoreNormalDispatchPriorityV1, AlwaysPrefersPeerOverResolver) {
    EXPECT_TRUE(aicore_normal_aiv_worker_precedes_v1(0, false, 0, true));
    EXPECT_FALSE(aicore_normal_aiv_worker_precedes_v1(0, true, 0, false));
    EXPECT_TRUE(aicore_normal_aiv_worker_precedes_v1(1, false, 0, true));
    EXPECT_FALSE(aicore_normal_aiv_worker_precedes_v1(0, true, 1, false));
}

TEST(AicoreNormalDispatchPriorityV1, LowerLoadBreaksTiesWithinSameRole) {
    EXPECT_TRUE(aicore_normal_aiv_worker_precedes_v1(0, false, 1, false));
    EXPECT_FALSE(aicore_normal_aiv_worker_precedes_v1(1, true, 0, true));
}

TEST(AicoreNormalDispatchPriorityV1, FillsBothPeerSlotsBeforeResolver) {
    uint32_t peer_occupied = 0;
    uint32_t resolver_occupied = 0;
    bool peer_selected = aicore_normal_aiv_worker_precedes_v1(peer_occupied, false, resolver_occupied, true);
    ASSERT_TRUE(peer_selected);
    ++peer_occupied;

    peer_selected = aicore_normal_aiv_worker_precedes_v1(peer_occupied, false, resolver_occupied, true);
    ASSERT_TRUE(peer_selected);
    ++peer_occupied;

    EXPECT_EQ(peer_occupied, AICORE_PENDING_SLOT_COUNT_V1);
    EXPECT_EQ(resolver_occupied, 0u);
}

TEST(AicoreNormalDispatchPriorityV1, EmptyReadyDirectorySkipsAllSlotClaims) {
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(1, 1, 0, &layout));
    void *sidecar = std::aligned_alloc(AICORE_SIDECAR_ALIGNMENT_V1, layout.total_size);
    ASSERT_NE(sidecar, nullptr);
    ASSERT_TRUE(aicore_sidecar_init_v1(sidecar, layout));

    auto *run_control = aicore_sidecar_at_v1<AicoreRunControlV1>(sidecar, layout.run_control_offset);
    auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(sidecar, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < 3; ++worker) {
        contexts[worker].active = 1;
        contexts[worker].worker_index = worker;
        contexts[worker].core_type = worker == 0 ? static_cast<int32_t>(AicoreRootCoreTypeV0::AIC) :
                                                  static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
        contexts[worker].worker_contexts_offset = layout.worker_contexts_offset;
        contexts[worker].dispatch_slots_offset = layout.dispatch_slots_offset;
        contexts[worker].ready_inboxes_offset = layout.ready_inboxes_offset;
        contexts[worker].ready_directory_offset = layout.ready_directory_offset;
        contexts[worker].gang_coordinator_offset = layout.gang_coordinator_offset;
        contexts[worker].runtime_worker_count = 3;
        contexts[worker].graph_task_count = 1;
    }
    AicoreWorkerContextV1 &resolver = contexts[1];
    resolver.is_resolver = 1;
    resolver.inbox_index = 0;
    resolver.resolver_count = 7;
    resolver.cluster_worker_ids[0] = 0;
    resolver.cluster_worker_ids[1] = 1;
    resolver.cluster_worker_ids[2] = 2;
    run_control->resolver_count = resolver.resolver_count;
    for (uint64_t worker = 0; worker < 3; ++worker) {
        for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot) {
            aicore_initialize_free_slot_v1(aicore_dispatch_slot_at_v1(sidecar, &resolver, worker, slot));
        }
    }

    uint64_t victim_cursors[AICORE_CORE_TYPE_COUNT_V1]{1, 1};
    AicoreReadyStatsV1 stats{};
    AicoreReadonlyGraphV0 graph{};
    graph.task_count = 1;
    EXPECT_FALSE(aicore_fill_cluster_normal_slots_v1(
        graph, sidecar, &resolver, run_control, victim_cursors, &stats, false
    ));
    EXPECT_EQ(victim_cursors[0], 1u);
    EXPECT_EQ(victim_cursors[1], 1u);
    std::free(sidecar);
}
