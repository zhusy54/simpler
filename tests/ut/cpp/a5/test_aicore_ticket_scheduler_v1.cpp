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
#include <atomic>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "aicore_execution_sidecar_v1.h"
#include "aicore_ticket_model_v1.h"
#include "aicore_ticket_scheduler_v1.h"
#include "callable.h"
#include "pto_runtime2_types.h"

namespace {

static_assert(sizeof(PTO2TaskDescriptor) == AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0);
static_assert(sizeof(PTO2TaskPayload) == AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0);
static_assert(offsetof(PTO2TaskDescriptor, kernel_id) == AICORE_GRAPH_KERNEL_IDS_OFFSET_V0);
static_assert(offsetof(PTO2TaskPayload, fanin_count) == AICORE_GRAPH_FANIN_COUNT_OFFSET_V0);
static_assert(offsetof(CoreCallable, resolved_addr_) == AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0);

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

TEST(AicoreSidecarV1, PlansOnlyTypedStreamsCompletionCellsAndTrace) {
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(5, 3, 2, &layout));
    EXPECT_EQ(layout.task_count, 5u);
    EXPECT_EQ(layout.aic_task_count, 3u);
    EXPECT_EQ(layout.aiv_task_count, 2u);
    EXPECT_EQ(layout.total_size % AICORE_SIDECAR_ALIGNMENT_V1, 0u);
    EXPECT_EQ(layout.completion_cells_offset % alignof(AicoreTaskCompletionCellV1), 0u);
    EXPECT_EQ(layout.aic_stream_offset % alignof(AicoreTaskStreamV1), 0u);
    EXPECT_EQ(layout.aiv_stream_offset % alignof(AicoreTaskStreamV1), 0u);
    EXPECT_EQ(layout.trace_cells_offset % alignof(AicoreTaskTraceCellV1), 0u);

    SidecarBuffer storage(layout);
    auto *completion = aicore_sidecar_at_v1<AicoreTaskCompletionCellV1>(storage.base(), layout.completion_cells_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(completion[task].completion, static_cast<int64_t>(AicoreTaskCompletionV1::NOT_DONE));
        if (task != 0) {
            EXPECT_EQ(
                reinterpret_cast<uintptr_t>(&completion[task]) - reinterpret_cast<uintptr_t>(&completion[task - 1]),
                128u
            );
        }
    }
}

TEST(AicoreSidecarV1, RejectsInvalidCountsAndOverflow) {
    AicoreExecutionSidecarLayoutV1 layout{};
    EXPECT_FALSE(aicore_sidecar_plan_v1(1, 1, 1, &layout));
    EXPECT_FALSE(aicore_sidecar_plan_v1(UINT64_MAX, UINT64_MAX, 0, &layout));
    uint64_t value = 0;
    EXPECT_FALSE(aicore_sidecar_checked_add_v1(UINT64_MAX, 1, &value));
    EXPECT_FALSE(aicore_sidecar_checked_mul_v1(UINT64_MAX, 2, &value));
    EXPECT_FALSE(aicore_sidecar_checked_align_v1(UINT64_MAX, 128, &value));
}

TEST(AicoreTicketSchedulerV1, TicketClaimsAreUniqueAndExhaustOncePerWorker) {
    constexpr uint64_t kTaskCount = 4096;
    constexpr int kWorkers = 8;
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(kTaskCount, kTaskCount, 0, &layout));
    SidecarBuffer storage(layout);
    auto *stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(storage.base(), layout.aic_stream_offset);
    auto *task_ids = aicore_sidecar_at_v1<uint32_t>(storage.base(), layout.aic_task_ids_offset);
    for (uint64_t task = 0; task < kTaskCount; ++task)
        task_ids[task] = static_cast<uint32_t>(task);

    std::vector<std::atomic<int>> seen(kTaskCount);
    std::vector<std::thread> workers;
    std::atomic<int> exhaustion_count{0};
    for (int worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                uint64_t index = 0;
                int64_t task_id = AICORE_TASK_ID_INVALID_V1;
                if (!aicore_claim_ticket_v1(storage.base(), stream, &index, &task_id)) {
                    exhaustion_count.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                seen[task_id].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread &worker : workers)
        worker.join();

    EXPECT_EQ(exhaustion_count.load(), kWorkers);
    EXPECT_GE(stream->next_index, kTaskCount);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1);
}

TEST(AicoreTicketSchedulerV1, ReadinessRemembersCompletedFaninPrefix) {
    alignas(64) PTO2TaskDescriptor descriptors[3]{};
    alignas(64) PTO2TaskPayload payloads[3]{};
    for (int64_t task = 0; task < 3; ++task) {
        descriptors[task].task_id = PTO2TaskId::make(0, task);
        descriptors[task].kernel_id[0] = 1;
        descriptors[task].kernel_id[1] = INVALID_KERNEL_ID;
        descriptors[task].kernel_id[2] = INVALID_KERNEL_ID;
    }
    payloads[2].fanin_count = 2;
    payloads[2].fanin_local_ids[0] = 0;
    payloads[2].fanin_local_ids[1] = 1;
    AicoreReadonlyGraphV0 graph{reinterpret_cast<uint64_t>(descriptors), reinterpret_cast<uint64_t>(payloads), 3, 3};
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(3, 3, 0, &layout));
    SidecarBuffer storage(layout);
    AicoreWorkerContextV1 context{};
    context.completion_cells_offset = layout.completion_cells_offset;
    auto *completion = aicore_sidecar_at_v1<AicoreTaskCompletionCellV1>(storage.base(), layout.completion_cells_offset);
    completion[0].completion = static_cast<int64_t>(AicoreTaskCompletionV1::DONE);
    AicorePendingSlotV1 pending{};
    ASSERT_EQ(
        aicore_pending_initialize_v1(graph, 2, 2, AicoreClaimKindV1::TICKET, 10, 11, &pending), AicoreRootStatusV0::OK
    );

    uint64_t loads = 0;
    EXPECT_EQ(
        aicore_pending_readiness_v1(graph, storage.base(), &context, &pending, &loads), AicorePendingStateV1::BLOCKED
    );
    EXPECT_EQ(pending.next_fanin_index, 1);
    EXPECT_EQ(pending.waiting_producer, 1);
    EXPECT_EQ(loads, 2u);
    EXPECT_EQ(
        aicore_pending_readiness_v1(graph, storage.base(), &context, &pending, &loads), AicorePendingStateV1::BLOCKED
    );
    EXPECT_EQ(loads, 3u);
    completion[1].completion = static_cast<int64_t>(AicoreTaskCompletionV1::DONE);
    EXPECT_EQ(
        aicore_pending_readiness_v1(graph, storage.base(), &context, &pending, &loads), AicorePendingStateV1::READY
    );
    EXPECT_EQ(loads, 4u);
}

TEST(AicoreTicketModelV1, CrossTypeDagCompletesExactlyOnceWithTwoPendingSlots) {
    std::vector<std::vector<int64_t>> fanins{{}, {}, {0}, {1}, {2, 3}, {4}};
    AicoreTicketModelV1 model(std::move(fanins), {0, 2, 4}, {1, 3, 5}, 2, 2);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    EXPECT_EQ(model.execution_count(), (std::vector<int>{1, 1, 1, 1, 1, 1}));
}

TEST(AicoreTicketModelV1, InlineCompletedTasksAreNotClaimed) {
    AicoreTicketModelV1 model({{}, {0}, {1}}, {2}, {1}, 2, 2);
    model.mark_inline_completed(0);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    EXPECT_EQ(model.execution_count(), (std::vector<int>{0, 1, 1}));
}

TEST(AicoreTicketModelV1, RandomTopologicalDagCompletesExactlyOnce) {
    constexpr int64_t kTaskCount = 128;
    std::vector<std::vector<int64_t>> fanins(kTaskCount);
    std::vector<int64_t> aic;
    std::vector<int64_t> aiv;
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t task = 0; task < kTaskCount; ++task) {
        ((task & 1) == 0 ? aic : aiv).push_back(task);
        if (task == 0) continue;
        random = random * UINT64_C(6364136223846793005) + 1;
        int64_t edge_count = static_cast<int64_t>(random % 5);
        for (int64_t edge = 0; edge < edge_count; ++edge) {
            random = random * UINT64_C(6364136223846793005) + 1;
            int64_t producer = static_cast<int64_t>(random % static_cast<uint64_t>(task));
            if (std::find(fanins[task].begin(), fanins[task].end(), producer) == fanins[task].end()) {
                fanins[task].push_back(producer);
            }
        }
    }
    AicoreTicketModelV1 model(std::move(fanins), std::move(aic), std::move(aiv), 8, 16);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    for (int count : model.execution_count())
        EXPECT_EQ(count, 1);
}

TEST(AicoreTicketModelV1, RejectsReverseEdgeAndMissingCoreType) {
    AicoreTicketModelV1 reverse_edge({{1}, {}}, {0, 1}, {}, 1, 0);
    EXPECT_FALSE(reverse_edge.valid());
    AicoreTicketModelV1 missing_aiv({{}, {}}, {0}, {1}, 1, 0);
    EXPECT_FALSE(missing_aiv.valid());
}

}  // namespace
