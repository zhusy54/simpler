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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "aicore_execution_sidecar_v1.h"
#include "aicore_ticket_stream_planner.h"
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

class GraphBuffer {
public:
    explicit GraphBuffer(size_t task_count) :
        inline_completed_(task_count, 0) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        descriptors_ = std::make_unique<PTO2TaskDescriptor[]>(capacity_);
        payloads_ = std::make_unique<PTO2TaskPayload[]>(capacity_);
        for (size_t task_id = 0; task_id < capacity_; ++task_id) {
            descriptors_[task_id].task_id = PTO2TaskId::make(0, static_cast<int64_t>(task_id));
            for (int slot = 0; slot < 3; ++slot)
                descriptors_[task_id].kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task_id, AicoreRootCoreTypeV0 core_type, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task_id, inline_completed_.size());
        ASSERT_LE(fanins.size(), static_cast<size_t>(AICORE_GRAPH_MAX_FANIN_V0));
        descriptors_[task_id].kernel_id[core_type == AicoreRootCoreTypeV0::AIC ? 0 : 1] = 1;
        payloads_[task_id].fanin_count = static_cast<int32_t>(fanins.size());
        std::copy(fanins.begin(), fanins.end(), payloads_[task_id].fanin_local_ids);
    }

    void inline_completed(size_t task_id, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task_id, inline_completed_.size());
        ASSERT_LE(fanins.size(), static_cast<size_t>(AICORE_GRAPH_MAX_FANIN_V0));
        inline_completed_[task_id] = 1;
        payloads_[task_id].fanin_count = static_cast<int32_t>(fanins.size());
        std::copy(fanins.begin(), fanins.end(), payloads_[task_id].fanin_local_ids);
    }

    AicoreReadonlyGraphV0 graph() {
        return {
            reinterpret_cast<uint64_t>(descriptors_.get()),
            reinterpret_cast<uint64_t>(payloads_.get()),
            inline_completed_.size(),
            capacity_ - 1,
        };
    }

    const std::vector<uint8_t> &inline_flags() const { return inline_completed_; }

private:
    size_t capacity_{1};
    std::unique_ptr<PTO2TaskDescriptor[]> descriptors_;
    std::unique_ptr<PTO2TaskPayload[]> payloads_;
    std::vector<uint8_t> inline_completed_;
};

std::vector<uint32_t> bottom_levels(const std::vector<std::vector<int64_t>> &fanins) {
    std::vector<uint32_t> result(fanins.size(), 0);
    for (size_t consumer = fanins.size(); consumer-- > 0;) {
        for (int64_t producer : fanins[consumer])
            result[static_cast<size_t>(producer)] =
                std::max(result[static_cast<size_t>(producer)], result[consumer] + 1);
    }
    return result;
}

void sort_by_priority(std::vector<int64_t> *tasks, const std::vector<uint32_t> &bottom_level) {
    std::sort(tasks->begin(), tasks->end(), [&](int64_t lhs, int64_t rhs) {
        if (bottom_level[static_cast<size_t>(lhs)] != bottom_level[static_cast<size_t>(rhs)]) {
            return bottom_level[static_cast<size_t>(lhs)] > bottom_level[static_cast<size_t>(rhs)];
        }
        return lhs < rhs;
    });
}

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

TEST(AicoreTicketStreamPlanner, EmptyGraphProducesEmptyStreams) {
    GraphBuffer storage(0);
    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    EXPECT_TRUE(streams.aic.empty());
    EXPECT_TRUE(streams.aiv.empty());
    EXPECT_TRUE(streams.bottom_level.empty());
}

TEST(AicoreTicketStreamPlanner, CriticalPathOrderPreservesCoverageCoreTypeAndPriority) {
    GraphBuffer storage(8);
    storage.executable(0, AicoreRootCoreTypeV0::AIC);
    storage.executable(1, AicoreRootCoreTypeV0::AIV, {0});
    storage.executable(2, AicoreRootCoreTypeV0::AIC, {1});
    storage.executable(3, AicoreRootCoreTypeV0::AIV, {2});
    storage.executable(4, AicoreRootCoreTypeV0::AIC);
    storage.executable(5, AicoreRootCoreTypeV0::AIV, {4});
    storage.executable(6, AicoreRootCoreTypeV0::AIC, {5});
    storage.executable(7, AicoreRootCoreTypeV0::AIV, {6});

    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    EXPECT_EQ(streams.aic, (std::vector<uint32_t>{0, 4, 2, 6}));
    EXPECT_EQ(streams.aiv, (std::vector<uint32_t>{1, 5, 3, 7}));
    EXPECT_EQ(streams.bottom_level, (std::vector<uint32_t>{3, 2, 1, 0, 3, 2, 1, 0}));
}

TEST(AicoreTicketStreamPlanner, UnevenChainsPrioritizeLongerRemainingPath) {
    GraphBuffer storage(8);
    storage.executable(0, AicoreRootCoreTypeV0::AIC);
    storage.executable(1, AicoreRootCoreTypeV0::AIV, {0});
    storage.executable(2, AicoreRootCoreTypeV0::AIC);
    storage.executable(3, AicoreRootCoreTypeV0::AIV, {2});
    storage.executable(4, AicoreRootCoreTypeV0::AIC, {3});
    storage.executable(5, AicoreRootCoreTypeV0::AIV, {4});
    storage.executable(6, AicoreRootCoreTypeV0::AIC, {5});
    storage.executable(7, AicoreRootCoreTypeV0::AIV, {6});

    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    ASSERT_FALSE(streams.aic.empty());
    EXPECT_EQ(streams.aic.front(), 2u);
    EXPECT_GT(streams.bottom_level[2], streams.bottom_level[0]);
}

TEST(AicoreTicketStreamPlanner, InlineCompletedTasksContributePriorityButAreNotStreamed) {
    GraphBuffer storage(5);
    storage.inline_completed(0);
    storage.executable(1, AicoreRootCoreTypeV0::AIC);
    storage.executable(2, AicoreRootCoreTypeV0::AIV, {1});
    storage.executable(3, AicoreRootCoreTypeV0::AIC, {2});
    storage.executable(4, AicoreRootCoreTypeV0::AIV, {0, 2, 3});

    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    EXPECT_EQ(streams.aic, (std::vector<uint32_t>{1, 3}));
    EXPECT_EQ(streams.aiv, (std::vector<uint32_t>{2, 4}));
    EXPECT_EQ(std::count(streams.aic.begin(), streams.aic.end(), 0), 0);
    EXPECT_EQ(std::count(streams.aiv.begin(), streams.aiv.end(), 0), 0);
    EXPECT_GT(streams.bottom_level[0], streams.bottom_level[4]);
}

TEST(AicoreTicketStreamPlanner, PagedAttentionSeedsSpanIndependentBatches) {
    constexpr size_t kBatchCount = 64;
    constexpr size_t kTasksPerBatch = 5;
    GraphBuffer storage(kBatchCount * kTasksPerBatch);
    for (size_t batch = 0; batch < kBatchCount; ++batch) {
        const int32_t base = static_cast<int32_t>(batch * kTasksPerBatch);
        storage.inline_completed(base);
        storage.executable(base + 1, AicoreRootCoreTypeV0::AIC);
        storage.executable(base + 2, AicoreRootCoreTypeV0::AIV, {base + 1});
        storage.executable(base + 3, AicoreRootCoreTypeV0::AIC, {base + 2});
        storage.executable(base + 4, AicoreRootCoreTypeV0::AIV, {base, base + 2, base + 3});
    }

    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    ASSERT_GE(streams.aic.size(), 28u);
    ASSERT_GE(streams.aiv.size(), 56u);
    for (size_t rank = 0; rank < 28; ++rank) {
        EXPECT_EQ(streams.aic[rank] % kTasksPerBatch, 1u);
        EXPECT_EQ(streams.aic[rank] / kTasksPerBatch, rank);
    }
    for (size_t rank = 0; rank < 56; ++rank) {
        EXPECT_EQ(streams.aiv[rank] % kTasksPerBatch, 2u);
        EXPECT_EQ(streams.aiv[rank] / kTasksPerBatch, rank);
    }
}

TEST(AicoreTicketStreamPlanner, RejectsInvalidInlineClassificationAndBitmapSize) {
    GraphBuffer storage(1);
    storage.executable(0, AicoreRootCoreTypeV0::AIC);
    AicoreTicketStreams streams;
    EXPECT_FALSE(build_aicore_ticket_streams(storage.graph(), {}, &streams));
    EXPECT_FALSE(build_aicore_ticket_streams(storage.graph(), {1}, &streams));
}

TEST(AicoreTicketStreamPlanner, SupportsMaximumLegalFanin) {
    constexpr int32_t kRootCount = AICORE_GRAPH_MAX_FANIN_V0;
    GraphBuffer storage(static_cast<size_t>(kRootCount) + 1);
    std::vector<int32_t> roots(static_cast<size_t>(kRootCount));
    std::iota(roots.begin(), roots.end(), 0);
    for (int32_t task_id : roots) {
        storage.executable(
            static_cast<size_t>(task_id), (task_id & 1) == 0 ? AicoreRootCoreTypeV0::AIC : AicoreRootCoreTypeV0::AIV
        );
    }
    storage.executable(static_cast<size_t>(kRootCount), AicoreRootCoreTypeV0::AIC, roots);

    AicoreTicketStreams streams;
    ASSERT_TRUE(build_aicore_ticket_streams(storage.graph(), storage.inline_flags(), &streams));
    EXPECT_EQ(streams.aic.size() + streams.aiv.size(), static_cast<size_t>(kRootCount) + 1);
    for (int32_t task_id : roots)
        EXPECT_EQ(streams.bottom_level[static_cast<size_t>(task_id)], 1u);
    EXPECT_EQ(streams.bottom_level[static_cast<size_t>(kRootCount)], 0u);
}

TEST(AicoreTicketModelV1, CrossTypeDagCompletesExactlyOnceWithTwoPendingSlots) {
    std::vector<std::vector<int64_t>> fanins{{}, {}, {0}, {1}, {2, 3}, {4}};
    std::vector<uint32_t> bottom_level = bottom_levels(fanins);
    AicoreTicketModelV1 model(std::move(fanins), std::move(bottom_level), {0, 2, 4}, {1, 3, 5}, 2, 2);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    EXPECT_EQ(model.execution_count(), (std::vector<int>{1, 1, 1, 1, 1, 1}));
}

TEST(AicoreTicketModelV1, InlineCompletedTasksAreNotClaimed) {
    std::vector<std::vector<int64_t>> fanins{{}, {0}, {1}};
    AicoreTicketModelV1 model(fanins, bottom_levels(fanins), {2}, {1}, 2, 2);
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
    std::vector<uint32_t> bottom_level = bottom_levels(fanins);
    sort_by_priority(&aic, bottom_level);
    sort_by_priority(&aiv, bottom_level);
    AicoreTicketModelV1 model(std::move(fanins), std::move(bottom_level), std::move(aic), std::move(aiv), 8, 16);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    for (int count : model.execution_count())
        EXPECT_EQ(count, 1);
}

TEST(AicoreTicketModelV1, NonMonotonicTaskIdsCompleteExactlyOnce) {
    std::vector<std::vector<int64_t>> fanins{{}, {0}, {1}, {}, {3}, {4}, {5}, {6}};
    std::vector<uint32_t> bottom_level = bottom_levels(fanins);
    std::vector<int64_t> aic{0, 2, 3, 5, 7};
    std::vector<int64_t> aiv{1, 4, 6};
    sort_by_priority(&aic, bottom_level);
    sort_by_priority(&aiv, bottom_level);
    ASSERT_FALSE(std::is_sorted(aic.begin(), aic.end()));
    AicoreTicketModelV1 model(fanins, bottom_level, aic, aiv, 2, 2);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    EXPECT_EQ(model.execution_count(), (std::vector<int>{1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(AicoreTicketModelV1, CrossTypeAlternatingChainDoesNotDeadlock) {
    constexpr int64_t kTaskCount = 64;
    std::vector<std::vector<int64_t>> fanins(kTaskCount);
    std::vector<int64_t> aic;
    std::vector<int64_t> aiv;
    for (int64_t task = 0; task < kTaskCount; ++task) {
        if (task != 0) fanins[task].push_back(task - 1);
        ((task & 1) == 0 ? aic : aiv).push_back(task);
    }
    std::vector<uint32_t> bottom_level = bottom_levels(fanins);
    AicoreTicketModelV1 model(fanins, bottom_level, aic, aiv, 8, 16);
    ASSERT_TRUE(model.valid());
    ASSERT_TRUE(model.run());
    for (int count : model.execution_count())
        EXPECT_EQ(count, 1);
}

TEST(AicoreTicketModelV1, RejectsReverseEdgeAndMissingCoreType) {
    AicoreTicketModelV1 reverse_edge({{1}, {}}, {1, 0}, {0, 1}, {}, 1, 0);
    EXPECT_FALSE(reverse_edge.valid());
    AicoreTicketModelV1 missing_aiv({{}, {}}, {0, 0}, {0}, {1}, 1, 0);
    EXPECT_FALSE(missing_aiv.valid());
}

TEST(AicoreTicketModelV1, RejectsNonMonotonicPriorityAndMisorderedStream) {
    AicoreTicketModelV1 nonmonotonic_priority({{}, {0}}, {0, 0}, {0}, {1}, 1, 1);
    EXPECT_FALSE(nonmonotonic_priority.valid());
    AicoreTicketModelV1 misordered_stream({{}, {}, {0}}, {1, 0, 0}, {2, 0}, {1}, 1, 1);
    EXPECT_FALSE(misordered_stream.valid());
}

}  // namespace
