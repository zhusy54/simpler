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
#include <thread>
#include <memory>
#include <vector>

#include "scheduler/scheduler.h"
#include "aicore_completion_mailbox.h"
#include "scheduler/scheduler_context.h"
#include "utils/queue_peaks.h"

namespace {
class QueuePeaksTest : public ::testing::Test {
protected:
    static constexpr uint64_t capacity = 16;
    ChipReadyQueue queue{};
    ChipReadyQueueSlot slots[capacity];
    ChipTaskSlotState tasks[capacity];
    simpler::queue_peaks::Workspace workspace;
    void SetUp() override {
        queue.slots = slots;
        queue.capacity = capacity;
        queue.mask = capacity - 1;
        queue.enqueue_pos.store(0);
        queue.dequeue_pos.store(0);
        for (uint64_t i = 0; i < capacity; ++i) {
            slots[i].sequence.store(i);
            tasks[i].logical_block_num = (i % 2 == 0) ? 1 : 32;
        }
        workspace.bind(queue, simpler::queue_peaks::Kind::Ready, 0, true);
    }
    void TearDown() override { workspace.reset(); }
};

TEST_F(QueuePeaksTest, FullFailedPushAndBatchWrap) {
    ChipTaskSlotState *input[capacity];
    ChipTaskSlotState *out[capacity];
    for (uint64_t i = 0; i < capacity; ++i)
        input[i] = &tasks[i];
    for (int lap = 0; lap < 4; ++lap) {
        ASSERT_TRUE(queue.push_batch(input, capacity));
        EXPECT_FALSE(queue.push(&tasks[0]));
        EXPECT_EQ(queue.queue_peaks->peak.load(), capacity);
        EXPECT_EQ(queue.queue_peaks->spmd_peak.load(), capacity / 2);
        EXPECT_EQ(queue.queue_peaks->normal_peak.load(), capacity / 2);
        ASSERT_EQ(queue.pop_batch(out, capacity), capacity);
        EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
        EXPECT_EQ(queue.pop(), nullptr);
        EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
    }
}

TEST_F(QueuePeaksTest, IndependentPeaksAreNotAdded) {
    for (int phase = 0; phase < 2; ++phase) {
        for (int i = 0; i < 6; ++i)
            ASSERT_TRUE(queue.push(&tasks[phase]));
        for (int i = 0; i < 6; ++i)
            ASSERT_NE(queue.pop(), nullptr);
    }
    EXPECT_EQ(queue.queue_peaks->peak.load(), 6u);
    EXPECT_EQ(queue.queue_peaks->normal_peak.load(), 6u);
    EXPECT_EQ(queue.queue_peaks->spmd_peak.load(), 6u);
}

TEST_F(QueuePeaksTest, ClassificationSurvivesTaskReuseAndTaggedRepush) {
    const auto tag = TaskId::invalid();
    ASSERT_TRUE(queue.push_tagged(&tasks[1], tag));
    tasks[1].logical_block_num = 1;
    uint8_t saved = 0;
    TaskId out_tag;
    auto *out = queue.pop_tagged(&out_tag, &saved);
    EXPECT_EQ(saved, 1u);
    EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
    ASSERT_TRUE(queue.push_tagged(out, out_tag, &saved));
    EXPECT_EQ(queue.queue_peaks->current.load(), uint64_t{1} << 32);
    ASSERT_EQ(queue.pop(), out);
    EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
}

TEST_F(QueuePeaksTest, ConcurrentMixedBatchesBalanceWithoutExceedingCapacity) {
    constexpr int per_producer = 4000;
    std::atomic<int> consumed{0};
    std::vector<std::thread> threads;
    for (int p = 0; p < 3; ++p)
        threads.emplace_back([&, p] {
            ChipTaskSlotState *items[] = {&tasks[2 * p], &tasks[2 * p + 1]};
            for (int i = 0; i < per_producer; i += 2) {
                while (!queue.push_batch(items, 2)) {}
            }
        });
    for (int c = 0; c < 3; ++c)
        threads.emplace_back([&] {
            ChipTaskSlotState *items[3];
            while (consumed.load() < 3 * per_producer) {
                consumed.fetch_add(queue.pop_batch(items, 3));
            }
        });
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(consumed.load(), 3 * per_producer);
    EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
    EXPECT_LE(queue.queue_peaks->peak.load(), capacity);
    EXPECT_LE(queue.queue_peaks->spmd_peak.load(), queue.queue_peaks->peak.load());
    EXPECT_LE(queue.queue_peaks->normal_peak.load(), queue.queue_peaks->peak.load());
}

TEST_F(QueuePeaksTest, SnapshotAndReset) {
    ASSERT_TRUE(queue.push(&tasks[1]));
    ASSERT_NE(queue.pop(), nullptr);
    simpler::queue_peaks::Report report{};
    workspace.snapshot(report);
    ASSERT_EQ(report.count, 1u);
    EXPECT_EQ(report.records[0].peak, 1u);
    EXPECT_EQ(report.records[0].spmd_peak, 1u);
    workspace.reset();
    workspace.bind(queue, simpler::queue_peaks::Kind::Ready, 0, true);
    EXPECT_EQ(queue.queue_peaks->peak.load(), 0u);
    EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
}

TEST(QueuePeakMailboxTest, FullWrapBothMessageKinds) {
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    mailbox->init_empty();
    simpler::queue_peaks::Counters counters;
    mailbox->queue_peaks = &counters;
    AICoreCompletionMsgView out;
    for (int lap = 0; lap < 2; ++lap) {
        for (uint32_t i = 0; i < AICORE_COMPLETION_MAILBOX_CAPACITY; ++i) {
            ASSERT_TRUE(
                i % 2 ? mailbox->try_push_normal_done(TaskId::invalid(), 0) :
                        mailbox->try_push_condition(TaskId::invalid(), 0, 1, 0, 0)
            );
        }
        EXPECT_FALSE(mailbox->try_push_normal_done(TaskId::invalid(), 0));
        EXPECT_EQ(counters.peak.load(), AICORE_COMPLETION_MAILBOX_CAPACITY);
        for (uint32_t i = 0; i < AICORE_COMPLETION_MAILBOX_CAPACITY; ++i)
            ASSERT_TRUE(mailbox->try_pop(out));
        EXPECT_FALSE(mailbox->try_pop(out));
        EXPECT_EQ(counters.current.load(), 0u);
    }
}

TEST(QueuePeakCompletionTest, IndependentQueueAndReset) {
    CompletedTaskQueue queue;
    queue.init(8);
    simpler::queue_peaks::Counters counters;
    queue.queue_peaks = &counters;
    ChipTaskSlotState task;
    for (int lap = 0; lap < 3; ++lap) {
        for (int i = 0; i < 8; ++i)
            queue.push(&task);
        EXPECT_EQ(counters.peak.load(), 8u);
        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(queue.pop(), &task);
        EXPECT_EQ(counters.current.load(), 0u);
    }
    queue.destroy();
}

#if SIMPLER_SCHED_PROFILING
TEST_F(QueuePeaksTest, ProfiledOverloadsAccountExactlyOnce) {
    uint64_t atomics = 0, wait = 0;
    ASSERT_TRUE(queue.push(&tasks[1], atomics, wait));
    ASSERT_TRUE(queue.push(&tasks[0], atomics, wait));
    EXPECT_EQ(queue.pop(atomics, wait), &tasks[1]);
    ChipTaskSlotState *out[2];
    EXPECT_EQ(queue.pop_batch(out, 2, atomics, wait), 1);
    EXPECT_EQ(queue.queue_peaks->current.load(), 0u);
    EXPECT_EQ(queue.queue_peaks->peak.load(), 2u);
    EXPECT_EQ(queue.queue_peaks->spmd_peak.load(), 1u);
    EXPECT_EQ(queue.queue_peaks->normal_peak.load(), 1u);
}
#endif

}  // namespace
