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
#include <cstdlib>
#include <thread>
#include <vector>

#include "aicore_execution_sidecar_v0.h"
#include "aicore_gm_atomic.h"
#include "aicore_ready_queue_v0.h"

namespace {

class SidecarBuffer {
public:
    explicit SidecarBuffer(const AicoreExecutionSidecarLayoutV0 &layout) :
        layout_(layout),
        base_(std::aligned_alloc(AICORE_SIDECAR_ALIGNMENT_V0, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(aicore_sidecar_init_v0(base_, layout_));
    }

    ~SidecarBuffer() { std::free(base_); }

    void *base() const { return base_; }

private:
    AicoreExecutionSidecarLayoutV0 layout_{};
    void *base_{nullptr};
};

TEST(AicoreSidecarV0, PlansAlignedGraphSizedRegions) {
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(5, 3, 2, &layout));
    EXPECT_EQ(layout.task_count, 5u);
    EXPECT_EQ(layout.aic_queue_capacity, 4u);
    EXPECT_EQ(layout.aiv_queue_capacity, 2u);
    EXPECT_EQ(layout.total_size % AICORE_SIDECAR_ALIGNMENT_V0, 0u);
    EXPECT_EQ(layout.task_controls_offset % alignof(AicoreTaskControlV0), 0u);
    EXPECT_EQ(layout.aic_queue_offset % alignof(AicoreReadyQueueV0), 0u);
    EXPECT_EQ(layout.aiv_queue_offset % alignof(AicoreReadyQueueV0), 0u);

    SidecarBuffer storage(layout);
    auto *controls = aicore_sidecar_at_v0<AicoreTaskControlV0>(storage.base(), layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        EXPECT_EQ(controls[i].completion, 0);
        EXPECT_EQ(controls[i].wake_list_head, AICORE_TASK_ID_INVALID_V0);
        EXPECT_EQ(controls[i].next_waiter, AICORE_TASK_ID_INVALID_V0);
        if (i != 0) {
            EXPECT_EQ(reinterpret_cast<uintptr_t>(&controls[i]) - reinterpret_cast<uintptr_t>(&controls[i - 1]), 128u);
        }
    }
}

TEST(AicoreSidecarV0, RejectsInvalidCountsAndOverflow) {
    AicoreExecutionSidecarLayoutV0 layout{};
    EXPECT_FALSE(aicore_sidecar_plan_v0(1, 1, 1, &layout));
    EXPECT_FALSE(aicore_sidecar_plan_v0(UINT64_MAX, UINT64_MAX, 0, &layout));

    uint64_t value = 0;
    EXPECT_FALSE(aicore_sidecar_checked_add_v0(UINT64_MAX, 1, &value));
    EXPECT_FALSE(aicore_sidecar_checked_mul_v0(UINT64_MAX, 2, &value));
    EXPECT_FALSE(aicore_sidecar_checked_align_v0(UINT64_MAX, 128, &value));
}

TEST(AicoreSidecarV0, TaskControlsKeepDcciFieldOffAtomicLineAndNeighbors) {
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(3, 3, 0, &layout));
    SidecarBuffer storage(layout);
    auto *controls = aicore_sidecar_at_v0<AicoreTaskControlV0>(storage.base(), layout.task_controls_offset);

    aicore_gm_store_v0(controls[0].completion, 11);
    EXPECT_EQ(aicore_gm_exchange_v0(controls[0].wake_list_head, 7), AICORE_TASK_ID_INVALID_V0);
    aicore_publish_next_waiter_v0(&controls[0], 9);
    aicore_gm_store_v0(controls[1].completion, 22);

    EXPECT_EQ(aicore_gm_load_v0(controls[0].completion), 11);
    EXPECT_EQ(aicore_gm_load_v0(controls[0].wake_list_head), 7);
    EXPECT_EQ(aicore_observe_next_waiter_v0(&controls[0]), 9);
    EXPECT_EQ(aicore_gm_load_v0(controls[1].completion), 22);
    EXPECT_EQ(controls[1].wake_list_head, AICORE_TASK_ID_INVALID_V0);
    EXPECT_EQ(controls[2].next_waiter, AICORE_TASK_ID_INVALID_V0);
}

TEST(AicoreSidecarV0, TaskIdQueueWrapsWithoutLoss) {
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(4, 4, 0, &layout));
    SidecarBuffer storage(layout);
    auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);

    for (int64_t round = 0; round < 20; ++round) {
        for (int64_t i = 0; i < 4; ++i)
            ASSERT_TRUE(aicore_ready_queue_push_v0(storage.base(), queue, round * 4 + i));
        EXPECT_FALSE(aicore_ready_queue_push_v0(storage.base(), queue, 1000));
        for (int64_t i = 0; i < 4; ++i) {
            int64_t task_id = AICORE_TASK_ID_INVALID_V0;
            ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), queue, &task_id));
            EXPECT_EQ(task_id, round * 4 + i);
        }
        int64_t task_id = AICORE_TASK_ID_INVALID_V0;
        EXPECT_FALSE(aicore_ready_queue_pop_v0(storage.base(), queue, &task_id));
    }
}

TEST(AicoreSidecarV0, TaskIdQueueMpmcIsExactlyOnce) {
    constexpr int64_t kTasks = 4096;
    constexpr int kProducers = 4;
    constexpr int kConsumers = 4;
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(kTasks, kTasks, 0, &layout));
    SidecarBuffer storage(layout);
    auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);

    std::vector<std::atomic<int>> seen(kTasks);
    for (auto &count : seen)
        count.store(0, std::memory_order_relaxed);
    std::atomic<int64_t> consumed{0};
    std::vector<std::thread> threads;
    for (int producer = 0; producer < kProducers; ++producer) {
        threads.emplace_back([&, producer] {
            for (int64_t task_id = producer; task_id < kTasks; task_id += kProducers) {
                while (!aicore_ready_queue_push_v0(storage.base(), queue, task_id))
                    std::this_thread::yield();
            }
        });
    }
    for (int consumer = 0; consumer < kConsumers; ++consumer) {
        threads.emplace_back([&] {
            while (consumed.load(std::memory_order_acquire) < kTasks) {
                int64_t task_id = AICORE_TASK_ID_INVALID_V0;
                if (!aicore_ready_queue_pop_v0(storage.base(), queue, &task_id)) {
                    std::this_thread::yield();
                    continue;
                }
                ASSERT_GE(task_id, 0);
                ASSERT_LT(task_id, kTasks);
                seen[task_id].fetch_add(1, std::memory_order_relaxed);
                consumed.fetch_add(1, std::memory_order_release);
            }
        });
    }
    for (std::thread &thread : threads)
        thread.join();

    EXPECT_EQ(consumed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1);
}

}  // namespace
