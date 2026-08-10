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
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "aicore_execution_sidecar_v0.h"
#include "aicore_dependency_model_v0.h"
#include "aicore_dependency_scheduler_v0.h"
#include "aicore_graph_view_v0.h"
#include "aicore_gm_atomic.h"
#include "aicore_ready_queue_v0.h"
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
    EXPECT_EQ(layout.completion_queue_capacity, 8u);
    EXPECT_EQ(layout.total_size % AICORE_SIDECAR_ALIGNMENT_V0, 0u);
    EXPECT_EQ(layout.task_controls_offset % alignof(AicoreTaskControlV0), 0u);
    EXPECT_EQ(layout.aic_queue_offset % alignof(AicoreReadyQueueV0), 0u);
    EXPECT_EQ(layout.aiv_queue_offset % alignof(AicoreReadyQueueV0), 0u);
    EXPECT_EQ(layout.completion_queue_offset % alignof(AicoreReadyQueueV0), 0u);

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

TEST(AicoreSidecarV0, TaskIdQueueHonorsVyukovSequenceInvariant) {
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(4, 4, 0, &layout));
    SidecarBuffer storage(layout);
    auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);
    auto *slots = aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(storage.base(), layout.aic_queue_slots_offset);

    // Push publishes sequence == pos + 1 after writing task_id; a full queue rejects further pushes.
    for (int64_t i = 0; i < 4; ++i)
        ASSERT_TRUE(aicore_ready_queue_push_v0(storage.base(), queue, i));
    EXPECT_FALSE(aicore_ready_queue_push_v0(storage.base(), queue, 1000));
    for (uint64_t i = 0; i < layout.aic_queue_capacity; ++i) {
        EXPECT_EQ(slots[i].sequence, static_cast<int64_t>(i + 1));
        EXPECT_EQ(slots[i].task_id, static_cast<int64_t>(i));
    }

    // Pop publishes sequence == pos + capacity, handing the slot back to the next round.
    int64_t task_id = AICORE_TASK_ID_INVALID_V0;
    for (int64_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), queue, &task_id));
        EXPECT_EQ(task_id, i);
    }
    for (uint64_t i = 0; i < layout.aic_queue_capacity; ++i)
        EXPECT_EQ(slots[i].sequence, static_cast<int64_t>(i + layout.aic_queue_capacity));

    // A second round wraps the cursor; sequence advances by another capacity step and stays monotonic.
    for (int64_t i = 0; i < 4; ++i)
        ASSERT_TRUE(aicore_ready_queue_push_v0(storage.base(), queue, 10 + i));
    for (uint64_t i = 0; i < layout.aic_queue_capacity; ++i) {
        EXPECT_EQ(slots[i].sequence, static_cast<int64_t>(i + layout.aic_queue_capacity + 1));
        EXPECT_EQ(slots[i].task_id, static_cast<int64_t>(10 + i));
    }
}

TEST(AicoreSidecarV0, TaskIdQueueLayoutDivergesFromHbgPointerSlot) {
    // The v0 slot is a pure task-id slot: two int64 words and no device pointer.
    // HBG's PTO2ReadyQueueSlot is a pointer-slot layout
    //   {atomic<int64> sequence; PTO2TaskSlotState* slot_state; uint64 task_id_snapshot};
    // the v0 queue borrows only the Vyukov algorithm from HBG, not that byte layout.
    static_assert(sizeof(AicoreReadyQueueSlotV0) == 16, "task-id slot must be exactly two int64 words");
    static_assert(offsetof(AicoreReadyQueueSlotV0, sequence) == 0, "sequence leads the slot");
    static_assert(offsetof(AicoreReadyQueueSlotV0, task_id) == 8, "task_id follows sequence");
    static_assert(std::is_standard_layout_v<AicoreReadyQueueSlotV0>, "slot must remain standard-layout");
    static_assert(std::is_trivially_copyable_v<AicoreReadyQueueSlotV0>, "slot must remain trivially copyable");

    EXPECT_EQ(sizeof(AicoreReadyQueueSlotV0), 2 * sizeof(int64_t));
    EXPECT_EQ(alignof(AicoreReadyQueueSlotV0), 16u);
}

TEST(AicoreSidecarV0, AdjacentQueueSlotsDoNotClobber) {
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(4, 4, 0, &layout));
    SidecarBuffer storage(layout);
    auto *slots = aicore_sidecar_at_v0<AicoreReadyQueueSlotV0>(storage.base(), layout.aic_queue_slots_offset);

    // Init state: slot[i].sequence == i, slot[i].task_id == INVALID.
    EXPECT_EQ(aicore_gm_exchange_v0(slots[0].sequence, 100), 0);
    aicore_gm_store_v0(slots[1].task_id, 7);
    aicore_gm_store_v0(slots[layout.aic_queue_capacity - 1].sequence, 200);

    EXPECT_EQ(slots[0].sequence, 100);
    EXPECT_EQ(slots[0].task_id, AICORE_TASK_ID_INVALID_V0);
    EXPECT_EQ(slots[1].sequence, 1);
    EXPECT_EQ(slots[1].task_id, 7);
    EXPECT_EQ(slots[2].sequence, 2);
    EXPECT_EQ(slots[2].task_id, AICORE_TASK_ID_INVALID_V0);
    EXPECT_EQ(slots[layout.aic_queue_capacity - 1].sequence, 200);
    EXPECT_EQ(slots[layout.aic_queue_capacity - 1].task_id, AICORE_TASK_ID_INVALID_V0);
}

TEST(AicoreSidecarV0, ClassifiesEmptyAicAndAivRoots) {
    alignas(64) PTO2TaskDescriptor descriptors[2]{};
    alignas(64) PTO2TaskPayload payloads[2]{};
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        0,
        1,
    };
    AicoreRootInfoV0 root{};
    EXPECT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::EMPTY);

    graph.task_count = 1;
    descriptors[0].task_id = PTO2TaskId::make(0, 0);
    descriptors[0].kernel_id[0] = 7;
    descriptors[0].kernel_id[1] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = INVALID_KERNEL_ID;
    payloads[0].fanin_count = 0;
    ASSERT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::OK);
    EXPECT_EQ(root.task_id, 0);
    EXPECT_EQ(root.kernel_id, 7);
    EXPECT_EQ(root.subtask_slot, 0);
    EXPECT_EQ(root.core_type, AicoreRootCoreTypeV0::AIC);

    descriptors[0].kernel_id[0] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = 9;
    ASSERT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::OK);
    EXPECT_EQ(root.kernel_id, 9);
    EXPECT_EQ(root.subtask_slot, 2);
    EXPECT_EQ(root.core_type, AicoreRootCoreTypeV0::AIV);
}

TEST(AicoreSidecarV0, RejectsNonRootAndUnsupportedShapes) {
    alignas(64) PTO2TaskDescriptor descriptors[2]{};
    alignas(64) PTO2TaskPayload payloads[2]{};
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        2,
        1,
    };
    AicoreRootInfoV0 root{};
    EXPECT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::INVALID_TASK_COUNT);

    graph.task_count = 1;
    descriptors[0].task_id = PTO2TaskId::make(0, 0);
    descriptors[0].kernel_id[0] = 1;
    descriptors[0].kernel_id[1] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = INVALID_KERNEL_ID;
    payloads[0].fanin_count = 1;
    EXPECT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::HAS_FANIN);

    payloads[0].fanin_count = 0;
    descriptors[0].kernel_id[1] = 2;
    EXPECT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::UNSUPPORTED_SHAPE);
}

TEST(AicoreSidecarV0, MapsPayloadArgumentsAndResolvesCallableAddress) {
    alignas(64) PTO2TaskDescriptor descriptors[1]{};
    alignas(64) PTO2TaskPayload payloads[1]{};
    descriptors[0].task_id = PTO2TaskId::make(0, 0);
    descriptors[0].kernel_id[0] = 3;
    descriptors[0].kernel_id[1] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = INVALID_KERNEL_ID;
    payloads[0].tensor_count = 2;
    payloads[0].scalar_count = 2;
    payloads[0].scalars[0] = UINT64_C(0x1234);
    payloads[0].scalars[1] = UINT64_C(0x5678);
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        1,
        0,
    };
    AicoreRootInfoV0 root{};
    ASSERT_EQ(aicore_classify_single_root_v0(graph, &root), AicoreRootStatusV0::OK);

    alignas(8) uint8_t callable_bytes[AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0 + sizeof(uint64_t)]{};
    *reinterpret_cast<uint64_t *>(callable_bytes + AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0) = UINT64_C(0xabcdef00);
    PTO2DispatchPayload dispatch{};
    ASSERT_EQ(
        aicore_materialize_root_payload_v0(graph, root, reinterpret_cast<uint64_t>(callable_bytes), &dispatch),
        AicoreRootStatusV0::OK
    );
    EXPECT_EQ(dispatch.function_bin_addr, UINT64_C(0xabcdef00));
    EXPECT_EQ(dispatch.args[0], reinterpret_cast<uint64_t>(&payloads[0].tensors[0]));
    EXPECT_EQ(dispatch.args[1], reinterpret_cast<uint64_t>(&payloads[0].tensors[1]));
    EXPECT_EQ(dispatch.args[2], UINT64_C(0x1234));
    EXPECT_EQ(dispatch.args[3], UINT64_C(0x5678));
    EXPECT_EQ(dispatch.args[PAYLOAD_LOCAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&dispatch.local_context));
    EXPECT_EQ(dispatch.args[PAYLOAD_GLOBAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&dispatch.global_context));
    EXPECT_EQ(dispatch.local_context.block_idx, 0);
    EXPECT_EQ(dispatch.local_context.block_num, 1);
    EXPECT_EQ(dispatch.local_context.async_ctx.task_token.raw, 0u);
}

TEST(AicoreSidecarV0, RoutesHomogeneousDagThroughWakeLists) {
    alignas(64) PTO2TaskDescriptor descriptors[4]{};
    alignas(64) PTO2TaskPayload payloads[4]{};
    for (int64_t task_id = 0; task_id < 4; ++task_id) {
        descriptors[task_id].task_id = PTO2TaskId::make(0, task_id);
        descriptors[task_id].kernel_id[0] = 3;
        descriptors[task_id].kernel_id[1] = INVALID_KERNEL_ID;
        descriptors[task_id].kernel_id[2] = INVALID_KERNEL_ID;
    }
    payloads[1].fanin_count = 1;
    payloads[1].fanin_local_ids[0] = 0;
    payloads[2].fanin_count = 1;
    payloads[2].fanin_local_ids[0] = 0;
    payloads[3].fanin_count = 2;
    payloads[3].fanin_local_ids[0] = 2;
    payloads[3].fanin_local_ids[1] = 1;
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        4,
        3,
    };

    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(4, 4, 0, &layout));
    SidecarBuffer storage(layout);
    auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
    AicoreWorkerContextV0 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.aic_queue_offset = layout.aic_queue_offset;
    context.aiv_queue_offset = layout.aiv_queue_offset;

    EXPECT_EQ(
        aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, 0), AicoreRouteResultV0::READY
    );
    for (int64_t task_id = 1; task_id < 4; ++task_id) {
        EXPECT_EQ(
            aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, task_id),
            AicoreRouteResultV0::WAITING
        );
    }
    auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);
    std::vector<int> seen(4, 0);
    int64_t task_id = AICORE_TASK_ID_INVALID_V0;
    while (aicore_ready_queue_pop_v0(storage.base(), queue, &task_id)) {
        ++seen[task_id];
        ASSERT_TRUE(aicore_complete_and_wake_v0(graph, storage.base(), &context, run_control, task_id));
    }

    EXPECT_EQ(seen, (std::vector<int>{1, 1, 1, 1}));
    EXPECT_EQ(context.ready_push_count, 4u);
    EXPECT_EQ(context.wake_close_count, 4u);
    EXPECT_EQ(context.wake_register_count, 4u);
    EXPECT_EQ(context.wake_reclassify_count, 4u);
    auto *controls = aicore_sidecar_at_v0<AicoreTaskControlV0>(storage.base(), layout.task_controls_offset);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_EQ(controls[i].completion, 1);
        EXPECT_EQ(controls[i].wake_list_head, AICORE_WAKE_LIST_CLOSED_V0);
    }
}

TEST(AicoreSidecarV0, SkipsPrecompletedInlineAllocationTasks) {
    alignas(64) PTO2TaskDescriptor descriptors[2]{};
    alignas(64) PTO2TaskPayload payloads[2]{};
    descriptors[0].task_id = PTO2TaskId::make(0, 0);
    descriptors[0].kernel_id[0] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[1] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = INVALID_KERNEL_ID;
    descriptors[1].task_id = PTO2TaskId::make(0, 1);
    descriptors[1].kernel_id[0] = INVALID_KERNEL_ID;
    descriptors[1].kernel_id[1] = 1;
    descriptors[1].kernel_id[2] = INVALID_KERNEL_ID;
    payloads[1].fanin_count = 1;
    payloads[1].fanin_local_ids[0] = 0;
    AicoreReadonlyGraphV0 graph{reinterpret_cast<uint64_t>(descriptors), reinterpret_cast<uint64_t>(payloads), 2, 1};

    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(2, 0, 1, &layout));
    SidecarBuffer storage(layout);
    auto *controls = aicore_sidecar_at_v0<AicoreTaskControlV0>(storage.base(), layout.task_controls_offset);
    controls[0].completion = 1;
    controls[0].wake_list_head = AICORE_WAKE_LIST_CLOSED_V0;

    auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
    AicoreWorkerContextV0 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.aic_queue_offset = layout.aic_queue_offset;
    context.aiv_queue_offset = layout.aiv_queue_offset;

    EXPECT_EQ(
        aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, 0), AicoreRouteResultV0::COMPLETED
    );
    EXPECT_EQ(
        aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, 1), AicoreRouteResultV0::READY
    );
    EXPECT_EQ(context.ready_push_count, 1u);
    EXPECT_EQ(run_control->classification_error, 0u);
}

TEST(AicoreSidecarV0, RejectsInvalidAndRoutesMixedDagTasks) {
    alignas(64) PTO2TaskDescriptor descriptors[2]{};
    alignas(64) PTO2TaskPayload payloads[2]{};
    descriptors[0].task_id = PTO2TaskId::make(0, 0);
    descriptors[0].kernel_id[0] = 0;
    descriptors[0].kernel_id[1] = INVALID_KERNEL_ID;
    descriptors[0].kernel_id[2] = INVALID_KERNEL_ID;
    descriptors[1].task_id = PTO2TaskId::make(0, 1);
    descriptors[1].kernel_id[0] = INVALID_KERNEL_ID;
    descriptors[1].kernel_id[1] = 1;
    descriptors[1].kernel_id[2] = INVALID_KERNEL_ID;
    payloads[1].fanin_count = 2;
    payloads[1].fanin_local_ids[0] = 0;
    payloads[1].fanin_local_ids[1] = 0;
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        2,
        1,
    };
    AicoreTaskInfoV0 task{};
    EXPECT_EQ(aicore_classify_task_v0(graph, 1, &task), AicoreRootStatusV0::INVALID_FANIN_ID);

    payloads[1].fanin_count = 1;
    EXPECT_EQ(aicore_classify_task_v0(graph, 1, &task), AicoreRootStatusV0::OK);
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(2, 1, 1, &layout));
    SidecarBuffer storage(layout);
    auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
    AicoreWorkerContextV0 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.aic_queue_offset = layout.aic_queue_offset;
    context.aiv_queue_offset = layout.aiv_queue_offset;
    EXPECT_EQ(
        aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, 0), AicoreRouteResultV0::READY
    );
    EXPECT_EQ(
        aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, 1), AicoreRouteResultV0::WAITING
    );
    auto *aic_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);
    auto *aiv_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aiv_queue_offset);
    int64_t task_id = AICORE_TASK_ID_INVALID_V0;
    ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), aic_queue, &task_id));
    ASSERT_EQ(task_id, 0);
    ASSERT_TRUE(aicore_complete_and_wake_v0(graph, storage.base(), &context, run_control, task_id));
    ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), aiv_queue, &task_id));
    EXPECT_EQ(task_id, 1);
    EXPECT_EQ(run_control->classification_error, 0u);
}

TEST(AicoreSidecarV0, CompletionQueueSeparatesExecutionFromWakeRelay) {
    alignas(64) PTO2TaskDescriptor descriptors[3]{};
    alignas(64) PTO2TaskPayload payloads[3]{};
    for (int64_t task_id = 0; task_id < 3; ++task_id) {
        descriptors[task_id].task_id = PTO2TaskId::make(0, task_id);
        descriptors[task_id].kernel_id[0] = (task_id & 1) == 0 ? 1 : INVALID_KERNEL_ID;
        descriptors[task_id].kernel_id[1] = (task_id & 1) == 0 ? INVALID_KERNEL_ID : 2;
        descriptors[task_id].kernel_id[2] = INVALID_KERNEL_ID;
        if (task_id != 0) {
            payloads[task_id].fanin_count = 1;
            payloads[task_id].fanin_local_ids[0] = task_id - 1;
        }
    }
    AicoreReadonlyGraphV0 graph{reinterpret_cast<uint64_t>(descriptors), reinterpret_cast<uint64_t>(payloads), 3, 3};
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(3, 2, 1, &layout));
    SidecarBuffer storage(layout);
    auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
    AicoreWorkerContextV0 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.aic_queue_offset = layout.aic_queue_offset;
    context.aiv_queue_offset = layout.aiv_queue_offset;
    context.completion_queue_offset = layout.completion_queue_offset;
    for (int64_t task_id = 0; task_id < 3; ++task_id) {
        ASSERT_NE(
            aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, task_id),
            AicoreRouteResultV0::ERROR
        );
    }

    auto *completion_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.completion_queue_offset);
    std::vector<int64_t> executed;
    for (int64_t expected = 0; expected < 3; ++expected) {
        uint64_t queue_offset = (expected & 1) == 0 ? layout.aic_queue_offset : layout.aiv_queue_offset;
        auto *ready_queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), queue_offset);
        int64_t task_id = AICORE_TASK_ID_INVALID_V0;
        ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), ready_queue, &task_id));
        ASSERT_EQ(task_id, expected);
        executed.push_back(task_id);
        ASSERT_TRUE(aicore_ready_queue_push_v0(storage.base(), completion_queue, task_id));
        int64_t completion = AICORE_TASK_ID_INVALID_V0;
        ASSERT_TRUE(aicore_ready_queue_pop_v0(storage.base(), completion_queue, &completion));
        ASSERT_EQ(completion, expected);
        ASSERT_TRUE(aicore_complete_and_wake_v0(graph, storage.base(), &context, run_control, completion));
    }
    EXPECT_EQ(executed, (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(context.ready_push_count, 3u);
    EXPECT_EQ(context.wake_close_count, 3u);
}

TEST(AicoreSidecarV0, ProductionWakeProtocolDrainsConcurrentWaitersExactlyOnce) {
    constexpr int64_t kConsumerCount = 16;
    constexpr int64_t kTaskCount = kConsumerCount + 1;
    alignas(64) PTO2TaskDescriptor descriptors[kTaskCount]{};
    alignas(64) PTO2TaskPayload payloads[kTaskCount]{};
    for (int64_t task_id = 0; task_id < kTaskCount; ++task_id) {
        descriptors[task_id].task_id = PTO2TaskId::make(0, task_id);
        descriptors[task_id].kernel_id[0] = 1;
        descriptors[task_id].kernel_id[1] = INVALID_KERNEL_ID;
        descriptors[task_id].kernel_id[2] = INVALID_KERNEL_ID;
        if (task_id != 0) {
            payloads[task_id].fanin_count = 1;
            payloads[task_id].fanin_local_ids[0] = 0;
        }
    }
    AicoreReadonlyGraphV0 graph{
        reinterpret_cast<uint64_t>(descriptors),
        reinterpret_cast<uint64_t>(payloads),
        kTaskCount,
        31,
    };
    AicoreExecutionSidecarLayoutV0 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v0(kTaskCount, kTaskCount, 0, &layout));
    SidecarBuffer storage(layout);
    auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
    AicoreWorkerContextV0 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.aic_queue_offset = layout.aic_queue_offset;
    context.aiv_queue_offset = layout.aiv_queue_offset;

    std::vector<AicoreRouteResultV0> routes(kConsumerCount, AicoreRouteResultV0::ERROR);
    std::vector<std::thread> threads;
    for (int64_t consumer = 1; consumer < kTaskCount; ++consumer) {
        threads.emplace_back([&, consumer] {
            routes[consumer - 1] = aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, consumer);
        });
    }
    for (std::thread &thread : threads)
        thread.join();
    for (AicoreRouteResultV0 route : routes)
        EXPECT_EQ(route, AicoreRouteResultV0::WAITING);

    ASSERT_TRUE(aicore_complete_and_wake_v0(graph, storage.base(), &context, run_control, 0));
    auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);
    std::vector<int> seen(kTaskCount, 0);
    int64_t task_id = AICORE_TASK_ID_INVALID_V0;
    while (aicore_ready_queue_pop_v0(storage.base(), queue, &task_id))
        ++seen[task_id];

    EXPECT_EQ(seen[0], 0);
    for (int64_t consumer = 1; consumer < kTaskCount; ++consumer)
        EXPECT_EQ(seen[consumer], 1);
    EXPECT_EQ(context.wake_register_count, static_cast<uint64_t>(kConsumerCount));
    EXPECT_EQ(context.wake_reclassify_count, static_cast<uint64_t>(kConsumerCount));
    EXPECT_EQ(context.ready_push_count, static_cast<uint64_t>(kConsumerCount));
    EXPECT_EQ(run_control->classification_error, 0u);
}

TEST(AicoreSidecarV0, ProductionWakeProtocolRacesCloseAgainstRegistration) {
    constexpr int64_t kConsumerCount = 16;
    constexpr int64_t kTaskCount = kConsumerCount + 1;
    constexpr int kRounds = 40;
    for (int round = 0; round < kRounds; ++round) {
        alignas(64) PTO2TaskDescriptor descriptors[kTaskCount]{};
        alignas(64) PTO2TaskPayload payloads[kTaskCount]{};
        for (int64_t task_id = 0; task_id < kTaskCount; ++task_id) {
            descriptors[task_id].task_id = PTO2TaskId::make(0, task_id);
            descriptors[task_id].kernel_id[0] = 1;
            descriptors[task_id].kernel_id[1] = INVALID_KERNEL_ID;
            descriptors[task_id].kernel_id[2] = INVALID_KERNEL_ID;
            if (task_id != 0) {
                payloads[task_id].fanin_count = 1;
                payloads[task_id].fanin_local_ids[0] = 0;
            }
        }
        AicoreReadonlyGraphV0 graph{
            reinterpret_cast<uint64_t>(descriptors),
            reinterpret_cast<uint64_t>(payloads),
            kTaskCount,
            31,
        };
        AicoreExecutionSidecarLayoutV0 layout{};
        ASSERT_TRUE(aicore_sidecar_plan_v0(kTaskCount, kTaskCount, 0, &layout));
        SidecarBuffer storage(layout);
        auto *run_control = aicore_sidecar_at_v0<AicoreRunControlV0>(storage.base(), layout.run_control_offset);
        AicoreWorkerContextV0 context{};
        context.task_controls_offset = layout.task_controls_offset;
        context.aic_queue_offset = layout.aic_queue_offset;
        context.aiv_queue_offset = layout.aiv_queue_offset;

        bool completion_ok = false;
        std::vector<AicoreRouteResultV0> routes(kConsumerCount, AicoreRouteResultV0::ERROR);
        std::vector<std::thread> threads;
        threads.emplace_back([&] {
            completion_ok = aicore_complete_and_wake_v0(graph, storage.base(), &context, run_control, 0);
        });
        for (int64_t consumer = 1; consumer < kTaskCount; ++consumer) {
            threads.emplace_back([&, consumer] {
                if ((round & 1) != 0) std::this_thread::yield();
                routes[consumer - 1] =
                    aicore_classify_and_route_v0(graph, storage.base(), &context, run_control, consumer);
            });
        }
        for (std::thread &thread : threads)
            thread.join();

        ASSERT_TRUE(completion_ok) << "round " << round;
        for (AicoreRouteResultV0 route : routes)
            ASSERT_NE(route, AicoreRouteResultV0::ERROR) << "round " << round;
        auto *queue = aicore_sidecar_at_v0<AicoreReadyQueueV0>(storage.base(), layout.aic_queue_offset);
        std::vector<int> seen(kTaskCount, 0);
        int64_t task_id = AICORE_TASK_ID_INVALID_V0;
        while (aicore_ready_queue_pop_v0(storage.base(), queue, &task_id))
            ++seen[task_id];
        for (int64_t consumer = 1; consumer < kTaskCount; ++consumer)
            ASSERT_EQ(seen[consumer], 1) << "round " << round << " consumer " << consumer;
        ASSERT_EQ(context.ready_push_count, static_cast<uint64_t>(kConsumerCount)) << "round " << round;
        ASSERT_EQ(run_control->classification_error, 0u) << "round " << round;
    }
}

TEST(AicoreDependencyModelV0, RegisterBeforeCloseWakesEveryWaiterExactlyOnce) {
    AicoreDependencyModelV0 model({{}, {0}, {0}});
    ASSERT_TRUE(model.classify_all());
    EXPECT_EQ(model.state(0), AicoreModelTaskStateV0::READY);
    EXPECT_EQ(model.state(1), AicoreModelTaskStateV0::WAITING);
    EXPECT_EQ(model.state(2), AicoreModelTaskStateV0::WAITING);

    int64_t task_id = -1;
    ASSERT_TRUE(model.pop_ready(&task_id));
    ASSERT_EQ(task_id, 0);
    ASSERT_TRUE(model.complete(task_id));
    EXPECT_EQ(model.ready_count(), 2u);
    EXPECT_TRUE(model.validate_invariants());

    std::vector<int64_t> completed;
    while (model.pop_ready(&task_id)) {
        completed.push_back(task_id);
        ASSERT_TRUE(model.complete(task_id));
    }
    EXPECT_EQ(completed, (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(model.completed_count(), 3u);
    EXPECT_TRUE(model.validate_invariants());
}

TEST(AicoreDependencyModelV0, CloseBeforeRegisterAndMultiFaninRehang) {
    AicoreDependencyModelV0 model({{}, {}, {0, 1}});
    ASSERT_TRUE(model.classify(0));
    int64_t task_id = -1;
    ASSERT_TRUE(model.pop_ready(&task_id));
    ASSERT_EQ(task_id, 0);
    ASSERT_TRUE(model.complete(task_id));

    ASSERT_TRUE(model.classify(2));
    EXPECT_EQ(model.state(2), AicoreModelTaskStateV0::WAITING);
    ASSERT_TRUE(model.classify(1));
    ASSERT_TRUE(model.pop_ready(&task_id));
    ASSERT_EQ(task_id, 1);
    ASSERT_TRUE(model.complete(task_id));
    EXPECT_EQ(model.state(2), AicoreModelTaskStateV0::READY);
    ASSERT_TRUE(model.pop_ready(&task_id));
    ASSERT_EQ(task_id, 2);
    ASSERT_TRUE(model.complete(task_id));
    EXPECT_EQ(model.completed_count(), 3u);
    EXPECT_TRUE(model.validate_invariants());
}

TEST(AicoreDependencyModelV0, FixedSeedRandomDagCompletesExactlyOnce) {
    constexpr int64_t kTaskCount = 64;
    std::vector<std::vector<int64_t>> fanins(kTaskCount);
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t task_id = 1; task_id < kTaskCount; ++task_id) {
        random = random * UINT64_C(6364136223846793005) + 1;
        int64_t fanin_count = static_cast<int64_t>(random % 5);
        for (int64_t edge = 0; edge < fanin_count; ++edge) {
            random = random * UINT64_C(6364136223846793005) + 1;
            int64_t producer = static_cast<int64_t>(random % static_cast<uint64_t>(task_id));
            if (std::find(fanins[task_id].begin(), fanins[task_id].end(), producer) == fanins[task_id].end()) {
                fanins[task_id].push_back(producer);
            }
        }
    }

    AicoreDependencyModelV0 model(std::move(fanins));
    ASSERT_TRUE(model.classify_all());
    std::vector<int> seen(kTaskCount, 0);
    while (model.completed_count() != kTaskCount) {
        int64_t task_id = -1;
        ASSERT_TRUE(model.pop_ready(&task_id));
        ASSERT_GE(task_id, 0);
        ASSERT_LT(task_id, kTaskCount);
        ++seen[task_id];
        ASSERT_TRUE(model.complete(task_id));
        ASSERT_TRUE(model.validate_invariants());
    }
    EXPECT_EQ(model.ready_count(), 0u);
    for (int count : seen)
        EXPECT_EQ(count, 1);
}

TEST(AicoreDependencyModelV0, LongFaninWaitsForEveryProducer) {
    constexpr int64_t kProducerCount = 32;
    constexpr int64_t kConsumer = kProducerCount;
    std::vector<std::vector<int64_t>> fanins(kProducerCount + 1);
    for (int64_t producer = 0; producer < kProducerCount; ++producer)
        fanins[kConsumer].push_back(producer);

    AicoreDependencyModelV0 model(std::move(fanins));
    ASSERT_TRUE(model.classify_all());
    EXPECT_EQ(model.ready_count(), static_cast<size_t>(kProducerCount));
    EXPECT_EQ(model.state(kConsumer), AicoreModelTaskStateV0::WAITING);

    for (int64_t producer = 0; producer < kProducerCount; ++producer) {
        int64_t task_id = -1;
        ASSERT_TRUE(model.pop_ready(&task_id));
        ASSERT_EQ(task_id, producer);
        ASSERT_TRUE(model.complete(task_id));
        ASSERT_TRUE(model.validate_invariants());
        EXPECT_EQ(
            model.state(kConsumer),
            producer + 1 == kProducerCount ? AicoreModelTaskStateV0::READY : AicoreModelTaskStateV0::WAITING
        );
    }

    int64_t task_id = -1;
    ASSERT_TRUE(model.pop_ready(&task_id));
    ASSERT_EQ(task_id, kConsumer);
    ASSERT_TRUE(model.complete(task_id));
    EXPECT_EQ(model.completed_count(), static_cast<size_t>(kProducerCount + 1));
    EXPECT_TRUE(model.validate_invariants());
}

}  // namespace
