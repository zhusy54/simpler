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
#include "aicore_worker_debug_policy_v1.h"
#include "callable.h"
#include "pto_runtime2_types.h"

namespace {

static_assert(sizeof(PTO2TaskDescriptor) == AICORE_GRAPH_TASK_DESCRIPTOR_STRIDE_V0);
static_assert(sizeof(PTO2TaskPayload) == AICORE_GRAPH_TASK_PAYLOAD_STRIDE_V0);
static_assert(sizeof(AicorePendingSlotV1) == 64);
static_assert(sizeof(AicoreTaskControlV1) == 128);
static_assert(sizeof(AicoreTaskTicketV1) == 16);
static_assert(alignof(AicoreTaskTicketV1) == 16);
static_assert(sizeof(AicoreTaskClaimBindingV1) == 64);
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
        executable(task_id, core_type == AicoreRootCoreTypeV0::AIC ? 0 : 1, 1, std::move(fanins));
    }

    void executable(size_t task_id, int32_t subtask_slot, int32_t kernel_id, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task_id, inline_completed_.size());
        ASSERT_GE(subtask_slot, 0);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(AICORE_GRAPH_MAX_FANIN_V0));
        descriptors_[task_id].kernel_id[subtask_slot] = kernel_id;
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

    void invalidate_stored_task_id(size_t task_id) {
        ASSERT_LT(task_id, inline_completed_.size());
        descriptors_[task_id].task_id = PTO2TaskId::make(0, static_cast<int64_t>(task_id + 1));
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

uint64_t fake_callable_address() {
    alignas(64) static uint8_t callable[256]{};
    *reinterpret_cast<uint64_t *>(callable + AICORE_CORE_CALLABLE_RESOLVED_ADDR_OFFSET_V0) = UINT64_C(0x12340000);
    return reinterpret_cast<uint64_t>(callable);
}

void publish_ready_candidate(
    const AicoreReadonlyGraphV0 &graph, void *sidecar, AicoreWorkerContextV1 *context,
    AicoreRunControlV1 *run_control, int64_t task_id, AicoreWakeStatsV1 *stats
) {
    ASSERT_EQ(
        aicore_route_task_v1(graph, sidecar, context, run_control, task_id, stats),
        AicoreRouteResultV1::READY_TO_PUBLISH
    );
    auto *control = aicore_task_control_at_v1(sidecar, context, task_id);
    aicore_gm_store_v0(control->state, static_cast<int64_t>(AicoreTaskStateV1::READY));
}

AicoreTaskClaimBindingV1 publish_test_binding(
    void *sidecar, const AicoreExecutionSidecarLayoutV1 &layout, AicoreWorkerContextV1 *context,
    AicoreRunControlV1 *run_control, int64_t task_id, uint8_t pending_slot = 0, uint16_t kernel_id = 1,
    uint8_t subtask_slot = 0
) {
    context->dispatch_payload_offset = layout.dispatch_payloads_offset;
    run_control->claim_bindings_offset = layout.claim_bindings_offset;
    AicoreTaskClaimBindingV1 binding = aicore_make_claim_binding_v1(
        context, task_id, kernel_id, subtask_slot, pending_slot, fake_callable_address()
    );
    aicore_publish_claim_binding_v1(sidecar, run_control, binding);
    return binding;
}

TEST(AicoreSidecarV1, PlansOnlyTypedStreamsTaskControlsAndTrace) {
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(5, 3, 2, &layout));
    EXPECT_EQ(layout.task_count, 5u);
    EXPECT_EQ(layout.aic_task_count, 3u);
    EXPECT_EQ(layout.aiv_task_count, 2u);
    EXPECT_EQ(layout.total_size % AICORE_SIDECAR_ALIGNMENT_V1, 0u);
    EXPECT_EQ(layout.task_controls_offset % alignof(AicoreTaskControlV1), 0u);
    EXPECT_EQ(layout.claim_bindings_offset % alignof(AicoreTaskClaimBindingV1), 0u);
    EXPECT_EQ(layout.completion_inboxes_offset % alignof(AicoreCompletionInboxV1), 0u);
    EXPECT_EQ(layout.aic_stream_offset % alignof(AicoreTaskStreamV1), 0u);
    EXPECT_EQ(layout.aic_tickets_offset % alignof(AicoreTaskTicketV1), 0u);
    EXPECT_EQ(layout.aiv_stream_offset % alignof(AicoreTaskStreamV1), 0u);
    EXPECT_EQ(layout.aiv_tickets_offset % alignof(AicoreTaskTicketV1), 0u);
    EXPECT_EQ(layout.trace_cells_offset % alignof(AicoreTaskTraceCellV1), 0u);
    EXPECT_EQ(sizeof(AicoreTaskTraceCellV1), 256u);

    SidecarBuffer storage(layout);
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.base(), layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(AicoreTaskStateV1::BLOCKED));
        EXPECT_EQ(controls[task].wake_list_head, AICORE_WAKE_LIST_OPEN_V1);
        if (task != 0) {
            EXPECT_EQ(reinterpret_cast<uintptr_t>(&controls[task]) - reinterpret_cast<uintptr_t>(&controls[task - 1]),
                      sizeof(AicoreTaskControlV1));
        }
    }
    auto *inboxes = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(storage.base(), layout.completion_inboxes_offset);
    for (uint64_t inbox = 0; inbox < AICORE_WORKER_CAPACITY_V1; ++inbox)
        EXPECT_EQ(inboxes[inbox].head, AICORE_COMPLETION_INBOX_EMPTY_V1);
    auto *aic_stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(storage.base(), layout.aic_stream_offset);
    auto *aiv_stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(storage.base(), layout.aiv_stream_offset);
    EXPECT_EQ(aic_stream->tickets_offset, layout.aic_tickets_offset);
    EXPECT_EQ(aiv_stream->tickets_offset, layout.aiv_tickets_offset);
    auto *bindings =
        aicore_sidecar_at_v1<AicoreTaskClaimBindingV1>(storage.base(), layout.claim_bindings_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task)
        EXPECT_EQ(bindings[task].task_id, AICORE_TASK_ID_INVALID_V1);
    EXPECT_EQ(
        layout.claim_bindings_offset - layout.dispatch_payloads_offset,
        AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1 * sizeof(PTO2DispatchPayload)
    );
}

TEST(AicoreTicketSchedulerV1, RouteDefersReadyPublicationUntilPayloadIsMaterialized) {
    GraphBuffer graph_storage(1);
    graph_storage.executable(0, AicoreRootCoreTypeV0::AIC);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(1, 1, 0, &layout));
    SidecarBuffer sidecar(layout);
    AicoreWorkerContextV1 context{};
    context.task_controls_offset = layout.task_controls_offset;
    AicoreRunControlV1 run_control{};
    AicoreWakeStatsV1 stats{};

    EXPECT_EQ(
        aicore_route_task_v1(graph, sidecar.base(), &context, &run_control, 0, &stats),
        AicoreRouteResultV1::READY_TO_PUBLISH
    );
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(sidecar.base(), layout.task_controls_offset);
    EXPECT_EQ(controls[0].state, static_cast<int64_t>(AicoreTaskStateV1::BLOCKED));
}

TEST(AicoreTicketSchedulerV1, ClaimedPendingSlotsMaterializeIndependentPayloads) {
    GraphBuffer graph_storage(2);
    graph_storage.executable(0, 0, 1);
    graph_storage.executable(1, 1, 2);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(2, 1, 1, &layout));
    SidecarBuffer sidecar(layout);
    AicoreWorkerContextV1 owner{};
    owner.task_controls_offset = layout.task_controls_offset;
    owner.dispatch_payload_offset = layout.dispatch_payloads_offset;
    owner.worker_index = 7;
    AicoreRunControlV1 run_control{};
    run_control.claim_bindings_offset = layout.claim_bindings_offset;
    AicoreWakeStatsV1 stats{};

    AicoreTaskClaimBindingV1 first = aicore_make_claim_binding_v1(&owner, 0, 1, 0, 0, fake_callable_address());
    AicoreTaskClaimBindingV1 second = aicore_make_claim_binding_v1(&owner, 1, 2, 1, 1, fake_callable_address());
    aicore_publish_claim_binding_v1(sidecar.base(), &run_control, first);
    aicore_publish_claim_binding_v1(sidecar.base(), &run_control, second);
    ASSERT_EQ(
        aicore_route_task_v1(graph, sidecar.base(), &owner, &run_control, 0, &stats),
        AicoreRouteResultV1::READY_TO_PUBLISH
    );
    ASSERT_EQ(
        aicore_route_task_v1(graph, sidecar.base(), &owner, &run_control, 1, &stats),
        AicoreRouteResultV1::READY_TO_PUBLISH
    );
    ASSERT_TRUE(aicore_finalize_ready_v1(graph, sidecar.base(), &owner, &run_control, first, false));
    ASSERT_TRUE(aicore_finalize_ready_v1(graph, sidecar.base(), &owner, &run_control, second, true));

    auto *first_payload = aicore_sidecar_at_v1<PTO2DispatchPayload>(sidecar.base(), first.dispatch_payload_offset);
    auto *second_payload = aicore_sidecar_at_v1<PTO2DispatchPayload>(sidecar.base(), second.dispatch_payload_offset);
    EXPECT_NE(first_payload, second_payload);
    EXPECT_EQ(first_payload->local_context.async_ctx.task_token.raw, 0u);
    EXPECT_EQ(second_payload->local_context.async_ctx.task_token.raw, 1u);
    EXPECT_EQ(first_payload->global_context.sub_block_id, 0u);
    EXPECT_EQ(second_payload->global_context.sub_block_id, 0u);
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

TEST(AicoreWorkerDebugPolicyV1, PublishesOnlyDiagnosticTransitions) {
    EXPECT_FALSE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::SEED_INITIALIZED));
    EXPECT_FALSE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::TICKET_CLAIMED));
    EXPECT_FALSE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::TASK_COMPLETED));
    EXPECT_TRUE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::WAITING_PRODUCER_CHANGED));
    EXPECT_TRUE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::CURSOR_EXHAUSTED));
    EXPECT_TRUE(aicore_worker_debug_requires_publish_v1(AicoreWorkerDebugEventV1::DRAINING));
}

TEST(AicoreTicketSchedulerV1, TicketClaimsAreUniqueAndExhaustOncePerWorker) {
    constexpr uint64_t kTaskCount = 4096;
    constexpr int kWorkers = 8;
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(kTaskCount, kTaskCount, 0, &layout));
    SidecarBuffer storage(layout);
    auto *stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(storage.base(), layout.aic_stream_offset);
    auto *tickets = aicore_sidecar_at_v1<AicoreTaskTicketV1>(storage.base(), layout.aic_tickets_offset);
    for (uint64_t task = 0; task < kTaskCount; ++task)
        tickets[task] = aicore_task_ticket_make_v1(static_cast<uint32_t>(task), 0, 0, 0);

    std::vector<std::atomic<int>> seen(kTaskCount);
    std::vector<std::thread> workers;
    std::atomic<int> exhaustion_count{0};
    for (int worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                uint64_t index = 0;
                AicoreTaskTicketV1 ticket{};
                if (!aicore_claim_ticket_v1(storage.base(), stream, &index, &ticket)) {
                    exhaustion_count.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                seen[aicore_task_ticket_task_id_v1(ticket)].fetch_add(1, std::memory_order_relaxed);
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

TEST(AicoreTicketSchedulerV1, WakeMigrationRemembersCompletedFaninPrefix) {
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
    context.task_controls_offset = layout.task_controls_offset;
    AicoreRunControlV1 run_control{};
    AicoreWakeStatsV1 stats{};
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(storage.base(), layout.task_controls_offset);
    publish_test_binding(storage.base(), layout, &context, &run_control, 2);
    EXPECT_EQ(
        aicore_route_task_v1(graph, storage.base(), &context, &run_control, 2, &stats), AicoreRouteResultV1::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 2);
    ASSERT_TRUE(aicore_complete_and_wake_v1(graph, storage.base(), &context, &run_control, 0, &stats));
    EXPECT_EQ(controls[2].state, static_cast<int64_t>(AicoreTaskStateV1::BLOCKED));
    EXPECT_EQ(controls[2].next_fanin_index, 1);
    EXPECT_EQ(controls[2].waiting_producer, 1);
    EXPECT_EQ(controls[1].wake_list_head, 2);
    ASSERT_TRUE(aicore_complete_and_wake_v1(graph, storage.base(), &context, &run_control, 1, &stats));
    EXPECT_EQ(controls[2].state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    EXPECT_EQ(controls[2].next_fanin_index, 2);
    EXPECT_EQ(stats.wake_register_count, 2u);
    EXPECT_EQ(stats.wake_migrate_count, 2u);
}

TEST(AicoreTicketSchedulerV1, CompletionInboxBatchesAndStealsWithoutLoss) {
    constexpr int64_t kTaskCount = 16;
    GraphBuffer graph_storage(kTaskCount);
    for (int64_t task = 0; task < kTaskCount; ++task)
        graph_storage.executable(static_cast<size_t>(task), AicoreRootCoreTypeV0::AIC);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(kTaskCount, kTaskCount, 0, &layout));
    SidecarBuffer sidecar(layout);
    AicoreWorkerContextV1 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.completion_inboxes_offset = layout.completion_inboxes_offset;
    context.core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    context.inbox_index = 0;
    AicoreRunControlV1 run_control{};
    run_control.active_worker_count = 2;
    run_control.aiv_active_worker_count = 2;
    AicoreWakeStatsV1 wake_stats{};
    AicoreCompletionStatsV1 completion_stats{};
    for (int64_t task = 0; task < kTaskCount; ++task) {
        publish_ready_candidate(graph, sidecar.base(), &context, &run_control, task, &wake_stats);
        ASSERT_TRUE(
            aicore_enqueue_completion_v1(
                graph, sidecar.base(), &context, &run_control, run_control.aiv_active_worker_count, task,
                &completion_stats
            )
        );
    }

    uint64_t victim_cursor = 1;
    while (run_control.resolved_task_count != static_cast<uint64_t>(kTaskCount)) {
        bool progress = false;
        ASSERT_TRUE(aicore_service_completion_inboxes_v1(
            graph, sidecar.base(), &context, &run_control, run_control.aiv_active_worker_count, &victim_cursor,
            &wake_stats, &completion_stats, &progress
        ));
        ASSERT_TRUE(progress);
    }
    EXPECT_EQ(completion_stats.enqueue_count, static_cast<uint64_t>(kTaskCount));
    EXPECT_EQ(completion_stats.resolve_count, static_cast<uint64_t>(kTaskCount));
    EXPECT_EQ(completion_stats.batch_count, 2u);
    EXPECT_EQ(completion_stats.steal_count, 1u);
    EXPECT_EQ(wake_stats.wake_close_count, static_cast<uint64_t>(kTaskCount));
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(sidecar.base(), layout.task_controls_offset);
    for (int64_t task = 0; task < kTaskCount; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
        EXPECT_EQ(controls[task].wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    }
}

TEST(AicoreTicketSchedulerV1, ConcurrentExecutorsPushOneInboxExactlyOnce) {
    constexpr int64_t kTaskCount = 64;
    constexpr int64_t kThreadCount = 8;
    GraphBuffer graph_storage(kTaskCount);
    for (int64_t task = 0; task < kTaskCount; ++task)
        graph_storage.executable(static_cast<size_t>(task), AicoreRootCoreTypeV0::AIC);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(kTaskCount, kTaskCount, 0, &layout));
    SidecarBuffer sidecar(layout);
    AicoreRunControlV1 run_control{};
    run_control.active_worker_count = 1;
    run_control.aiv_active_worker_count = 1;
    std::vector<AicoreWorkerContextV1> contexts(kThreadCount);
    std::vector<AicoreWakeStatsV1> wake_stats(kThreadCount);
    std::vector<AicoreCompletionStatsV1> completion_stats(kThreadCount);
    for (int64_t thread = 0; thread < kThreadCount; ++thread) {
        contexts[thread].task_controls_offset = layout.task_controls_offset;
        contexts[thread].completion_inboxes_offset = layout.completion_inboxes_offset;
    }
    contexts[0].core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    std::vector<std::thread> producers;
    for (int64_t thread = 0; thread < kThreadCount; ++thread) {
        producers.emplace_back([&, thread] {
            for (int64_t task = thread; task < kTaskCount; task += kThreadCount) {
                publish_ready_candidate(
                    graph, sidecar.base(), &contexts[thread], &run_control, task, &wake_stats[thread]
                );
                EXPECT_TRUE(aicore_enqueue_completion_v1(
                    graph, sidecar.base(), &contexts[thread], &run_control, run_control.aiv_active_worker_count, task,
                    &completion_stats[thread]
                ));
            }
        });
    }
    for (std::thread &producer : producers)
        producer.join();

    uint64_t victim_cursor = 0;
    bool progress = false;
    ASSERT_TRUE(aicore_service_completion_inboxes_v1(
        graph, sidecar.base(), &contexts[0], &run_control, run_control.aiv_active_worker_count, &victim_cursor,
        &wake_stats[0], &completion_stats[0], &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(run_control.resolved_task_count, static_cast<uint64_t>(kTaskCount));
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(sidecar.base(), layout.task_controls_offset);
    for (int64_t task = 0; task < kTaskCount; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
        EXPECT_EQ(controls[task].wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
    }
}

TEST(AicoreTicketSchedulerV1, ResolverWaitsForDetachedCompletionLinkPublication) {
    GraphBuffer graph_storage(1);
    graph_storage.executable(0, AicoreRootCoreTypeV0::AIC);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(1, 1, 0, &layout));
    SidecarBuffer sidecar(layout);
    AicoreWorkerContextV1 context{};
    context.task_controls_offset = layout.task_controls_offset;
    context.completion_inboxes_offset = layout.completion_inboxes_offset;
    context.core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    context.inbox_index = 0;
    AicoreRunControlV1 run_control{};
    run_control.active_worker_count = 1;
    run_control.aiv_active_worker_count = 1;
    AicoreWakeStatsV1 wake_stats{};
    AicoreCompletionStatsV1 completion_stats{};
    publish_ready_candidate(graph, sidecar.base(), &context, &run_control, 0, &wake_stats);

    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(sidecar.base(), layout.task_controls_offset);
    auto *inboxes = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(sidecar.base(), layout.completion_inboxes_offset);
    aicore_gm_store_v0(controls[0].state, static_cast<int64_t>(AicoreTaskStateV1::DONE));
    ASSERT_EQ(aicore_gm_exchange_v0(inboxes[0].head, 0), AICORE_COMPLETION_INBOX_EMPTY_V1);
    ASSERT_EQ(controls[0].completion_next, AICORE_COMPLETION_LINK_UNPUBLISHED_V1);

    std::atomic<bool> publisher_ready{false};
    std::thread publisher([&] {
        publisher_ready.store(true, std::memory_order_release);
        while (aicore_gm_load_v0(inboxes[0].head) != AICORE_COMPLETION_INBOX_EMPTY_V1) {}
        aicore_gm_store_v0(controls[0].completion_next, AICORE_COMPLETION_INBOX_EMPTY_V1);
        aicore_publish_cache_line_v0(&controls[0].next_waiter);
    });
    while (!publisher_ready.load(std::memory_order_acquire)) {}

    uint64_t victim_cursor = 0;
    bool progress = false;
    bool service_ok = aicore_service_completion_inboxes_v1(
        graph, sidecar.base(), &context, &run_control, run_control.aiv_active_worker_count, &victim_cursor, &wake_stats,
        &completion_stats, &progress
    );
    publisher.join();

    ASSERT_TRUE(service_ok);
    EXPECT_TRUE(progress);
    EXPECT_EQ(run_control.resolved_task_count, 1u);
    EXPECT_EQ(completion_stats.resolve_count, 1u);
    EXPECT_EQ(controls[0].wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
}

TEST(AicoreTicketSchedulerV1, AicWorkerCannotResolveCompletionInbox) {
    GraphBuffer graph_storage(1);
    graph_storage.executable(0, AicoreRootCoreTypeV0::AIC);
    AicoreReadonlyGraphV0 graph = graph_storage.graph();
    AicoreExecutionSidecarLayoutV1 layout{};
    ASSERT_TRUE(aicore_sidecar_plan_v1(1, 1, 0, &layout));
    SidecarBuffer sidecar(layout);
    AicoreWorkerContextV1 aic_context{};
    aic_context.task_controls_offset = layout.task_controls_offset;
    aic_context.completion_inboxes_offset = layout.completion_inboxes_offset;
    aic_context.core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIC);
    AicoreWorkerContextV1 aiv_context = aic_context;
    aiv_context.core_type = static_cast<int32_t>(AicoreRootCoreTypeV0::AIV);
    aiv_context.inbox_index = 0;
    AicoreRunControlV1 run_control{};
    run_control.active_worker_count = 2;
    run_control.aiv_active_worker_count = 1;
    AicoreWakeStatsV1 wake_stats{};
    AicoreCompletionStatsV1 completion_stats{};
    publish_ready_candidate(graph, sidecar.base(), &aic_context, &run_control, 0, &wake_stats);
    ASSERT_TRUE(aicore_enqueue_completion_v1(
        graph, sidecar.base(), &aic_context, &run_control, run_control.aiv_active_worker_count, 0, &completion_stats
    ));

    uint64_t victim_cursor = 0;
    bool progress = true;
    ASSERT_TRUE(aicore_service_completion_inboxes_v1(
        graph, sidecar.base(), &aic_context, &run_control, run_control.aiv_active_worker_count, &victim_cursor,
        &wake_stats, &completion_stats, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(run_control.resolved_task_count, 0u);
    auto *inboxes = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(sidecar.base(), layout.completion_inboxes_offset);
    EXPECT_EQ(inboxes[0].head, 0);

    ASSERT_TRUE(aicore_service_completion_inboxes_v1(
        graph, sidecar.base(), &aiv_context, &run_control, run_control.aiv_active_worker_count, &victim_cursor,
        &wake_stats, &completion_stats, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(run_control.resolved_task_count, 1u);
    EXPECT_EQ(inboxes[0].head, AICORE_COMPLETION_INBOX_EMPTY_V1);
}

TEST(AicoreTicketSchedulerV1, RegistrationRacingCompletionCannotLoseWakeup) {
    for (int iteration = 0; iteration < 128; ++iteration) {
        GraphBuffer graph_storage(2);
        graph_storage.executable(0, AicoreRootCoreTypeV0::AIC);
        graph_storage.executable(1, AicoreRootCoreTypeV0::AIV, {0});
        AicoreReadonlyGraphV0 graph = graph_storage.graph();
        AicoreExecutionSidecarLayoutV1 layout{};
        ASSERT_TRUE(aicore_sidecar_plan_v1(2, 1, 1, &layout));
        SidecarBuffer sidecar(layout);
        AicoreWorkerContextV1 route_context{};
        AicoreWorkerContextV1 complete_context{};
        route_context.task_controls_offset = layout.task_controls_offset;
        complete_context.task_controls_offset = layout.task_controls_offset;
        AicoreRunControlV1 run_control{};
        AicoreWakeStatsV1 route_stats{};
        AicoreWakeStatsV1 complete_stats{};
        publish_ready_candidate(graph, sidecar.base(), &complete_context, &run_control, 0, &complete_stats);
        publish_test_binding(sidecar.base(), layout, &route_context, &run_control, 1, 1, 1, 1);
        std::atomic<bool> start{false};
        std::thread route([&] {
            while (!start.load(std::memory_order_acquire)) {}
            AicoreRouteResultV1 result =
                aicore_route_task_v1(graph, sidecar.base(), &route_context, &run_control, 1, &route_stats);
            if (result == AicoreRouteResultV1::READY_TO_PUBLISH) {
                AicoreTaskClaimBindingV1 binding{};
                ASSERT_TRUE(aicore_observe_claim_binding_v1(sidecar.base(), &run_control, 1, &binding));
                EXPECT_TRUE(aicore_finalize_ready_v1(
                    graph, sidecar.base(), &route_context, &run_control, binding, false
                ));
            } else {
                EXPECT_EQ(result, AicoreRouteResultV1::WAITING);
            }
        });
        std::thread complete([&] {
            while (!start.load(std::memory_order_acquire)) {}
            EXPECT_TRUE(
                aicore_complete_and_wake_v1(graph, sidecar.base(), &complete_context, &run_control, 0, &complete_stats)
            );
        });
        start.store(true, std::memory_order_release);
        route.join();
        complete.join();
        auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(sidecar.base(), layout.task_controls_offset);
        EXPECT_EQ(controls[0].wake_list_head, AICORE_WAKE_LIST_CLOSED_V1);
        EXPECT_EQ(controls[1].state, static_cast<int64_t>(AicoreTaskStateV1::READY));
    }
}

TEST(AicoreTicketSchedulerV1, PendingCachesTaskClassification) {
    GraphBuffer storage(3);
    storage.executable(0, 0, 17);
    storage.executable(1, 1, 18);
    storage.executable(2, 2, 19);
    AicoreReadonlyGraphV0 graph = storage.graph();

    const struct {
        int64_t task_id;
        AicoreRootCoreTypeV0 core_type;
        uint16_t kernel_id;
        uint8_t subtask_slot;
    } cases[] = {
        {0, AicoreRootCoreTypeV0::AIC, 17, 0},
        {1, AicoreRootCoreTypeV0::AIV, 18, 1},
        {2, AicoreRootCoreTypeV0::AIV, 19, 2},
    };
    for (const auto &test_case : cases) {
        AicoreTaskTicketV1 ticket{};
        ASSERT_EQ(
            aicore_make_task_ticket_v1(graph, test_case.task_id, test_case.core_type, 1024, &ticket),
            AicoreRootStatusV0::OK
        );
        AicorePendingSlotV1 pending{};
        aicore_pending_initialize_v1(ticket, 7, AicoreClaimKindV1::TICKET, 10, 11, &pending);
        EXPECT_EQ(pending.task_id, test_case.task_id);
        EXPECT_EQ(pending.fanin_count, aicore_task_ticket_fanin_count_v1(ticket));
        EXPECT_EQ(pending.kernel_id, test_case.kernel_id);
        EXPECT_EQ(pending.subtask_slot, test_case.subtask_slot);
    }
}

TEST(AicoreTicketSchedulerV1, HostTicketBuilderRejectsClassificationThatWorkerCannotExecute) {
    GraphBuffer storage(3);
    storage.executable(0, 0, 1);
    storage.executable(1, 1, 1024);
    storage.executable(2, 0, 1);
    storage.invalidate_stored_task_id(2);
    AicoreReadonlyGraphV0 graph = storage.graph();
    AicoreTaskTicketV1 ticket{};

    EXPECT_EQ(
        aicore_make_task_ticket_v1(graph, 0, AicoreRootCoreTypeV0::AIV, 1024, &ticket),
        AicoreRootStatusV0::UNSUPPORTED_SHAPE
    );
    EXPECT_EQ(
        aicore_make_task_ticket_v1(graph, 1, AicoreRootCoreTypeV0::AIV, 1024, &ticket),
        AicoreRootStatusV0::UNSUPPORTED_SHAPE
    );
    EXPECT_EQ(
        aicore_make_task_ticket_v1(graph, 2, AicoreRootCoreTypeV0::AIC, 1024, &ticket),
        AicoreRootStatusV0::INVALID_TASK_ID
    );
}

TEST(AicoreTicketSchedulerV1, PendingClearInvalidatesCachedClassification) {
    AicorePendingSlotV1 pending{};
    aicore_pending_clear_v1(&pending);
    EXPECT_EQ(pending.task_id, AICORE_TASK_ID_INVALID_V1);
    EXPECT_EQ(pending.kernel_id, UINT16_MAX);
    EXPECT_EQ(pending.subtask_slot, UINT8_MAX);
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
    EXPECT_GT(model.wake_register_count(), 0u);
    EXPECT_EQ(model.wake_register_count(), model.wake_migrate_count());
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
