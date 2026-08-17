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

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "pto2_dispatch_payload.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __host__
#define __host__
#endif

inline constexpr uint64_t AICORE_SIDECAR_ALIGNMENT_V1 = 128;
inline constexpr uint64_t AICORE_WORKER_CAPACITY_V1 = 108;
inline constexpr uint32_t AICORE_PENDING_SLOT_COUNT_V1 = 2;
inline constexpr int64_t AICORE_TASK_ID_INVALID_V1 = -1;
inline constexpr int64_t AICORE_WAKE_LIST_OPEN_V1 = -1;
inline constexpr int64_t AICORE_WAKE_LIST_CLOSED_V1 = -2;
inline constexpr int64_t AICORE_COMPLETION_INBOX_EMPTY_V1 = -1;
inline constexpr int64_t AICORE_COMPLETION_LINK_UNPUBLISHED_V1 = -2;
inline constexpr int64_t AICORE_ROOT_PREPARE_INBOX_EMPTY_V1 = -1;

enum class AicoreTaskStateV1 : int64_t {
    BLOCKED = 0,
    READY = 1,
    DONE = 2,
    PREPARING = 3,
};

enum class AicoreClaimKindV1 : uint64_t {
    SEED = 0,
    TICKET = 1,
};

struct alignas(128) AicoreTaskControlV1 {
    // Cross-core state publication and wake-list RMWs share the first line.
    volatile int64_t state;
    volatile int64_t wake_list_head;
    uint8_t atomic_line_padding[48];

    // A consumer is linked to at most one producer. The current list owner is
    // the only writer of this line until it relinks or publishes READY.
    int64_t next_waiter;
    int32_t next_fanin_index;
    int32_t waiting_producer;
    // RootPrepare and Completion reuse this intrusive link sequentially. The
    // RootPrepare resolver restores UNPUBLISHED before it publishes READY, so
    // execution completion can safely reuse the same field.
    volatile int64_t completion_next;
    uint64_t completion_enqueue_cycles;
    uint64_t completion_resolve_start_cycles;
    uint64_t completion_resolve_end_cycles;
    uint64_t ready_publish_cycles;
    uint64_t resolver_worker_id;
};

struct alignas(128) AicoreCompletionInboxV1 {
    volatile int64_t head;
    uint8_t atomic_line_padding[120];
};

struct alignas(128) AicoreRootPrepareInboxV1 {
    volatile int64_t head;
    uint8_t atomic_line_padding[120];
};

struct alignas(64) AicoreTaskClaimBindingV1 {
    int64_t task_id;
    uint64_t callable_address;
    uint64_t dispatch_payload_offset;
    uint64_t owner_worker_id;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t pending_slot;
    uint32_t reserved0;
    uint64_t root_prepare_start_cycles;
    uint64_t root_prepare_end_cycles;
    uint64_t root_prepare_resolver_worker_id;
};

struct alignas(8) AicoreTaskTicketV1 {
    // Eight tickets fit exactly in one A5 64-byte data-cache line. The full
    // fanin count stays in the graph payload; the ticket only distinguishes
    // the zero-fanin local-ready path from dependency-routed work.
    uint64_t packed;
};

inline __host__ __aicore__ AicoreTaskTicketV1
aicore_task_ticket_make_v1(uint32_t task_id, uint16_t kernel_id, uint8_t subtask_slot, bool has_fanin) {
    return {
        static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(kernel_id) << 32) |
            (static_cast<uint64_t>(subtask_slot) << 48) | (static_cast<uint64_t>(has_fanin ? 1 : 0) << 56),
    };
}

inline __host__ __aicore__ uint32_t aicore_task_ticket_task_id_v1(const AicoreTaskTicketV1 &ticket) {
    return static_cast<uint32_t>(ticket.packed);
}

inline __host__ __aicore__ bool aicore_task_ticket_has_fanin_v1(const AicoreTaskTicketV1 &ticket) {
    return ((ticket.packed >> 56) & UINT64_C(1)) != 0;
}

inline __host__ __aicore__ uint16_t aicore_task_ticket_kernel_id_v1(const AicoreTaskTicketV1 &ticket) {
    return static_cast<uint16_t>(ticket.packed >> 32);
}

inline __host__ __aicore__ uint8_t aicore_task_ticket_subtask_slot_v1(const AicoreTaskTicketV1 &ticket) {
    return static_cast<uint8_t>(ticket.packed >> 48);
}

struct alignas(128) AicoreTaskStreamV1 {
    uint64_t tickets_offset;
    uint64_t task_count;
    uint64_t initial_ticket_count;
    uint8_t metadata_padding[104];

    volatile uint64_t next_index;
    uint8_t cursor_padding[120];
};

struct alignas(128) AicoreRunControlV1 {
    uint64_t config_reserved_prefix[2];
    uint64_t active_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t aic_active_worker_count;
    uint64_t aiv_active_worker_count;
    uint64_t claim_bindings_offset;
    uint64_t expected_root_prepare_count;
    uint64_t config_reserved[7];

    volatile uint64_t executed_task_count;
    volatile uint64_t executor_drained_worker_count;
    volatile uint64_t resolved_task_count;
    uint64_t completion_wait_start_cycles;
    uint64_t all_executors_drained_cycles;
    uint64_t all_tasks_resolved_cycles;
    uint64_t shutdown_ready_cycles;
    uint64_t completion_poll_count;
    uint64_t completion_poll_cycles;
    uint64_t error_poll_count;
    uint64_t lifecycle_reserved[6];

    volatile uint64_t error_claimed;
    volatile uint64_t scheduler_error;
    volatile uint64_t error_task_id;
    volatile uint64_t error_core_id;
    volatile uint64_t error_core_type;
    volatile uint64_t error_graph_task_count;
    volatile uint64_t error_descriptors_address;
    volatile uint64_t error_payloads_address;
    volatile uint64_t error_task_window_mask;
    uint64_t error_reserved[7];
};

struct alignas(128) AicpuCoreLifecycleTraceV1 {
    uint64_t worker_id;
    uint64_t aicpu_thread_id;
    uint64_t core_type;
    uint64_t physical_core_id;
    uint64_t register_release_cycles;
    uint64_t exit_signal_cycles;
    uint64_t exit_ack_cycles;
    uint64_t reserved[9];
};

struct alignas(128) AicoreWorkerContextV1 {
    int32_t core_type;
    int32_t physical_core_id;
    int32_t type_rank;
    int32_t active;
    uint64_t run_control_offset;
    uint64_t task_controls_offset;
    uint64_t aic_stream_offset;
    uint64_t aiv_stream_offset;
    uint64_t graph_descriptors_address;
    uint64_t graph_payloads_address;
    uint64_t sidecar_base_address;
    uint64_t dispatch_payload_offset;
    uint64_t trace_cells_offset;
    uint64_t task_window_mask;
    uint64_t graph_task_count;
    uint64_t worker_index;
    uint64_t completion_inboxes_offset;
    uint64_t inbox_index;

    int64_t pending_task_id[AICORE_PENDING_SLOT_COUNT_V1];
    int32_t pending_next_fanin[AICORE_PENDING_SLOT_COUNT_V1];
    int32_t pending_waiting_producer[AICORE_PENDING_SLOT_COUNT_V1];
    uint64_t cursor_exhausted;
    uint64_t lifecycle_state;
    uint64_t root_prepare_inboxes_offset;
    uint8_t debug_padding[72];

    uint64_t seeded_task_count;
    uint64_t ticket_claim_count;
    uint64_t ticket_exhaustion_count;
    uint64_t executed_task_count;
    uint64_t pending_peak;
    uint64_t task_state_poll_count;
    uint64_t fanin_state_load_count;
    uint64_t wake_register_count;
    uint64_t idle_iteration_count;
    uint64_t backoff_cycles;
    uint64_t claim_cycles;
    uint64_t dependency_wait_cycles;
    uint64_t payload_cycles;
    uint64_t kernel_cycles;
    uint64_t completion_enqueue_cycles;
    uint64_t drain_cycles;

    uint64_t drain_start_cycles;
    uint64_t drain_end_cycles;
    uint64_t wake_cas_retry_count;
    uint64_t wake_closed_retry_count;
    uint64_t wake_migrate_count;
    uint64_t wake_close_count;
    uint64_t completion_enqueue_count;
    uint64_t completion_batch_count;
    uint64_t completion_resolve_count;
    uint64_t completion_steal_count;
    uint64_t completion_link_wait_count;
    uint64_t completion_link_wait_max;
    uint64_t completion_lag_cycles;
    uint64_t completion_lag_max_cycles;
    uint64_t ready_to_kernel_cycles;
    uint64_t ready_to_kernel_max_cycles;

    uint64_t executor_drain_publish_start_cycles;
    uint64_t executor_drain_publish_end_cycles;
    uint64_t exit_wait_start_cycles;
    uint64_t exit_observed_cycles;
    uint64_t final_stats_publish_start_cycles;
    uint64_t final_stats_publish_end_cycles;
    uint64_t exit_ack_publish_cycles;
    uint64_t root_prepare_enqueue_count;
    uint64_t root_prepare_batch_count;
    uint64_t root_prepare_resolve_count;
    uint64_t root_prepare_steal_count;
    uint64_t root_prepare_link_wait_count;
    uint64_t root_prepare_link_wait_max;
    uint64_t root_prepare_cycles;
    uint64_t termination_reserved[2];
};

struct alignas(128) AicoreTaskTraceCellV1 {
    volatile uint64_t valid;
    uint64_t claim_kind;
    uint64_t worker_id;
    uint64_t core_type;
    uint64_t task_id;
    uint64_t stream_index;
    uint64_t claim_start_cycles;
    uint64_t claim_end_cycles;
    uint64_t pending_wait_start_cycles;
    uint64_t pending_wait_end_cycles;
    uint64_t payload_start_cycles;
    uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t completion_end_cycles;
    uint64_t ready_scan_start_cycles;
    uint64_t ready_observe_cycles;
    uint64_t completion_bookkeeping_end_cycles;
    uint64_t completion_service_start_cycles;
    uint64_t completion_service_end_cycles;
    uint64_t previous_trace_commit_end_cycles;
    uint64_t initialize_end_cycles;
    uint64_t route_end_cycles;
    uint64_t startup_reserved[2];
    uint64_t aicore_entry_cycles;
    uint64_t handshake_publish_cycles;
    uint64_t register_release_cycles;
    uint64_t descriptor_cache_observed_cycles;
    uint64_t reserved[4];
};

struct AicoreExecutionSidecarLayoutV1 {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t run_control_offset;
    uint64_t aicpu_lifecycle_traces_offset;
    uint64_t worker_contexts_offset;
    uint64_t dispatch_payloads_offset;
    uint64_t claim_bindings_offset;
    uint64_t task_controls_offset;
    uint64_t completion_inboxes_offset;
    uint64_t root_prepare_inboxes_offset;
    uint64_t aic_stream_offset;
    uint64_t aic_tickets_offset;
    uint64_t aiv_stream_offset;
    uint64_t aiv_tickets_offset;
    uint64_t trace_cells_offset;
};

static_assert(sizeof(AicoreTaskControlV1) == 128, "task control layout changed");
static_assert(alignof(AicoreTaskControlV1) == 128, "task control alignment changed");
static_assert(offsetof(AicoreTaskControlV1, state) == 0, "task state offset changed");
static_assert(offsetof(AicoreTaskControlV1, wake_list_head) == 8, "wake head offset changed");
static_assert(offsetof(AicoreTaskControlV1, next_waiter) == 64, "waiter metadata needs its own line");
static_assert(offsetof(AicoreTaskControlV1, completion_next) == 80, "completion link offset changed");
static_assert(sizeof(AicoreCompletionInboxV1) == 128, "completion inbox layout changed");
static_assert(alignof(AicoreCompletionInboxV1) == 128, "completion inbox alignment changed");
static_assert(sizeof(AicoreRootPrepareInboxV1) == 128, "root prepare inbox layout changed");
static_assert(alignof(AicoreRootPrepareInboxV1) == 128, "root prepare inbox alignment changed");
static_assert(sizeof(AicoreTaskClaimBindingV1) == 64, "claim binding must occupy one cache line");
static_assert(alignof(AicoreTaskClaimBindingV1) == 64, "claim binding must be cache-line aligned");
static_assert(sizeof(AicoreTaskTicketV1) == 8, "task ticket layout changed");
static_assert(alignof(AicoreTaskTicketV1) == 8, "task tickets must not straddle 8-byte boundaries");
static_assert(sizeof(AicoreTaskStreamV1) == 256, "task stream layout changed");
static_assert(offsetof(AicoreTaskStreamV1, next_index) == 128, "ticket cursor must have its own line");
static_assert(sizeof(AicoreRunControlV1) == 384, "run control layout changed");
static_assert(offsetof(AicoreRunControlV1, executed_task_count) == 128, "lifecycle atomics need their own line");
static_assert(offsetof(AicoreRunControlV1, error_claimed) == 256, "error state needs its own line");
static_assert(sizeof(AicpuCoreLifecycleTraceV1) == 128, "AICPU lifecycle trace layout changed");
static_assert(alignof(AicpuCoreLifecycleTraceV1) == 128, "AICPU lifecycle trace must be cache-line aligned");
static_assert(sizeof(AicoreWorkerContextV1) == 640, "worker context layout changed");
static_assert(offsetof(AicoreWorkerContextV1, pending_task_id) == 128, "debug snapshot offset changed");
static_assert(offsetof(AicoreWorkerContextV1, seeded_task_count) == 256, "worker stats offset changed");
static_assert(offsetof(AicoreWorkerContextV1, drain_start_cycles) == 384, "worker trace offset changed");
static_assert(
    offsetof(AicoreWorkerContextV1, executor_drain_publish_start_cycles) == 512,
    "worker termination trace offset changed"
);
static_assert(sizeof(AicoreTaskTraceCellV1) == 256, "task trace layout changed");
static_assert(
    offsetof(AicoreTaskTraceCellV1, completion_bookkeeping_end_cycles) == 128,
    "post-completion trace needs its own line"
);
static_assert(offsetof(AicoreTaskTraceCellV1, aicore_entry_cycles) == 192, "startup trace needs its own line");

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<AicoreTaskControlV1>);
static_assert(std::is_trivially_copyable_v<AicoreTaskControlV1>);
static_assert(std::is_standard_layout_v<AicoreCompletionInboxV1>);
static_assert(std::is_trivially_copyable_v<AicoreCompletionInboxV1>);
static_assert(std::is_standard_layout_v<AicoreRootPrepareInboxV1>);
static_assert(std::is_trivially_copyable_v<AicoreRootPrepareInboxV1>);
static_assert(std::is_standard_layout_v<AicoreTaskClaimBindingV1>);
static_assert(std::is_trivially_copyable_v<AicoreTaskClaimBindingV1>);
static_assert(std::is_standard_layout_v<AicoreTaskTicketV1>);
static_assert(std::is_trivially_copyable_v<AicoreTaskTicketV1>);
static_assert(std::is_standard_layout_v<AicoreTaskStreamV1>);
static_assert(std::is_trivially_copyable_v<AicoreTaskStreamV1>);
static_assert(std::is_standard_layout_v<AicoreRunControlV1>);
static_assert(std::is_trivially_copyable_v<AicoreRunControlV1>);
static_assert(std::is_standard_layout_v<AicpuCoreLifecycleTraceV1>);
static_assert(std::is_trivially_copyable_v<AicpuCoreLifecycleTraceV1>);
static_assert(std::is_standard_layout_v<AicoreWorkerContextV1>);
static_assert(std::is_trivially_copyable_v<AicoreWorkerContextV1>);
static_assert(std::is_standard_layout_v<AicoreTaskTraceCellV1>);
static_assert(std::is_trivially_copyable_v<AicoreTaskTraceCellV1>);
#endif

inline bool aicore_sidecar_checked_add_v1(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool aicore_sidecar_checked_mul_v1(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool aicore_sidecar_checked_align_v1(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!aicore_sidecar_checked_add_v1(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool aicore_sidecar_reserve_v1(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !aicore_sidecar_checked_align_v1(*cursor, alignment, &aligned)) {
        return false;
    }
    uint64_t end = 0;
    if (!aicore_sidecar_checked_add_v1(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool aicore_sidecar_plan_v1(
    uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, AicoreExecutionSidecarLayoutV1 *layout
) {
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count ||
        aic_task_count > task_count - aiv_task_count) {
        return false;
    }

    AicoreExecutionSidecarLayoutV1 next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    uint64_t cursor = 0;
    uint64_t bytes = 0;
    if (!aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreRunControlV1), alignof(AicoreRunControlV1), &next.run_control_offset
        ) ||
        !aicore_sidecar_checked_mul_v1(AICORE_WORKER_CAPACITY_V1, sizeof(AicpuCoreLifecycleTraceV1), &bytes) ||
        !aicore_sidecar_reserve_v1(
            &cursor, bytes, alignof(AicpuCoreLifecycleTraceV1), &next.aicpu_lifecycle_traces_offset
        ) ||
        !aicore_sidecar_checked_mul_v1(AICORE_WORKER_CAPACITY_V1, sizeof(AicoreWorkerContextV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreWorkerContextV1), &next.worker_contexts_offset) ||
        !aicore_sidecar_checked_mul_v1(
            AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1, sizeof(PTO2DispatchPayload), &bytes
        ) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(PTO2DispatchPayload), &next.dispatch_payloads_offset) ||
        !aicore_sidecar_checked_mul_v1(task_count, sizeof(AicoreTaskClaimBindingV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreTaskClaimBindingV1), &next.claim_bindings_offset) ||
        !aicore_sidecar_checked_mul_v1(task_count, sizeof(AicoreTaskControlV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreTaskControlV1), &next.task_controls_offset) ||
        !aicore_sidecar_checked_mul_v1(AICORE_WORKER_CAPACITY_V1, sizeof(AicoreCompletionInboxV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreCompletionInboxV1), &next.completion_inboxes_offset) ||
        !aicore_sidecar_checked_mul_v1(AICORE_WORKER_CAPACITY_V1, sizeof(AicoreRootPrepareInboxV1), &bytes) ||
        !aicore_sidecar_reserve_v1(
            &cursor, bytes, alignof(AicoreRootPrepareInboxV1), &next.root_prepare_inboxes_offset
        ) ||
        !aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreTaskStreamV1), alignof(AicoreTaskStreamV1), &next.aic_stream_offset
        ) ||
        !aicore_sidecar_checked_mul_v1(aic_task_count, sizeof(AicoreTaskTicketV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreTaskTicketV1), &next.aic_tickets_offset) ||
        !aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreTaskStreamV1), alignof(AicoreTaskStreamV1), &next.aiv_stream_offset
        ) ||
        !aicore_sidecar_checked_mul_v1(aiv_task_count, sizeof(AicoreTaskTicketV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreTaskTicketV1), &next.aiv_tickets_offset) ||
        !aicore_sidecar_checked_mul_v1(task_count, sizeof(AicoreTaskTraceCellV1), &bytes) ||
        !aicore_sidecar_reserve_v1(&cursor, bytes, alignof(AicoreTaskTraceCellV1), &next.trace_cells_offset) ||
        !aicore_sidecar_checked_align_v1(cursor, AICORE_SIDECAR_ALIGNMENT_V1, &next.total_size)) {
        return false;
    }
    *layout = next;
    return true;
}

template <typename T>
inline __host__ __aicore__ __gm__ T *aicore_sidecar_at_v1(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

inline bool aicore_sidecar_init_v1(void *base, const AicoreExecutionSidecarLayoutV1 &layout) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (AICORE_SIDECAR_ALIGNMENT_V1 - 1)) != 0) {
        return false;
    }
    __builtin_memset(base, 0, static_cast<size_t>(layout.total_size));
    auto *bindings = aicore_sidecar_at_v1<AicoreTaskClaimBindingV1>(base, layout.claim_bindings_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i)
        bindings[i].task_id = AICORE_TASK_ID_INVALID_V1;
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].state = static_cast<int64_t>(AicoreTaskStateV1::BLOCKED);
        controls[i].wake_list_head = AICORE_WAKE_LIST_OPEN_V1;
        controls[i].next_waiter = AICORE_TASK_ID_INVALID_V1;
        controls[i].next_fanin_index = 0;
        controls[i].waiting_producer = static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1);
        controls[i].completion_next = AICORE_COMPLETION_LINK_UNPUBLISHED_V1;
    }
    auto *inboxes = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(base, layout.completion_inboxes_offset);
    for (uint64_t i = 0; i < AICORE_WORKER_CAPACITY_V1; ++i)
        inboxes[i].head = AICORE_COMPLETION_INBOX_EMPTY_V1;
    auto *root_prepare_inboxes =
        aicore_sidecar_at_v1<AicoreRootPrepareInboxV1>(base, layout.root_prepare_inboxes_offset);
    for (uint64_t i = 0; i < AICORE_WORKER_CAPACITY_V1; ++i)
        root_prepare_inboxes[i].head = AICORE_ROOT_PREPARE_INBOX_EMPTY_V1;
    auto *aic_stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(base, layout.aic_stream_offset);
    aic_stream->tickets_offset = layout.aic_tickets_offset;
    aic_stream->task_count = layout.aic_task_count;
    auto *aiv_stream = aicore_sidecar_at_v1<AicoreTaskStreamV1>(base, layout.aiv_stream_offset);
    aiv_stream->tickets_offset = layout.aiv_tickets_offset;
    aiv_stream->task_count = layout.aiv_task_count;
    auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(base, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < AICORE_WORKER_CAPACITY_V1; ++worker) {
        contexts[worker].physical_core_id = -1;
        for (uint32_t slot = 0; slot < AICORE_PENDING_SLOT_COUNT_V1; ++slot) {
            contexts[worker].pending_task_id[slot] = AICORE_TASK_ID_INVALID_V1;
            contexts[worker].pending_waiting_producer[slot] = static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1);
        }
    }
    return true;
}
