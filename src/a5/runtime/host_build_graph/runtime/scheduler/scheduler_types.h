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

#include "dispatch_payload.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __host__
#define __host__
#endif

inline constexpr uint64_t SCHEDULER_STATE_ALIGNMENT = 128;
inline constexpr uint64_t SCHEDULER_WORKER_CAPACITY = 108;
inline constexpr uint32_t SCHEDULER_PENDING_SLOT_COUNT = 2;
inline constexpr uint32_t SCHEDULER_CALLABLE_CAPACITY = 1024;
inline constexpr uint32_t SCHEDULER_CORE_TYPE_COUNT = 2;
inline constexpr uint32_t SCHEDULER_CLUSTER_CAPACITY = SCHEDULER_WORKER_CAPACITY / 3;
inline constexpr uint32_t SCHEDULER_GANG_COHORT_COUNT = 2;
inline constexpr uint32_t SCHEDULER_READY_DIRECTORY_RESOLVERS_PER_SHARD = 7;
inline constexpr uint32_t SCHEDULER_READY_DIRECTORY_SHARD_COUNT =
    (SCHEDULER_CLUSTER_CAPACITY + SCHEDULER_READY_DIRECTORY_RESOLVERS_PER_SHARD - 1) /
    SCHEDULER_READY_DIRECTORY_RESOLVERS_PER_SHARD;
inline constexpr int64_t SCHEDULER_TASK_ID_INVALID = -1;
inline constexpr int64_t SCHEDULER_WAKE_LIST_OPEN = -1;
inline constexpr int64_t SCHEDULER_WAKE_LIST_CLOSED = -2;
inline constexpr int64_t SCHEDULER_INBOX_EMPTY = -1;
inline constexpr int64_t SCHEDULER_INBOX_LINK_UNPUBLISHED = -2;
inline constexpr uint64_t SCHEDULER_READY_PENDING_EMPTY = UINT64_MAX;

enum class SchedulerTaskState : int64_t {
    BLOCKED = 0,
    READY = 1,
    DONE = 2,
    DISPATCHING = 3,
};

enum class SchedulerReadySource : uint8_t {
    LOCAL = 0,
    STOLEN = 1,
};

enum class SchedulerDispatchSlotState : uint8_t {
    EMPTY = 0,
    FREE = 1,
    FILLING = 2,
    READY = 3,
    GATED = 4,
};

enum SchedulerTaskMetadataFlags : uint8_t {
    SCHEDULER_TASK_EXECUTABLE = 1U << 0,
    SCHEDULER_TASK_HAS_FANIN = 1U << 1,
    SCHEDULER_TASK_MIX = 1U << 2,
    SCHEDULER_TASK_SPMD = 1U << 3,
    SCHEDULER_TASK_SYNC_START = 1U << 4,
    SCHEDULER_TASK_INLINE = 1U << 5,
    SCHEDULER_TASK_HAS_PREDICATE = 1U << 6,
};

constexpr int32_t SCHEDULER_TASK_TIMING_SLOT_COUNT = 16;

struct alignas(16) SchedulerTaskMetadata {
    uint16_t kernel_ids[3];
    uint8_t active_mask;
    uint8_t flags;
    uint16_t logical_block_num;
    uint16_t total_required_subtasks;
    int32_t timing_slot;
};

inline __host__ __aicore__ bool scheduler_task_is_executable(uint8_t flags) {
    return (flags & SCHEDULER_TASK_EXECUTABLE) != 0;
}

inline __host__ __aicore__ bool scheduler_task_has_fanin(uint8_t flags) {
    return (flags & SCHEDULER_TASK_HAS_FANIN) != 0;
}

inline __host__ __aicore__ bool scheduler_task_is_mix(uint8_t flags) { return (flags & SCHEDULER_TASK_MIX) != 0; }

inline __host__ __aicore__ bool scheduler_task_is_spmd(uint8_t flags) { return (flags & SCHEDULER_TASK_SPMD) != 0; }

inline __host__ __aicore__ bool scheduler_task_requires_sync_start(uint8_t flags) {
    return (flags & SCHEDULER_TASK_SYNC_START) != 0;
}

inline __host__ __aicore__ bool scheduler_task_is_inline(uint8_t flags) { return (flags & SCHEDULER_TASK_INLINE) != 0; }

inline __host__ __aicore__ bool scheduler_task_has_predicate(uint8_t flags) {
    return (flags & SCHEDULER_TASK_HAS_PREDICATE) != 0;
}

inline __host__ __aicore__ bool scheduler_task_is_gang(uint8_t flags) {
    return (flags & (SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD)) != 0;
}

inline __host__ __aicore__ uint32_t scheduler_task_priority_bit(uint8_t flags) {
    if (scheduler_task_requires_sync_start(flags)) return 1U;
    if (scheduler_task_is_mix(flags)) return 2U;
    if (scheduler_task_is_spmd(flags)) return 4U;
    return 0;
}

struct alignas(128) SchedulerTaskControl {
    // Cross-core state publication and wake-list RMWs share the first line.
    volatile int64_t state;
    volatile int64_t wake_list_head;
    uint8_t atomic_line_padding[48];

    // next_waiter links the dependency wake list while BLOCKED and the Ready
    // inbox after routing.
    int64_t next_waiter;
    int32_t next_fanin_index;
    int32_t waiting_producer;
    uint64_t completion_resolve_start_cycles;
    uint64_t completion_resolve_end_cycles;
    uint64_t ready_publish_cycles;
    uint64_t resolver_worker_id;
    uint8_t scheduler_line_padding[16];
};

struct alignas(64) SchedulerCompletionInbox {
    volatile uint32_t completed_generations[SCHEDULER_PENDING_SLOT_COUNT];
    uint8_t completion_line_padding[56];
};

struct alignas(128) SchedulerReadyInbox {
    volatile int64_t head;
    uint8_t atomic_line_padding[120];
};

struct alignas(64) SchedulerReadyOwnerQueue {
    // Head and tail are one owner-only device word so an ld_dev cannot
    // observe endpoints from different updates.
    volatile uint64_t pending_endpoints{SCHEDULER_READY_PENDING_EMPTY};
    volatile uint64_t advertised{0};
    uint8_t owner_line_padding[48];
};

struct alignas(128) SchedulerReadyOwnerState {
    SchedulerReadyOwnerQueue queues[SCHEDULER_CORE_TYPE_COUNT];
};

enum class SchedulerGangCohortState : uint64_t {
    FREE = 0,
    DRAINING = 1,
    STAGING = 2,
    RELEASING = 3,
    DISPATCHING = 4,
    EXECUTING = 5,
    RETIRING = 6,
};

struct alignas(128) SchedulerGangCoordinator {
    volatile uint64_t ready_priority_bits;
    uint8_t priority_line_padding[56];

    volatile uint64_t active_dispatch_cohort;
    uint64_t next_generation;
    uint64_t scan_cursor;
    uint64_t gang_task_count;
    uint64_t resolver_count;
    uint64_t cohort_count;
    uint64_t reserved0;
    uint64_t owner_reserved;

    uint64_t admitted_count;
    uint64_t sync_drain_count;
    uint64_t mix_dispatch_count;
    uint64_t spmd_dispatch_count;
    uint64_t capacity_reject_count;
    uint64_t reserved[3];
};

struct alignas(128) SchedulerGangCohort {
    volatile uint64_t state;
    int64_t task_id;
    uint64_t generation;
    uint64_t priority_bit;
    uint64_t active_mask;
    uint64_t logical_block_num;
    uint64_t participant_count;
    uint64_t local_stride;
    uint64_t admitted_cycles;
    uint64_t drain_complete_cycles;
    uint64_t stage_complete_cycles;
    uint64_t dispatch_complete_cycles;
    uint64_t completion_cycles;
    uint64_t reserved[3];
};

// One Resolver owns one participant cell. The second line contains generation
// tokens observed only by its parent in the binary Resolver tree.
struct alignas(128) SchedulerGangParticipant {
    volatile uint64_t config_generation;
    int64_t task_id;
    uint32_t active_mask;
    uint32_t logical_block_num;
    uint32_t local_expected_subtasks;
    uint32_t local_published_subtasks;
    uint32_t local_completed_subtasks;
    uint32_t block_stride;
    uint32_t next_block[2];
    uint32_t participant_count;
    uint32_t forwarded_state;
    uint64_t forwarded_generation;

    volatile uint64_t drain_local_token;
    volatile uint64_t drain_subtree_token;
    volatile uint64_t stage_local_token;
    volatile uint64_t stage_subtree_token;
    volatile uint64_t dispatch_local_token;
    volatile uint64_t dispatch_subtree_token;
    volatile uint64_t completion_local_token;
    volatile uint64_t completion_subtree_token;
};

// Each Resolver polls its own command line. Resolver0 seeds the root and every
// parent forwards transitions to two children, avoiding a globally contended
// cohort line and bounding sync-start release skew by the tree depth.
struct alignas(128) SchedulerGangCommand {
    volatile uint64_t generation[SCHEDULER_GANG_COHORT_COUNT];
    volatile uint64_t state[SCHEDULER_GANG_COHORT_COUNT];
    uint8_t padding[96];
};

struct alignas(64) SchedulerReadyDirectoryShard {
    volatile uint64_t bits;
    uint8_t cache_line_padding[64 - sizeof(uint64_t)];
};

struct alignas(128) SchedulerReadyDirectory {
    SchedulerReadyDirectoryShard core_types[SCHEDULER_CORE_TYPE_COUNT][SCHEDULER_READY_DIRECTORY_SHARD_COUNT];
    volatile uint64_t bootstrap_ready_types[SCHEDULER_WORKER_CAPACITY];
};

// Resolver-owned metadata occupies the first line. The Executor polls only
// publication in the second line.
struct alignas(128) SchedulerDispatchSlot {
    int64_t task_id;
    uint64_t ready_inbox_index;
    uint64_t claim_start_cycles;
    uint64_t claim_end_cycles;
    uint64_t claim_worker_id;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t has_fanin;
    uint8_t ready_source;
    uint8_t pending_slot;
    uint16_t block_num;
    uint32_t generation;
    uint32_t block_idx;
    uint32_t cohort_generation;
    uint8_t cohort_index;
    uint8_t gang;
    uint8_t metadata_padding[2];

    volatile uint64_t publication;
    uint8_t publication_padding[56];
};

struct alignas(128) SchedulerRunControl {
    uint64_t config_reserved_prefix[2];
    uint64_t active_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t aic_active_worker_count;
    uint64_t aiv_active_worker_count;
    uint64_t dispatch_reserved;
    volatile uint64_t dispatch_payloads_offset;
    uint64_t task_metadata_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_directory_offset;
    uint64_t directory_reserved;
    uint64_t gang_coordinator_offset;
    uint64_t gang_cohorts_offset;
    uint64_t resolver_count;

    volatile uint64_t executed_task_count;
    volatile uint64_t resolved_task_count;
    volatile uint64_t bootstrap_arrived_count;
    volatile uint64_t bootstrap_complete;
    uint64_t completion_wait_start_cycles;
    uint64_t bootstrap_complete_cycles;
    uint64_t all_tasks_resolved_cycles;
    uint64_t shutdown_ready_cycles;
    uint64_t completion_poll_count;
    uint64_t completion_poll_cycles;
    uint64_t error_poll_count;
    volatile uint64_t bootstrap_scan_arrived_count;
    volatile uint64_t bootstrap_scan_complete;
    uint64_t lifecycle_reserved[3];

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

struct alignas(128) AicpuCoreLifecycleTrace {
    uint64_t worker_id;
    uint64_t aicpu_thread_id;
    uint64_t core_type;
    uint64_t physical_core_id;
    uint64_t handshake_observed_cycles;
    uint64_t handshake_partition_complete_cycles;
    uint64_t config_start_cycles;
    uint64_t topology_complete_cycles;
    uint64_t context_publish_complete_cycles;
    uint64_t bootstrap_wait_start_cycles;
    uint64_t bootstrap_complete_cycles;
    uint64_t register_release_cycles;
    uint64_t exit_signal_cycles;
    uint64_t exit_ack_cycles;
    uint64_t reserved[2];
};

struct alignas(128) SchedulerTailTrace {
    volatile uint64_t valid;
    uint64_t start_cycles;
    uint64_t end_cycles;
    uint64_t completion_scan_cycles;
    uint64_t completion_consume_cycles;
    uint64_t completion_resolve_cycles;
    uint64_t completion_ready_publish_cycles;
    uint64_t completion_refill_cycles;
    uint64_t completion_finalize_cycles;
    uint64_t gang_service_cycles;
    uint64_t dispatch_probe_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t dispatch_claim_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t dispatch_prepare_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t dispatch_materialize_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t dispatch_publish_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t ready_poll_cycles;
    uint64_t backoff_cycles;
};

struct alignas(128) SchedulerWorkerContext {
    volatile int32_t core_type;
    int32_t physical_core_id;
    volatile int32_t type_rank;
    volatile int32_t active;
    volatile uint64_t run_control_offset;
    volatile uint64_t task_controls_offset;
    volatile uint64_t graph_descriptors_address;
    volatile uint64_t graph_payloads_address;
    volatile uint64_t scheduler_state_base_address;
    volatile uint64_t dispatch_payload_offset;
    volatile uint64_t trace_cells_offset;
    volatile uint64_t task_window_mask;
    volatile uint64_t graph_task_count;
    volatile uint64_t worker_index;
    volatile uint64_t completion_inboxes_offset;
    volatile uint64_t inbox_index;
    volatile uint64_t ready_owner_states_offset;
    uint64_t runtime_offset_padding;

    volatile uint64_t task_metadata_offset;
    volatile uint64_t ready_inboxes_offset;
    volatile uint64_t ready_directory_offset;
    uint64_t scheduling_reserved;
    volatile uint64_t worker_contexts_offset;
    volatile uint64_t dispatch_slots_offset;
    volatile uint64_t callable_addresses_offset;
    volatile uint64_t runtime_worker_count;
    volatile uint64_t bootstrap_done;
    uint64_t bootstrap_scan_end_cycles;
    uint64_t target_bootstrap_start_cycles;
    uint64_t target_bootstrap_end_cycles;
    uint64_t bootstrap_target_aic_cycles;
    uint64_t bootstrap_target_aiv_cycles;
    uint64_t bootstrap_ready_claim_aic_cycles;
    uint64_t bootstrap_ready_claim_aiv_cycles;

    volatile uint64_t gang_coordinator_offset;
    volatile uint64_t gang_cohorts_offset;
    volatile uint64_t gang_participants_offset;
    volatile uint64_t gang_commands_offset;
    volatile uint64_t resolver_count;
    volatile uint64_t cluster_count;
    volatile uint64_t cluster_index;
    volatile uint64_t resolver_index;
    volatile uint64_t resolver_worker_id;
    volatile uint64_t is_resolver;
    volatile uint64_t cluster_worker_ids[3];
    uint64_t topology_reserved[3];

    uint64_t bootstrap_task_count;
    uint64_t ready_enqueue_count;
    uint64_t ready_batch_count;
    uint64_t ready_pop_count;
    uint64_t ready_steal_count;
    uint64_t ready_cas_retry_count;
    uint64_t ready_link_wait_count;
    uint64_t ready_link_wait_max;
    uint64_t ready_stats_reserved[2];
    uint64_t executed_task_count;
    uint64_t task_state_poll_count;
    uint64_t fanin_state_load_count;
    uint64_t wake_register_count;
    uint64_t idle_iteration_count;
    uint64_t backoff_cycles;

    uint64_t wake_cas_retry_count;
    uint64_t wake_closed_retry_count;
    uint64_t wake_migrate_count;
    uint64_t wake_close_count;
    uint64_t completion_enqueue_count;
    uint64_t completion_resolve_count;
    uint64_t completion_stats_reserved[6];
    uint64_t ready_to_kernel_cycles;
    uint64_t ready_to_kernel_max_cycles;
    uint64_t payload_cycles;
    uint64_t kernel_cycles;

    uint64_t completion_enqueue_cycles;
    uint64_t bootstrap_start_cycles;
    uint64_t bootstrap_end_cycles;
    uint64_t drain_start_cycles;
    uint64_t drain_end_cycles;
    uint64_t exit_wait_start_cycles;
    uint64_t exit_observed_cycles;
    uint64_t final_stats_publish_start_cycles;
    uint64_t final_stats_publish_end_cycles;
    uint64_t exit_ack_publish_cycles;
    uint64_t bootstrap_slot_fill_aic_cycles;
    uint64_t bootstrap_slot_fill_aiv_cycles;
    uint64_t termination_reserved[4];

    SchedulerTailTrace scheduler_tail_trace;
};

struct alignas(128) SchedulerTaskTrace {
    volatile uint64_t valid;
    uint64_t ready_source;
    uint64_t worker_id;
    uint64_t task_id;
    uint64_t claim_worker_id;
    uint64_t claim_start_cycles;
    uint64_t claim_end_cycles;
    uint64_t previous_trace_commit_end_cycles;

    uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t completion_end_cycles;
    uint64_t ready_scan_start_cycles;
    uint64_t ready_observe_cycles;
    uint64_t completion_bookkeeping_end_cycles;
    uint64_t completion_id;
    uint64_t completion_inbox_index;

    uint64_t ready_transition_cycles;
    uint64_t inter_task_completion_service_cycles;
    uint64_t inter_task_dispatch_aic_cycles;
    uint64_t inter_task_dispatch_aiv_cycles;
    uint64_t inter_task_ready_poll_cycles;
    uint64_t inter_task_backoff_cycles;
    uint64_t aicore_entry_cycles;
    uint64_t handshake_publish_cycles;

    uint64_t register_release_cycles;
    uint64_t descriptor_cache_observed_cycles;
    uint64_t completion_prepare_start_cycles;
    uint64_t refill_resolver_worker_id;
    uint64_t refill_start_cycles;
    uint64_t refill_end_cycles;
    uint64_t refill_task_id;
    uint64_t inter_task_completion_refill_cycles;

    uint64_t inter_task_completion_scan_cycles;
    uint64_t inter_task_completion_consume_cycles;
    uint64_t inter_task_completion_resolve_cycles;
    uint64_t inter_task_completion_ready_publish_cycles;
    uint64_t inter_task_completion_finalize_cycles;
    uint64_t inter_task_gang_service_cycles;
    uint64_t inter_task_dispatch_probe_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t inter_task_dispatch_claim_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t inter_task_dispatch_prepare_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t inter_task_dispatch_materialize_cycles[SCHEDULER_CORE_TYPE_COUNT];
    uint64_t inter_task_dispatch_publish_cycles[SCHEDULER_CORE_TYPE_COUNT];
};

struct SchedulerLayout {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t run_control_offset;
    uint64_t aicpu_lifecycle_traces_offset;
    uint64_t worker_contexts_offset;
    uint64_t dispatch_payloads_offset;
    uint64_t dispatch_slots_offset;
    uint64_t callable_addresses_offset;
    uint64_t task_metadata_offset;
    uint64_t task_controls_offset;
    uint64_t completion_inboxes_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_owner_states_offset;
    uint64_t ready_directory_offset;
    uint64_t trace_cells_offset;
    uint64_t gang_coordinator_offset;
    uint64_t gang_cohorts_offset;
    uint64_t gang_participants_offset;
    uint64_t gang_commands_offset;
    uint64_t executable_task_count;
    uint64_t executable_subtask_count;
    uint64_t gang_task_count;
    uint64_t aic_worker_demand;
    uint64_t aiv_worker_demand;
};

static_assert(sizeof(SchedulerTaskMetadata) == 16, "task metadata layout changed");
static_assert(sizeof(SchedulerTaskControl) == 128, "task control layout changed");
static_assert(offsetof(SchedulerTaskControl, next_waiter) == 64, "waiter metadata needs its own line");
static_assert(sizeof(SchedulerCompletionInbox) == 64, "completion line must occupy one cache line");
static_assert(offsetof(SchedulerCompletionInbox, completed_generations) == 0, "completion generations must lead line");
static_assert(
    sizeof(decltype(SchedulerCompletionInbox::completed_generations)) == sizeof(uint64_t),
    "completion generations must fit one 64-bit device load"
);
static_assert(sizeof(SchedulerReadyInbox) == 128, "ready inbox layout changed");
static_assert(sizeof(SchedulerReadyOwnerQueue) == 64, "ready owner queue must occupy one cache line");
static_assert(sizeof(SchedulerReadyOwnerState) == 128, "ready owner state must occupy two cache lines");
static_assert(sizeof(SchedulerGangCoordinator) == 256, "gang coordinator layout changed");
static_assert(
    offsetof(SchedulerGangCoordinator, active_dispatch_cohort) == 64,
    "owner-only gang state must not share the priority line"
);
static_assert(sizeof(SchedulerGangCohort) == 128, "gang cohort layout changed");
static_assert(sizeof(SchedulerGangParticipant) == 128, "gang participant layout changed");
static_assert(sizeof(SchedulerGangCommand) == 128, "gang command layout changed");
static_assert(sizeof(SchedulerReadyDirectoryShard) == 64, "ready directory shard must occupy one cache line");
static_assert(
    offsetof(SchedulerReadyDirectory, bootstrap_ready_types) ==
        SCHEDULER_CORE_TYPE_COUNT * SCHEDULER_READY_DIRECTORY_SHARD_COUNT * 64,
    "bootstrap flags must follow the ready directory shards"
);
static_assert(
    sizeof(SchedulerReadyDirectory) == ((offsetof(SchedulerReadyDirectory, bootstrap_ready_types) +
                                         SCHEDULER_WORKER_CAPACITY * sizeof(uint64_t) + 127) /
                                        128 * 128),
    "ready directory layout changed"
);
static_assert(sizeof(SchedulerDispatchSlot) == 128, "dispatch slot must occupy two cache lines");
static_assert(offsetof(SchedulerDispatchSlot, publication) == 64, "dispatch publication needs its own line");
static_assert(sizeof(SchedulerRunControl) == 384, "run control layout changed");
static_assert(offsetof(SchedulerRunControl, executed_task_count) == 128, "lifecycle atomics need their own line");
static_assert(offsetof(SchedulerRunControl, error_claimed) == 256, "error state needs its own line");
static_assert(sizeof(AicpuCoreLifecycleTrace) == 128, "AICPU lifecycle trace layout changed");
static_assert(sizeof(SchedulerTailTrace) == 256, "scheduler tail trace layout changed");
static_assert(sizeof(SchedulerWorkerContext) == 1024, "worker context layout changed");
static_assert(offsetof(SchedulerWorkerContext, task_metadata_offset) == 128, "runtime offsets changed");
static_assert(offsetof(SchedulerWorkerContext, gang_coordinator_offset) == 256, "topology offsets changed");
static_assert(offsetof(SchedulerWorkerContext, bootstrap_task_count) == 384, "worker stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, wake_cas_retry_count) == 512, "wake stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, completion_enqueue_cycles) == 640, "termination stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, scheduler_tail_trace) == 768, "scheduler tail trace offset changed");
static_assert(sizeof(SchedulerTaskTrace) == 384, "task trace layout changed");

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<SchedulerTaskMetadata> && std::is_trivially_copyable_v<SchedulerTaskMetadata>);
static_assert(std::is_standard_layout_v<SchedulerTaskControl> && std::is_trivially_copyable_v<SchedulerTaskControl>);
static_assert(
    std::is_standard_layout_v<SchedulerCompletionInbox> && std::is_trivially_copyable_v<SchedulerCompletionInbox>
);
static_assert(std::is_standard_layout_v<SchedulerReadyInbox> && std::is_trivially_copyable_v<SchedulerReadyInbox>);
static_assert(
    std::is_standard_layout_v<SchedulerReadyOwnerState> && std::is_trivially_copyable_v<SchedulerReadyOwnerState>
);
static_assert(
    std::is_standard_layout_v<SchedulerGangCoordinator> && std::is_trivially_copyable_v<SchedulerGangCoordinator>
);
static_assert(std::is_standard_layout_v<SchedulerGangCohort> && std::is_trivially_copyable_v<SchedulerGangCohort>);
static_assert(
    std::is_standard_layout_v<SchedulerGangParticipant> && std::is_trivially_copyable_v<SchedulerGangParticipant>
);
static_assert(std::is_standard_layout_v<SchedulerGangCommand> && std::is_trivially_copyable_v<SchedulerGangCommand>);
static_assert(
    std::is_standard_layout_v<SchedulerReadyDirectory> && std::is_trivially_copyable_v<SchedulerReadyDirectory>
);
static_assert(std::is_standard_layout_v<SchedulerDispatchSlot> && std::is_trivially_copyable_v<SchedulerDispatchSlot>);
static_assert(std::is_standard_layout_v<SchedulerRunControl> && std::is_trivially_copyable_v<SchedulerRunControl>);
static_assert(
    std::is_standard_layout_v<SchedulerWorkerContext> && std::is_trivially_copyable_v<SchedulerWorkerContext>
);
static_assert(std::is_standard_layout_v<SchedulerTailTrace> && std::is_trivially_copyable_v<SchedulerTailTrace>);
#endif

inline bool scheduler_layout_checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool scheduler_layout_checked_mul(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool scheduler_layout_checked_align(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!scheduler_layout_checked_add(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool scheduler_layout_reserve(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !scheduler_layout_checked_align(*cursor, alignment, &aligned))
        return false;
    uint64_t end = 0;
    if (!scheduler_layout_checked_add(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool
scheduler_plan_layout(uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, SchedulerLayout *layout) {
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count) return false;
    SchedulerLayout next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define SCHEDULER_RESERVE_ARRAY(count, type, field)                 \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(&cursor, bytes, alignof(type), &next.field))
    if (!scheduler_layout_reserve(
            &cursor, sizeof(SchedulerRunControl), alignof(SchedulerRunControl), &next.run_control_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, AicpuCoreLifecycleTrace, aicpu_lifecycle_traces_offset) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerWorkerContext, worker_contexts_offset) ||
        !SCHEDULER_RESERVE_ARRAY(
            SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT, DispatchPayload, dispatch_payloads_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(
            SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT, SchedulerDispatchSlot, dispatch_slots_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_CALLABLE_CAPACITY, uint64_t, callable_addresses_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskMetadata, task_metadata_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskControl, task_controls_offset) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerCompletionInbox, completion_inboxes_offset) ||
        !SCHEDULER_RESERVE_ARRAY(
            SCHEDULER_CORE_TYPE_COUNT * SCHEDULER_WORKER_CAPACITY, SchedulerReadyInbox, ready_inboxes_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_CLUSTER_CAPACITY, SchedulerReadyOwnerState, ready_owner_states_offset) ||
        !scheduler_layout_reserve(
            &cursor, sizeof(SchedulerReadyDirectory), alignof(SchedulerReadyDirectory), &next.ready_directory_offset
        ) ||
        !scheduler_layout_reserve(
            &cursor, sizeof(SchedulerGangCoordinator), alignof(SchedulerGangCoordinator), &next.gang_coordinator_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_GANG_COHORT_COUNT, SchedulerGangCohort, gang_cohorts_offset) ||
        !SCHEDULER_RESERVE_ARRAY(
            SCHEDULER_GANG_COHORT_COUNT * SCHEDULER_CLUSTER_CAPACITY, SchedulerGangParticipant, gang_participants_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_CLUSTER_CAPACITY, SchedulerGangCommand, gang_commands_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskTrace, trace_cells_offset) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) {
#undef SCHEDULER_RESERVE_ARRAY
        return false;
    }
#undef SCHEDULER_RESERVE_ARRAY
    *layout = next;
    return true;
}

template <typename T>
inline __host__ __aicore__ __gm__ T *scheduler_state_at(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

inline bool scheduler_init_data_from_layout(void *base, const SchedulerLayout &layout) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0) return false;
    __builtin_memset(base, 0, static_cast<size_t>(layout.total_size));
    auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        controls[i].wake_list_head = SCHEDULER_WAKE_LIST_OPEN;
        controls[i].next_waiter = SCHEDULER_TASK_ID_INVALID;
        controls[i].waiting_producer = static_cast<int32_t>(SCHEDULER_TASK_ID_INVALID);
    }
    auto *slots = scheduler_state_at<SchedulerDispatchSlot>(base, layout.dispatch_slots_offset);
    for (uint64_t i = 0; i < SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT; ++i) {
        slots[i].task_id = SCHEDULER_TASK_ID_INVALID;
        slots[i].publication = static_cast<uint64_t>(SchedulerDispatchSlotState::EMPTY);
    }
    auto *ready = scheduler_state_at<SchedulerReadyInbox>(base, layout.ready_inboxes_offset);
    for (uint64_t i = 0; i < SCHEDULER_CORE_TYPE_COUNT * SCHEDULER_WORKER_CAPACITY; ++i)
        ready[i].head = SCHEDULER_INBOX_EMPTY;
    auto *ready_owners = scheduler_state_at<SchedulerReadyOwnerState>(base, layout.ready_owner_states_offset);
    for (uint64_t owner = 0; owner < SCHEDULER_CLUSTER_CAPACITY; ++owner) {
        for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type)
            ready_owners[owner].queues[type].pending_endpoints = SCHEDULER_READY_PENDING_EMPTY;
    }
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(base, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        contexts[worker].physical_core_id = -1;
        contexts[worker].cluster_index = UINT64_MAX;
        contexts[worker].resolver_index = UINT64_MAX;
        contexts[worker].resolver_worker_id = UINT64_MAX;
        contexts[worker].cluster_worker_ids[0] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[1] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[2] = UINT64_MAX;
    }
    auto *coordinator = scheduler_state_at<SchedulerGangCoordinator>(base, layout.gang_coordinator_offset);
    coordinator->active_dispatch_cohort = UINT64_MAX;
    coordinator->cohort_count = SCHEDULER_GANG_COHORT_COUNT;
    auto *cohorts = scheduler_state_at<SchedulerGangCohort>(base, layout.gang_cohorts_offset);
    for (uint32_t i = 0; i < SCHEDULER_GANG_COHORT_COUNT; ++i) {
        cohorts[i].state = static_cast<uint64_t>(SchedulerGangCohortState::FREE);
        cohorts[i].task_id = SCHEDULER_TASK_ID_INVALID;
    }
    auto *participants = scheduler_state_at<SchedulerGangParticipant>(base, layout.gang_participants_offset);
    for (uint32_t i = 0; i < SCHEDULER_GANG_COHORT_COUNT * SCHEDULER_CLUSTER_CAPACITY; ++i)
        participants[i].task_id = SCHEDULER_TASK_ID_INVALID;
    return true;
}
