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
inline constexpr uint32_t AICORE_CALLABLE_CAPACITY_V1 = 1024;
inline constexpr uint32_t AICORE_CORE_TYPE_COUNT_V1 = 2;
inline constexpr uint32_t AICORE_CLUSTER_CAPACITY_V1 = AICORE_WORKER_CAPACITY_V1 / 3;
inline constexpr uint32_t AICORE_GANG_COHORT_COUNT_V1 = 2;
inline constexpr uint32_t AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1 = 7;
inline constexpr uint32_t AICORE_READY_DIRECTORY_SHARD_COUNT_V1 =
    (AICORE_CLUSTER_CAPACITY_V1 + AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1 - 1) /
    AICORE_READY_DIRECTORY_RESOLVERS_PER_SHARD_V1;
inline constexpr uint32_t AICORE_FREE_SLOT_DIRECTORY_WORD_COUNT_V1 =
    (AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1 + 63) / 64;
inline constexpr int64_t AICORE_TASK_ID_INVALID_V1 = -1;
inline constexpr int64_t AICORE_WAKE_LIST_OPEN_V1 = -1;
inline constexpr int64_t AICORE_WAKE_LIST_CLOSED_V1 = -2;
inline constexpr int64_t AICORE_INBOX_EMPTY_V1 = -1;
inline constexpr int64_t AICORE_INBOX_LINK_UNPUBLISHED_V1 = -2;

enum class AicoreTaskStateV1 : int64_t {
    BLOCKED = 0,
    READY = 1,
    DONE = 2,
    DISPATCHING = 3,
};

enum class AicoreReadySourceV1 : uint8_t {
    LOCAL = 0,
    STOLEN = 1,
};

enum class AicoreDispatchPublicationV1 : uint8_t {
    EMPTY = 0,
    FREE = 1,
    FILLING = 2,
    READY = 3,
    GATED = 4,
};

enum AicoreTaskMetadataFlagsV1 : uint8_t {
    AICORE_TASK_EXECUTABLE_V1 = 1U << 0,
    AICORE_TASK_HAS_FANIN_V1 = 1U << 1,
    AICORE_TASK_MIX_V1 = 1U << 2,
    AICORE_TASK_SPMD_V1 = 1U << 3,
    AICORE_TASK_SYNC_START_V1 = 1U << 4,
};

struct alignas(16) AicoreTaskMetadataV1 {
    uint16_t kernel_ids[3];
    uint8_t active_mask;
    uint8_t flags;
    uint16_t logical_block_num;
    uint16_t total_required_subtasks;
    uint32_t reserved;
};

inline __host__ __aicore__ bool aicore_task_is_executable_v1(uint8_t flags) {
    return (flags & AICORE_TASK_EXECUTABLE_V1) != 0;
}

inline __host__ __aicore__ bool aicore_task_has_fanin_v1(uint8_t flags) {
    return (flags & AICORE_TASK_HAS_FANIN_V1) != 0;
}

inline __host__ __aicore__ bool aicore_task_is_mix_v1(uint8_t flags) { return (flags & AICORE_TASK_MIX_V1) != 0; }

inline __host__ __aicore__ bool aicore_task_is_spmd_v1(uint8_t flags) { return (flags & AICORE_TASK_SPMD_V1) != 0; }

inline __host__ __aicore__ bool aicore_task_requires_sync_start_v1(uint8_t flags) {
    return (flags & AICORE_TASK_SYNC_START_V1) != 0;
}

inline __host__ __aicore__ bool aicore_task_is_gang_v1(uint8_t flags) {
    return (flags & (AICORE_TASK_MIX_V1 | AICORE_TASK_SPMD_V1)) != 0;
}

inline __host__ __aicore__ uint32_t aicore_task_priority_bit_v1(uint8_t flags) {
    if (aicore_task_requires_sync_start_v1(flags)) return 1U;
    if (aicore_task_is_mix_v1(flags)) return 2U;
    if (aicore_task_is_spmd_v1(flags)) return 4U;
    return 0;
}

struct alignas(128) AicoreTaskControlV1 {
    // Cross-core state publication and wake-list RMWs share the first line.
    volatile int64_t state;
    volatile int64_t wake_list_head;
    uint8_t atomic_line_padding[48];

    // next_waiter links the dependency wake list while BLOCKED and the Ready
    // inbox after routing. inbox_next belongs exclusively to Completion, so a
    // Ready pop never mutates a link that competing consumers may still read.
    int64_t next_waiter;
    int32_t next_fanin_index;
    int32_t waiting_producer;
    volatile int64_t inbox_next;
    uint64_t completion_enqueue_cycles;
    uint64_t completion_resolve_start_cycles;
    uint64_t completion_resolve_end_cycles;
    uint64_t ready_publish_cycles;
    uint64_t resolver_worker_id;
};

struct alignas(128) AicoreCompletionInboxV1 {
    volatile int64_t head;
    uint8_t legacy_line_padding[56];
    volatile uint64_t completed_generations[AICORE_PENDING_SLOT_COUNT_V1];
    uint8_t completion_line_padding[48];
};

struct alignas(128) AicoreReadyInboxV1 {
    volatile int64_t head;
    uint8_t atomic_line_padding[120];
};

enum class AicoreGangCohortStateV1 : uint64_t {
    FREE = 0,
    DRAINING = 1,
    STAGING = 2,
    RELEASING = 3,
    DISPATCHING = 4,
    EXECUTING = 5,
    RETIRING = 6,
};

struct alignas(128) AicoreGangCoordinatorV1 {
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

struct alignas(128) AicoreGangCohortV1 {
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
struct alignas(128) AicoreGangParticipantV1 {
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
struct alignas(128) AicoreGangCommandV1 {
    volatile uint64_t generation[AICORE_GANG_COHORT_COUNT_V1];
    volatile uint64_t state[AICORE_GANG_COHORT_COUNT_V1];
    uint8_t padding[96];
};

struct alignas(64) AicoreReadyDirectoryShardV1 {
    volatile uint64_t bits;
    uint8_t cache_line_padding[64 - sizeof(uint64_t)];
};

struct alignas(128) AicoreReadyDirectoryV1 {
    AicoreReadyDirectoryShardV1 core_types[AICORE_CORE_TYPE_COUNT_V1][AICORE_READY_DIRECTORY_SHARD_COUNT_V1];
    volatile uint64_t bootstrap_ready_types[AICORE_WORKER_CAPACITY_V1];
};

struct alignas(128) AicoreFreeSlotDirectoryV1 {
    volatile uint64_t words[AICORE_CORE_TYPE_COUNT_V1][AICORE_FREE_SLOT_DIRECTORY_WORD_COUNT_V1];
    uint8_t padding[64];
};

struct alignas(64) AicoreTaskClaimBindingV1 {
    int64_t task_id;
    uint64_t callable_address;
    uint64_t dispatch_payload_offset;
    uint64_t owner_worker_id;
    uint16_t kernel_id;
    uint8_t subtask_slot;
    uint8_t pending_slot;
    uint32_t dispatch_generation;
    uint64_t ready_claim_start_cycles;
    uint64_t ready_claim_end_cycles;
    uint64_t ready_claim_resolver_worker_id;
};

// Resolver-owned metadata occupies the first line. The Executor polls only
// publication in the second line.
struct alignas(128) AicoreDispatchSlotV1 {
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

struct alignas(128) AicoreRunControlV1 {
    uint64_t config_reserved_prefix[2];
    uint64_t active_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t aic_active_worker_count;
    uint64_t aiv_active_worker_count;
    uint64_t claim_bindings_offset;
    volatile uint64_t dispatch_payloads_offset;
    uint64_t task_metadata_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_directory_offset;
    uint64_t free_slot_directory_offset;
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

struct alignas(128) AicpuCoreLifecycleTraceV1 {
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

struct alignas(128) AicoreWorkerContextV1 {
    volatile int32_t core_type;
    int32_t physical_core_id;
    volatile int32_t type_rank;
    volatile int32_t active;
    volatile uint64_t run_control_offset;
    volatile uint64_t task_controls_offset;
    volatile uint64_t graph_descriptors_address;
    volatile uint64_t graph_payloads_address;
    volatile uint64_t sidecar_base_address;
    volatile uint64_t dispatch_payload_offset;
    volatile uint64_t trace_cells_offset;
    volatile uint64_t task_window_mask;
    volatile uint64_t graph_task_count;
    volatile uint64_t worker_index;
    volatile uint64_t completion_inboxes_offset;
    volatile uint64_t inbox_index;
    uint64_t runtime_offset_padding[2];

    volatile uint64_t task_metadata_offset;
    volatile uint64_t ready_inboxes_offset;
    volatile uint64_t ready_directory_offset;
    volatile uint64_t free_slot_directory_offset;
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
    uint64_t free_slot_claim_count;
    uint64_t free_slot_stale_count;
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
    uint64_t completion_batch_count;
    uint64_t completion_resolve_count;
    uint64_t completion_steal_count;
    uint64_t completion_link_wait_count;
    uint64_t completion_link_wait_max;
    uint64_t completion_lag_cycles;
    uint64_t completion_lag_max_cycles;
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
    uint64_t bootstrap_free_advertise_aic_cycles;
    uint64_t bootstrap_free_advertise_aiv_cycles;
    uint64_t termination_reserved[2];
};

struct alignas(128) AicoreTaskTraceCellV1 {
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
    uint64_t inter_task_dispatch_probe_cycles[AICORE_CORE_TYPE_COUNT_V1];
    uint64_t inter_task_dispatch_claim_cycles[AICORE_CORE_TYPE_COUNT_V1];
    uint64_t inter_task_dispatch_prepare_cycles[AICORE_CORE_TYPE_COUNT_V1];
    uint64_t inter_task_dispatch_materialize_cycles[AICORE_CORE_TYPE_COUNT_V1];
    uint64_t inter_task_dispatch_publish_cycles[AICORE_CORE_TYPE_COUNT_V1];
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
    uint64_t dispatch_slots_offset;
    uint64_t claim_bindings_offset;
    uint64_t callable_addresses_offset;
    uint64_t task_metadata_offset;
    uint64_t task_controls_offset;
    uint64_t completion_inboxes_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_directory_offset;
    uint64_t free_slot_directory_offset;
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

static_assert(sizeof(AicoreTaskMetadataV1) == 16, "task metadata layout changed");
static_assert(sizeof(AicoreTaskControlV1) == 128, "task control layout changed");
static_assert(offsetof(AicoreTaskControlV1, next_waiter) == 64, "waiter metadata needs its own line");
static_assert(offsetof(AicoreTaskControlV1, inbox_next) == 80, "inbox link offset changed");
static_assert(sizeof(AicoreCompletionInboxV1) == 128, "completion inbox layout changed");
static_assert(
    offsetof(AicoreCompletionInboxV1, completed_generations) == 64, "SPSC completion line must be cache aligned"
);
static_assert(sizeof(AicoreReadyInboxV1) == 128, "ready inbox layout changed");
static_assert(sizeof(AicoreGangCoordinatorV1) == 256, "gang coordinator layout changed");
static_assert(
    offsetof(AicoreGangCoordinatorV1, active_dispatch_cohort) == 64,
    "owner-only gang state must not share the priority line"
);
static_assert(sizeof(AicoreGangCohortV1) == 128, "gang cohort layout changed");
static_assert(sizeof(AicoreGangParticipantV1) == 128, "gang participant layout changed");
static_assert(sizeof(AicoreGangCommandV1) == 128, "gang command layout changed");
static_assert(sizeof(AicoreReadyDirectoryShardV1) == 64, "ready directory shard must occupy one cache line");
static_assert(
    offsetof(AicoreReadyDirectoryV1, bootstrap_ready_types) ==
        AICORE_CORE_TYPE_COUNT_V1 * AICORE_READY_DIRECTORY_SHARD_COUNT_V1 * 64,
    "bootstrap flags must follow the ready directory shards"
);
static_assert(
    sizeof(AicoreReadyDirectoryV1) == ((offsetof(AicoreReadyDirectoryV1, bootstrap_ready_types) +
                                        AICORE_WORKER_CAPACITY_V1 * sizeof(uint64_t) + 127) /
                                       128 * 128),
    "ready directory layout changed"
);
static_assert(sizeof(AicoreFreeSlotDirectoryV1) == 128, "free-slot directory layout changed");
static_assert(sizeof(AicoreTaskClaimBindingV1) == 64, "claim binding must occupy one cache line");
static_assert(sizeof(AicoreDispatchSlotV1) == 128, "dispatch slot must occupy two cache lines");
static_assert(offsetof(AicoreDispatchSlotV1, publication) == 64, "dispatch publication needs its own line");
static_assert(sizeof(AicoreRunControlV1) == 384, "run control layout changed");
static_assert(offsetof(AicoreRunControlV1, executed_task_count) == 128, "lifecycle atomics need their own line");
static_assert(offsetof(AicoreRunControlV1, error_claimed) == 256, "error state needs its own line");
static_assert(sizeof(AicpuCoreLifecycleTraceV1) == 128, "AICPU lifecycle trace layout changed");
static_assert(sizeof(AicoreWorkerContextV1) == 768, "worker context layout changed");
static_assert(offsetof(AicoreWorkerContextV1, task_metadata_offset) == 128, "runtime offsets changed");
static_assert(offsetof(AicoreWorkerContextV1, gang_coordinator_offset) == 256, "topology offsets changed");
static_assert(offsetof(AicoreWorkerContextV1, bootstrap_task_count) == 384, "worker stats offset changed");
static_assert(offsetof(AicoreWorkerContextV1, wake_cas_retry_count) == 512, "wake stats offset changed");
static_assert(offsetof(AicoreWorkerContextV1, completion_enqueue_cycles) == 640, "termination stats offset changed");
static_assert(sizeof(AicoreTaskTraceCellV1) == 384, "task trace layout changed");

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<AicoreTaskMetadataV1> && std::is_trivially_copyable_v<AicoreTaskMetadataV1>);
static_assert(std::is_standard_layout_v<AicoreTaskControlV1> && std::is_trivially_copyable_v<AicoreTaskControlV1>);
static_assert(
    std::is_standard_layout_v<AicoreCompletionInboxV1> && std::is_trivially_copyable_v<AicoreCompletionInboxV1>
);
static_assert(std::is_standard_layout_v<AicoreReadyInboxV1> && std::is_trivially_copyable_v<AicoreReadyInboxV1>);
static_assert(
    std::is_standard_layout_v<AicoreGangCoordinatorV1> && std::is_trivially_copyable_v<AicoreGangCoordinatorV1>
);
static_assert(std::is_standard_layout_v<AicoreGangCohortV1> && std::is_trivially_copyable_v<AicoreGangCohortV1>);
static_assert(
    std::is_standard_layout_v<AicoreGangParticipantV1> && std::is_trivially_copyable_v<AicoreGangParticipantV1>
);
static_assert(std::is_standard_layout_v<AicoreGangCommandV1> && std::is_trivially_copyable_v<AicoreGangCommandV1>);
static_assert(
    std::is_standard_layout_v<AicoreReadyDirectoryV1> && std::is_trivially_copyable_v<AicoreReadyDirectoryV1>
);
static_assert(
    std::is_standard_layout_v<AicoreFreeSlotDirectoryV1> && std::is_trivially_copyable_v<AicoreFreeSlotDirectoryV1>
);
static_assert(
    std::is_standard_layout_v<AicoreTaskClaimBindingV1> && std::is_trivially_copyable_v<AicoreTaskClaimBindingV1>
);
static_assert(std::is_standard_layout_v<AicoreDispatchSlotV1> && std::is_trivially_copyable_v<AicoreDispatchSlotV1>);
static_assert(std::is_standard_layout_v<AicoreRunControlV1> && std::is_trivially_copyable_v<AicoreRunControlV1>);
static_assert(std::is_standard_layout_v<AicoreWorkerContextV1> && std::is_trivially_copyable_v<AicoreWorkerContextV1>);
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
    if (cursor == nullptr || offset == nullptr || !aicore_sidecar_checked_align_v1(*cursor, alignment, &aligned))
        return false;
    uint64_t end = 0;
    if (!aicore_sidecar_checked_add_v1(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool aicore_sidecar_plan_v1(
    uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, AicoreExecutionSidecarLayoutV1 *layout
) {
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count) return false;
    AicoreExecutionSidecarLayoutV1 next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define AICORE_RESERVE_ARRAY(count, type, field)                     \
    (aicore_sidecar_checked_mul_v1((count), sizeof(type), &bytes) && \
     aicore_sidecar_reserve_v1(&cursor, bytes, alignof(type), &next.field))
    if (!aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreRunControlV1), alignof(AicoreRunControlV1), &next.run_control_offset
        ) ||
        !AICORE_RESERVE_ARRAY(AICORE_WORKER_CAPACITY_V1, AicpuCoreLifecycleTraceV1, aicpu_lifecycle_traces_offset) ||
        !AICORE_RESERVE_ARRAY(AICORE_WORKER_CAPACITY_V1, AicoreWorkerContextV1, worker_contexts_offset) ||
        !AICORE_RESERVE_ARRAY(
            AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1, PTO2DispatchPayload, dispatch_payloads_offset
        ) ||
        !AICORE_RESERVE_ARRAY(
            AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1, AicoreDispatchSlotV1, dispatch_slots_offset
        ) ||
        !AICORE_RESERVE_ARRAY(task_count, AicoreTaskClaimBindingV1, claim_bindings_offset) ||
        !AICORE_RESERVE_ARRAY(AICORE_CALLABLE_CAPACITY_V1, uint64_t, callable_addresses_offset) ||
        !AICORE_RESERVE_ARRAY(task_count, AicoreTaskMetadataV1, task_metadata_offset) ||
        !AICORE_RESERVE_ARRAY(task_count, AicoreTaskControlV1, task_controls_offset) ||
        !AICORE_RESERVE_ARRAY(AICORE_WORKER_CAPACITY_V1, AicoreCompletionInboxV1, completion_inboxes_offset) ||
        !AICORE_RESERVE_ARRAY(
            AICORE_CORE_TYPE_COUNT_V1 * AICORE_WORKER_CAPACITY_V1, AicoreReadyInboxV1, ready_inboxes_offset
        ) ||
        !aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreReadyDirectoryV1), alignof(AicoreReadyDirectoryV1), &next.ready_directory_offset
        ) ||
        !aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreFreeSlotDirectoryV1), alignof(AicoreFreeSlotDirectoryV1),
            &next.free_slot_directory_offset
        ) ||
        !aicore_sidecar_reserve_v1(
            &cursor, sizeof(AicoreGangCoordinatorV1), alignof(AicoreGangCoordinatorV1), &next.gang_coordinator_offset
        ) ||
        !AICORE_RESERVE_ARRAY(AICORE_GANG_COHORT_COUNT_V1, AicoreGangCohortV1, gang_cohorts_offset) ||
        !AICORE_RESERVE_ARRAY(
            AICORE_GANG_COHORT_COUNT_V1 * AICORE_CLUSTER_CAPACITY_V1, AicoreGangParticipantV1, gang_participants_offset
        ) ||
        !AICORE_RESERVE_ARRAY(AICORE_CLUSTER_CAPACITY_V1, AicoreGangCommandV1, gang_commands_offset) ||
        !AICORE_RESERVE_ARRAY(task_count, AicoreTaskTraceCellV1, trace_cells_offset) ||
        !aicore_sidecar_checked_align_v1(cursor, AICORE_SIDECAR_ALIGNMENT_V1, &next.total_size)) {
#undef AICORE_RESERVE_ARRAY
        return false;
    }
#undef AICORE_RESERVE_ARRAY
    *layout = next;
    return true;
}

template <typename T>
inline __host__ __aicore__ __gm__ T *aicore_sidecar_at_v1(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

inline bool aicore_sidecar_init_v1(void *base, const AicoreExecutionSidecarLayoutV1 &layout) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (AICORE_SIDECAR_ALIGNMENT_V1 - 1)) != 0) return false;
    __builtin_memset(base, 0, static_cast<size_t>(layout.total_size));
    auto *bindings = aicore_sidecar_at_v1<AicoreTaskClaimBindingV1>(base, layout.claim_bindings_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i)
        bindings[i].task_id = AICORE_TASK_ID_INVALID_V1;
    auto *controls = aicore_sidecar_at_v1<AicoreTaskControlV1>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].state = static_cast<int64_t>(AicoreTaskStateV1::BLOCKED);
        controls[i].wake_list_head = AICORE_WAKE_LIST_OPEN_V1;
        controls[i].next_waiter = AICORE_TASK_ID_INVALID_V1;
        controls[i].waiting_producer = static_cast<int32_t>(AICORE_TASK_ID_INVALID_V1);
        controls[i].inbox_next = AICORE_INBOX_LINK_UNPUBLISHED_V1;
    }
    auto *slots = aicore_sidecar_at_v1<AicoreDispatchSlotV1>(base, layout.dispatch_slots_offset);
    for (uint64_t i = 0; i < AICORE_WORKER_CAPACITY_V1 * AICORE_PENDING_SLOT_COUNT_V1; ++i) {
        slots[i].task_id = AICORE_TASK_ID_INVALID_V1;
        slots[i].publication = static_cast<uint64_t>(AicoreDispatchPublicationV1::EMPTY);
    }
    auto *completion = aicore_sidecar_at_v1<AicoreCompletionInboxV1>(base, layout.completion_inboxes_offset);
    for (uint64_t i = 0; i < AICORE_WORKER_CAPACITY_V1; ++i)
        completion[i].head = AICORE_INBOX_EMPTY_V1;
    auto *ready = aicore_sidecar_at_v1<AicoreReadyInboxV1>(base, layout.ready_inboxes_offset);
    for (uint64_t i = 0; i < AICORE_CORE_TYPE_COUNT_V1 * AICORE_WORKER_CAPACITY_V1; ++i)
        ready[i].head = AICORE_INBOX_EMPTY_V1;
    auto *contexts = aicore_sidecar_at_v1<AicoreWorkerContextV1>(base, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < AICORE_WORKER_CAPACITY_V1; ++worker) {
        contexts[worker].physical_core_id = -1;
        contexts[worker].cluster_index = UINT64_MAX;
        contexts[worker].resolver_index = UINT64_MAX;
        contexts[worker].resolver_worker_id = UINT64_MAX;
        contexts[worker].cluster_worker_ids[0] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[1] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[2] = UINT64_MAX;
    }
    auto *coordinator = aicore_sidecar_at_v1<AicoreGangCoordinatorV1>(base, layout.gang_coordinator_offset);
    coordinator->active_dispatch_cohort = UINT64_MAX;
    coordinator->cohort_count = AICORE_GANG_COHORT_COUNT_V1;
    auto *cohorts = aicore_sidecar_at_v1<AicoreGangCohortV1>(base, layout.gang_cohorts_offset);
    for (uint32_t i = 0; i < AICORE_GANG_COHORT_COUNT_V1; ++i) {
        cohorts[i].state = static_cast<uint64_t>(AicoreGangCohortStateV1::FREE);
        cohorts[i].task_id = AICORE_TASK_ID_INVALID_V1;
    }
    auto *participants = aicore_sidecar_at_v1<AicoreGangParticipantV1>(base, layout.gang_participants_offset);
    for (uint32_t i = 0; i < AICORE_GANG_COHORT_COUNT_V1 * AICORE_CLUSTER_CAPACITY_V1; ++i)
        participants[i].task_id = AICORE_TASK_ID_INVALID_V1;
    return true;
}
