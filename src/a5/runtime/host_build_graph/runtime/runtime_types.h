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

/**
 * PTO Runtime2 - Core Type Definitions
 *
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - simpler::hbg::Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "profiling_config.h"
#include "constants.h"
#include "runtime_status.h"
// NOTE (host_build_graph divergence from tensormap_and_ringbuffer): the
// dispatch_payload.h include is intentionally dropped here. This header is
// pulled in by the platform's args_dump.h via a hardcoded
// "host_build_graph/runtime/runtime_types.h" path, and dispatch_payload.h
// uses #pragma once (path-keyed), so leaving it in double-defines DispatchPayload
// against tensormap_and_ringbuffer's copy inside the shared host-dispatcher TU.
// runtime_types.h never references DispatchPayload itself; consumers that
// need it include it via runtime.h directly.
#include "common/args_dump_task_metadata.h"
#include "host_build_graph/self_relative_ptr.h"
#include "submit_types.h"
#include "task_id.h"
#include "types.h"

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
//
// The task table is a flat array of slots, indexed directly by local task id:
// ids start at 0, are never recycled, and alloc() caps them at the table's size,
// so there is no wrap and no slot mask. The size need not be a power of two —
// nothing masks with it.
//
// This is the default; `CallConfig.runtime_env.ring_task_window` overrides it per
// task. The host mirror is allocated at whatever size is in effect and committed
// by first touch, so a run pays only for the slots and argument-pool bytes it
// actually writes. Raising it costs virtual address space, bounded by the int32
// reach of a payload's self-relative region deltas (checked in
// SharedMemoryHandle::init).
#define CHIP_DEFAULT_GRAPH_TASKS 16384

// host_build_graph carries no per-scope-depth task partition: host-orch builds
// the whole graph on the host and the device runs it once without reclaim. tmr's
// multi-ring design existed only to let inner scopes reclaim independently under
// small rings; with no reclaim and a whole-graph-resident task table, per-depth
// isolation is moot. The RuntimeEnv ABI still carries RUNTIME_ENV_RING_COUNT
// slots because it is shared with tmr; this runtime reads none of them and warns
// when one is set (see bind_callable_to_runtime_impl).

// Memory pools (total = value)
#define CHIP_TENSORMAP_POOL_SIZE (65536)  // TensorMap entry pool
#define CHIP_TENSORMAP_NUM_BUCKETS 4096   // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Three address classes coexist during orchestration, in windows the two constants
// below keep disjoint: real device addresses stay below HEAP_VIRTUAL_BASE, since
// Ascend VA is 48-bit and the asserts named further down hold caller-owned ones
// there; the graph heap spans HEAP_VIRTUAL_BASE up to GRAPH_RECORD_VIRTUAL_BASE;
// Graph recording takes everything above.
//
// Base of the window the graph heap is allocated out of during orchestration.
// The heap's device region is committed only once orchestration has run and its
// exact size is known, so the addresses handed out while the graph is being
// built cannot be the device ones; compact_live_image rewrites them to the real
// base before the image travels. Nothing dereferences an address in this window.
inline constexpr uint64_t HEAP_VIRTUAL_BASE = 1ULL << 62;

// Base of the address range Graph recording hands to an internal node's packed
// outputs. Recorded addresses are never dereferenced: they exist so
// graph_classify_tensor can tell an internal producer's output from a boundary
// tensor by address-range containment alone, and the Definition stores them as
// offsets. That classification is only sound while the range is disjoint from
// every graph-heap address, which TaskAllocator::init() asserts, and from
// every real device address, which the two asserts in the host's bind path
// (the acquired heap base, and each caller tensor as it enters device_args)
// keep below HEAP_VIRTUAL_BASE.
inline constexpr uint64_t GRAPH_RECORD_VIRTUAL_BASE = 1ULL << 63;

// Span of the graph-heap window: everything between the two virtual bases. This
// is the bound orchestration allocates against, so a graph is limited by what
// the device can commit afterwards rather than by a configured heap size.
inline constexpr uint64_t HEAP_VIRTUAL_CAPACITY = GRAPH_RECORD_VIRTUAL_BASE - HEAP_VIRTUAL_BASE;

// Scope management
#define CHIP_MAX_SCOPE_DEPTH 64  // Maximum nesting depth

// Fanin storage
#define CHIP_FANIN_INLINE_CAP 64

// The AICore scheduler reads producer ids directly from the compact fanin pool.
// A task's fanin degree is hard-capped here and must cover the densest workload.
#define CHIP_MAX_FANIN 128

// Alignment of every per-task region inside an argument pool. Each region starts
// and ends on a cache line so TaskPayload::init's round-up scalar memcpy stays
// inside the task's own region — see its comment. simpler::hbg::Tensor is already 2 cache
// lines, so only the fanin and scalar regions need the round-up.
inline constexpr int32_t ARG_POOL_ALIGN = 64;

// Dependency-degree diagnostic: warn once when a task's fanin or a producer's
// fanout first exceeds this degree, so dense dependency graphs surface without
// flooding the AICPU hot-path device log.
#define CHIP_DEP_DEGREE_WARN_THRESHOLD 16

// get_tensor_data/set_tensor_data spin-wait timeout, expressed in time. The cycle
// count (TENSOR_DATA_TIMEOUT_CYCLES) is derived from this in runtime_core.cpp
// — its only user — by scaling with the platform counter frequency, like
// SCHEDULER_TIMEOUT_CYCLES, so it reaps at the same wall-clock on every arch (a
// fixed raw cycle count would be 15 s on a5 at 1 GHz but 300 s on a2a3 at 50 MHz).
// PLATFORM_PROF_SYS_CNT_FREQ is deliberately NOT pulled into this header: it is
// included by orchestrations that define that constant locally, so doing so caused
// a redefinition conflict. See issue #1189.
constexpr uint64_t TENSOR_DATA_TIMEOUT_MS = 15000;  // 15 s

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 *
 * State transitions:
 *   PENDING -> COMPLETED
 *
 * Executable tasks remain PENDING in the host graph image. Hidden allocation
 * tasks transition to COMPLETED during orchestration and are represented as
 * already resolved when the scheduler state is built.
 *
 * COMPLETED is terminal because host_build_graph does not recycle graph slots.
 */
typedef enum {
    CHIP_TASK_PENDING = 0,    // Submitted; awaiting fanin, queued, or dispatched
    CHIP_TASK_COMPLETED = 1,  // Execution finished, output may still be in use
    CHIP_TASK_CONSUMED = 2    // Unused: host_build_graph never advances past COMPLETED
} ChipTaskState;

/**
 * Result of a unified task allocation.
 *
 * There is no separate slot: a task id indexes the task table directly.
 */
struct TaskAllocResult {
    int32_t task_id;    // Task id, which is also its task-table index
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const { return task_id < 0; }
};

enum class TaskKind : uint8_t {
    KERNEL = 0,
    DUMMY = 1,
    GRAPH = 2,
    GRAPH_NODE = 3,
};

struct OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

// =============================================================================
// Dependency List Entry
// =============================================================================

struct ChipTaskSlotState;  // Forward declaration (defined below)

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in ChipTaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct TaskDescriptor {
    // Task identity. See src/common/host_build_graph/task_id_encoding.h: the
    // upper 32 bits are this runtime's id space, not a ring index.
    TaskId task_id;

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// A 4-byte alignment pad follows kernel_id[3]; the AICore graph ABI depends on
// the descriptor size and packed_buffer_base offset staying fixed.
static_assert(sizeof(TaskDescriptor) == 40, "TaskDescriptor size is part of the shared-memory ABI");
static_assert(offsetof(TaskDescriptor, packed_buffer_base) == 24, "packed_buffer_base offset must be unchanged");

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + inline fanin packed in the first 9 cache lines, followed
 * by bulk tensor and scalar data. Small fanins stay fully inline; larger
 * fanins spill into a per-ring ring buffer slice.
 */
struct TaskPayload {
    // === Cache line 0 (64B) — the dispatch path's own line ===
    // sizeof is independent of CHIP_MAX_FANIN / MAX_TENSOR_ARGS / MAX_SCALAR_ARGS:
    // widening a cap costs pool bytes for the tasks that need them, not a control
    // block on every task.
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_count{0};  // Producer dependency count (raw, no +1 redundance)

    // This task's three argument regions, each in a pool outside this struct and
    // named by a delta from the naming field's own address. A delta holds only for
    // the layout it was taken in, so every one is bound twice on the host: against
    // the mirror when the slot is claimed, and against the image in
    // compact_live_image, which re-pitches the segments.
    //
    // fanin holds flat position-independent producer local task ids. Single-ring
    // hbg: every producer is ring 0, so no per-edge ring id is stored. Scanned by
    // classify_fanin_state against the ring completion_flags. Hard-capped at
    // CHIP_MAX_FANIN (no dep-pool spill). Unbound on a Graph node, whose
    // dependencies live in the Definition's fanin CSR instead.
    simpler::hbg::SelfRelativePtr<simpler::hbg::Tensor> tensors;
    simpler::hbg::SelfRelativePtr<uint64_t> scalars;
    simpler::hbg::SelfRelativePtr<int32_t> fanin;

    // === Cache line 1 — dispatch predicate + dump metadata ===
    // The AICore scheduler evaluates the predicate immediately before dispatch.
    alignas(64) DispatchPredicate predicate;
    ArgsDumpTaskMetadata dump_metadata;

    // --- Argument region access ---
    // Each accessor resolves its delta once and hands back the region's first
    // element, so a caller indexes the region directly. Deliberately no per-element
    // accessor: one inside a loop would re-resolve the delta on every iteration, and
    // a store through an unrelated pointer in the loop body is enough to stop the
    // compiler hoisting that load — build_payload's args[] writes are exactly that.
    simpler::hbg::Tensor *tensor_data() { return tensors.get(); }
    const simpler::hbg::Tensor *tensor_data() const { return tensors.get(); }
    uint64_t *scalar_data() { return scalars.get(); }
    const uint64_t *scalar_data() const { return scalars.get(); }
    int32_t *fanin_data() { return fanin.get(); }
    const int32_t *fanin_data() const { return fanin.get(); }

    /**
     * Point this payload's three argument regions at pool-resident storage. Must run
     * before prefetch() and init(), which dereference them.
     *
     * A Graph node passes nullptr for fanin: its dependencies come from the
     * Definition's CSR, so the region does not exist and fanin_count stays 0.
     */
    void bind_regions(simpler::hbg::Tensor *tensor_region, uint64_t *scalar_region, int32_t *fanin_region) {
        tensors.set(tensor_region);
        scalars.set(scalar_region);
        fanin.set(fanin_region);
    }

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). A member fn lowers to the same prefetch
     * instructions as a free function (`this` is just a register), no cache impact.
     */
    void prefetch(int32_t tensor_count, int32_t scalar_count) const {
        const simpler::hbg::Tensor *t = tensor_data();
        for (int32_t i = 0; i < tensor_count; i++) {
            __builtin_prefetch(&t[i], 1, 3);
            __builtin_prefetch(reinterpret_cast<const char *>(&t[i]) + 64, 1, 3);
        }
        const uint64_t *s = scalar_data();
        for (int32_t i = 0; i < scalar_count; i += 8) {
            __builtin_prefetch(&s[i], 1, 3);
        }
        __builtin_prefetch(this, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 64, 1, 3);
    }

    /**
     * Initialize payload: copy tensors, store scalars.
     *
     * For each param slot, the tensor source is determined by TensorArgType:
     * - OUTPUT -> use materialized_outputs.output_ptr(out_idx++)
     * - INPUT / INOUT -> use refs[i].tensor
     *
     * @param args                Task arguments (tensors + scalars)
     * @param result  Materialized output tensors (from TensorCreateInfo path)
     */
    void
    init(const CoreTaskArgs &args, TaskOutputTensors &result, TaskAllocResult &alloc_result, OutputLayout &layout) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // bind_regions must already have run: an unbound region reads back as null and
        // the stores below would go through it. A count of zero needs no region, so a
        // task with neither argument kind may leave both unbound.
        debug_assert(args.tensor_count() == 0 || tensor_data() != nullptr);
        debug_assert(args.scalar_count() == 0 || scalar_data() != nullptr);

        simpler::hbg::Tensor *dst = tensor_data();
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) {
                dst[i].copy(args.tensor(i).ref());
            } else {
                init_tensor_from_create_info(
                    dst[i], args.tensor(i).create_info(),
                    reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]),
                    layout.buffer_sizes[i]
                );
                dst[i].owner_task_id = result.task_id();
                result.materialize_output(dst[i]);
            }
        }
        // Round up to cache line boundary. Every scalar region is a whole number of
        // cache lines (ARG_POOL_ALIGN), so the rounded copy stays inside this
        // task's own region. Eliminates branches; extra bytes within the same CL have
        // zero additional cost.
        memcpy(scalar_data(), args.scalars(), CHIP_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));

        // The ring's payload storage is reused raw memory that no constructor runs
        // over, so an unset predicate reads back as whatever the slot last held —
        // and compact_live_image translates predicate.addr as a graph-heap address
        // for every submitted slot. An ordinary task overwrites this right
        // after init(); a hidden-alloc task has nothing following it, so this is
        // the only value its predicate ever gets.
        predicate = DispatchPredicate{};
        dump_metadata = {};
#if SIMPLER_DFX
        dump_metadata.dump_arg_mask = args.dump_arg_mask();
        dump_metadata.dump_arg_flags = args.dump_arg_index_ambiguous_mask();
        memcpy(dump_metadata.scalar_dtypes, args.scalar_dtypes(), args.scalar_count() * sizeof(uint8_t));
#endif
    }
};

// TaskPayload layout verification (offsetof requires complete type). The counts
// and region deltas share the first cache line; the predicate and dump metadata
// own the second.
static_assert(offsetof(TaskPayload, tensors) == 12, "region deltas must follow the three counts");
static_assert(
    offsetof(TaskPayload, fanin) + sizeof(simpler::hbg::SelfRelativePtr<int32_t>) <= 64,
    "counts + region deltas must fit the first cache line"
);
static_assert(offsetof(TaskPayload, predicate) == 64, "dispatch predicate owns cache line 1");
static_assert(
    offsetof(TaskPayload, dump_metadata) + sizeof(ArgsDumpTaskMetadata) <= 128,
    "dump metadata must fit the predicate's cache line"
);
static_assert(sizeof(TaskPayload) == 128, "TaskPayload is two cache lines and independent of every argument cap");
// compact_live_image restacks the payload segment with one memcpy and the device
// copy moves the whole image, so the payload has to be a POD wire struct. Deleting
// SelfRelativePtr's copy operations does not cost it that — a deleted special member
// is trivial — but only an assertion keeps a future member from doing so silently.
static_assert(
    std::is_trivially_copyable_v<TaskPayload> && std::is_standard_layout_v<TaskPayload>,
    "TaskPayload crosses to the device by memcpy"
);
static_assert(sizeof(simpler::hbg::Tensor) == 128, "simpler::hbg::Tensor must be 2 cache lines");

/**
 * Per-task host orchestration state
 *
 * 64 bytes = one cache line. The host uses this metadata while constructing
 * the compact scheduler state and servicing orchestration-time waits.
 */
struct alignas(64) ChipTaskSlotState {
    // Highest local task id among this slot's consumers. Reclaim gate: the slot
    // is safe to retire once the per-ring completed_watermark reaches this id.
    // Whole-graph-resident hbg never reclaims at runtime, so this is
    // inert-but-scaffolded for parity. Seeded to own local_id in prepare_task;
    // bumped via max() at submit for each consumer.
    int32_t last_consumer_local_id;

    // Host graph classification state. Executable tasks remain PENDING; inline
    // allocation tasks become COMPLETED before the scheduler state is built.
    std::atomic<ChipTaskState> task_state;

    // --- Per-slot constant, re-bound by orch::prepare_task each submit ---
    // Self-relative, so the SM image needs no pointer fix-up on its way to the
    // device: both targets sit in the same block as this field.
    simpler::hbg::SelfRelativePtr<TaskPayload> payload;
    simpler::hbg::SelfRelativePtr<TaskDescriptor> task;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    // Single per-task attributes byte (early-dispatch hint, sync_start,
    // has_predicate, selective timing tag). Plain-write once at submit, before
    // the scheduler state builder reads it.
    TaskAttrs task_attrs{};
    TaskKind task_kind{TaskKind::KERNEL};

    int16_t total_required_subtasks{0};  // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};        // Total logical blocks (set by orchestrator)

    // Graph scheduling metadata occupies the slot's tail padding. Ordinary
    // Ordinary tasks keep the index invalid and the context null.
    int32_t graph_node_index{-1};
    void *graph_context{nullptr};

    void bind_buffers(TaskPayload *p, TaskDescriptor *t) {
        payload.set(p);
        task.set(t);
    }

    // Host-visible completion mirror for orchestration-time waits.
    void mark_completed() { task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release); }

    /**
     * Reset dynamic scheduling fields to their pristine values. Called once per
     * slot as the orchestrator claims it in prepare_task, and again as a Graph
     * node's storage is materialized — whole-graph-resident hbg has no
     * execution-time slot recycle. Skips payload/task (bound once) and
     * task_state (the orchestrator sets PENDING when it populates the slot).
     */
    void reset_for_reuse() {
        graph_node_index = -1;
        graph_context = nullptr;
        task_kind = TaskKind::KERNEL;
        // Note: active_mask and task_attrs are per-submit-constant fields
        // rewritten in prepare_task on every reuse, so they are not reset here.
        // last_consumer_local_id is seeded in prepare_task once the id is known.
    }
};

static_assert(sizeof(ChipTaskSlotState) == 64);
