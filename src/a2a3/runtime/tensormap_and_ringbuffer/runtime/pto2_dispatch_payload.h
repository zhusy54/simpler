/**
 * @file pto2_dispatch_payload.h
 * @brief Dispatch descriptor for AICore kernel execution
 *
 * PTO2DispatchDesc is embedded in PTO2TaskPayload and built by the Orchestrator
 * at submit time. It contains per-slot function addresses and a unified args[]
 * array (tensor pointers + scalar values).
 *
 * Dispatch uses a static per-core descriptor array: AICPU copies the descriptor
 * into s_dispatch_desc_per_core[core_id] before each dispatch, then writes a
 * monotonically increasing reg_task_id (with slot_idx) to DATA_MAIN_BASE.
 * AICore caches its per-core descriptor pointer at startup and reads from it
 * on each dispatch.
 */

#ifndef RT2_PTO2_DISPATCH_PAYLOAD_H_
#define RT2_PTO2_DISPATCH_PAYLOAD_H_

#include <stdint.h>

#include "pto_submit_types.h"

/** Max arguments per task; must match RUNTIME_MAX_ARGS and PTO2_MAX_OUTPUTS */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS 128
#endif

// =============================================================================
// Register Encoding Constants (Per-Core Array Scheme)
// =============================================================================

/**
 * DATA_MAIN_BASE register bit layout (AICPU → AICore):
 *
 *   [31:2]  reg_task_id (30 bits, monotonically increasing per core)
 *   [1:0]   slot_idx  (2 bits: 0=AIC, 1=AIV0, 2=AIV1)
 *
 * reg_task_id wraps around at 0x1FFFFFFC to avoid producing sentinel values:
 *   AICORE_EXIT_SIGNAL  = 0x7FFFFFF0  (reg_task_id = 0x1FFFFFFC)
 *   AICPU_IDLE_TASK_ID  = 0x7FFFFFFD  (reg_task_id = 0x1FFFFFFF)
 *   AICORE_IDLE_TASK_ID = 0x7FFFFFFF  (reg_task_id = 0x1FFFFFFF)
 */
constexpr uint32_t PTO2_REG_SLOTIDX_MASK = 0x3;
constexpr uint32_t PTO2_REG_TASKID_SHIFT = 2;

/** Encode reg_task_id and slot_idx into a 32-bit register value. */
static inline uint32_t pto2_reg_encode(uint32_t reg_task_id, uint32_t slot_idx) {
    return (reg_task_id << PTO2_REG_TASKID_SHIFT) | slot_idx;
}

/** Decode slot_idx (subtask slot: 0=AIC, 1=AIV0, 2=AIV1) from register value. */
static inline uint32_t pto2_reg_decode_slotidx(uint32_t reg_val) {
    return reg_val & PTO2_REG_SLOTIDX_MASK;
}

/**
 * Advance reg_task_id, wrapping before the sentinel-collision zone.
 * Sentinel values occupy reg_task_id >= 0x1FFFFFFC, so we wrap to 1 there.
 * Starting from 1 (not 0) ensures the first encoded reg_val is never 0.
 */
static inline uint32_t pto2_next_reg_task_id(uint32_t current) {
    uint32_t next = current + 1;
    if (next >= 0x1FFFFFFCu) next = 1;
    return next;
}

// =============================================================================
// Dispatch Descriptor
// =============================================================================

/**
 * Dispatch descriptor: execution interface for AICore.
 *
 * Layout: per-slot function_bin_addrs[] followed by unified args[].
 * AICore reads function_bin_addrs[slot_idx], casts to UnifiedKernelFunc,
 * and calls with args (tensor GM pointers followed by scalar values).
 *
 * Built once by the Orchestrator during submit_mixed_task(); the Scheduler
 * copies it into the per-core static array before dispatching.
 */
struct PTO2DispatchDesc {
    /** Per-slot kernel entry addresses in GM (AIC, AIV0, AIV1); 0 = inactive */
    uint64_t function_bin_addrs[PTO2_SUBTASK_SLOT_COUNT];
    /** Kernel arguments: tensor GM pointers first, then scalar values */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
