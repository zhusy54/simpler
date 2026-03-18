/**
 * @file pto2_dispatch_payload.h
 * @brief Dispatch descriptor for AICore kernel execution
 *
 * PTO2DispatchDesc is embedded in PTO2TaskPayload and built by the Orchestrator
 * at submit time. It contains per-slot function addresses and a unified args[]
 * array (tensor pointers + scalar values). The Scheduler encodes a flat byte
 * offset (from a single dispatch base address) into the DATA_MAIN_BASE register;
 * AICore decodes the offset and adds it to a cached base to get the descriptor
 * address, achieving zero GM access on the scheduler hot path.
 *
 * PTO2DispatchInitInfo is a one-shot initialization struct passed to AICore
 * via Handshake.task during startup. It provides the single dispatch base
 * address needed for address computation.
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
// Register Encoding Constants (Flat Byte Offset Scheme)
// =============================================================================

/**
 * DATA_MAIN_BASE register bit layout (AICPU → AICore):
 *
 *   [31]    unused (COND register uses bit 31 for ACK/FIN state)
 *   [30]    toggle bit (alternates per core to guarantee uniqueness)
 *   [29:2]  offset_field = (desc_byte_offset >> 3) + 1  (28 bits)
 *           +1 reserves 0 for idle; >>3 because PTO2DispatchDesc is 8-byte aligned
 *   [1:0]   slot_idx  (2 bits: 0=AIC, 1=AIV0, 2=AIV1)
 *
 * Max safe offset_field = 0x0FFFFFFB (avoid sentinel collision when toggle=1).
 * Max encodable byte offset ≈ 2GB — well beyond any practical shared memory size.
 *
 * Sentinel values (must never be produced by encoding):
 *   AICORE_EXIT_SIGNAL   = 0x7FFFFFF0
 *   AICPU_IDLE_TASK_ID   = 0x7FFFFFFD
 *   AICORE_IDLE_TASK_ID  = 0x7FFFFFFF
 */
constexpr uint32_t PTO2_REG_SLOTIDX_MASK      = 0x3;
constexpr uint32_t PTO2_REG_OFFSET_SHIFT       = 2;
constexpr uint32_t PTO2_REG_OFFSET_BITS        = 28;
constexpr uint32_t PTO2_REG_OFFSET_MASK        = (1u << PTO2_REG_OFFSET_BITS) - 1;  // 0x0FFFFFFF
constexpr uint32_t PTO2_REG_TOGGLE_BIT         = 30;
constexpr uint32_t PTO2_REG_ALIGN_SHIFT        = 3;   // PTO2DispatchDesc is 8-byte aligned
constexpr uint32_t PTO2_REG_MAX_OFFSET_FIELD   = 0x0FFFFFFBu;  // sentinel-safe upper bound

/**
 * Encode dispatch info into a 32-bit register value.
 *
 * @param desc_byte_offset  Byte offset of PTO2DispatchDesc from dispatch_base (must be 8-byte aligned)
 * @param slot_idx          Subtask slot (0=AIC, 1=AIV0, 2=AIV1)
 * @param toggle            Toggle bit value (bit 30, alternated per dispatch per core)
 */
static inline uint32_t pto2_reg_encode(uint64_t desc_byte_offset,
                                       uint32_t slot_idx, uint32_t toggle) {
    uint64_t offset_field_64 = (desc_byte_offset >> PTO2_REG_ALIGN_SHIFT) + 1;
    uint32_t offset_field = static_cast<uint32_t>(offset_field_64);
    return (offset_field << PTO2_REG_OFFSET_SHIFT)
         | slot_idx
         | toggle;
}

/**
 * Decode desc byte offset from register value.
 * Returns the byte offset to add to dispatch_base to get the PTO2DispatchDesc address.
 */
static inline uint64_t pto2_reg_decode_offset(uint32_t reg_val) {
    uint32_t offset_field = (reg_val >> PTO2_REG_OFFSET_SHIFT) & PTO2_REG_OFFSET_MASK;
    return static_cast<uint64_t>(offset_field - 1) << PTO2_REG_ALIGN_SHIFT;
}

/** Decode slot_idx (subtask slot: 0=AIC, 1=AIV0, 2=AIV1) from register value. */
static inline uint32_t pto2_reg_decode_slotidx(uint32_t reg_val) {
    return reg_val & PTO2_REG_SLOTIDX_MASK;
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
 * never touches it — it only writes a register-encoded value.
 */
struct PTO2DispatchDesc {
    /** Per-slot kernel entry addresses in GM (AIC, AIV0, AIV1); 0 = inactive */
    uint64_t function_bin_addrs[PTO2_SUBTASK_SLOT_COUNT];
    /** Kernel arguments: tensor GM pointers first, then scalar values */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
};

// =============================================================================
// Dispatch Initialization Info
// =============================================================================

/**
 * One-shot initialization info passed from AICPU to AICore.
 *
 * Lifecycle:
 *   1. AICPU sets Handshake.task = 0 during handshake (AICore not yet ready)
 *   2. AICPU creates PTO2Runtime, computes dispatch_base from shared memory
 *   3. AICPU writes Handshake.task = &init_info for all cores
 *   4. AICore waits for Handshake.task != 0, reads and caches dispatch_base
 *   5. AICore clears Handshake.task = 0 (init_info no longer needed)
 *
 * During the main loop, AICore computes the dispatch desc address as:
 *   dispatch_base + pto2_reg_decode_offset(reg_val)
 *
 * This avoids any GM read on the scheduler hot path — the scheduler only
 * writes a register-encoded byte offset value.
 */
struct PTO2DispatchInitInfo {
    /** Base address of the first PTO2DispatchDesc (ring 0, slot 0) */
    uint64_t dispatch_base;
};

#endif  // RT2_PTO2_DISPATCH_PAYLOAD_H_
