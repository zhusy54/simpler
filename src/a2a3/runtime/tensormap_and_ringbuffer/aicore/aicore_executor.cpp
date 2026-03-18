#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"
#include "common/perf_profiling.h"
#include "aicore/performance_collector_aicore.h"
#include "common/platform_config.h"  // Register-based communication

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Execute task from PTO2DispatchDesc.
 *
 * Reads function_bin_addrs[slot_idx] and args from the dispatch descriptor.
 * The descriptor is pre-built by the Orchestrator at submit time, so this
 * function performs no address computation—just a function pointer call.
 *
 * @param desc     Pointer to PTO2DispatchDesc in global memory
 * @param slot_idx Subtask slot index (0=AIC, 1=AIV0, 2=AIV1)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(
    __gm__ PTO2DispatchDesc* desc, uint32_t slot_idx
) {
    if (desc == nullptr) {
        return;
    }
    uint64_t func_addr = desc->function_bin_addrs[slot_idx];
    if (func_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)func_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(desc->args));
    FULL_MEMORY_BARRIER();
}

/**
 * AICore main execution loop
 *
 * Implements the AICPU-AICore register-based dispatch protocol:
 * 1. Wait for AICPU ready signal via handshake buffer
 * 2. Report physical core ID and core type, signal AICore ready
 * 3. Read PTO2DispatchInitInfo from hank->task (one-shot, wait for non-zero)
 * 4. Poll DATA_MAIN_BASE register for task dispatch until exit signal
 *
 * Register encoding (set by AICPU scheduler) — see pto2_dispatch_payload.h:
 *   bit  [30]   = toggle bit (alternates per core, ignored during decode)
 *   bits [29:2] = offset_field = (desc_byte_offset >> 3) + 1  (28 bits, 0 = idle)
 *   bits [1:0]  = slot_idx  (2 bits: 0=AIC, 1=AIV0, 2=AIV1)
 *
 * Dispatch desc address = dispatch_base + decoded byte offset
 *
 * @param runtime Pointer to Runtime in global memory
 * @param block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
    }

    // Phase 2: Report physical core ID, signal ready
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->aicore_regs_ready = 1;
    dcci(&my_hank->aicore_regs_ready, SINGLE_CACHE_LINE, CACHELINE_OUT);
    while (my_hank->aicpu_regs_ready == 0) {
        dcci(&my_hank->aicpu_regs_ready, SINGLE_CACHE_LINE);
    }
    // Report initial idle status via register
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // Phase 3: Report core type, signal ready
    my_hank->core_type = core_type;
    STORE_RELEASE_FENCE();
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Phase 3.5: Cache dispatch init info from AICPU.
    //
    // Why this wait is necessary:
    //   hank->task is set to 0 during handshake (Phase 1) because
    //   PTO2DispatchInitInfo requires shared memory addresses that don't
    //   exist yet — PTO2Runtime is created AFTER handshake completes.
    //   AICPU writes &init_info to hank->task once PTO2Runtime is ready.
    //   dcci is needed because hank->task lives in GM; without cache
    //   invalidation, AICore would keep reading a stale cached zero.
    while (my_hank->task == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
    }
    __gm__ PTO2DispatchInitInfo* init_info =
        reinterpret_cast<__gm__ PTO2DispatchInitInfo*>(my_hank->task);
    // init_info points to a separate GM object — invalidate its cache line
    // so we read the values AICPU wrote, not stale data.
    dcci(init_info, SINGLE_CACHE_LINE);

    uint64_t dispatch_base = init_info->dispatch_base;
    my_hank->task = 0;  // Clear after reading (no longer needed)
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    bool profiling_enabled = runtime->enable_profiling;
    uint64_t kernel_ready_time = get_sys_cnt_aicore();

    // Phase 4: Main execution loop - poll register for tasks until exit signal
    uint32_t reg_val = AICPU_IDLE_TASK_ID;
    uint32_t last_reg_val = AICPU_IDLE_TASK_ID;

    while (true) {
        reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            // Signal exit acknowledgment to AICPU
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }

        // Skip idle (0 or AICPU_IDLE_TASK_ID) or duplicate dispatch
        if (reg_val == 0 || reg_val == AICPU_IDLE_TASK_ID || reg_val == last_reg_val) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            // Decode register value using named constants from pto2_dispatch_payload.h.
            // Inline decode instead of calling pto2_reg_decode_*() because ccec does not
            // allow [aicore] code to call [host]-annotated functions.
            uint32_t offset_field = (reg_val >> PTO2_REG_OFFSET_SHIFT) & PTO2_REG_OFFSET_MASK;
            uint64_t desc_byte_offset = static_cast<uint64_t>(offset_field - 1) << PTO2_REG_ALIGN_SHIFT;
            uint32_t slot_idx = reg_val & PTO2_REG_SLOTIDX_MASK;

            // Compute dispatch descriptor address from cached base + decoded offset
            __gm__ PTO2DispatchDesc* desc = reinterpret_cast<__gm__ PTO2DispatchDesc*>(
                dispatch_base + desc_byte_offset);

            // Invalidate data cache to ensure fresh read of dispatch descriptor
            dcci(desc, ENTIRE_DATA_CACHE);

            write_reg(RegId::COND, MAKE_ACK_VALUE(reg_val));

            // Performance profiling: record start time
            uint64_t start_time = get_sys_cnt_aicore();

            // Execute the task
            execute_task(desc, slot_idx);

            // Performance profiling: record task execution
            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;
                perf_aicore_record_task(perf_buf, reg_val,
                                       start_time, end_time, kernel_ready_time);
            }

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(reg_val));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
