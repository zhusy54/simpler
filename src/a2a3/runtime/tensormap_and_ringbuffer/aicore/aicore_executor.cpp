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
 * 3. Read per-core PTO2DispatchDesc pointer from hank->task (wait for non-zero)
 * 4. Poll DATA_MAIN_BASE register for task dispatch until exit signal
 *
 * Register encoding (set by AICPU scheduler) — see pto2_dispatch_payload.h:
 *   bits [31:2] = reg_task_id (monotonically increasing per core)
 *   bits [1:0]  = slot_idx  (2 bits: 0=AIC, 1=AIV0, 2=AIV1)
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
    my_hank->aicore_done = 1;
    dcci(&my_hank->aicore_done, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Phase 3.5: Wait for per-core dispatch descriptor pointer from AICPU.
    // AICPU writes &s_dispatch_desc_per_core[i] to hank->task after PTO2Runtime
    // is created. AICore caches this pointer for the entire execution lifetime.
    while (my_hank->task == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
    }
    __gm__ PTO2DispatchDesc* my_desc =
        reinterpret_cast<__gm__ PTO2DispatchDesc*>(my_hank->task);
    my_hank->task = 0;
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
            // Decode slot_idx from register value (inline, ccec cannot call host functions)
            uint32_t slot_idx = reg_val & PTO2_REG_SLOTIDX_MASK;

            // Invalidate data cache to ensure fresh read of dispatch descriptor
            dcci(my_desc, ENTIRE_DATA_CACHE);

            write_reg(RegId::COND, MAKE_ACK_VALUE(reg_val));

            // Performance profiling: record start time
            uint64_t start_time = get_sys_cnt_aicore();

            // Execute the task
            execute_task(my_desc, slot_idx);

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
