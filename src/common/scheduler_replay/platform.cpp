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
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "aicpu/platform_regs.h"
#include "common/unified_log.h"
#include "aicpu/aicpu_device_config.h"
#include <stdexcept>

extern "C" uint64_t get_platform_phase_base() { return 0; }
unsigned long long get_dma_workspace_addr(int) { return 0; }
unsigned long long get_dev_l2_cache_offset() { return 0; }
uint64_t platform_aicore_exit_deadline() { throw std::runtime_error("unexpected device retirement in replay"); }
int32_t platform_retire_aicore_group(const AicoreExitTarget *, size_t, uint64_t, bool *) {
    throw std::runtime_error("unexpected device retirement in replay");
}
[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::runtime_error(condition); }
extern "C" void unified_log_error(const char *func, const char *fmt, ...) {
    std::fprintf(stderr, "%s: ", func);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
}
extern "C" void unified_log_warn(const char *, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_debug(const char *, const char *, ...) {}

uint64_t get_sys_cnt_aicpu() { return 0; }
uint64_t get_sys_cnt_aicpu_observed() { return 0; }
uint32_t reg_load_acquire(const volatile uint32_t *p) { return *p; }
void reg_store_release(volatile uint32_t *p, uint32_t v) { *p = v; }
volatile uint32_t *get_reg_ptr(uint64_t address, RegId reg) {
    return reinterpret_cast<volatile uint32_t *>(address) + static_cast<size_t>(reg);
}
uint64_t read_reg(uint64_t address, RegId reg) { return *get_reg_ptr(address, reg); }
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *, size_t) {}
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance
