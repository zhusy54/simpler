/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int64_t *state = reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
    const int64_t index = args[1];
    const int64_t iterations = args[2];
    volatile uint32_t digest = 0x13579BDFu;
    for (int64_t i = 0; i < iterations; ++i) {
        digest = digest * 1664525u + 1013904223u;
    }
    const int64_t token = (static_cast<int64_t>(digest & 0x7FFFFFFFu) << 32) | (index + 1);
    __gm__ int64_t *slot = &state[(index + 1) * 8];
    *slot = token;
    dcci(slot, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
