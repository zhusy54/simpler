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

#include <stdint.h>

#include "aicore_execution_sidecar_v1.h"

inline __aicore__ void aicore_writeback_cache_line_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    (void)address;
#endif
}

inline __aicore__ void aicore_writeback_cache_line_v0(volatile __gm__ void *address) {
    aicore_writeback_cache_line_v0(const_cast<__gm__ void *>(address));
}

inline __aicore__ void aicore_cache_barrier_v0() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_publish_cache_line_v0(__gm__ void *address) {
    aicore_writeback_cache_line_v0(address);
    aicore_cache_barrier_v0();
}

inline __aicore__ void aicore_publish_cache_line_v0(volatile __gm__ void *address) {
    aicore_publish_cache_line_v0(const_cast<__gm__ void *>(address));
}

inline __aicore__ void aicore_invalidate_cache_line_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE);
#else
    (void)address;
#endif
}

inline __aicore__ void aicore_invalidate_cache_line_v0(volatile __gm__ void *address) {
    aicore_invalidate_cache_line_v0(const_cast<__gm__ void *>(address));
}

inline __aicore__ void aicore_observe_cache_line_v0(__gm__ void *address) {
    aicore_invalidate_cache_line_v0(address);
    aicore_cache_barrier_v0();
}

inline __aicore__ void aicore_observe_cache_line_v0(volatile __gm__ void *address) {
    aicore_observe_cache_line_v0(const_cast<__gm__ void *>(address));
}

inline __aicore__ void aicore_observe_data_cache_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_publish_dispatch_payload_v1(__gm__ PTO2DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 0; offset < sizeof(PTO2DispatchPayload); offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(payload) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb((mem_dsb_t)0);
#else
    (void)payload;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_observe_dispatch_payload_control_v1(__gm__ PTO2DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    dcci(payload, SINGLE_CACHE_LINE);
#else
    (void)payload;
#endif
}

inline __aicore__ void aicore_observe_dispatch_payload_arguments_v1(__gm__ PTO2DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 64; offset < sizeof(PTO2DispatchPayload); offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(payload) + offset, SINGLE_CACHE_LINE);
    }
#else
    (void)payload;
#endif
}

inline __aicore__ void aicore_observe_dispatch_payload_barrier_v1() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_observe_dispatch_payload_v1(__gm__ PTO2DispatchPayload *payload) {
    aicore_observe_dispatch_payload_control_v1(payload);
    aicore_observe_dispatch_payload_arguments_v1(payload);
    aicore_observe_dispatch_payload_barrier_v1();
}

// A5 ld_dev bypasses the scalar DCache. It is only an observation/prefilter
// primitive; ownership changes use the CAS/exchange/fetch wrappers below.
// Publication-protected metadata is invalidated separately before consumption.
inline __aicore__ int64_t aicore_gm_query_v0(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
    __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
    return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint64_t aicore_gm_query_v0(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ void
aicore_gm_store_v0(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    st_dev(
        static_cast<uint64_t>(desired), reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&value)), 0
    );
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
aicore_gm_store_v0(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    st_dev(desired, const_cast<__gm__ uint64_t *>(&value), 0);
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
aicore_gm_publish_v0(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    (void)atomicExch(const_cast<__gm__ uint64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ int64_t
aicore_gm_exchange_v0(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    return atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
    return __atomic_exchange_n(&value, desired, order);
#endif
}

inline __aicore__ uint64_t
aicore_gm_fetch_add_v0(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    return atomicAdd(const_cast<__gm__ uint64_t *>(&value), delta);
#else
    return __atomic_fetch_add(&value, delta, order);
#endif
}

inline __aicore__ int64_t aicore_gm_compare_exchange_v0(
    __gm__ volatile int64_t &value, int64_t expected, int64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
#if defined(__CCE_AICORE__)
    (void)success_order;
    (void)failure_order;
    return atomicCAS(const_cast<__gm__ int64_t *>(&value), expected, desired);
#else
    int64_t observed = expected;
    (void)__atomic_compare_exchange_n(&value, &observed, desired, false, success_order, failure_order);
    return observed;
#endif
}

inline __aicore__ uint64_t aicore_gm_compare_exchange_v0(
    __gm__ volatile uint64_t &value, uint64_t expected, uint64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
#if defined(__CCE_AICORE__)
    (void)success_order;
    (void)failure_order;
    return atomicCAS(const_cast<__gm__ uint64_t *>(&value), expected, desired);
#else
    uint64_t observed = expected;
    (void)__atomic_compare_exchange_n(&value, &observed, desired, false, success_order, failure_order);
    return observed;
#endif
}

inline __aicore__ uint64_t
aicore_gm_fetch_or_v0(__gm__ volatile uint64_t &value, uint64_t bits, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    uint64_t observed = aicore_gm_query_v0(value);
    while ((observed & bits) != bits) {
        uint64_t actual = aicore_gm_compare_exchange_v0(value, observed, observed | bits);
        if (actual == observed) break;
        observed = actual;
    }
    return observed;
#else
    return __atomic_fetch_or(&value, bits, order);
#endif
}

inline __aicore__ uint64_t
aicore_gm_fetch_and_v0(__gm__ volatile uint64_t &value, uint64_t bits, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    uint64_t observed = aicore_gm_query_v0(value);
    while ((observed & bits) != observed) {
        uint64_t actual = aicore_gm_compare_exchange_v0(value, observed, observed & bits);
        if (actual == observed) break;
        observed = actual;
    }
    return observed;
#else
    return __atomic_fetch_and(&value, bits, order);
#endif
}
