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

#include "aicore_execution_sidecar_v0.h"

inline __aicore__ void aicore_publish_cache_line_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_observe_cache_line_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void aicore_publish_data_cache_v0(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
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

// These wrappers are a runtime-local subset of simpler-dist FDWIC's validated
// A5 raw-GM atomic protocol. They intentionally expose the observed old value
// returned by the hardware instead of emulating std::atomic's API.
inline __aicore__ int64_t aicore_gm_load_v0(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    constexpr int64_t identity = (-9223372036854775807LL - 1LL);
    return atomicMax(const_cast<__gm__ int64_t *>(&value), identity);
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint64_t aicore_gm_load_v0(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    return atomicAdd(const_cast<__gm__ uint64_t *>(&value), UINT64_C(0));
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

inline __aicore__ void aicore_publish_next_waiter_v0(__gm__ AicoreTaskControlV0 *control, int64_t next_waiter) {
    control->next_waiter = next_waiter;
    aicore_publish_cache_line_v0(&control->next_waiter);
}

inline __aicore__ int64_t aicore_observe_next_waiter_v0(__gm__ AicoreTaskControlV0 *control) {
    aicore_observe_cache_line_v0(&control->next_waiter);
    return control->next_waiter;
}
