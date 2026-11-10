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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace wse::temporary {

enum class CounterStatus { OK, READY, NOT_READY, INVALID_ARGUMENT, OVERFLOW };

// The caller owns live uint64_t objects; concurrent accesses use atomic_ref exclusively.
struct CounterView {
    uint64_t *data = nullptr;
    size_t size = 0;
};
struct CounterSignalResult {
    CounterStatus status;
    uint64_t old_value = 0;
    uint64_t new_value = 0;
};
struct CounterPollResult {
    CounterStatus status;
    uint64_t observed = 0;
};

inline bool valid_counter_word(CounterView view, size_t index) noexcept {
    constexpr auto max = std::numeric_limits<uintptr_t>::max();
    const auto base = reinterpret_cast<uintptr_t>(view.data);
    if (!base || index >= view.size || view.size > (max - base) / sizeof(uint64_t)) return false;
    return (base + index * sizeof(uint64_t)) % std::atomic_ref<uint64_t>::required_alignment == 0;
}

inline CounterSignalResult counter_signal(CounterView view, size_t index, uint64_t delta = 1) noexcept {
    if (!delta || !valid_counter_word(view, index)) return {CounterStatus::INVALID_ARGUMENT};
    std::atomic_ref<uint64_t> word(view.data[index]);
    auto old = word.load(std::memory_order_relaxed);
    for (;;) {
        if (delta > std::numeric_limits<uint64_t>::max() - old) return {CounterStatus::OVERFLOW, old, old};
        if (word.compare_exchange_weak(old, old + delta, std::memory_order_acq_rel, std::memory_order_relaxed))
            return {CounterStatus::OK, old, old + delta};
    }
}

inline CounterPollResult counter_poll(CounterView view, size_t index, uint64_t target, size_t max_reads = 1) noexcept {
    if (!max_reads || !valid_counter_word(view, index)) return {CounterStatus::INVALID_ARGUMENT};
    std::atomic_ref<uint64_t> word(view.data[index]);
    uint64_t observed = 0;
    for (size_t i = 0; i < max_reads; ++i) {
        observed = word.load(std::memory_order_acquire);
        if (observed >= target) return {CounterStatus::READY, observed};
    }
    return {CounterStatus::NOT_READY, observed};
}

}  // namespace wse::temporary
