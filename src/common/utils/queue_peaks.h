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
#include <cstdint>
#include <type_traits>

namespace simpler::queue_peaks {

enum class Kind : uint32_t { Ready, Sync, Early, Dummy, EarlySync, GraphPrepare, PublishDrain, Completion, Mailbox };

inline const char *name(Kind kind) {
    constexpr const char *names[] = {
        "ready",         "ready_sync",       "early_dispatch", "dummy",  "early_sync_start",
        "graph_prepare", "ed_publish_drain", "completion",     "mailbox"
    };
    return names[static_cast<uint32_t>(kind)];
}

struct alignas(64) Counters {
    // Low word: ordinary entries (or all entries in an unclassified queue).
    // High word: SPMD entries. A slot is accounted before publication and
    // unaccounted before reuse, so neither component can underflow.
    std::atomic<uint64_t> current{0};
    std::atomic<uint64_t> peak{0};
    std::atomic<uint64_t> spmd_peak{0};
    std::atomic<uint64_t> normal_peak{0};

    static uint64_t packed(uint32_t count, uint32_t spmd) {
        return (static_cast<uint64_t>(spmd) << 32) | (count - spmd);
    }
    static void raise(std::atomic<uint64_t> &target, uint64_t value) {
        uint64_t old = target.load(std::memory_order_relaxed);
        while (value > old && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {}
    }
    void reset() {
        current.store(0, std::memory_order_relaxed);
        peak.store(0, std::memory_order_relaxed);
        spmd_peak.store(0, std::memory_order_relaxed);
        normal_peak.store(0, std::memory_order_relaxed);
    }
    void push(uint32_t count, uint32_t spmd = 0, bool classified = false) {
        const uint64_t delta = packed(count, spmd);
        const uint64_t now = current.fetch_add(delta, std::memory_order_relaxed) + delta;
        const uint64_t normal = static_cast<uint32_t>(now);
        const uint64_t parallel = now >> 32;
        raise(peak, normal + parallel);
        if (classified) {
            raise(spmd_peak, parallel);
            raise(normal_peak, normal);
        }
    }
    void pop(uint32_t count, uint32_t spmd = 0) { current.fetch_sub(packed(count, spmd), std::memory_order_relaxed); }
};
static_assert(sizeof(Counters) == 64);

struct Record {
    Kind kind;
    int32_t index;
    uint64_t capacity;
    uint64_t peak;
    uint64_t spmd_peak;
    uint64_t normal_peak;
    uint32_t classified;
    uint32_t reserved;
};
constexpr uint32_t MAX_QUEUES = 32;
struct alignas(64) Report {
    uint64_t epoch;
    uint32_t version;
    uint32_t count;
    Record records[MAX_QUEUES];
};
static_assert(std::is_standard_layout_v<Report> && std::is_trivially_copyable_v<Report>);

// Owned by one AICPU executor; only bind/snapshot/reset run outside queue ops.
class Workspace {
public:
    Counters *add(Kind kind, int32_t index, uint64_t capacity, bool classified = false) {
        if (count_ == MAX_QUEUES) return nullptr;
        auto &record = records_[count_];
        record = {kind, index, capacity, 0, 0, 0, static_cast<uint32_t>(classified), 0};
        states_[count_].reset();
        return &states_[count_++];
    }
    template <class Queue>
    void bind(Queue &queue, Kind kind, int32_t index, bool classified = false) {
        const uint32_t slot = count_;
        queue.queue_peaks = add(kind, index, queue.capacity, classified);
        if (queue.queue_peaks == nullptr) {
            queue.entry_classes = nullptr;
            return;
        }
        classes_[slot] = classified ? new uint8_t[queue.capacity] : nullptr;
        queue.entry_classes = classes_[slot];
    }
    void snapshot(Report &report) const {
        report = {};
        report.version = 1;
        report.count = count_;
        for (uint32_t i = 0; i < count_; ++i) {
            report.records[i] = records_[i];
            report.records[i].peak = states_[i].peak.load(std::memory_order_relaxed);
            report.records[i].spmd_peak = states_[i].spmd_peak.load(std::memory_order_relaxed);
            report.records[i].normal_peak = states_[i].normal_peak.load(std::memory_order_relaxed);
        }
    }
    void reset() {
        for (uint32_t i = 0; i < count_; ++i) {
            delete[] classes_[i];
            classes_[i] = nullptr;
        }
        count_ = 0;
    }

private:
    uint32_t count_{0};
    Counters states_[MAX_QUEUES];
    Record records_[MAX_QUEUES]{};
    uint8_t *classes_[MAX_QUEUES]{};
};

}  // namespace simpler::queue_peaks
