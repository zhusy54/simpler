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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

enum class AicoreModelTaskStateV0 : uint8_t {
    UNCLASSIFIED = 0,
    WAITING = 1,
    READY = 2,
    RUNNING = 3,
    COMPLETED = 4,
};

class AicoreDependencyModelV0 {
public:
    explicit AicoreDependencyModelV0(std::vector<std::vector<int64_t>> fanins) :
        fanins_(std::move(fanins)),
        states_(fanins_.size(), AicoreModelTaskStateV0::UNCLASSIFIED),
        wake_heads_(fanins_.size(), kInvalidTask),
        next_waiters_(fanins_.size(), kInvalidTask),
        queued_(fanins_.size(), false) {
        for (size_t task_id = 0; task_id < fanins_.size(); ++task_id) {
            std::vector<int64_t> &task_fanins = fanins_[task_id];
            std::sort(task_fanins.begin(), task_fanins.end());
            if (std::adjacent_find(task_fanins.begin(), task_fanins.end()) != task_fanins.end()) valid_ = false;
            for (int64_t producer : task_fanins) {
                if (producer < 0 || producer >= static_cast<int64_t>(task_id)) valid_ = false;
            }
        }
    }

    bool classify_all() {
        if (!valid_) return false;
        for (int64_t task_id = 0; task_id < static_cast<int64_t>(fanins_.size()); ++task_id) {
            if (!classify(task_id)) return false;
        }
        return true;
    }

    bool classify(int64_t task_id) {
        if (!valid_task(task_id) || states_[task_id] == AicoreModelTaskStateV0::COMPLETED ||
            states_[task_id] == AicoreModelTaskStateV0::RUNNING) {
            return false;
        }
        for (int64_t producer : fanins_[task_id]) {
            if (states_[producer] == AicoreModelTaskStateV0::COMPLETED) continue;
            next_waiters_[task_id] = wake_heads_[producer];
            if (wake_heads_[producer] == kClosedWakeList) continue;
            wake_heads_[producer] = task_id;
            states_[task_id] = AicoreModelTaskStateV0::WAITING;
            return true;
        }
        return enqueue_ready(task_id);
    }

    bool pop_ready(int64_t *task_id) {
        if (task_id == nullptr || ready_.empty()) return false;
        *task_id = ready_.front();
        ready_.pop_front();
        queued_[*task_id] = false;
        states_[*task_id] = AicoreModelTaskStateV0::RUNNING;
        return true;
    }

    bool complete(int64_t task_id) {
        if (!valid_task(task_id) || states_[task_id] != AicoreModelTaskStateV0::RUNNING) return false;
        states_[task_id] = AicoreModelTaskStateV0::COMPLETED;
        ++completed_count_;
        int64_t waiter = wake_heads_[task_id];
        wake_heads_[task_id] = kClosedWakeList;
        while (waiter >= 0) {
            int64_t next = next_waiters_[waiter];
            next_waiters_[waiter] = kInvalidTask;
            if (!classify(waiter)) return false;
            waiter = next;
        }
        return true;
    }

    AicoreModelTaskStateV0 state(int64_t task_id) const {
        return valid_task(task_id) ? states_[task_id] : AicoreModelTaskStateV0::UNCLASSIFIED;
    }

    size_t ready_count() const { return ready_.size(); }
    size_t completed_count() const { return completed_count_; }

    bool validate_invariants() const {
        if (!valid_) return false;
        std::vector<int> ready_occurrences(fanins_.size(), 0);
        for (int64_t task_id : ready_) {
            if (!valid_task(task_id)) return false;
            ++ready_occurrences[task_id];
        }

        std::vector<int> waiter_occurrences(fanins_.size(), 0);
        for (size_t producer = 0; producer < wake_heads_.size(); ++producer) {
            int64_t waiter = wake_heads_[producer];
            size_t steps = 0;
            while (waiter >= 0) {
                if (!valid_task(waiter) || ++steps > fanins_.size()) return false;
                ++waiter_occurrences[waiter];
                waiter = next_waiters_[waiter];
            }
        }

        size_t completed = 0;
        for (size_t task_id = 0; task_id < states_.size(); ++task_id) {
            if (ready_occurrences[task_id] > 1 || waiter_occurrences[task_id] > 1) return false;
            switch (states_[task_id]) {
            case AicoreModelTaskStateV0::WAITING:
                if (waiter_occurrences[task_id] != 1 || queued_[task_id]) return false;
                break;
            case AicoreModelTaskStateV0::READY:
                if (ready_occurrences[task_id] != 1 || !queued_[task_id]) return false;
                for (int64_t producer : fanins_[task_id]) {
                    if (states_[producer] != AicoreModelTaskStateV0::COMPLETED) return false;
                }
                break;
            case AicoreModelTaskStateV0::COMPLETED:
                ++completed;
                if (wake_heads_[task_id] != kClosedWakeList) return false;
                break;
            case AicoreModelTaskStateV0::UNCLASSIFIED:
            case AicoreModelTaskStateV0::RUNNING:
                if (ready_occurrences[task_id] != 0 || waiter_occurrences[task_id] != 0 || queued_[task_id]) {
                    return false;
                }
                break;
            }
        }
        return completed == completed_count_;
    }

private:
    static constexpr int64_t kInvalidTask = -1;
    static constexpr int64_t kClosedWakeList = -2;

    bool valid_task(int64_t task_id) const { return task_id >= 0 && task_id < static_cast<int64_t>(fanins_.size()); }

    bool enqueue_ready(int64_t task_id) {
        if (queued_[task_id] || states_[task_id] == AicoreModelTaskStateV0::READY) return false;
        states_[task_id] = AicoreModelTaskStateV0::READY;
        queued_[task_id] = true;
        ready_.push_back(task_id);
        return true;
    }

    std::vector<std::vector<int64_t>> fanins_;
    std::vector<AicoreModelTaskStateV0> states_;
    std::vector<int64_t> wake_heads_;
    std::vector<int64_t> next_waiters_;
    std::vector<bool> queued_;
    std::deque<int64_t> ready_;
    size_t completed_count_{0};
    bool valid_{true};
};
