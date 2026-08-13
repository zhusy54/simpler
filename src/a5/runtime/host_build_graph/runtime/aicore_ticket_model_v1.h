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
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

class AicoreTicketModelV1 {
public:
    AicoreTicketModelV1(
        std::vector<std::vector<int64_t>> fanins, std::vector<uint32_t> bottom_level, std::vector<int64_t> aic_tasks,
        std::vector<int64_t> aiv_tasks, size_t aic_workers, size_t aiv_workers, size_t pending_slots = 2
    ) :
        fanins_(std::move(fanins)),
        bottom_level_(std::move(bottom_level)),
        state_(fanins_.size(), TaskState::BLOCKED),
        next_fanin_(fanins_.size(), 0),
        wake_lists_(fanins_.size()),
        wake_closed_(fanins_.size(), false),
        execution_count_(fanins_.size(), 0),
        pending_slots_(pending_slots) {
        valid_ = pending_slots_ != 0 && bottom_level_.size() == fanins_.size();
        validate_graph();
        streams_.reserve(2);
        add_workers(std::move(aic_tasks), aic_workers);
        add_workers(std::move(aiv_tasks), aiv_workers);
    }

    void mark_inline_completed(int64_t task_id) {
        if (!valid_task(task_id) || stream_task_[task_id]) {
            valid_ = false;
            return;
        }
        state_[task_id] = TaskState::DONE;
        wake_closed_[task_id] = true;
    }

    bool run() {
        if (!valid_) return false;
        seed_workers();
        size_t no_progress_rounds = 0;
        while (true) {
            bool all_drained = true;
            bool progress = false;
            for (Worker &worker : workers_) {
                if (worker.drained) continue;
                all_drained = false;
                if (advance(worker)) progress = true;
            }
            if (all_drained) break;
            if (progress) {
                no_progress_rounds = 0;
            } else if (++no_progress_rounds > fanins_.size() + workers_.size() + 1) {
                return false;
            }
        }
        return std::all_of(state_.begin(), state_.end(), [](TaskState state) {
            return state == TaskState::DONE;
        });
    }

    bool valid() const { return valid_; }
    const std::vector<int> &execution_count() const { return execution_count_; }
    size_t wake_register_count() const { return wake_register_count_; }
    size_t wake_migrate_count() const { return wake_migrate_count_; }

private:
    enum class TaskState {
        BLOCKED,
        READY,
        DONE,
    };
    struct Pending {
        int64_t task_id{-1};
    };
    struct Stream {
        std::vector<int64_t> tasks;
        size_t cursor{0};
        size_t active_workers{0};
    };
    struct Worker {
        Stream *stream{nullptr};
        size_t rank{0};
        std::vector<Pending> pending;
        size_t scan_start{0};
        bool exhausted{false};
        bool drained{false};
    };

    bool valid_task(int64_t task_id) const { return task_id >= 0 && static_cast<size_t>(task_id) < fanins_.size(); }

    bool priority_before(int64_t lhs, int64_t rhs) const {
        if (bottom_level_[lhs] != bottom_level_[rhs]) return bottom_level_[lhs] > bottom_level_[rhs];
        return lhs < rhs;
    }

    void validate_graph() {
        stream_task_.assign(fanins_.size(), false);
        for (size_t task_id = 0; task_id < fanins_.size(); ++task_id) {
            auto &fanin = fanins_[task_id];
            std::sort(fanin.begin(), fanin.end());
            if (std::adjacent_find(fanin.begin(), fanin.end()) != fanin.end()) valid_ = false;
            for (int64_t producer : fanin) {
                if (producer < 0 || producer >= static_cast<int64_t>(task_id)) {
                    valid_ = false;
                } else if (bottom_level_.size() == fanins_.size() &&
                           bottom_level_[producer] <= bottom_level_[task_id]) {
                    valid_ = false;
                }
            }
        }
    }

    void add_workers(std::vector<int64_t> tasks, size_t worker_count) {
        bool valid_stream = true;
        for (int64_t task_id : tasks) {
            if (!valid_task(task_id) || stream_task_[task_id]) {
                valid_ = false;
                valid_stream = false;
            } else {
                stream_task_[task_id] = true;
            }
        }
        if (valid_stream && bottom_level_.size() == fanins_.size() &&
            !std::is_sorted(tasks.begin(), tasks.end(), [&](int64_t lhs, int64_t rhs) {
                return priority_before(lhs, rhs);
            })) {
            valid_ = false;
        }
        streams_.push_back({std::move(tasks), 0, 0});
        Stream &stream = streams_.back();
        stream.active_workers = std::min(worker_count, stream.tasks.size());
        stream.cursor = stream.active_workers;
        for (size_t rank = 0; rank < stream.active_workers; ++rank) {
            workers_.push_back({&stream, rank, std::vector<Pending>(pending_slots_)});
        }
        if (!stream.tasks.empty() && stream.active_workers == 0) valid_ = false;
    }

    void seed_workers() {
        for (Worker &worker : workers_) {
            worker.pending[0].task_id = worker.stream->tasks[worker.rank];
            route(worker.pending[0].task_id);
        }
    }

    void route(int64_t task_id) {
        while (next_fanin_[task_id] < fanins_[task_id].size()) {
            int64_t producer = fanins_[task_id][next_fanin_[task_id]];
            if (state_[producer] == TaskState::DONE) {
                ++next_fanin_[task_id];
                continue;
            }
            if (wake_closed_[producer]) continue;
            wake_lists_[producer].push_back(task_id);
            ++wake_register_count_;
            return;
        }
        state_[task_id] = TaskState::READY;
    }

    void complete(int64_t task_id) {
        state_[task_id] = TaskState::DONE;
        wake_closed_[task_id] = true;
        std::vector<int64_t> waiters = std::move(wake_lists_[task_id]);
        for (int64_t waiter : waiters) {
            ++wake_migrate_count_;
            route(waiter);
        }
    }

    bool advance(Worker &worker) {
        for (size_t offset = 0; offset < worker.pending.size(); ++offset) {
            size_t slot = (worker.scan_start + offset) % worker.pending.size();
            Pending &pending = worker.pending[slot];
            if (pending.task_id >= 0 && state_[pending.task_id] == TaskState::READY) {
                ++execution_count_[pending.task_id];
                complete(pending.task_id);
                pending = {};
                worker.scan_start = (slot + 1) % worker.pending.size();
                return true;
            }
        }
        auto free = std::find_if(worker.pending.begin(), worker.pending.end(), [](const Pending &p) {
            return p.task_id < 0;
        });
        if (free != worker.pending.end() && !worker.exhausted) {
            size_t index = worker.stream->cursor++;
            if (index < worker.stream->tasks.size()) {
                free->task_id = worker.stream->tasks[index];
                route(free->task_id);
                return true;
            }
            worker.exhausted = true;
        }
        bool empty = std::all_of(worker.pending.begin(), worker.pending.end(), [](const Pending &pending) {
            return pending.task_id < 0;
        });
        if (worker.exhausted && empty) {
            worker.drained = true;
            return true;
        }
        return false;
    }

    std::vector<std::vector<int64_t>> fanins_;
    std::vector<uint32_t> bottom_level_;
    std::vector<TaskState> state_;
    std::vector<size_t> next_fanin_;
    std::vector<std::vector<int64_t>> wake_lists_;
    std::vector<bool> wake_closed_;
    std::vector<bool> stream_task_;
    std::vector<int> execution_count_;
    std::vector<Stream> streams_;
    std::vector<Worker> workers_;
    size_t pending_slots_{2};
    size_t wake_register_count_{0};
    size_t wake_migrate_count_{0};
    bool valid_{true};
};
