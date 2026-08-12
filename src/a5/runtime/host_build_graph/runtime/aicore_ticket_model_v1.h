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
        std::vector<std::vector<int64_t>> fanins, std::vector<int64_t> aic_tasks, std::vector<int64_t> aiv_tasks,
        size_t aic_workers, size_t aiv_workers, size_t pending_slots = 2
    ) :
        fanins_(std::move(fanins)),
        completed_(fanins_.size(), false),
        execution_count_(fanins_.size(), 0),
        pending_slots_(pending_slots) {
        valid_ = pending_slots_ != 0;
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
        completed_[task_id] = true;
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
        return std::all_of(completed_.begin(), completed_.end(), [](bool value) {
            return value;
        });
    }

    bool valid() const { return valid_; }
    const std::vector<int> &execution_count() const { return execution_count_; }

private:
    struct Pending {
        int64_t task_id{-1};
        size_t next_fanin{0};
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

    void validate_graph() {
        stream_task_.assign(fanins_.size(), false);
        for (size_t task_id = 0; task_id < fanins_.size(); ++task_id) {
            auto &fanin = fanins_[task_id];
            std::sort(fanin.begin(), fanin.end());
            if (std::adjacent_find(fanin.begin(), fanin.end()) != fanin.end()) valid_ = false;
            for (int64_t producer : fanin) {
                if (producer < 0 || producer >= static_cast<int64_t>(task_id)) valid_ = false;
            }
        }
    }

    void add_workers(std::vector<int64_t> tasks, size_t worker_count) {
        if (!std::is_sorted(tasks.begin(), tasks.end())) valid_ = false;
        for (int64_t task_id : tasks) {
            if (!valid_task(task_id) || stream_task_[task_id]) {
                valid_ = false;
            } else {
                stream_task_[task_id] = true;
            }
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
        for (Worker &worker : workers_)
            worker.pending[0].task_id = worker.stream->tasks[worker.rank];
    }

    bool ready(Pending &pending) {
        while (pending.next_fanin < fanins_[pending.task_id].size()) {
            int64_t producer = fanins_[pending.task_id][pending.next_fanin];
            if (!completed_[producer]) return false;
            ++pending.next_fanin;
        }
        return true;
    }

    bool advance(Worker &worker) {
        for (size_t offset = 0; offset < worker.pending.size(); ++offset) {
            size_t slot = (worker.scan_start + offset) % worker.pending.size();
            Pending &pending = worker.pending[slot];
            if (pending.task_id >= 0 && ready(pending)) {
                ++execution_count_[pending.task_id];
                completed_[pending.task_id] = true;
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
    std::vector<bool> completed_;
    std::vector<bool> stream_task_;
    std::vector<int> execution_count_;
    std::vector<Stream> streams_;
    std::vector<Worker> workers_;
    size_t pending_slots_{2};
    bool valid_{true};
};
