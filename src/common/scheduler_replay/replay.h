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
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace simpler::replay {

struct Task {
    uint64_t token{};
    uint32_t mask{};
    std::vector<int32_t> predecessors;
    std::vector<uint32_t> dependency_flags;
    uint64_t duration[3]{};
};

struct Input {
    uint32_t clusters{}, threads{};
    uint32_t task_window{16384}, dependency_capacity{16384};
    uint32_t reverse_order{};
    std::vector<Task> tasks;
    explicit Input(std::istream &stream) {
        uint32_t version{}, count{};
        if (!(stream >> version >> clusters >> threads >> count) || version < 1 || version > 4 || clusters == 0 ||
            clusters > 64 || threads == 0 || threads > clusters || count == 0 || count > 1000000)
            throw std::invalid_argument("invalid replay header");
        if (version >= 2 &&
            (!(stream >> task_window >> dependency_capacity) || task_window < 2 || (task_window & (task_window - 1)) ||
             task_window > 1048576 || dependency_capacity < 2 || dependency_capacity > 1048576))
            throw std::invalid_argument("invalid native resource capacity");
        if (version >= 4 && (!(stream >> reverse_order) || reverse_order > 1))
            throw std::invalid_argument("invalid scheduler arbitration order");
        tasks.resize(count);
        for (size_t i = 0; i < count; ++i) {
            auto &task = tasks[i];
            size_t fanin{};
            if (!(stream >> task.token >> task.mask >> task.duration[0] >> task.duration[1] >> task.duration[2] >>
                  fanin) ||
                task.mask > 7 || fanin > count)
                throw std::invalid_argument("invalid task row");
            for (uint32_t sub = 0; sub < 3; ++sub)
                if (((task.mask >> sub) & 1) != (task.duration[sub] != 0))
                    throw std::invalid_argument("active fragment needs a positive duration");
            task.predecessors.resize(fanin);
            task.dependency_flags.resize(fanin, 3);
            for (size_t edge = 0; edge < fanin; ++edge) {
                auto &pred = task.predecessors[edge];
                if (!(stream >> pred) || pred < 0 || static_cast<size_t>(pred) >= i ||
                    (version >= 3 && (!(stream >> task.dependency_flags[edge]) || task.dependency_flags[edge] > 3)))
                    throw std::invalid_argument("tasks must be in dependency order");
            }
        }
        std::string extra;
        if (stream >> extra) throw std::invalid_argument("unexpected trailing input");
    }
};

struct Event {
    uint64_t end{};
    uint32_t core{}, owner{}, lane{}, slot{}, generation{}, task{}, sub{};
    bool operator>(const Event &other) const { return std::tie(end, core) > std::tie(other.end, other.core); }
};

class Clock {
public:
    explicit Clock(const Input &input) :
        input_(input),
        busy_(input.clusters * 3, false) {}
    bool busy(uint32_t core) const { return busy_.at(core); }
    void dispatch(uint32_t core, uint32_t generation) { dispatches_.emplace(std::make_pair(core, generation), now); }
    void admit(uint32_t task) { admissions_.emplace(task, now); }
    bool start(Event event) {
        if (busy_.at(event.core)) return false;
        const auto duration = input_.tasks.at(event.task).duration[event.sub];
        if ((!duration && input_.tasks.at(event.task).mask != 0) ||
            duration > std::numeric_limits<uint64_t>::max() - now)
            throw std::overflow_error("invalid execution duration");
        busy_[event.core] = true;
        event.end = now + duration;
        events_.push(event);
        rows.push_back({event, now});
        return true;
    }
    std::vector<Event> advance() {
        if (events_.empty()) throw std::runtime_error("scheduler stalled without a future completion");
        now = events_.top().end;
        std::vector<Event> batch;
        while (!events_.empty() && events_.top().end == now) {
            auto event = events_.top();
            events_.pop();
            busy_[event.core] = false;
            batch.push_back(event);
        }
        return batch;
    }
    void write() const {
        std::cout << "time " << now << '\n';
        for (const auto &[event, start] : rows)
            std::cout << input_.tasks[event.task].token << ' ' << event.sub << ' ' << event.core << ' ' << start << ' '
                      << event.end << ' ' << event.owner << ' '
                      << (input_.tasks[event.task].mask ? dispatches_.at({event.core, event.generation}) : start) << ' '
                      << (admissions_.count(event.task) ? admissions_.at(event.task) : 0) << '\n';
    }
    uint64_t now{};
    std::vector<std::pair<Event, uint64_t>> rows;

private:
    const Input &input_;
    std::vector<bool> busy_;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> dispatches_;
    std::map<uint32_t, uint64_t> admissions_;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events_;
};

}  // namespace simpler::replay
