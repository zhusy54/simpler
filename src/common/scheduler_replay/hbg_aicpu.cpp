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
#include "replay.h"

#include <array>
#include <memory>

#include "callable.h"
#include "scheduler/scheduler_context.h"
#include "host_build_graph/runtime_core.h"

class SchedulerReplayAdapter {
public:
    explicit SchedulerReplayAdapter(const simpler::replay::Input &input) :
        input_(input),
        clock_(input),
        context_(std::make_unique<SchedulerContext>()),
        registers_(input.clusters * 3),
        launched_(input.clusters * 3) {
        if (input.threads >= MAX_AICPU_THREADS || input.clusters * 3 > RUNTIME_MAX_WORKER ||
            (input.clusters + input.threads - 1) / input.threads > CoreTracker::MAX_CLUSTERS)
            throw std::invalid_argument("invalid native scheduler topology");
        const auto count = input.tasks.size();
        auto bytes = SharedMemoryHandle::calculate_size(count);
        size_t edges = 0;
        for (const auto &task : input.tasks)
            edges += task.predecessors.size();
        const auto image_bytes = (bytes + edges * sizeof(int32_t) + 63) & ~size_t(63);
        image_.reset(std::aligned_alloc(64, image_bytes));
        if (!image_ || !shared_.init(image_.get(), image_bytes, count)) throw std::bad_alloc();
        auto *fanin = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(image_.get()) + bytes);
        auto layout = SchedulerState::reserve_layout(arena_);
        if (!arena_.commit() || !scheduler_.init_data_from_layout(layout, arena_, image_.get()))
            throw std::runtime_error("HBG scheduler initialization failed");
        scheduler_.wire_arena_pointers(layout, arena_);
        scheduler_.seed_queue_slots();
        auto &ctx = *context_;
        ctx.sched_ = &scheduler_;
        ctx.active_sched_threads_ = input.threads;
        ctx.aicpu_thread_num_ = input.threads + 1;
        ctx.p_thread_idx_ = input.threads;
        ctx.cores_total_num_ = input.clusters * 3;
        ctx.aic_count_ = input.clusters;
        ctx.aiv_count_ = input.clusters * 2;
        auto *callable = reinterpret_cast<CoreCallable *>(callable_.data());
        callable->set_resolved_addr(0x1000);
        callable_address_ = reinterpret_cast<uint64_t>(callable);
        ctx.func_id_to_addr_ = &callable_address_;
        ctx.func_id_to_addr_count_ = 1;
        uint64_t capacity = 1;
        while (capacity < count)
            capacity *= 2;
        for (uint32_t thread = 0; thread < input.threads; ++thread)
            ctx.sp_queues_[thread].init(capacity);
        for (uint32_t core = 0; core < input.clusters * 3; ++core) {
            auto &state = ctx.core_exec_states_[core];
            state.reg_addr = reinterpret_cast<uint64_t>(registers_[core].data());
            state.cond_ptr = get_reg_ptr(state.reg_addr, RegId::COND);
            state.running_reg_task_id = AICPU_TASK_INVALID;
            state.pending_reg_task_id = AICPU_TASK_INVALID;
            ctx.core_type_compact_[core] = static_cast<uint8_t>(core < input.clusters ? CoreType::AIC : CoreType::AIV);
        }
        for (uint32_t core = 0; core < input.clusters; ++core)
            ctx.aic_worker_ids_[core] = core;
        for (uint32_t core = 0; core < input.clusters * 2; ++core)
            ctx.aiv_worker_ids_[core] = input.clusters + core;
        if (!ctx.assign_cores_to_threads()) throw std::runtime_error("HBG native core assignment failed");
        auto &tasks = shared_.header->tasks;
        tasks.total_tasks = count;
        for (size_t i = 0; i < count; ++i) {
            auto &entry = tasks.storage_at(i);
            new (&entry) ChipTaskStorage{};
            const auto &task = input.tasks[i];
            entry.task.task_id = TaskId::make_global(i);
            for (uint32_t sub = 0; sub < 3; ++sub)
                entry.task.kernel_id[sub] = (task.mask & (1U << sub)) ? 0 : -1;
            entry.payload.bind_regions(nullptr, nullptr, fanin);
            entry.payload.fanin_count = task.predecessors.size();
            std::copy(task.predecessors.begin(), task.predecessors.end(), fanin);
            fanin += task.predecessors.size();
            entry.slot.active_mask = ActiveMask(task.mask);
            entry.slot.total_required_subtasks = __builtin_popcount(task.mask);
            entry.slot.logical_block_num = 1;
            tasks.reset_task_state(i);
        }
        for (size_t i = 0; i < count; ++i) {
            auto &slot = tasks.get_slot_state_by_task_id(i);
            auto unmet = scheduler_.classify_fanin_state(&slot);
            if (unmet < 0) scheduler_.push_ready_routed(&slot);
            else
                scheduler_.register_wake(
                    &tasks.get_slot_state_by_task_id(slot.to_payload().fanin_data()[unmet]), &slot
                );
        }
    }
    ~SchedulerReplayAdapter() {
        for (uint32_t thread = 0; thread < input_.threads; ++thread)
            context_->sp_queues_[thread].destroy();
    }
    void run() {
        while (completed_ < input_.tasks.size()) {
            bool progress;
            do {
                const auto before = completed_;
                progress = false;
                for (uint32_t thread = 0; thread < input_.threads; ++thread) {
                    int done = 0, current = 0;
                    bool made_progress = false;
                    context_->check_running_cores_for_completion(thread, nullptr, done, current, made_progress);
                }
                resolve();
                for (uint32_t thread = 0; thread < input_.threads; ++thread) {
                    bool made_progress = false, pushed = false;
                    auto index = input_.reverse_order ? input_.threads - 1 - thread : thread;
                    context_->dispatch_ready_tasks(
                        index, context_->core_trackers_[index], false, made_progress, pushed
                    );
                }
                for (uint32_t core = 0; core < input_.clusters * 3; ++core)
                    progress = launch(core) || progress;
                progress = progress || completed_ != before;
                if (shared_.header->sched_error_code.load()) throw std::runtime_error("native HBG scheduler error");
            } while (progress);
            if (completed_ == input_.tasks.size()) break;
            for (const auto &event : clock_.advance())
                *context_->core_exec_states_[event.core].cond_ptr = MAKE_FIN_VALUE(event.generation);
        }
        clock_.write();
    }

private:
    void resolve() {
        for (uint32_t thread = 0; thread < input_.threads; ++thread)
            while (auto *slot = context_->sp_queues_[thread].pop()) {
                scheduler_.on_task_complete(*slot);
                ++completed_;
            }
        ChipTaskSlotState *dummy[256];
        const auto count = scheduler_.dummy_ready_queue.pop_batch(dummy, 256);
        for (int i = 0; i < count; ++i) {
            auto task = static_cast<uint32_t>(TaskId::to_uint64(dummy[i]->to_descriptor().task_id));
            scheduler_.on_task_complete(*dummy[i]);
            clock_.rows.push_back({{clock_.now, input_.clusters, input_.threads, 0, 0, 0, task, 1}, clock_.now});
            ++completed_;
        }
    }
    bool launch(uint32_t core) {
        auto &state = context_->core_exec_states_[core];
        if (state.running_slot_state) clock_.dispatch(core, state.running_reg_task_id);
        if (state.pending_slot_state) clock_.dispatch(core, state.pending_reg_task_id);
        auto *slot = state.running_slot_state;
        auto token = state.running_reg_task_id;
        auto sub = state.running_subslot;
        if (launched_[core] == token && state.pending_slot_state && state.pending_reg_task_id != launched_[core]) {
            slot = state.pending_slot_state;
            token = state.pending_reg_task_id;
            sub = state.pending_subslot;
        }
        if (!slot || token == AICPU_TASK_INVALID || launched_[core] == token) return false;
        auto task = static_cast<uint32_t>(TaskId::to_uint64(slot->to_descriptor().task_id));
        auto owner = (core < input_.clusters ? core : (core - input_.clusters) / 2) % input_.threads;
        if (!clock_.start({0, core, owner, 0, 0, static_cast<uint32_t>(token), task, static_cast<uint32_t>(sub)}))
            return false;
        launched_[core] = token;
        *state.cond_ptr = MAKE_ACK_VALUE(token);
        return true;
    }
    const simpler::replay::Input &input_;
    simpler::replay::Clock clock_;
    std::unique_ptr<SchedulerContext> context_;
    std::vector<std::array<uint64_t, 32>> registers_;
    std::vector<int32_t> launched_;
    std::unique_ptr<void, decltype(&std::free)> image_{nullptr, &std::free};
    SharedMemoryHandle shared_{};
    DeviceArena arena_;
    SchedulerState scheduler_{};
    std::vector<uint8_t> callable_{make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0)};
    uint64_t callable_address_{};
    size_t completed_{};
};

int main() {
    try {
        const simpler::replay::Input input(std::cin);
        SchedulerReplayAdapter(input).run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
