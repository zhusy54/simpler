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
#include <unordered_map>

#include "callable.h"
#include "scheduler/scheduler_context.h"
#include "runtime_core.h"
#include "tmr_traits.h"
#include "submission_actor.h"

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
            throw std::invalid_argument("scheduler thread/cluster configuration exceeds native capacity");
        const size_t ring_size = SharedMemoryHandle::calculate_size(input.task_window);
        sm_.reset(
            std::aligned_alloc(CHIP_ALIGN_SIZE, (ring_size + CHIP_ALIGN_SIZE - 1) & ~size_t(CHIP_ALIGN_SIZE - 1))
        );
        if (!sm_ || !shared_.init(sm_.get(), ring_size, input.task_window, 0))
            throw std::runtime_error("TMR shared memory initialization failed");
        auto layout = runtime_reserve_layout(arena_, input.task_window, input.dependency_capacity);
        if (!arena_.commit()) throw std::runtime_error("TMR scheduler initialization failed");
        runtime_ = runtime_init_data_from_layout(
            arena_, layout, MODE_EXECUTE, sm_.get(), ring_size, empty_heap_.data(), uint64_t(0)
        );
        if (!runtime_) throw std::runtime_error("TMR arena initialization failed");
        runtime_wire_arena_pointers(arena_, layout, runtime_);
        *runtime_->sm_handle = shared_;
        runtime_->orchestrator.total_cluster_count = input.clusters;
        runtime_->orchestrator.total_aiv_count = input.clusters * 2;
        if (runtime_->orchestrator.scope_stack_top < 0) runtime_->orchestrator.begin_scope(ScopeMode::AUTO);
        scheduler_ = &runtime_->scheduler;
        auto &ctx = *context_;
        ctx.sched_ = scheduler_;
        ctx.rt_ = runtime_;
        ctx.active_sched_threads_ = input.threads;
        ctx.sched_thread_num_ = input.threads;
        ctx.aicpu_thread_num_ = input.threads + 1;
        ctx.cores_total_num_ = input.clusters * 3;
        ctx.aic_count_ = input.clusters;
        ctx.aiv_count_ = input.clusters * 2;
        auto *callable = reinterpret_cast<CoreCallable *>(callable_.data());
        callable->set_resolved_addr(0x1000);
        callable_address_ = reinterpret_cast<uint64_t>(callable);
        ctx.func_id_to_addr_ = &callable_address_;
        ctx.func_id_to_addr_count_ = 1;
        for (uint32_t core = 0; core < input.clusters * 3; ++core) {
            auto &state = ctx.core_exec_states_[core];
            state.reg_addr = reinterpret_cast<uint64_t>(registers_[core].data());
            state.cond_ptr = get_reg_ptr(state.reg_addr, RegId::COND);
            state.running_reg_task_id = AICPU_TASK_INVALID;
            state.pending_reg_task_id = AICPU_TASK_INVALID;
            ctx.core_type_compact_[core] = static_cast<uint8_t>(core < input.clusters ? CoreType::AIC : CoreType::AIV);
        }
        for (uint32_t thread = 0; thread < input.threads; ++thread)
            replay_assign_clusters(ctx, thread, 0);
        producer_ = std::make_unique<simpler::replay::SubmissionActor>([this] {
            supply();
        });
    }

    void run() {
        while (completed_ < input_.tasks.size()) {
            bool progress;
            do {
                const size_t before = clock_.rows.size();
                const size_t completed_before = completed_;
                const auto submitted_before = submitted_;
                producer_->resume();
                progress = submitted_ != submitted_before;
                for (uint32_t pass = 0; pass < input_.threads; ++pass)
                    step(input_.reverse_order ? input_.threads - 1 - pass : pass);
                for (uint32_t core = 0; core < input_.clusters * 3; ++core)
                    progress = launch(core) || progress;
                producer_->resume();
                progress = progress || submitted_ != submitted_before;
                progress = progress || before != clock_.rows.size() || completed_before != completed_;
            } while (progress);
            if (completed_ == input_.tasks.size()) break;
            std::vector<simpler::replay::Event> events;
            try {
                events = clock_.advance();
            } catch (const std::runtime_error &) {
                auto &head = shared_.header->rings[1].get_slot_state_by_task_id(
                    scheduler_->ring_sched_states[1].last_task_alive
                );
                throw std::runtime_error(
                    "TMR stalled: submitted=" + std::to_string(submitted_) +
                    " completed=" + std::to_string(completed_) +
                    " dependency slots=" + std::to_string(scheduler_->ring_sched_states[1].dep_pool.available()) +
                    " ring1 head=" + std::to_string(scheduler_->ring_sched_states[1].last_task_alive) +
                    " state=" + std::to_string(head.task_state.load()) +
                    " fanout=" + std::to_string(head.fanout_refcount.load()) + "/" + std::to_string(head.fanout_count)
                );
            }
            for (const auto &event : events) {
                auto &state = context_->core_exec_states_[event.core];
                *state.cond_ptr = MAKE_FIN_VALUE(event.generation);
            }
        }
        clock_.write();
    }

private:
    void step(uint32_t thread) {
        auto &ctx = *context_;
        int32_t completed = 0, current = 0, deferred_count = 0;
        std::array<ChipTaskSlotState *, 256> deferred{};
        bool progress = false, pushed = false;
        ctx.check_running_cores_for_completion(
            thread, nullptr, completed, current, progress, deferred.data(), deferred_count
        );
        completed_ += completed;
        ChipTaskSlotState *dummy[256];
        int got = scheduler_->dummy_ready_queue.pop_batch(dummy, 256);
        for (int i = 0; i < got; ++i) {
            scheduler_->on_task_complete(*dummy[i]);
            const auto index = identity_.at(TaskId::to_uint64(dummy[i]->task->task_id));
            clock_.rows.push_back(
                {{clock_.now, input_.clusters, thread, 0, 0, 0, static_cast<uint32_t>(index), 1}, clock_.now}
            );
            ++completed_;
            deferred[deferred_count++] = dummy[i];
            if (deferred_count == 256)
                replay_release(scheduler_, deferred.data(), deferred_count, ctx.orchestrator_done_.load(), 0);
        }
        replay_release(scheduler_, deferred.data(), deferred_count, ctx.orchestrator_done_.load(), 0);
        for (auto &ring : scheduler_->ring_sched_states)
            replay_advance_ring(ring, 0);
        ctx.dispatch_ready_tasks(thread, ctx.core_trackers_[thread], false, progress, pushed);
    }
    bool launch(uint32_t core) {
        auto &ctx = *context_;
        auto &state = ctx.core_exec_states_[core];
        if (state.running_slot_state) clock_.dispatch(core, state.running_reg_task_id);
        if (state.pending_slot_state) clock_.dispatch(core, state.pending_reg_task_id);
        auto *slot = state.running_slot_state;
        auto token = state.running_reg_task_id;
        auto sub = state.running_subslot;
        if (launched_[core] == token && state.pending_slot_state != nullptr &&
            state.pending_reg_task_id != launched_[core]) {
            slot = state.pending_slot_state;
            token = state.pending_reg_task_id;
            sub = state.pending_subslot;
        }
        if (slot == nullptr || token == AICPU_TASK_INVALID || launched_[core] == token) return false;
        auto index = identity_.at(TaskId::to_uint64(slot->task->task_id));
        if (!clock_.start(
                {0, core,
                 core < input_.clusters ? core % input_.threads : ((core - input_.clusters) / 2) % input_.threads, 0, 0,
                 static_cast<uint32_t>(token), static_cast<uint32_t>(index), static_cast<uint32_t>(sub)}
            ))
            return false;
        launched_[core] = token;
        *state.cond_ptr = MAKE_ACK_VALUE(token);
        return true;
    }
    void supply() {
        auto &orch = runtime_->orchestrator;
        while (submitted_ < input_.tasks.size()) {
            const auto &task = input_.tasks[submitted_];
            const auto ring = static_cast<uint32_t>(task.token >> 32);
            if (ring > 1) throw std::invalid_argument("nested scopes require an explicit scope event stream");
            while (orch.current_ring_id() > ring)
                orch.end_scope();
            while (orch.current_ring_id() < ring)
                orch.begin_scope(ScopeMode::MANUAL);
            std::vector<TaskId> deps;
            std::vector<DepFlags> kinds;
            for (auto pred : task.predecessors)
                deps.push_back(tokens_.at(pred));
            for (auto flags : task.dependency_flags)
                kinds.push_back(static_cast<DepFlags>(flags));
            CoreTaskArgs args;
            args.set_dependencies_with_kinds(deps.data(), kinds.data(), deps.size());
            MixedKernels kernels;
            kernels.aic_kernel_id = (task.mask & 1) ? 0 : -1;
            kernels.aiv0_kernel_id = (task.mask & 2) ? 0 : -1;
            kernels.aiv1_kernel_id = (task.mask & 4) ? 0 : -1;
            auto output = task.mask ? orch.submit_task(kernels, args) : orch.submit_dummy_task(args);
            if (orch.fatal || output.task_id() == TaskId::invalid())
                throw std::runtime_error("native TMR submission failed");
            clock_.admit(submitted_);
            tokens_.push_back(output.task_id());
            identity_.emplace(TaskId::to_uint64(output.task_id()), submitted_);
            auto &submitted_slot =
                shared_.header->rings[output.task_id().ring()].get_slot_state_by_task_id(output.task_id().local_id());
            if (!task.mask && submitted_slot.task_state.load() >= CHIP_TASK_COMPLETED) {
                clock_.rows.push_back(
                    {{clock_.now, input_.clusters, 0, 0, 0, 0, static_cast<uint32_t>(submitted_), 1}, clock_.now}
                );
                ++completed_;
            }
            ++submitted_;
        }
        if (submitted_ == input_.tasks.size() && !context_->orchestrator_done_.load()) {
            orch.mark_done();
            context_->orchestrator_done_.store(true);
        }
    }
    const simpler::replay::Input &input_;
    simpler::replay::Clock clock_;
    std::unique_ptr<SchedulerContext> context_;
    std::vector<std::array<uint64_t, 32>> registers_;
    std::vector<int32_t> launched_;
    std::unique_ptr<void, decltype(&std::free)> sm_{nullptr, &std::free};
    SharedMemoryHandle shared_{};
    DeviceArena arena_;
    std::array<uint8_t, 64> empty_heap_{};
    RuntimeContext *runtime_{};
    SchedulerState *scheduler_{};
    std::vector<TaskId> tokens_;
    std::unordered_map<uint64_t, uint32_t> identity_;
    std::vector<uint8_t> callable_{make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0)};
    uint64_t callable_address_{};
    size_t completed_{};
    size_t submitted_{};
    std::unique_ptr<simpler::replay::SubmissionActor> producer_;
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
