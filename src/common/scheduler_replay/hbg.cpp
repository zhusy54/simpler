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
#include <cstdlib>
#include <memory>

#include "runtime_types.h"
#include "scheduler/scheduler_mix.h"

namespace simpler::replay {

struct Cluster {
    SchedulerLocalState local;
    SchedulerDeferredAivQueue deferred;
    std::array<uint64_t, SCHEDULER_READY_QUEUE_COUNT> cursors{};
    std::vector<uint8_t> ssbuf;
    std::array<std::array<uint32_t, 2>, 3> launched{};
    Cluster() :
        ssbuf(SCHEDULER_SSBUF_HARDWARE_SIZE + 64) {}
    SchedulerSsbufRegion *region() {
        return scheduler_ssbuf_region((reinterpret_cast<uintptr_t>(ssbuf.data()) + 63) & ~uintptr_t(63));
    }
};

class Hbg {
public:
    explicit Hbg(const Input &input) :
        input_(input),
        clock_(input),
        clusters_(input.clusters) {
        const auto count = input.tasks.size();
        if (input.clusters * 3 > SCHEDULER_WORKER_CAPACITY || input.clusters > SCHEDULER_CAPACITY)
            throw std::invalid_argument("HBG topology exceeds native capacity");
        if (!scheduler_plan_layout(count, count, count, &layout_, false, count * 3))
            throw std::runtime_error("scheduler layout failed");
        state_.reset(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout_.total_size));
        if (!state_ || !scheduler_init_data_from_layout(base(), layout_)) throw std::bad_alloc();
        size_t edge_count = 0;
        for (const auto &task : input.tasks)
            edge_count += task.predecessors.size();
        const size_t bytes = (count * sizeof(ChipTaskStorage) + edge_count * sizeof(int32_t) + 63) & ~size_t(63);
        graph_image_.reset(std::aligned_alloc(64, bytes));
        if (!graph_image_) throw std::bad_alloc();
        storage_ = static_cast<ChipTaskStorage *>(graph_image_.get());
        for (size_t i = 0; i < count; ++i)
            new (&storage_[i]) ChipTaskStorage{};
        auto *fanin_region = reinterpret_cast<int32_t *>(storage_ + count);
        graph_ = {reinterpret_cast<uint64_t>(storage_), 0, count, count - 1};
        auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base(), layout_.task_metadata_offset);
        auto *contexts = scheduler_state_at<SchedulerWorkerContext>(base(), layout_.worker_contexts_offset);
        control_ = scheduler_state_at<SchedulerRunControl>(base(), layout_.run_control_offset);
        control_->expected_task_count = count;
        control_->scheduler_count = input.clusters;
        control_->active_worker_count = input.clusters * 3;
        control_->aic_active_worker_count = input.clusters;
        control_->aiv_active_worker_count = input.clusters * 2;
        bool mix = false;
        for (size_t i = 0; i < count; ++i) {
            const auto &task = input.tasks[i];
            if (task.predecessors.size() > SCHEDULER_GRAPH_MAX_FANIN)
                throw std::invalid_argument("HBG fanin exceeds native capacity");
            auto &entry = storage_[i];
            entry.task.task_id = TaskId::make_global(static_cast<int32_t>(i));
            std::copy(task.predecessors.begin(), task.predecessors.end(), fanin_region);
            entry.payload.bind_regions(nullptr, nullptr, fanin_region);
            entry.payload.fanin_count = task.predecessors.size();
            fanin_region += task.predecessors.size();
            auto &m = metadata[i];
            m.active_mask = task.mask ? task.mask : 2;
            m.flags = SCHEDULER_TASK_EXECUTABLE;
            if (!task.mask) m.flags |= SCHEDULER_TASK_INLINE;
            if (!task.predecessors.empty()) m.flags |= SCHEDULER_TASK_HAS_FANIN;
            if (__builtin_popcount(task.mask) > 1) {
                m.flags |= SCHEDULER_TASK_MIX;
                mix = true;
            }
            m.trace_index_base = i * 3;
            m.timing_slot = -1;
            for (uint32_t sub = 0; sub < 3; ++sub) {
                entry.task.kernel_id[sub] = (task.mask & (1U << sub)) ? 1 : INVALID_KERNEL_ID;
                m.kernel_ids[sub] = (m.active_mask & (1U << sub)) ? 1 : UINT16_MAX;
            }
            if (!task.mask && task.predecessors.empty()) {
                m.flags = 0;
                auto *task_control = scheduler_state_at<SchedulerTaskControl>(base(), layout_.task_controls_offset) + i;
                task_control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
                task_control->wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;
                ++control_->resolved_task_count;
                clock_.rows.push_back({{0, input.clusters, 0, 0, 0, 0, static_cast<uint32_t>(i), 1}, 0});
            }
        }
        auto *coordinator = scheduler_state_at<SchedulerGangCoordinator>(base(), layout_.gang_coordinator_offset);
        coordinator->mix_task_count = mix;
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < input.tasks[i].predecessors.size(); ++j)
                if (scheduler_graph_fanin_id(graph_, i, j) != input.tasks[i].predecessors[j])
                    throw std::runtime_error("HBG graph image fanin mismatch");
        for (uint32_t cluster = 0; cluster < input.clusters; ++cluster) {
            const uint32_t workers[3] = {cluster, input.clusters + cluster * 2, input.clusters + cluster * 2 + 1};
            for (uint32_t lane = 0; lane < 3; ++lane) {
                auto &c = contexts[workers[lane]];
                c.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
                c.worker_index = workers[lane];
                c.runtime_worker_count = input.clusters * 3;
                c.scheduler_index = cluster;
                c.scheduler_count = input.clusters;
                c.scheduler_worker_id = workers[1];
                c.is_scheduler = lane == 1;
                c.task_controls_offset = layout_.task_controls_offset;
                c.task_metadata_offset = layout_.task_metadata_offset;
                c.ready_inboxes_offset = layout_.ready_inboxes_offset;
                c.ready_directory_offset = layout_.ready_directory_offset;
                c.worker_contexts_offset = layout_.worker_contexts_offset;
                c.gang_coordinator_offset = layout_.gang_coordinator_offset;
                c.callable_addresses_address = reinterpret_cast<uint64_t>(callables_.data());
                c.callable_addresses_count = callables_.size();
                c.dispatch_payload_offset =
                    layout_.dispatch_payloads_offset + workers[lane] * 2 * sizeof(DispatchPayload);
                std::copy(std::begin(workers), std::end(workers), c.cluster_worker_ids);
            }
            auto &cluster_state = clusters_[cluster];
            if (!scheduler_initialize_local_config(base(), &contexts[workers[1]], &graph_, &cluster_state.local))
                throw std::runtime_error("invalid HBG topology");
            for (auto &lane : cluster_state.local.slots)
                for (auto &slot : lane)
                    scheduler_initialize_free_slot(&slot);
            cluster_state.cursors.fill((cluster + 1) % input.clusters);
            scheduler_ssbuf_initialize(cluster_state.region(), 1);
        }
        bootstrap();
    }

    void run() {
        size_t finished = control_->resolved_task_count;
        while (finished < input_.tasks.size()) {
            bool progress;
            do {
                progress = false;
                for (size_t pass = 0; pass < clusters_.size(); ++pass)
                    progress = step(clusters_[input_.reverse_order ? clusters_.size() - 1 - pass : pass]) || progress;
                if (control_->scheduler_error) throw std::runtime_error("native HBG scheduler rejected replay state");
            } while (progress);
            auto events = clock_.advance();
            for (const auto &event : events) {
                auto &cluster = clusters_[event.owner];
                if (event.lane == cluster.local.config.self_lane)
                    cluster.local.local_completed_generations[event.slot] = event.generation;
                else {
                    auto &pub = cluster.region()->lanes[event.lane].completion.publication;
                    const uint64_t mask = UINT64_C(0xffffffff) << (event.slot * 32);
                    pub = (pub & ~mask) | (static_cast<uint64_t>(event.generation) << (event.slot * 32));
                }
            }
            for (size_t pass = 0; pass < clusters_.size(); ++pass)
                step(clusters_[input_.reverse_order ? clusters_.size() - 1 - pass : pass]);
            finished = control_->resolved_task_count;
            if (finished == input_.tasks.size()) break;
        }
        clock_.write();
    }

private:
    void bootstrap() {
        const auto count = input_.tasks.size();
        for (uint32_t index = 0; index < clusters_.size(); ++index) {
            auto &local = clusters_[index].local;
            SchedulerReadyBatch batches[SCHEDULER_READY_QUEUE_COUNT]{};
            const auto first = index * (count / clusters_.size()) + std::min<size_t>(index, count % clusters_.size());
            const auto last = first + count / clusters_.size() + (index < count % clusters_.size());
            for (size_t task = first; task < last; ++task) {
                if (!input_.tasks[task].mask && input_.tasks[task].predecessors.empty()) continue;
                auto route = input_.tasks[task].predecessors.empty() ?
                                 SchedulerRouteResult::READY_TO_ENQUEUE :
                                 scheduler_bootstrap_route_task(graph_, base(), &local, control_, task, nullptr);
                if (route == SchedulerRouteResult::ERROR) throw std::runtime_error("HBG dependency routing failed");
                if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                    auto *m = scheduler_task_metadata_at(base(), &local, task);
                    auto type = scheduler_task_ready_queue(m->flags, m->active_mask);
                    if (!scheduler_bootstrap_ready_batch_append(base(), &local, task, &batches[type], nullptr, 0))
                        throw std::runtime_error("HBG bootstrap enqueue failed");
                }
            }
            uint64_t types = 0;
            for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type)
                if (!scheduler_bootstrap_ready_batch_publish(
                        base(), &local, type, index, &batches[type], nullptr, &types
                    ))
                    throw std::runtime_error("HBG bootstrap publication failed");
            scheduler_ready_directory_at(base(), &local)->bootstrap_ready_types[index] = types;
            local.owner_ready_queue_mask = types;
        }
        scheduler_bootstrap_ready_directory_publish(base(), &clusters_[0].local, clusters_.size());
    }

    bool step(Cluster &cluster) {
        auto *local = &cluster.local;
        bool progress = false;
        uint64_t skip{};
        SchedulerRefillCandidates candidates(local->has_mix);
        progress = scheduler_service_cluster_completions(
            graph_, base(), local, control_, nullptr, nullptr, nullptr, cluster.cursors.data(), 0, &skip,
            cluster.region(), local->has_mix ? &candidates : nullptr
        );
        if (!scheduler_fill_mix_after_completions(
                graph_, base(), local, control_, &candidates.mix_ready, &cluster.cursors[SCHEDULER_MIX_QUEUE], nullptr,
                0, cluster.deferred.count == 0, cluster.region(), &progress
            ))
            throw std::runtime_error("HBG MIX dispatch failed");
        if (local->has_mix && !scheduler_publish_refill_candidates(
                                  graph_, base(), local, control_, &candidates, nullptr, 0, cluster.region()
                              ))
            throw std::runtime_error("HBG refill failed");
        bool failed = false;
        progress = scheduler_fill_cluster_normal_slots(
                       graph_, base(), local, control_, cluster.cursors.data(), nullptr, 0, skip, &cluster.deferred,
                       &failed, cluster.region()
                   ) ||
                   progress;
        if (failed) throw std::runtime_error("HBG dispatch failed");
        if (cluster.deferred.count) {
            auto before = cluster.deferred.count;
            if (!scheduler_drain_deferred_aiv_to_peer(
                    graph_, base(), local, control_, &cluster.deferred, nullptr, nullptr, nullptr, 0, cluster.region()
                ))
                throw std::runtime_error("HBG late binding failed");
            progress = progress || before != cluster.deferred.count;
            if (cluster.deferred.count) {
                uint32_t published = UINT32_MAX;
                if (!scheduler_publish_deferred_aiv_local(
                        graph_, base(), local, control_, &cluster.deferred, 0, &published, cluster.region()
                    ))
                    throw std::runtime_error("HBG local publication failed");
                progress = progress || published != UINT32_MAX;
            }
        }
        scheduler_flush_completions(control_, local);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            for (const auto &slot : local->slots[lane])
                if (slot.state == SchedulerDispatchSlotState::READY)
                    clock_.dispatch(local->config.worker_ids[lane], slot.generation);
            uint32_t selected = 2;
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const auto &s = local->slots[lane][slot];
                if (s.state == SchedulerDispatchSlotState::READY && s.generation != cluster.launched[lane][slot] &&
                    (selected == 2 || s.generation < local->slots[lane][selected].generation))
                    selected = slot;
            }
            if (selected == 2) continue;
            const auto &s = local->slots[lane][selected];
            if (clock_.start(
                    {0, static_cast<uint32_t>(local->config.worker_ids[lane]), local->config.scheduler_index, lane,
                     selected, s.generation, static_cast<uint32_t>(s.task_id), s.subtask_slot}
                )) {
                cluster.launched[lane][selected] = s.generation;
                local->local_ready_mask &= lane == local->config.self_lane ? ~(1U << selected) : UINT8_MAX;
                progress = true;
            }
        }
        return progress;
    }
    void *base() { return state_.get(); }
    const Input &input_;
    Clock clock_;
    AicoreSchedulerLayout layout_{};
    std::unique_ptr<void, decltype(&std::free)> state_{nullptr, &std::free};
    std::unique_ptr<void, decltype(&std::free)> graph_image_{nullptr, &std::free};
    ChipTaskStorage *storage_{};
    std::vector<Cluster> clusters_;
    std::array<uint64_t, 2> callables_{0, 0x1000};
    SchedulerGraphView graph_{};
    SchedulerRunControl *control_{};
};
}  // namespace simpler::replay

int main() {
    try {
        const simpler::replay::Input input(std::cin);
        simpler::replay::Hbg(input).run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
