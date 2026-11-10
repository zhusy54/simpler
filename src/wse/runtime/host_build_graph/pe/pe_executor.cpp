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
#include "wse_launch.h"
#include "cpu_sim_context.h"
#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler_graph_binding.h"

#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
struct Work {
    uint64_t task;
    uint32_t block;
};
struct Node {
    uint32_t remaining_dependencies{0};
    uint32_t remaining_blocks{0};
    std::vector<uint64_t> successors;
};
struct Core {
    bool initialized{false};
    uint32_t ready{0};
    uint32_t quiet{0};
    bool retired{false};
    uint64_t expected_tasks{0};
};
}  // namespace

struct WseRun {
    const WseGraphDescriptor *descriptor;
    const uint32_t core_count;
    const uint64_t epoch;
    std::mutex mutex;
    std::condition_variable changed;
    bool configured{false};
    bool go{false};
    bool stop{false};
    int init_error{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> sequence{0};
    std::vector<Core> cores;
    std::vector<Node> nodes;
    std::vector<std::deque<Work>> queues;
    std::vector<std::vector<WseEvent>> events;

    WseRun(const WseGraphDescriptor *d, uint32_t c, uint64_t e) :
        descriptor(d),
        core_count(c),
        epoch(e),
        events(c * WSE_PE_PER_CORE) {}

    void record(
        uint32_t worker, WseEventKind kind, uint64_t task = UINT64_MAX, uint32_t block = 0, uint64_t value = 0,
        uint32_t slot = 0, uint64_t generation = 0
    ) {
        events[worker].push_back(
            {sequence.fetch_add(1), task, value, worker / WSE_PE_PER_CORE, worker % WSE_PE_PER_CORE, block, kind, slot,
             generation}
        );
    }
    const WseNodePlan *plans() const { return reinterpret_cast<const WseNodePlan *>(descriptor->nodes_address); }
    void enqueue(uint64_t ref) {
        for (uint32_t b = 0; b < plans()[ref].blocks; ++b)
            queues[(b % core_count) * WSE_PE_PER_CORE + (b / core_count) % WSE_PE_PER_CORE].push_back({ref, b});
    }
    void bootstrap() {
        clear_cpu_sim_shared_storage();
        const SchedulerGraphAccess graph(descriptor->graph);
        if (!scheduler_graph_view_valid(graph)) throw std::invalid_argument("invalid WSE graph");
        cores.resize(core_count);
        nodes.resize(graph.task_count);
        queues.resize(core_count * WSE_PE_PER_CORE);
        uint64_t expected = 0;
        for (uint64_t ref = 0; ref < graph.task_count; ++ref) {
            if (plans()[ref].host_done) continue;
            ++expected;
            if (!plans()[ref].blocks) throw std::invalid_argument("empty WSE task block set");
            nodes[ref].remaining_blocks = plans()[ref].blocks;
            if (plans()[ref].entry && scheduler_bind_graph_arguments(graph, ref) != SchedulerGraphBindingError::NONE)
                throw std::invalid_argument("invalid WSE graph argument binding");
            if (plans()[ref].entry) {
                SchedulerTaskPayloadArguments arguments{};
                if (scheduler_parse_task_payload_arguments(graph, ref, &arguments) != SchedulerGraphResult::OK)
                    throw std::invalid_argument("invalid WSE task arguments");
                if (plans()[ref].has_predicate) {
                    const auto *predicate = scheduler_node_predicate(graph, ref);
                    if (!predicate || !scheduler_dispatch_predicate_metadata_valid(
                                          predicate->addr, predicate->elem_size, predicate->op
                                      ))
                        throw std::invalid_argument("invalid WSE task predicate");
                }
            }
            uint32_t fanins = 0;
            if (scheduler_node_fanin_count(graph, ref, &fanins) != SchedulerGraphResult::OK)
                throw std::invalid_argument("invalid WSE dependencies");
            for (uint32_t i = 0; i < fanins; ++i) {
                int64_t producer = -1;
                const auto status = scheduler_node_fanin_ref(graph, ref, i, &producer);
                if (status != SchedulerGraphResult::OK || producer < 0 || static_cast<uint64_t>(producer) >= ref) {
                    std::fprintf(
                        stderr, "WSE dependency error: ref=%llu index=%u count=%u producer=%lld status=%llu\n",
                        (unsigned long long)ref, i, fanins, (long long)producer, (unsigned long long)status
                    );
                    throw std::invalid_argument("invalid WSE dependency order");
                }
                if (!plans()[producer].host_done) {
                    ++nodes[ref].remaining_dependencies;
                    nodes[producer].successors.push_back(ref);
                }
            }
        }
        if (expected != descriptor->expected_tasks) throw std::invalid_argument("WSE task count mismatch");
        for (uint64_t ref = 0; ref < graph.task_count; ++ref)
            if (!plans()[ref].host_done && nodes[ref].remaining_dependencies == 0) enqueue(ref);
        completed.store(0, std::memory_order_relaxed);
        record(0, WseEventKind::CONFIG_READY, UINT64_MAX, 0, descriptor->expected_tasks);
        configured = true;
    }
};

namespace {
struct OperationBinding {
    WseRun *run;
    uint32_t worker;
    uint64_t task;
    uint32_t block;
};
void operation_trace(void *opaque, wse::OperationEvent kind, uint32_t slot, uint64_t generation) {
    auto &binding = *static_cast<OperationBinding *>(opaque);
    binding.run->record(
        binding.worker, static_cast<WseEventKind>(kind), binding.task, binding.block, 0, slot, generation
    );
}
static_assert(static_cast<uint32_t>(wse::OperationEvent::PUBLISH) == static_cast<uint32_t>(WseEventKind::OP_PUBLISH));
static_assert(static_cast<uint32_t>(wse::OperationEvent::CONSUMED) == static_cast<uint32_t>(WseEventKind::OP_CONSUMED));
}  // namespace

extern "C" WseRun *wse_create_run(const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch) {
    if (!d || !cores || cores > WSE_DEFAULT_CORE_COUNT || !epoch) return nullptr;
    try {
        return new WseRun(d, cores, epoch);
    } catch (...) {
        return nullptr;
    }
}
extern "C" void wse_destroy_run(WseRun *run) { delete run; }
extern "C" uint64_t wse_completed_tasks(const WseRun *run) { return run->completed.load(std::memory_order_acquire); }
extern "C" const WseEvent *wse_run_events(const WseRun *run, uint32_t worker, uint64_t *count) {
    if (!run || !count || worker >= run->events.size()) return nullptr;
    *count = run->events[worker].size();
    return run->events[worker].data();
}

extern "C" int wse_execute_pe(WseRun *run, uint32_t worker) {
    const uint32_t core = worker / WSE_PE_PER_CORE;
    const uint32_t pe = worker % WSE_PE_PER_CORE;
    std::unique_lock lock(run->mutex);
    if (worker == 0) {
        try {
            run->bootstrap();
        } catch (const std::exception &error) {
            std::fprintf(stderr, "WSE bootstrap rejected: %s\n", error.what());
            run->init_error = -1;
        }
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->configured || run->init_error;
    });
    if (run->init_error) return run->init_error;
    auto &local = run->cores[core];
    if (pe == 0) {
        local.ready = 0;
        local.quiet = 0;
        local.retired = false;
        local.expected_tasks = run->descriptor->expected_tasks;
        local.initialized = true;
        run->record(worker, WseEventKind::CORE_INIT, UINT64_MAX, 0, local.expected_tasks);
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return local.initialized;
    });
    sim_context_set_cluster_id(core);
    sim_context_set_subblock_id(0);
    run->record(worker, WseEventKind::PE_READY);
    ++local.ready;
    run->changed.notify_all();
    if (worker == 0) {
        run->changed.wait(lock, [&] {
            for (const auto &c : run->cores)
                if (c.ready != WSE_PE_PER_CORE) return false;
            return true;
        });
        run->record(worker, WseEventKind::GO);
        run->go = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->go;
    });
    const SchedulerGraphAccess graph(run->descriptor->graph);
    while (true) {
        run->changed.wait(lock, [&] {
            return !run->queues[worker].empty() ||
                   run->completed.load(std::memory_order_acquire) == local.expected_tasks;
        });
        if (run->completed.load(std::memory_order_acquire) == local.expected_tasks) break;
        const Work work = run->queues[worker].front();
        run->queues[worker].pop_front();
        lock.unlock();
        run->record(worker, WseEventKind::BLOCK_BEGIN, work.task, work.block);
        const auto &plan = run->plans()[work.task];
        bool execute = true;
        if (plan.has_predicate) {
            const auto &predicate = *scheduler_node_predicate(graph, work.task);
            execute =
                DispatchPredicate{
                    predicate.addr, predicate.target, predicate.elem_size, static_cast<PredicateOp>(predicate.op)
                }
                    .pass();
        }
        if (!execute) run->record(worker, WseEventKind::BLOCK_SKIPPED, work.task, work.block);
        if (plan.entry && execute) {
            SchedulerTaskPayloadArguments arguments{};
            if (scheduler_parse_task_payload_arguments(graph, work.task, &arguments) != SchedulerGraphResult::OK)
                std::terminate();  // Admission and bootstrap validate immutable arguments before GO.
            int64_t args[SPMD_GLOBAL_CONTEXT_INDEX + 1]{};
            int index = 0;
            for (int i = 0; i < arguments.tensor_count; ++i)
                args[index++] = reinterpret_cast<int64_t>(arguments.tensors + i * TASKPAYLOAD_TENSOR_STRIDE);
            for (int i = 0; i < arguments.scalar_count; ++i)
                args[index++] = arguments.scalars[i];
            LocalContext context{};
            context.block_idx = work.block;
            context.block_num = plan.blocks;
            GlobalContext global{};
            global.core_id = core;
            global.pe_id = pe;
            OperationBinding binding{run, worker, work.task, work.block};
            global.operation_trace = {&binding, operation_trace};
            args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
            args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
            reinterpret_cast<void (*)(int64_t *)>(plan.entry)(args);
        }
        run->record(worker, WseEventKind::BLOCK_END, work.task, work.block);
        lock.lock();
        auto &node = run->nodes[work.task];
        if (--node.remaining_blocks == 0) {
            // All block writes acquire through this lock before successors become runnable.
            for (const auto next : node.successors)
                if (--run->nodes[next].remaining_dependencies == 0) run->enqueue(next);
            const uint64_t count = run->completed.fetch_add(1, std::memory_order_acq_rel) + 1;
            run->record(worker, WseEventKind::TASK_DONE, work.task, 0, count);
            if (count == local.expected_tasks) {
                run->changed.notify_all();
                break;
            }
        }
        run->changed.notify_all();
    }
    if (worker == 0) {
        run->record(worker, WseEventKind::STOP);
        run->stop = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->stop;
    });
    run->record(worker, WseEventKind::QUIESCENT);
    ++local.quiet;
    run->changed.notify_all();
    if (pe == 0) {
        run->changed.wait(lock, [&] {
            return local.quiet == WSE_PE_PER_CORE;
        });
        local.retired = true;
        run->record(worker, WseEventKind::CORE_RETIRED);
        run->changed.notify_all();
    }
    if (worker == 0) {
        run->changed.wait(lock, [&] {
            for (const auto &c : run->cores)
                if (!c.retired) return false;
            return true;
        });
        run->record(worker, WseEventKind::RUN_DONE, UINT64_MAX, 0, run->completed.load());
    }
    return 0;
}
