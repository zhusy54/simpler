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
#include "sram_layout.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <iostream>
#include <string>
#include "cpu_sim_context.h"
#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler_graph_binding.h"

#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

static_assert(alignof(wse::RunControl) >= std::atomic_ref<uint64_t>::required_alignment);

namespace {
class ImageBuilder {
public:
    explicit ImageBuilder(const wse::SramRegion *regions, uint32_t count) :
        regions_(regions, regions + count) {
        for (const auto &r : regions_)
            cursor_.push_back(r.capacity);
    }
    template <class T>
    uint64_t allocate(uint32_t core, uint64_t count = 1, uint32_t kind = 0) {
        const auto &r = regions_.at(core);
        if (count > SIZE_MAX / sizeof(T)) throw std::invalid_argument("WSE SRAM layout multiplication overflow");
        const uint64_t bytes = count * sizeof(T);
        if (bytes > cursor_[core]) exhausted(core, bytes);
        const auto offset = (cursor_[core] - bytes) & ~(uint64_t{alignof(T)} - 1);
        if (offset < r.capacity - r.reserve) exhausted(core, r.capacity - offset);
        cursor_[core] = offset;
        if (bytes) sections.push_back({r.base + offset, bytes, core, kind});
        auto *objects = reinterpret_cast<T *>(r.base + offset);
        for (uint64_t i = 0; i < count; ++i)
            new (objects + i) T{};
        return offset;
    }
    template <class T>
    T *at(uint32_t core, uint64_t offset) {
        return reinterpret_cast<T *>(regions_[core].base + offset);
    }
    std::vector<WseMemorySection> sections;
    std::vector<uint64_t> usage() const {
        std::vector<uint64_t> result;
        result.reserve(regions_.size());
        for (size_t i = 0; i < regions_.size(); ++i)
            result.push_back(regions_[i].capacity - cursor_[i]);
        return result;
    }

private:
    [[noreturn]] void exhausted(uint32_t core, uint64_t required) const {
        throw std::runtime_error(
            "WSE Runtime SRAM capacity exhausted: core=" + std::to_string(core) +
            " required=" + std::to_string(required) + " available=" + std::to_string(regions_[core].reserve)
        );
    }
    std::vector<wse::SramRegion> regions_;
    std::vector<uint64_t> cursor_;
};
}  // namespace

struct WseRun {
    uint64_t base;
    wse::RunControl *control;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<uint64_t> sequence{0};
    std::vector<std::vector<WseEvent>> events;
    std::vector<uint64_t> used;
    std::vector<WseMemorySection> sections;

    template <class T>
    T *at(uint64_t offset) const {
        return reinterpret_cast<T *>(base + offset);
    }
    const wse::SramRegion *regions() const { return at<wse::SramRegion>(control->access.regions_offset); }
    wse::CoreControl &local(uint32_t core) const {
        return *reinterpret_cast<wse::CoreControl *>(
            regions()[core].base + at<uint64_t>(control->core_offsets_offset)[core]
        );
    }
    wse::TaskRecord *plans() const { return at<wse::TaskRecord>(control->tasks_offset); }
    auto completion() const { return std::atomic_ref<uint64_t>(control->completed); }
    wse::Queue &queue(uint32_t worker) const {
        return local(worker / WSE_PE_PER_CORE).queues[worker % WSE_PE_PER_CORE];
    }
    bool has_work(uint32_t worker) const {
        const auto &q = queue(worker);
        return q.head != q.tail;
    }
    wse::Work take_work(uint32_t worker) {
        auto &q = queue(worker);
        return reinterpret_cast<wse::Work *>(regions()[worker / WSE_PE_PER_CORE].base + q.work_offset)[q.head++];
    }
    const uint64_t *successors(const wse::TaskRecord &task) const { return at<uint64_t>(task.successors_offset); }

    WseRun(
        const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *storage,
        uint32_t region_count, const wse::SramAllocation *allocations, uint32_t allocation_count
    ) :
        base(storage[0].base),
        events(cores * WSE_PE_PER_CORE) {
        const SchedulerGraphAccess graph(d->graph);
        if (!scheduler_graph_view_valid(graph)) throw std::invalid_argument("invalid WSE graph");
        const auto *source_plans = reinterpret_cast<const WseNodePlan *>(d->nodes_address);
        const auto count = graph.task_count;
        ImageBuilder image(storage, region_count);
        control = image.at<wse::RunControl>(0, image.allocate<wse::RunControl>(0, 1, 1));
        control->epoch = epoch;
        control->expected_tasks = d->expected_tasks;
        control->core_count = cores;
        control->task_count = count;
        control->tasks_offset = image.allocate<wse::TaskRecord>(0, count, 2);
        control->core_offsets_offset = image.allocate<uint64_t>(0, cores, 3);
        control->access.region_count = region_count;
        control->access.regions_offset = image.allocate<wse::SramRegion>(0, region_count, 4);
        std::copy(storage, storage + region_count, at<wse::SramRegion>(control->access.regions_offset));
        std::vector<std::vector<uint64_t>> edges(count);
        std::vector<uint64_t> queue_sizes(cores * WSE_PE_PER_CORE);
        std::vector<wse::SramAllocation> used_allocations;
        auto validate_storage = [&](uint64_t address, uint64_t bytes) {
            if (bytes > UINT64_MAX - address) throw std::invalid_argument("WSE tensor address overflow");
            bool is_sram = false;
            for (uint32_t i = 0; i < region_count; ++i) {
                const auto &r = storage[i];
                if (address < r.base + r.capacity && (address >= r.base || address + bytes > r.base)) is_sram = true;
            }
            if (!is_sram) return;
            for (uint32_t i = 0; i < allocation_count; ++i) {
                const auto &a = allocations[i];
                if (address < a.base || address - a.base > a.bytes || bytes > a.bytes - (address - a.base)) continue;
                if (std::none_of(used_allocations.begin(), used_allocations.end(), [&](const auto &old) {
                        return old.base == a.base;
                    }))
                    used_allocations.push_back(a);
                return;
            }
            throw std::invalid_argument("WSE tensor names unallocated or private SRAM");
        };
        uint64_t expected = 0;
        for (uint64_t ref = 0; ref < count; ++ref) {
            const auto &source = source_plans[ref];
            auto &task = plans()[ref];
            task.entry = source.entry;
            task.blocks = source.blocks;
            task.host_done = source.host_done;
            task.has_predicate = source.has_predicate;
            if (source.host_done) continue;
            ++expected;
            if (!source.blocks) throw std::invalid_argument("empty WSE task block set");
            if (source.entry && scheduler_bind_graph_arguments(graph, ref) != SchedulerGraphBindingError::NONE)
                throw std::invalid_argument("invalid WSE graph argument binding");
            SchedulerTaskPayloadArguments args{};
            if (source.entry) {
                if (scheduler_parse_task_payload_arguments(graph, ref, &args) != SchedulerGraphResult::OK)
                    throw std::invalid_argument("invalid WSE task arguments");
                task.tensor_count = args.tensor_count;
                task.scalar_count = args.scalar_count;
                task.tensors_offset = image.allocate<Tensor>(0, task.tensor_count, 5);
                task.scalars_offset = image.allocate<uint64_t>(0, task.scalar_count, 6);
                for (uint32_t i = 0; i < task.tensor_count; ++i) {
                    auto &tensor = at<Tensor>(task.tensors_offset)[i];
                    std::memcpy(&tensor, args.tensors + i * TASKPAYLOAD_TENSOR_STRIDE, sizeof(Tensor));
                    uint64_t address = 0, bytes = 0;
                    if (!wse::tensor_span(tensor, address, bytes))
                        throw std::invalid_argument("invalid WSE Tensor span");
                    validate_storage(tensor.buffer.addr, tensor.buffer.size);
                }
                if (task.scalar_count)
                    std::copy(args.scalars, args.scalars + task.scalar_count, at<uint64_t>(task.scalars_offset));
                if (source.has_predicate) {
                    const auto *predicate = scheduler_node_predicate(graph, ref);
                    if (!predicate || !scheduler_dispatch_predicate_metadata_valid(
                                          predicate->addr, predicate->elem_size, predicate->op
                                      ))
                        throw std::invalid_argument("invalid WSE predicate");
                    task.predicate = *predicate;
                    validate_storage(predicate->addr, predicate->elem_size);
                }
            }
            uint32_t fanins = 0;
            if (scheduler_node_fanin_count(graph, ref, &fanins) != SchedulerGraphResult::OK)
                throw std::invalid_argument("invalid WSE dependencies");
            for (uint32_t i = 0; i < fanins; ++i) {
                int64_t producer = -1;
                if (scheduler_node_fanin_ref(graph, ref, i, &producer) != SchedulerGraphResult::OK || producer < 0 ||
                    static_cast<uint64_t>(producer) >= ref)
                    throw std::invalid_argument("invalid WSE dependency order");
                if (!source_plans[producer].host_done) {
                    ++task.initial_dependencies;
                    edges[producer].push_back(ref);
                }
            }
            const uint64_t workers = queue_sizes.size();
            for (uint64_t worker = 0; worker < workers; ++worker) {
                const uint64_t first = (worker % WSE_PE_PER_CORE) * cores + worker / WSE_PE_PER_CORE;
                const uint64_t amount = source.blocks / workers + (first < source.blocks % workers);
                if (amount > UINT64_MAX - queue_sizes[worker])
                    throw std::invalid_argument("WSE block capacity overflow");
                queue_sizes[worker] += amount;
            }
        }
        if (expected != d->expected_tasks) throw std::invalid_argument("WSE task count mismatch");
        for (uint64_t ref = 0; ref < count; ++ref) {
            auto &task = plans()[ref];
            task.successor_count = edges[ref].size();
            task.successors_offset = image.allocate<uint64_t>(0, task.successor_count, 7);
            std::copy(edges[ref].begin(), edges[ref].end(), at<uint64_t>(task.successors_offset));
        }
        for (uint32_t core = 0; core < cores; ++core) {
            const auto offset = image.allocate<wse::CoreControl>(core, 1, 8);
            at<uint64_t>(control->core_offsets_offset)[core] = offset;
            auto &c = local(core);
            for (uint32_t pe = 0; pe < WSE_PE_PER_CORE; ++pe) {
                auto &q = c.queues[pe];
                q.capacity = queue_sizes[core * WSE_PE_PER_CORE + pe];
                q.work_offset = image.allocate<wse::Work>(core, q.capacity, 9);
            }
        }
        control->access.allocation_count = used_allocations.size();
        control->access.allocations_offset = image.allocate<wse::SramAllocation>(0, used_allocations.size(), 10);
        std::copy(
            used_allocations.begin(), used_allocations.end(),
            at<wse::SramAllocation>(control->access.allocations_offset)
        );
        used = image.usage();
        sections = std::move(image.sections);
    }
    void record(
        uint32_t worker, WseEventKind kind, uint64_t task = UINT64_MAX, uint32_t block = 0, uint64_t value = 0,
        uint32_t slot = 0, uint64_t generation = 0
    ) {
        events[worker].push_back(
            {sequence.fetch_add(1), task, value, worker / WSE_PE_PER_CORE, worker % WSE_PE_PER_CORE, block, kind, slot,
             generation}
        );
    }
    void enqueue(uint64_t ref) {
        for (uint32_t b = 0; b < plans()[ref].blocks; ++b) {
            const uint32_t core = b % control->core_count;
            const uint32_t pe = (b / control->core_count) % WSE_PE_PER_CORE;
            auto &q = local(core).queues[pe];
            if (q.tail >= q.capacity) throw std::logic_error("WSE SRAM queue capacity invariant");
            reinterpret_cast<wse::Work *>(regions()[core].base + q.work_offset)[q.tail++] = {ref, b, 0};
        }
    }
    void bootstrap() {
        clear_cpu_sim_shared_storage();
        for (uint32_t i = 0; i < control->task_count; ++i) {
            auto &task = plans()[i];
            task.remaining_dependencies = task.initial_dependencies;
            task.remaining_blocks = task.blocks;
        }
        completion().store(0, std::memory_order_relaxed);
        record(0, WseEventKind::CONFIG_READY, UINT64_MAX, 0, control->expected_tasks);
        control->configured = true;
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

extern "C" WseRun *wse_create_run(
    const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *regions, uint32_t region_count,
    const wse::SramAllocation *allocations, uint32_t allocation_count, char *error, size_t error_size
) {
    try {
        if (!d || !cores || cores > region_count || region_count > wse::MAX_CORE_COUNT || !epoch || !regions)
            throw std::invalid_argument("invalid WSE SRAM run descriptor");
        return new WseRun(d, cores, epoch, regions, region_count, allocations, allocation_count);
    } catch (const std::exception &e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return nullptr;
    }
}
extern "C" const WseMemorySection *wse_memory_sections(const WseRun *run, uint64_t *count) {
    *count = run->sections.size();
    return run->sections.data();
}
extern "C" uint64_t wse_runtime_used(const WseRun *run, uint32_t core) {
    return core < run->used.size() ? run->used[core] : 0;
}
extern "C" uint64_t wse_control_address(const WseRun *run, uint32_t core) {
    return core < run->control->core_count ? reinterpret_cast<uint64_t>(&run->local(core)) : 0;
}
extern "C" void wse_destroy_run(WseRun *run) { delete run; }
extern "C" uint64_t wse_completed_tasks(const WseRun *run) { return run->completion().load(std::memory_order_acquire); }
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
            std::cerr << "WSE bootstrap rejected: " << error.what() << '\n';
            run->control->init_error = -1;
        }
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->configured || run->control->init_error;
    });
    if (run->control->init_error) return run->control->init_error;
    auto &local = run->local(core);
    if (pe == 0) {
        for (auto &q : local.queues)
            q.head = q.tail = 0;
        local.ready = 0;
        local.quiet = 0;
        local.retired = false;
        local.expected_tasks = run->control->expected_tasks;
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
            for (uint32_t c = 0; c < run->control->core_count; ++c)
                if (run->local(c).ready != WSE_PE_PER_CORE) return false;
            return true;
        });
        for (uint32_t i = 0; i < run->control->task_count; ++i)
            if (!run->plans()[i].host_done && run->plans()[i].remaining_dependencies == 0) run->enqueue(i);
        run->record(worker, WseEventKind::GO);
        run->control->go = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->go;
    });
    while (true) {
        run->changed.wait(lock, [&] {
            return run->has_work(worker) || run->completion().load(std::memory_order_acquire) == local.expected_tasks;
        });
        if (run->completion().load(std::memory_order_acquire) == local.expected_tasks) break;
        const wse::Work work = run->take_work(worker);
        lock.unlock();
        run->record(worker, WseEventKind::BLOCK_BEGIN, work.task, work.block);
        const auto &plan = run->plans()[work.task];
        bool execute = true;
        if (plan.has_predicate) {
            const auto &predicate = plan.predicate;
            execute =
                DispatchPredicate{
                    predicate.addr, predicate.target, predicate.elem_size, static_cast<PredicateOp>(predicate.op)
                }
                    .pass();
        }
        if (!execute) run->record(worker, WseEventKind::BLOCK_SKIPPED, work.task, work.block);
        if (plan.entry && execute) {
            int64_t args[SPMD_GLOBAL_CONTEXT_INDEX + 1]{};
            int index = 0;
            for (uint32_t i = 0; i < plan.tensor_count; ++i)
                args[index++] = reinterpret_cast<int64_t>(run->at<Tensor>(plan.tensors_offset) + i);
            for (uint32_t i = 0; i < plan.scalar_count; ++i)
                args[index++] = run->at<uint64_t>(plan.scalars_offset)[i];
            LocalContext context{};
            context.block_idx = work.block;
            context.block_num = plan.blocks;
            GlobalContext global{};
            global.core_id = core;
            global.pe_id = pe;
            global.sram_base = run->base;
            global.sram_access_offset = reinterpret_cast<uint64_t>(&run->control->access) - run->base;
            OperationBinding binding{run, worker, work.task, work.block};
            global.operation_trace = {&binding, operation_trace};
            args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
            args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
            reinterpret_cast<void (*)(int64_t *)>(plan.entry)(args);
        }
        run->record(worker, WseEventKind::BLOCK_END, work.task, work.block);
        lock.lock();
        auto &node = run->plans()[work.task];
        if (--node.remaining_blocks == 0) {
            // All block writes acquire through this lock before successors become runnable.
            for (uint32_t i = 0; i < node.successor_count; ++i) {
                const auto next = run->successors(node)[i];
                if (--run->plans()[next].remaining_dependencies == 0) run->enqueue(next);
            }
            const uint64_t count = run->completion().fetch_add(1, std::memory_order_acq_rel) + 1;
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
        run->control->stop = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->stop;
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
            for (uint32_t c = 0; c < run->control->core_count; ++c)
                if (!run->local(c).retired) return false;
            return true;
        });
        run->record(worker, WseEventKind::RUN_DONE, UINT64_MAX, 0, run->completion().load());
    }
    return 0;
}
