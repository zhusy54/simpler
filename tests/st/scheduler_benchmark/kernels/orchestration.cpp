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

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicKernel = 0;
constexpr int32_t kAivKernel = 1;
constexpr int32_t kMaxTasks = 32768;
constexpr int32_t kMaxWidth = 288;      // Four times A5's maximum 72 AIV executors.
constexpr int32_t kMaxConsumers = 144;  // Four times A5's maximum 36 AIC executors.
constexpr int32_t kMaxProducers = 64;
constexpr int64_t kShortIters = 1;
constexpr int64_t kLongIters = 2048;

struct Spec {
    int64_t family;
    int64_t count;
    int64_t width;
    int64_t length;
    int64_t producers;
    int64_t consumers;
    int64_t repeats;
    int64_t core_mode;
    int64_t work_mode;
    int64_t k_c;
    int64_t k_v;
};

bool valid(const Spec &spec) {
    if (spec.count < 1 || spec.count > kMaxTasks || spec.k_c < 1 || spec.k_v < 1 ||
        spec.k_c > rt_available_cluster_count() || spec.k_v > rt_available_aiv_count()) {
        return false;
    }
    if (spec.family == 1) {
        return spec.length == spec.count && spec.core_mode >= 0 && spec.core_mode <= 3 && spec.work_mode == 0;
    }
    if (spec.family == 2) {
        return spec.width > 0 && spec.width <= kMaxWidth && spec.length > 0 && spec.width * spec.length == spec.count &&
               spec.core_mode >= 0 && spec.core_mode <= 1 && spec.work_mode >= 0 && spec.work_mode <= 4;
    }
    if (spec.family == 3) {
        return spec.count >= 4096 && spec.count % 4096 == 0 && spec.core_mode >= 0 && spec.core_mode <= 1 &&
               spec.work_mode == 0;
    }
    if (spec.family == 4) {
        return spec.producers > 0 && spec.producers <= kMaxProducers && spec.consumers > 0 &&
               spec.consumers <= kMaxConsumers && spec.repeats >= 32 &&
               spec.count == spec.repeats * (spec.producers + spec.consumers) + spec.repeats - 1 &&
               spec.core_mode == 0 && spec.work_mode == 0;
    }
    return false;
}

bool uses_aiv(const Spec &spec, int64_t logical_index) {
    if (spec.core_mode == 1) return true;
    if (spec.core_mode == 2) return (logical_index & 1) != 0;
    if (spec.core_mode == 3) return (logical_index & 1) == 0;
    return false;
}

bool uses_long_work(const Spec &spec, int64_t logical_index) {
    if (spec.work_mode == 1) return true;
    if (spec.work_mode == 2) return logical_index % 5 == 0;
    if (spec.work_mode == 3) return logical_index % 2 == 0;
    if (spec.work_mode == 4) return logical_index % 5 != 0;
    return false;
}

template <typename TensorT>
TaskId
submit_kernel(const TensorT &state, const Spec &spec, int64_t logical_index, const TaskId *deps, uint32_t dep_count) {
    CoreTaskArgs args;
    args.add_no_dep(state);
    args.add_scalar(logical_index);
    args.add_scalar(uses_long_work(spec, logical_index) ? kLongIters : kShortIters);
    args.set_dependencies(deps, dep_count);
    return uses_aiv(spec, logical_index) ? rt_submit_aiv_task(kAivKernel, args).task_id() :
                                           rt_submit_aic_task(kAicKernel, args).task_id();
}

template <typename TensorT>
void build_chain(const TensorT &state, const Spec &spec) {
    TaskId previous;
    for (int64_t index = 0; index < spec.count; ++index) {
        previous = submit_kernel(state, spec, index, index == 0 ? nullptr : &previous, index == 0 ? 0 : 1);
    }
}

template <typename TensorT>
void build_parallel_chains(const TensorT &state, const Spec &spec) {
    TaskId previous[kMaxWidth];
    for (int64_t depth = 0; depth < spec.length; ++depth) {
        for (int64_t chain = 0; chain < spec.width; ++chain) {
            const int64_t index = depth * spec.width + chain;
            previous[chain] =
                submit_kernel(state, spec, index, depth == 0 ? nullptr : &previous[chain], depth == 0 ? 0 : 1);
        }
    }
}

template <typename TensorT>
void build_roots(const TensorT &state, const Spec &spec) {
    for (int64_t index = 0; index < spec.count; ++index) {
        (void)submit_kernel(state, spec, index, nullptr, 0);
    }
}

template <typename TensorT>
void build_fanin_units(const TensorT &state, const Spec &spec) {
    TaskId producers[kMaxProducers];
    TaskId consumers[kMaxConsumers];
    TaskId previous_gate;
    int64_t index = 0;
    for (int64_t unit = 0; unit < spec.repeats; ++unit) {
        // A TMR producer cannot finish until every consumer dependency in this
        // unit has been registered. HBG builds the same logical scope.
        rt_scope_begin(ScopeMode::MANUAL);
        for (int64_t producer = 0; producer < spec.producers; ++producer) {
            producers[producer] =
                submit_kernel(state, spec, index++, unit == 0 ? nullptr : &previous_gate, unit == 0 ? 0 : 1);
        }
        for (int64_t consumer = 0; consumer < spec.consumers; ++consumer) {
            consumers[consumer] = submit_kernel(state, spec, index++, producers, static_cast<uint32_t>(spec.producers));
        }
        rt_scope_end();
        if (unit + 1 < spec.repeats) {
            CoreTaskArgs gate_args;
            gate_args.set_dependencies(consumers, static_cast<uint32_t>(spec.consumers));
            previous_gate = rt_submit_dummy_task(gate_args).task_id();
            ++index;
        }
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 12};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &state = args.tensor(0).ref();
    const Spec spec{
        args.scalar<int64_t>(0), args.scalar<int64_t>(1), args.scalar<int64_t>(2),  args.scalar<int64_t>(3),
        args.scalar<int64_t>(4), args.scalar<int64_t>(5), args.scalar<int64_t>(6),  args.scalar<int64_t>(7),
        args.scalar<int64_t>(8), args.scalar<int64_t>(9), args.scalar<int64_t>(10),
    };
    if (!valid(spec)) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "invalid scheduler benchmark case or executor counts");
        return;
    }
    switch (spec.family) {
    case 1:
        build_chain(state, spec);
        break;
    case 2:
        build_parallel_chains(state, spec);
        break;
    case 3:
        build_roots(state, spec);
        break;
    case 4:
        build_fanin_units(state, spec);
        break;
    }
}

}  // extern "C"
