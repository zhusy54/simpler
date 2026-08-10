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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kFuncAdd = 0;
constexpr int32_t kFuncAddScalar = 1;

union ScalarBits {
    float f32;
    uint64_t u64;
};

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return PTO2OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const ChipTensor &a = args.tensor(0).ref();
    const ChipTensor &b = args.tensor(1).ref();
    const ChipTensor &out = args.tensor(2).ref();
    uint32_t intermediate_shapes[1] = {a.shapes[0]};
    TensorCreateInfo intermediate(intermediate_shapes, 1, DataType::FLOAT32);

    CoreTaskArgs add_args;
    add_args.add_input(a);
    add_args.add_input(b);
    add_args.add_output(intermediate);
    TaskOutputTensors add_outputs = rt_submit_aiv_task(kFuncAdd, add_args);
    const ChipTensor &sum = add_outputs.get_ref(0);

    CoreTaskArgs increment_args;
    increment_args.add_input(sum);
    increment_args.add_output(intermediate);
    ScalarBits scalar{};
    scalar.f32 = 1.0f;
    increment_args.add_scalar(scalar.u64);
    TaskOutputTensors increment_outputs = rt_submit_aiv_task(kFuncAddScalar, increment_args);
    const ChipTensor &incremented = increment_outputs.get_ref(0);

    CoreTaskArgs output_args;
    output_args.add_input(incremented);
    output_args.add_output(out);
    scalar.f32 = 2.0f;
    output_args.add_scalar(scalar.u64);
    (void)rt_submit_aiv_task(kFuncAddScalar, output_args);
}

}  // extern "C"
