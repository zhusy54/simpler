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
#include "orchestration_api.h"
namespace {
void graph(const GraphTaskArgs &a) {
    for (int stage = 0; stage < 2; ++stage) {
        CoreTaskArgs task;
        task.add_inout(a.tensor(0).ref());
        task.add_inout(a.tensor(1).ref());
        task.add_inout(a.tensor(2).ref());
        task.add_scalar(stage);
        task.launch_spec.set_block_num(a.tensor(0).ref().shapes[0]);
        rt_submit_wse_task(0, task);
    }
}
}  // namespace
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 4}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    GraphTaskArgs args;
    for (int i = 0; i < 3; ++i)
        args.add_inout(a.tensor(i).ref());
    args.add_scalar(a.scalar(0));
    rt_submit_graph(&graph, args);
}
