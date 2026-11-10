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
#include <stdexcept>
extern "C" OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) { return {.expected_arg_count = 2}; }
extern "C" void aicpu_orchestration_entry(const ChipTaskArgs &a) {
    if (rt_available_core_count() != 1) throw std::invalid_argument("probe requires one core");
    CoreTaskArgs producer;
    producer.add_input(a.tensor(0).ref());
    rt_submit_wse_task(0, producer);
    CoreTaskArgs consumer;
    consumer.add_output(a.tensor(1).ref());
    consumer.launch_spec.set_block_num(4);
    rt_submit_wse_task(1, consumer);
}
