#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual M1 gates for one dependency-free AIC or AIV root."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestHbgAicoreSingleAicRoot(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_aic_root_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../tensormap_and_ringbuffer/dummy_task/kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            }
        ],
    }

    CASES = [
        {
            "name": "single_aic_root",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
            "manual": True,
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("out", torch.zeros(16, dtype=torch.float32)))

    def compute_golden(self, args, params):
        args.out[0] = 42.0


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestHbgAicoreSingleAivRoot(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_aiv_root_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../host_build_graph/vector_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT],
            }
        ],
    }

    CASES = [
        {
            "name": "single_aiv_root",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
            "manual": True,
        },
    ]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("out", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out[:] = args.a + args.b


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
