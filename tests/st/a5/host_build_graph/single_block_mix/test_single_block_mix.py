#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-block Mix scheduling coverage for dependent and independent DAGs."""

import ctypes

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

MATMUL_SIZE = 128
TILE_ELEMS = MATMUL_SIZE * MATMUL_SIZE
TASK_COUNT = 16


@scene_test(level=2, runtime="host_build_graph")
class TestSingleBlockMixHostBuildGraphA5(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/single_block_mix_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MATMUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aic/kernel_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "MUL",
                "source": "../../tensormap_and_ringbuffer/mixed_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "chain16",
            "platforms": ["a5sim", "a5"],
            "params": {"graph_case": 0},
        },
        {
            "name": "burst16",
            "platforms": ["a5sim", "a5"],
            "params": {"graph_case": 1},
        },
    ]

    def generate_args(self, params):
        torch.manual_seed(42)
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, dtype=torch.float32) * 0.01
        d = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        e = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        g = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01
        h = torch.randn(TILE_ELEMS, dtype=torch.float32) * 0.01

        def output():
            return torch.zeros(TASK_COUNT * TILE_ELEMS, dtype=torch.float32)

        return TaskArgsBuilder(
            TensorArg("a", a.flatten()),
            TensorArg("b", b.flatten()),
            TensorArg("c", output()),
            TensorArg("d", d),
            TensorArg("e", e),
            TensorArg("f", output()),
            TensorArg("g", g),
            TensorArg("h", h),
            TensorArg("i", output()),
            Scalar("graph_case", ctypes.c_int64(params["graph_case"])),
        )

    def compute_golden(self, args, params):
        del params
        matmul = torch.matmul(
            args.a.reshape(MATMUL_SIZE, MATMUL_SIZE), args.b.reshape(MATMUL_SIZE, MATMUL_SIZE)
        ).flatten()
        args.c.reshape(TASK_COUNT, TILE_ELEMS)[:] = matmul
        args.f.reshape(TASK_COUNT, TILE_ELEMS)[:] = args.d + args.e
        args.i.reshape(TASK_COUNT, TILE_ELEMS)[:] = args.g * args.h


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
