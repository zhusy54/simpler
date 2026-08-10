#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real AIV dataflow DAGs for HBG-AICore and the HBG semantic oracle."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


def _aiv_callable(orchestration_source, orchestration_signature):
    return {
        "orchestration": {
            "source": orchestration_source,
            "function_name": "aicpu_orchestration_entry",
            "signature": orchestration_signature,
        },
        "incores": [
            {
                "func_id": 0,
                "source": "../../host_build_graph/vector_example/kernels/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": "../../host_build_graph/vector_example/kernels/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": "../../host_build_graph/vector_example/kernels/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }


def _manual_case(name):
    return [
        {
            "name": name,
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {},
            "manual": True,
        }
    ]


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestHbgAicoreRealChainDag(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5
    CALLABLE = _aiv_callable("kernels/orchestration/chain_orch.cpp", [D.IN, D.IN, D.OUT])
    CASES = _manual_case("real_chain")

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("out", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out[:] = args.a + args.b + 3


@scene_test(level=2, runtime="host_build_graph")
class TestHbgRealChainDagOracle(TestHbgAicoreRealChainDag):
    CASES = _manual_case("oracle_real_chain")


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestHbgAicoreRealDiamondDag(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5
    CALLABLE = _aiv_callable("kernels/orchestration/diamond_orch.cpp", [D.IN, D.IN, D.OUT])
    CASES = _manual_case("real_diamond")

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("out", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out[:] = (args.a + args.b + 1) * (args.a + args.b + 2)


@scene_test(level=2, runtime="host_build_graph")
class TestHbgRealDiamondDagOracle(TestHbgAicoreRealDiamondDag):
    CASES = _manual_case("oracle_real_diamond")


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestHbgAicoreRealMultiRootDag(SceneTestCase):
    RTOL = 1e-5
    ATOL = 1e-5
    CALLABLE = _aiv_callable("kernels/orchestration/multi_root_orch.cpp", [D.IN, D.IN, D.OUT, D.IN, D.IN, D.OUT])
    CASES = _manual_case("real_multi_root")

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a1", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b1", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("out1", torch.zeros(size, dtype=torch.float32)),
            Tensor("a2", torch.full((size,), 5.0, dtype=torch.float32)),
            Tensor("b2", torch.full((size,), 7.0, dtype=torch.float32)),
            Tensor("out2", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.out1[:] = (args.a1 + args.b1 + 1) * (args.a1 + args.b1 + 2)
        args.out2[:] = (args.a2 + args.b2 + 1) * (args.a2 + args.b2 + 2)


@scene_test(level=2, runtime="host_build_graph")
class TestHbgRealMultiRootDagOracle(TestHbgAicoreRealMultiRootDag):
    CASES = _manual_case("oracle_real_multi_root")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
