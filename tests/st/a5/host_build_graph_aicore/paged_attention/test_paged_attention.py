#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Production-scale paged attention on the HBG-AICore runtime."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs


@scene_test(level=2, runtime="host_build_graph_aicore")
class TestPagedAttentionHostBuildGraphAicoreA5(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "../../host_build_graph/paged_attention/kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "build_paged_attention_graph",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": "../../host_build_graph/paged_attention/kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "PV",
                "source": "../../host_build_graph/paged_attention/kernels/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "SF",
                "source": "../../host_build_graph/paged_attention/kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 3,
                "name": "UP",
                "source": "../../host_build_graph/paged_attention/kernels/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a5"],
            "manual": True,
            "config": {
                "runtime_env": {
                    "ring_task_window": 131072,
                    "ring_heap": 512 * 1024 * 1024,
                }
            },
            "params": {
                "batch": 256,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8192,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "SmallCase1",
            "platforms": ["a5sim", "a5"],
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 33,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        inputs = _pa_generate_inputs(params)
        specs = []
        for name, val in inputs:
            if isinstance(val, torch.Tensor):
                specs.append(Tensor(name, val))
            else:
                specs.append(Scalar(name, val))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, Tensor)}
        _pa_compute_golden(tensors, params)
        for spec in args.specs:
            if isinstance(spec, Tensor) and spec.name in tensors:
                getattr(args, spec.name)[:] = tensors[spec.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
