#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG-carried M0 hardware gate for production HBG-AICore sidecar primitives."""

import struct

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

PROBE_BYTES = 256 * 1024


@scene_test(level=2, runtime="host_build_graph")
class TestAicoreSidecarProbe(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/sidecar_probe_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": "kernels/aic/sidecar_probe.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            }
        ],
    }

    CASES = [
        {
            "name": "production_sidecar_primitives",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(Tensor("storage", torch.zeros(PROBE_BYTES, dtype=torch.uint8)))

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        raw = bytes(test_args.storage.tolist())
        sidecar_offset, controls_offset, _queue_offset, results_offset, block_num = struct.unpack_from("<QQQQQ", raw)
        assert block_num > 1
        for block_idx in range(block_num):
            control_offset = sidecar_offset + controls_offset + block_idx * 128
            completion, wake_head = struct.unpack_from("<qq", raw, control_offset)
            (next_waiter,) = struct.unpack_from("<q", raw, control_offset + 64)
            assert completion == 0x10000 + block_idx
            assert wake_head == 0x20000 + block_idx
            assert next_waiter == 0x30000 + block_idx

        popped = struct.unpack_from(f"<{block_num}q", raw, sidecar_offset + results_offset)
        assert sorted(popped) == list(range(block_num))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
