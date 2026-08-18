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

import json
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSingleAicRoot(SceneTestCase):
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

    def _build_config(self, config_dict, *args, **kwargs):
        config = super()._build_config(config_dict, *args, **kwargs)
        self._profiling_level = int(kwargs.get("enable_chip_swimlane", args[0] if args else 0))
        output_prefix = kwargs.get("output_prefix", "")
        self._profiling_path = Path(output_prefix) / "chip_swimlane_records.json" if output_prefix else None
        return config

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        if getattr(self, "_profiling_level", 0) == 0:
            return
        assert self._profiling_level == 1
        path = self._profiling_path
        assert path is not None and path.exists()
        with path.open() as f:
            capture = json.load(f)
        assert capture["chip_swimlane_level"] == 1
        assert len(capture["aicore_tasks"]) == 1
        assert {phase["phase"] for phase in capture["aicore_scheduler_phases"]} == {
            "AICoreEntryToHandshake",
            "HandshakeToRegisterRelease",
            "RegisterReleaseToDescriptorReady",
            "DescriptorReadyToSeedClaim",
            "SeedClaim",
            "PayloadBaseAtomicLoad",
            "PayloadCacheObserveCL0",
            "PayloadCacheObserveCL1To7",
            "PayloadCacheObserveBarrier",
            "PayloadSetup",
            "Kernel",
            "CompletionEnqueue",
            "PostCompletion",
            "ReadyScan",
            "ReadyToPayload",
            "ExecutorDrainPublish",
            "WaitForExit",
            "FinalStatsPublish",
            "ExitAckPublish",
            "Drain",
        }
        assert {phase["phase"] for phase in capture["aicpu_lifecycle_phases"]} == {
            "WaitExecutors",
            "WaitResolved",
            "CompletionDecision",
            "RegisterRelease",
            "ExitSignalToAck",
        }
        assert "aicore_resolve_phases" not in capture
        assert capture["aicpu_tasks"] == []
        assert "aicpu_scheduler_phases" not in capture
        assert "aicpu_orchestrator_phases" not in capture
        assert not Path(f"{path}.tmp").exists()


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSingleAivRoot(SceneTestCase):
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
                "source": "../vector_example/kernels/aiv/kernel_add.cpp",
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
