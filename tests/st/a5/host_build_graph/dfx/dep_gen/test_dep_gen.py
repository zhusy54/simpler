#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 host_build_graph dependency-capture integration tests."""

import ctypes
import json
import time

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

VECTOR_KERNELS = "../../vector_example/kernels"
DAG_KERNELS = "../../single_core_dag/kernels"


def _load_deps(test_cls_name, case_name, run_marker):
    safe_label = _sanitize_for_filename(f"{test_cls_name}_{case_name}")
    matches = [p for p in _outputs_dir().glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
    assert matches, f"no diagnostic output directory was created for {case_name!r}"
    out_dir = max(matches, key=lambda p: p.stat().st_mtime)
    deps_path = out_dir / "deps.json"
    assert deps_path.exists(), f"host-direct dependency capture did not produce {deps_path}"
    return json.loads(deps_path.read_text())


def _edges_by_position(deps):
    position = {int(task["task_id"]): idx for idx, task in enumerate(deps["tasks"])}
    unknown = {
        (edge["pred"], edge["succ"])
        for edge in deps["edges"]
        if int(edge["pred"]) not in position or int(edge["succ"]) not in position
    }
    assert not unknown, f"dependency edges reference unknown tasks: {unknown}"
    return {(position[int(edge["pred"])], position[int(edge["succ"])], edge["source"]) for edge in deps["edges"]}


def _assert_task_metadata(deps):
    assert all(len(task.get("kernel_ids", [])) == 3 for task in deps["tasks"])
    assert all(int(task.get("block_num", 0)) > 0 for task in deps["tasks"])


def _assert_tensor_annotations(deps):
    tensor_ids = {int(tensor["tensor_id"]) for tensor in deps.get("tensors", [])}
    for edge in deps["edges"]:
        if edge["source"] == "explicit":
            continue
        assert int(edge["tensor_id"]) in tensor_ids
        assert {"consumer_shape", "consumer_start_offset", "consumer_strides"} <= edge.keys()


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphA5Creator(SceneTestCase):
    """The vector DAG records creator edges and per-task kernel metadata."""

    CALLABLE = {
        "orchestration": {
            "source": f"{VECTOR_KERNELS}/orchestration/example_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{VECTOR_KERNELS}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{VECTOR_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{VECTOR_KERNELS}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }
    CASES = [{"name": "creator_diamond", "platforms": ["a5sim", "a5"], "params": {}}]

    def generate_args(self, params):
        size = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((size,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((size,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(size, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2)

    def test_run(self, st_platform, st_worker, request):
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        deps = _load_deps(type(self).__name__, self.CASES[0]["name"], run_marker)
        assert len(deps.get("tasks", [])) == 4
        assert _edges_by_position(deps) == {
            (0, 1, "creator"),
            (0, 2, "creator"),
            (1, 3, "creator"),
            (2, 3, "creator"),
        }
        _assert_task_metadata(deps)
        _assert_tensor_annotations(deps)


@scene_test(level=2, runtime="host_build_graph")
class TestDepGenHostBuildGraphA5Explicit(SceneTestCase):
    """The explicit chain records every declared predecessor edge."""

    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": {
            "source": f"{DAG_KERNELS}/orchestration/single_core_dag_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{DAG_KERNELS}/aic/check_dag.cpp",
                "core_type": "aic",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "source": f"{DAG_KERNELS}/aiv/check_dag.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }
    CASES = [
        {
            "name": "explicit_chain",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2},
            "params": {},
        }
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            Tensor("task_state", torch.zeros(64 * 8, dtype=torch.int64)),
            Scalar("graph_case", ctypes.c_int64(0)),
            Scalar("core_type", ctypes.c_int64(1)),
        )

    def compute_golden(self, args, params):
        args.task_state[::8] = torch.arange(1, 65, dtype=torch.int64)

    def test_run(self, st_platform, st_worker, request):
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        deps = _load_deps(type(self).__name__, self.CASES[0]["name"], run_marker)
        assert len(deps.get("tasks", [])) == 64
        assert _edges_by_position(deps) == {(idx, idx + 1, "explicit") for idx in range(63)}
        _assert_task_metadata(deps)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
