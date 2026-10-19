# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene-test adapter and independent DAG oracle for the scheduling matrix."""

import ctypes
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename
from tests.st.scheduler_benchmark.matrix import (
    CASE_IDS,
    LONG_ITERS,
    SHORT_ITERS,
    CaseSpec,
    expand_case,
    is_long,
    scale_case,
    task_roles,
    validate_deps,
)

_KERNELS = Path(__file__).parent / "kernels"


def case_list(platforms: list[str]) -> list[dict]:
    return [
        {"name": case_id, "platforms": platforms, "manual": True, "params": {"case_id": case_id}}
        for case_id in CASE_IDS
    ]


def _counts() -> tuple[int, int]:
    try:
        k_c = int(os.environ["SIMPLER_SCHED_BENCH_K_C"])
        k_v = int(os.environ["SIMPLER_SCHED_BENCH_K_V"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "set SIMPLER_SCHED_BENCH_K_C and SIMPLER_SCHED_BENCH_K_V using "
            "python -m tests.st.scheduler_benchmark.run_cases"
        ) from exc
    if k_c < 1 or k_v < 1:
        raise ValueError("executor counts must be positive")
    return k_c, k_v


@lru_cache(maxsize=2)
def _digest(iterations: int) -> int:
    state = 0x13579BDF
    for _ in range(iterations):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state & 0x7FFFFFFF


class SchedulingBenchmarkScene(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": str(_KERNELS / "orchestration.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT],
        },
        "incores": [
            {"func_id": 0, "source": str(_KERNELS / "write_token.cpp"), "core_type": "aic", "signature": [D.INOUT]},
            {"func_id": 1, "source": str(_KERNELS / "write_token.cpp"), "core_type": "aiv", "signature": [D.INOUT]},
        ],
    }

    def _spec(self, params: dict) -> CaseSpec:
        return scale_case(expand_case(params["case_id"], *_counts()), params.get("task_scale", 1))

    def generate_args(self, params):
        spec = self._spec(params)
        return TaskArgsBuilder(
            TensorArg("state", torch.zeros((spec.task_count + 1) * 8, dtype=torch.int64)),
            *(
                Scalar(name, ctypes.c_int64(getattr(spec, name)))
                for name in (
                    "family",
                    "task_count",
                    "width",
                    "length",
                    "producers",
                    "consumers",
                    "repeats",
                    "core_mode",
                    "work_mode",
                )
            ),
            Scalar("k_c", ctypes.c_int64(_counts()[0])),
            Scalar("k_v", ctypes.c_int64(_counts()[1])),
        )

    def compute_golden(self, args, params):
        spec = self._spec(params)
        roles = task_roles(spec)
        for long_work, iterations in ((False, SHORT_ITERS), (True, LONG_ITERS)):
            indexes = [
                index for index, role in enumerate(roles) if role != "dummy" and is_long(spec, index) == long_work
            ]
            if not indexes:
                continue
            positions = torch.tensor([(index + 1) * 8 for index in indexes], dtype=torch.long)
            tokens = torch.tensor([(_digest(iterations) << 32) | (index + 1) for index in indexes], dtype=torch.int64)
            args.state[positions] = tokens

    def test_run(self, st_platform, st_worker, request):
        selected = self._matching_cases(st_platform, request)
        before = {}
        if self._effective_enable_dep_gen(request):
            for case in selected:
                label = _sanitize_for_filename(f"{type(self).__name__}_{case['name']}")
                before[case["name"]] = set(_outputs_dir().glob(f"{label}_*"))
        super().test_run(st_platform, st_worker, request)
        if not self._effective_enable_dep_gen(request):
            return
        for case in selected:
            label = _sanitize_for_filename(f"{type(self).__name__}_{case['name']}")
            fresh = set(_outputs_dir().glob(f"{label}_*")) - before[case["name"]]
            assert len(fresh) == 1, f"{case['name']}: expected one new dep_gen capture, got {len(fresh)}"
            deps_path = fresh.pop() / "deps.json"
            assert deps_path.exists(), f"{case['name']}: dep_gen did not write {deps_path}"
            validate_deps(json.loads(deps_path.read_text()), self._spec(case["params"]))


def execute_scaled_scene(module_name: str, scale: int) -> None:
    module = importlib.import_module(module_name)
    for value in vars(module).values():
        if isinstance(value, type) and issubclass(value, SchedulingBenchmarkScene) and value.__module__ == module_name:
            value.CASES = [{**case, "params": {**case["params"], "task_scale": scale}} for case in value.CASES]
    SceneTestCase.run_module(module_name)
