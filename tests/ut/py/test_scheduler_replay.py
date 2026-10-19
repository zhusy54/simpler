# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native zero-cost scheduling, input rejection, and Perfetto integration."""

import copy
import json

import pytest

from simpler_setup.tools.scheduler_replay import (
    UnsupportedReplay,
    build_backend,
    execute,
    normalize_input,
    replay,
    validate_timeline,
)
from simpler_setup.tools.scheduler_replay_matrix import replay_manifests

BACKENDS = [
    ("a5", "host_build_graph", "aicore"),
    ("a5", "host_build_graph", "aicpu"),
    ("a2a3", "host_build_graph", "aicpu"),
    ("a5", "tensormap_and_ringbuffer", "aicpu"),
    ("a2a3", "tensormap_and_ringbuffer", "aicpu"),
]


def model(tasks, *, backend=BACKENDS[0], clusters=2, config=None):
    arch, runtime, location = backend
    return {
        "arch": arch,
        "runtime": runtime,
        "scheduler_location": location,
        "clock_freq_hz": 1_000_000_000,
        "clusters": clusters,
        "scheduler_threads": 1,
        "tasks": tasks,
        "config": config or {},
    }


def task(index, duration, pred=(), sub=0):
    durations = [0, 0, 0]
    durations[sub] = duration
    return {
        "task_id": str(index),
        "mask": (1 << sub) if duration else 0,
        "durations_ticks": durations,
        "predecessors": list(pred),
    }


@pytest.fixture(scope="module")
def backends():
    return {backend: build_backend(model([], backend=backend)) for backend in BACKENDS}


@pytest.mark.parametrize("backend", BACKENDS)
def test_chain_eliminates_all_handoffs(backends, backend):
    tasks = [task(i, 7 + i, [i - 1] if i else [], sub=i % 2) for i in range(80)]
    total, rows = execute(model(tasks, backend=backend), backends[backend])
    assert total == pytest.approx(sum(7 + i for i in range(80)) / 1000)
    assert len(rows) == 80
    assert rows[1]["start_ticks"] == rows[0]["end_ticks"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_independent_equal_work_keeps_capacity_and_pending_wait(backends, backend):
    total, rows = execute(model([task(i, 10) for i in range(100)], backend=backend), backends[backend])
    assert total == pytest.approx(0.5)
    assert {row["core_id"] for row in rows} == {0, 1}
    assert any(row["dispatch_ticks"] < row["start_ticks"] for row in rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fanin_control_and_fanout(backends, backend):
    tasks = [task(0, 5), task(1, 20), task(2, 0, [0, 1]), task(3, 10, [2]), task(4, 10, [2])]
    total, rows = execute(model(tasks, backend=backend), backends[backend])
    assert 0.03 <= total <= 0.045
    control = next(row for row in rows if row["task_id"] == "2")
    parents_end = max(row["end_ticks"] for row in rows if row["task_id"] in {"0", "1"})
    assert control["start_ticks"] == control["end_ticks"] == parents_end


@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic_queue_choices(backends, backend):
    data = model([task(i, (i * 7) % 31 + 1, [i - 4] if i >= 4 else []) for i in range(180)], backend=backend)
    assert execute(data, backends[backend]) == execute(data, backends[backend])


@pytest.mark.parametrize("backend", BACKENDS)
def test_reverse_scheduler_interleaving_remains_a_legal_schedule(backends, backend):
    data = model([task(i, i % 9 + 1, [i - 7] if i >= 7 else []) for i in range(130)], backend=backend)
    data["scheduler_threads"] = 2
    data["config"] = {"scheduler_order": "descending"}
    first = execute(data, backends[backend])
    assert first == execute(data, backends[backend])


@pytest.mark.parametrize("backend", [BACKENDS[3], BACKENDS[4]])
def test_retention_only_dependency_does_not_delay_ready(backends, backend):
    tasks = [task(0, 1000), task(1, 1, [0])]
    tasks[1]["dependency_flags"] = [2]
    _, rows = execute(model(tasks, backend=backend), backends[backend])
    producer = next(row for row in rows if row["task_id"] == "0")
    consumer = next(row for row in rows if row["task_id"] == "1")
    assert consumer["start_ticks"] < producer["end_ticks"]


@pytest.mark.parametrize("backend", [BACKENDS[3], BACKENDS[4]])
@pytest.mark.parametrize("pool_capacity", [64, 256])
def test_native_tmr_dependency_pool_reclaims_between_manual_scopes(backends, backend, pool_capacity):
    tasks = []
    previous_gate = None
    local_id = 0
    for unit in range(25):
        first = len(tasks)
        for _ in range(4):
            tasks.append(task((1 << 32) + local_id, 10, [] if previous_gate is None else [previous_gate]))
            local_id += 1
        for _ in range(4):
            tasks.append(task((1 << 32) + local_id, 10, range(first, first + 4)))
            local_id += 1
        previous_gate = len(tasks)
        tasks.append(task(unit, 0, range(first + 4, first + 8)))
    data = model(tasks, backend=backend, config={"ring_dep_pool": pool_capacity, "ring_task_window": 32})
    if pool_capacity == 64:
        with pytest.raises(RuntimeError, match="TMR stalled"):
            execute(data, backends[backend])
        return
    total, rows = execute(data, backends[backend])
    assert total == pytest.approx(1.0)
    assert len(rows) == len(tasks)


@pytest.mark.parametrize("backend", BACKENDS)
def test_partial_mix_completion_waits_for_both_subslots(backends, backend):
    mixed = task(0, 10)
    mixed["mask"] = 3
    mixed["durations_ticks"] = [10, 30, 0]
    data = model([mixed, task(1, 7, [0])], backend=backend)
    total, rows = execute(data, backends[backend])
    assert total == pytest.approx(0.037)
    successor = next(row for row in rows if row["task_id"] == "1")
    assert successor["start_ticks"] == 30


def test_tmr_allocating_output_requires_additional_events(tmp_path):
    records, graph, raw, deps = capture(tmp_path)
    raw["metadata"]["runtime"] = deps["runtime"] = "tensormap_and_ringbuffer"
    raw["scheduler_tasks"]["producer"] = "aicpu"
    deps["tasks"][0]["args"] = [{"type": "OUTPUT"}]
    records.write_text(json.dumps(raw))
    graph.write_text(json.dumps(deps))
    with pytest.raises(UnsupportedReplay, match="allocation/lifetime"):
        normalize_input(records, graph)


def capture(tmp_path):
    raw = {
        "chip_swimlane_level": 2,
        "metadata": {
            "runtime": "host_build_graph",
            "clock_freq_hz": 1_000_000_000,
            "num_cores": 3,
            "core_types": ["aic", "aiv", "aiv"],
        },
        "aicore_tasks": [[0, 0, 0, 100, 110, 0, 1], [0, 1, 1, 500, 520, 0, 1]],
        "scheduler_tasks": {"producer": "aicore", "records": [[0, 0, 90, 400, 1], [0, 1, 450, 700, 1]]},
        "task_transitions": {
            "pre_registered": True,
            "root_release_cycles": 0,
            "fragment_records": [[0, 0, 1, 90, None, None], [0, 1, 1, 450, None, None]],
        },
    }
    deps = {
        "runtime": "host_build_graph",
        "tasks": [{"task_id": str(i), "kernel_ids": [0, -1, -1], "block_num": 1} for i in range(2)],
        "edges": [{"pred": "0", "succ": "1"}],
    }
    records, graph = tmp_path / "records.json", tmp_path / "deps.json"
    records.write_text(json.dumps(raw))
    graph.write_text(json.dumps(deps))
    return records, graph, raw, deps


def test_missing_release_and_long_scheduler_completion_do_not_affect_ideal(tmp_path, backends):
    records, graph, raw, _ = capture(tmp_path)
    normalized, _, _ = normalize_input(records, graph)
    assert execute(normalized, backends[BACKENDS[0]])[0] == pytest.approx(0.03)
    raw["scheduler_tasks"]["records"][0][3] = 490
    records.write_text(json.dumps(raw))
    revised, _, _ = normalize_input(records, graph)
    assert revised["tasks"] == normalized["tasks"]


@pytest.mark.parametrize(
    "defect", ["cycle", "missing_execution", "wrong_runtime", "multiblock", "drop", "control_execution", "edge_flags"]
)
def test_reject_incomplete_or_unrepresentable_input(tmp_path, defect):
    records, graph, raw, deps = capture(tmp_path)
    if defect == "cycle":
        deps["edges"].append({"pred": "1", "succ": "0"})
    elif defect == "missing_execution":
        deps["tasks"].append({"task_id": "2", "kernel_ids": [0, -1, -1], "block_num": 1})
    elif defect == "wrong_runtime":
        deps["runtime"] = "tensormap_and_ringbuffer"
    elif defect == "multiblock":
        deps["tasks"][0]["block_num"] = 2
    elif defect == "control_execution":
        deps["tasks"][0]["kernel_ids"] = [-1, -1, -1]
    elif defect == "edge_flags":
        deps["edges"][0]["flags"] = ["unknown"]
    else:
        raw["task_transitions"]["capture"] = {"complete": False}
    records.write_text(json.dumps(raw))
    graph.write_text(json.dumps(deps))
    with pytest.raises(UnsupportedReplay):
        normalize_input(records, graph)


def test_timeline_validation_detects_dependency_violation(backends):
    data = model([task(0, 10), task(1, 20, [0])])
    _, rows = execute(data, backends[BACKENDS[0]])
    bad = copy.deepcopy(rows)
    bad[1]["start_ticks"] = 0
    bad[1]["end_ticks"] = 20
    with pytest.raises(RuntimeError, match="dependency"):
        validate_timeline(data, bad)


def test_perfetto_comparison_preserves_source_and_synthetic_identity(tmp_path, backends):
    records, graph, _, _ = capture(tmp_path)
    original = records.read_bytes(), graph.read_bytes()
    result = replay(records, graph, tmp_path / "result")
    assert result["T_ideal_us"] == pytest.approx(0.03)
    trace = json.loads((tmp_path / "result/comparison.json").read_text())
    events = trace["traceEvents"]
    assert any(event.get("args", {}).get("synthetic") for event in events)
    assert any(event.get("ph") == "s" for event in events)
    assert (records.read_bytes(), graph.read_bytes()) == original
    assert len({event["pid"] // 100 for event in events}) == 2


def test_manifest_matrix_keeps_all_ids_and_rejects_source_mixing(tmp_path):
    manifest = {
        "platform": "a5",
        "runtime": "host_build_graph",
        "source_patch_sha256": "a",
        "cases": [],
        "case_results": {"C1-01": {"status": "passed", "diagnostic_paths": []}},
    }
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(manifest))
    manifest["source_patch_sha256"] = "b"
    second.write_text(json.dumps(manifest))
    report = replay_manifests([first, second], tmp_path / "results")
    assert len(report["results"]) == 41
    row = next(row for row in report["results"] if row["case_id"] == "C1-01")
    assert row["status"] == "unsupported"
    assert "incompatible" in row["reason"]
