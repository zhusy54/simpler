# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hand-calculated event timelines for the scheduling benchmark metrics."""

import copy

import pytest
from _capture_builder import Capture

from simpler_setup.tools.scheduler_metrics import analyze, capture_overhead, scale_calibration
from simpler_setup.tools.swimlane_converter import _decode_perf_data


def _row(tid, r, d, a, s, e, f, core=0, ty="aic"):
    return {
        "task_id": tid,
        "run_epoch": 1,
        "reg_task_id": tid,
        "core_id": core,
        "core_type": ty,
        "ready_time_us": r,
        "dispatch_time_us": d,
        "acquisition_time_us": a,
        "start_time_us": s,
        "end_time_us": e,
        "finish_time_us": f,
        "registered_time_us": 0.1,
        "gate_time_us": None,
    }


def _chain():
    data = {
        "tasks": [_row(0, 0.5, 1, 2, 3, 5, 6), _row(1, 8, 10, 11, 12, 14, 15)],
        "core_types": ["aic", "aic"],
        "scheduler_task_producer": "aicpu",
        "task_transition_metadata": {"capture": {"complete": True}},
    }
    deps = {
        "tasks": [{"task_id": i, "kernel_ids": [0, -1, -1], "block_num": 1} for i in range(2)],
        "edges": [{"pred": 0, "succ": 1}],
    }
    return data, deps


def test_serial_chain_stage_depth_idle_and_critical_path():
    data, deps = _chain()
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    assert run["T_exec_us"] == 11
    assert run["phases"]["E_F"]["distribution_us"]["median"] == 1
    assert run["phases"]["F_R"]["distribution_us"]["median"] == 2
    assert run["phases"]["R_D"]["distribution_us"]["median"] == 1.25
    assert run["phases"]["D_A"]["distribution_us"]["median"] == 1
    assert run["phases"]["A_S"]["distribution_us"]["median"] == 1
    assert run["I_sched"]["idle_ready_us"]["aic"] == 4
    assert run["I_data"]["idle_ready_us"]["aic"] == 7
    assert run["I_sched"]["ready_depth"]["mean"] == pytest.approx(4 / 11)
    assert run["utilization"]["aic"]["fraction"] == pytest.approx(2 / 11)
    path = run["critical_path"]
    assert path["static_duration_us"] == 4
    assert path["observed_contribution_us"] == {
        "execution": 4,
        "unclassified": 0,
        "E_F": 1,
        "F_R": 2,
        "R_D": 2,
        "D_A": 1,
        "A_S": 1,
    }
    assert sum(path["observed_contribution_us"].values()) == 11


def test_early_dispatch_keeps_parked_wait_and_omits_negative_R_D():
    data, deps = _chain()
    data["tasks"][1].update(dispatch_time_us=4, acquisition_time_us=4.5, gate_time_us=9)
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    phases = run["task_metrics"][1]["fragments"][0]
    assert phases["dispatch_mode"] == "early"
    assert "R_D_us" not in phases
    assert phases["A_S_us"] == 7.5
    assert run["phases"]["R_G"]["distribution_us"]["median"] == 1
    assert run["phases"]["gate_to_start"]["distribution_us"]["median"] == 3


def test_multi_core_completion_uses_last_required_slice():
    data, deps = _chain()
    data["tasks"].append(_row(0, 0.5, 1, 2, 3, 7, 7.5, core=1))
    deps["tasks"][0]["block_num"] = 2
    deps["tasks"][0]["sync_start"] = True
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    first = run["task_metrics"][0]
    assert first["E_us"] == 7
    assert first["F_us"] == 7.5
    assert run["phases"]["F_R"]["distribution_us"]["median"] == 0.5


def test_mix_requires_both_engines_and_enough_vector_cores():
    data, deps = _chain()
    data["core_types"] = ["aic", "aiv", "aiv"]
    data["tasks"] = [
        _row(0, 0.5, 1, 2, 3, 5, 6, core=0),
        _row(0, 0.5, 1, 2, 3, 9, 9.5, core=1, ty="aiv"),
        _row(0, 0.5, 1, 2, 3, 5, 6, core=2, ty="aiv"),
        _row(1, 7, 10, 11, 12, 14, 15, core=0),
        _row(1, 7, 10, 11, 12, 14, 15, core=1, ty="aiv"),
        _row(1, 7, 10, 11, 12, 14, 15, core=2, ty="aiv"),
    ]
    # Independent MIX task is Ready at 7; one required AIV remains busy until 9.
    deps["edges"] = []
    for node in deps["tasks"]:
        node["kernel_ids"] = [0, 1, 2]
        node["resource_affinity"] = [[0, 1, 2]]
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    assert run["I_sched"]["idle_ready_us"] == {"aic": 3, "aiv": 3}


def test_missing_ready_and_unverified_DAG_are_not_substituted():
    data, deps = _chain()
    data["tasks"][1]["ready_time_us"] = None
    run = analyze(data, deps=deps)["runs"][0]
    assert run["phases"]["F_R"]["status"] == "unmeasured"
    assert run["I_sched"]["status"] == "unmeasured"
    assert run["I_data"]["status"] == "unmeasured"
    assert run["critical_path"]["status"] == "unmeasured"
    assert run["phases"]["E_F"]["status"] == "measured"


def test_drop_counters_invalidate_aggregate_metrics():
    data, deps = _chain()
    data["task_transition_metadata"]["capture"]["complete"] = False
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    assert run["phases"]["E_F"]["status"] == "unmeasured"
    assert run["I_sched"]["status"] == "unmeasured"
    assert run["utilization"]["aic"]["status"] == "unmeasured"


def test_repeated_task_ids_are_partitioned_by_run():
    data, deps = _chain()
    more = copy.deepcopy(data["tasks"])
    for row in more:
        row["run_epoch"] = 2
    data["tasks"].extend(more)
    runs = analyze(data, deps=deps, dag_verified=True)["runs"]
    assert [r["run_epoch"] for r in runs] == [1, 2]
    assert all(r["coverage"]["observed_logical_tasks"] == 2 for r in runs)


def test_transition_reader_joins_by_epoch_core_and_dispatch_token():
    raw = (
        Capture(runtime="host_build_graph", core_types=["aic"], scheduler_producer="aicore")
        .task(task_id=7, reg_task_id=9, start=120, end=180, dispatch=110, finish=190, receive_to_start=5, run_epoch=3)
        .build()
    )
    raw["task_transitions"] = {"schema_version": 1, "fragment_records": [[0, 9, 3, 100, 90, None]]}
    decoded = _decode_perf_data(raw)
    row = decoded["tasks"][0]
    assert row["ready_time_us"] < row["dispatch_time_us"]
    assert row["registered_time_us"] < row["ready_time_us"]
    raw["task_transitions"]["fragment_records"].append([0, 9, 3, 100, 90, None])
    with pytest.raises(ValueError, match="duplicate task transition"):
        _decode_perf_data(raw)


def test_receive_delta_overflow_is_explicitly_unmeasured():
    raw = (
        Capture(runtime="host_build_graph", core_types=["aic"])
        .task(task_id=7, start=120, end=180, dispatch=110, finish=190)
        .build()
    )
    raw["aicore_tasks"][0][5] = 0xFFFFFFFF
    row = _decode_perf_data(raw)["tasks"][0]
    assert row["acquisition_time_us"] is None
    assert row["receive_delta_valid"] is False


def test_fixed_cost_fit_and_incremental_capture_overhead():
    fit = scale_calibration([(100, 110), (200, 210)])
    assert fit["fixed_us"] == 10
    assert fit["unit_us"] == 1
    assert fit["points"][0]["below_5_percent"] is False
    assert fit["points"][1]["below_5_percent"] is True
    assert scale_calibration([(100, 1)])["status"] == "unmeasured"
    overhead = capture_overhead([100, 102, 98], [110, 112, 108], [115, 117, 113])
    assert overhead["increment_us"] == 5
    assert overhead["total_fraction_of_baseline"] == 0.15


def test_mix_without_placement_has_depth_but_no_idle_attribution():
    data, deps = _chain()
    deps["tasks"][1]["kernel_ids"] = [0, 1, -1]
    data["core_types"] = ["aic", "aic", "aiv"]
    data["tasks"].append(_row(1, 8, 10, 11, 12, 14, 15, core=2, ty="aiv"))
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    assert run["I_sched"]["status"] == "unmeasured"
    assert run["I_sched"]["idle_ready_fraction"] is None
    assert run["I_sched"]["ready_depth"]["status"] == "measured"


def test_unassociated_multiple_fragments_keep_depth_but_not_idle_attribution():
    data, _ = _chain()
    fragment = copy.deepcopy(data["tasks"][1])
    fragment["core_id"] = 1
    data["tasks"].append(fragment)
    run = analyze(data)["runs"][0]
    assert run["I_sched"]["status"] == "unmeasured"
    assert run["I_sched"]["ready_depth"]["status"] == "measured"
    assert run["I_sched"]["idle_ready_us"] is None


def test_missing_control_marker_does_not_hide_complete_executor_occupancy():
    data, deps = _chain()
    deps["tasks"].append({"task_id": 2, "kernel_ids": [-1, -1, -1]})
    deps["edges"].append({"pred": 1, "succ": 2})
    run = analyze(data, deps=deps, dag_verified=True)["runs"][0]
    assert run["coverage"]["missing_task_ids"] == [2]
    assert run["utilization"]["aic"]["status"] == "measured"
    assert run["I_data"]["status"] == "unmeasured"
