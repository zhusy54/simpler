# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Matrix reporting rejects mixed builds and preserves unmeasured configurations."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.st.scheduler_benchmark.matrix import expand_case, scale_case
from tests.st.scheduler_benchmark.report import build_report
from tests.st.scheduler_benchmark.run_cases import _round_timings, _with_host_log_directory


def _manifest(scale=1, device_us=110, source_hash="same"):
    spec = scale_case(expand_case("C1-01", 32, 64), scale)
    return {
        "commit": "test",
        "source_patch_sha256": source_hash,
        "platform": "a5",
        "runtime": "host_build_graph",
        "soc": "test",
        "device": 0,
        "k_c": 32,
        "k_v": 64,
        "short_iterations": 1,
        "long_iterations": 2048,
        "cann_version": "test",
        "pto_isa_pin": "test",
        "task_scale": scale,
        "cases": [spec.manifest()],
        "case_results": {
            "C1-01": {"status": "passed", "measurement_mode": "timing", "rounds": [{"device_us": device_us}]}
        },
    }


def test_report_has_all_41_ids_and_calibrates_fixed_cost():
    report = build_report([(Path("a"), _manifest()), (Path("b"), _manifest(scale=2, device_us=210))])
    assert len({row["case_id"] for row in report["results"]}) == 41
    assert len(report["missing_case_ids"]) == 40
    calibration = report["scale_calibration"][0]
    assert calibration["fixed_us"] == 10
    assert calibration["points"][1]["below_5_percent"] is True


def test_different_source_builds_are_not_merged():
    report = build_report([(Path("a"), _manifest()), (Path("b"), _manifest(source_hash="different"))])
    row = next(row for row in report["results"] if row["case_id"] == "C1-01")
    assert row["comparison_compatible"] is False
    assert row["timing"]["status"] == "unmeasured"


def test_report_does_not_infer_golden_validation_for_legacy_results():
    from tests.st.scheduler_benchmark.report import report_markdown  # noqa: PLC0415

    legacy = _manifest()
    current = _manifest()
    current["case_results"]["C1-01"]["golden_validation"] = "skipped"
    report = build_report([(Path("legacy"), legacy), (Path("current"), current)])
    row = next(row for row in report["results"] if row["case_id"] == "C1-01")
    assert row["golden_validation"] == ["skipped", "unrecorded"]
    assert "skipped, unrecorded" in report_markdown(report)


def test_scale_fit_does_not_mix_builds_between_scales():
    report = build_report([(Path("a"), _manifest()), (Path("b"), _manifest(scale=2, source_hash="different"))])
    assert all(row["status"] == "unmeasured" for row in report["scale_calibration"])


def test_scale_preserves_width_fanin_fanout_and_mix_ratio():
    for cid in ("C1-02", "C2-03", "C2-13", "C3-10"):
        base = expand_case(cid, 32, 64)
        scaled = scale_case(base, 2)
        assert scaled.width == base.width
        assert scaled.producers == base.producers
        assert scaled.consumers == base.consumers
        if base.family in (1, 2):
            assert scaled.length == base.length * 2
        else:
            assert scaled.repeats == base.repeats * 2


def test_diagnostic_host_logs_survive_process_exit(tmp_path):
    scene = tmp_path / "emit_trace.py"
    scene.write_text("""from _task_interface import _initialize_host_log, _emit_host_span, _flush_host_log
_initialize_host_log(25)
_emit_host_span('chip.run', 1, 0, 0, 100000, 3000, '')
_emit_host_span('chip.run.runner_run.device_wall', 1, 0, 2, 0, 2000, 'clk=dev')
assert _flush_host_log(1000)
""")
    for command in ([sys.executable, str(scene)], [sys.executable, "-c", scene.read_text()]):
        directory = tmp_path / ("inline" if command[1] == "-c" else "script")
        subprocess.run(_with_host_log_directory(command, directory), check=True)
        result = _round_timings(directory, 1)
        assert result[0]["device_us"] == 2
        assert result[0]["host_us"] == 3


def _diagnostic_manifest(level, device_us, source_hash="same"):
    manifest = _manifest(source_hash=source_hash)
    manifest["swimlane_level"] = level
    manifest["case_results"]["C1-01"] = {
        "status": "passed",
        "measurement_mode": "diagnostic",
        "diagnostic_device_us": device_us,
        "diagnostic_paths": ["capture/chip_swimlane_records.json"],
    }
    return manifest


def test_overhead_comparison_keeps_capture_levels_separate():
    from tests.st.scheduler_benchmark.report import profiling_overhead_report  # noqa: PLC0415

    baseline = [(Path("old_time"), _manifest()), (Path("old_diag"), _diagnostic_manifest(2, 120))]
    candidate = [
        (Path("time"), _manifest()),
        (Path("diag2"), _diagnostic_manifest(2, 130)),
        (Path("diag3"), _diagnostic_manifest(3, 1000)),
    ]
    rows = profiling_overhead_report(baseline, candidate)
    level2 = next(row for row in rows if row["swimlane_level"] == 2)
    assert level2["increment_us"] == 10
    assert level2["extended"]["count"] == 1
    assert next(row for row in rows if row["swimlane_level"] == 3)["status"] == "unmeasured"


def test_overhead_comparison_rejects_mixed_candidate_builds():
    from tests.st.scheduler_benchmark.report import profiling_overhead_report  # noqa: PLC0415

    baseline = [(Path("old_time"), _manifest()), (Path("old_diag"), _diagnostic_manifest(2, 120))]
    candidate = [(Path("time"), _manifest()), (Path("diag"), _diagnostic_manifest(2, 130, source_hash="other"))]
    row = profiling_overhead_report(baseline, candidate)[0]
    assert row["status"] == "unmeasured"
    assert "increment_us" not in row


def test_report_rejects_mixed_execution_models():
    legacy = _manifest()
    persistent = _manifest()
    persistent["execution_model"] = "persistent_process_worker_per_phase_v1"
    row = build_report([(Path("legacy"), legacy), (Path("persistent"), persistent)])["results"][0]
    assert row["comparison_compatible"] is False
    assert row["timing"]["status"] == "unmeasured"


@pytest.mark.parametrize("identities", [[], [{}, {}], [{"run_epoch": 8}], [{"run_epoch": 9}] * 2])
def test_capture_identity_rejects_missing_mismatched_or_duplicate_epochs(identities):
    from tests.st.scheduler_benchmark.report import _validate_capture_identity  # noqa: PLC0415

    data = {"tasks": [{"run_epoch": 9}], "runtime": "host_build_graph"}
    with pytest.raises(ValueError, match="identity|mismatch"):
        _validate_capture_identity(data, identities, "host_build_graph")


def test_capture_identity_checks_runtime_and_accepts_matching_fragments():
    from tests.st.scheduler_benchmark.report import _validate_capture_identity  # noqa: PLC0415

    data = {"tasks": [{"run_epoch": 9}] * 4, "runtime": "host_build_graph"}
    _validate_capture_identity(data, [{"run_epoch": 9}], "host_build_graph")
    with pytest.raises(ValueError, match="Runtime mismatch"):
        _validate_capture_identity(data, [{"run_epoch": 9}], "tensormap_and_ringbuffer")


def test_single_capture_without_host_epoch_does_not_guess_inv_mapping():
    from tests.st.scheduler_benchmark.report import _validate_capture_identity  # noqa: PLC0415

    data = {"tasks": [{"run_epoch": 9}] * 4, "runtime": "host_build_graph"}
    identity = _validate_capture_identity(data, [{"pid": 77, "inv": 1}], "host_build_graph")
    check = identity["host_epoch_check"]
    assert isinstance(check, dict) and check["status"] == "unmeasured"
    assert identity["device_run_epoch"] == 9
    with pytest.raises(ValueError, match="ambiguous"):
        _validate_capture_identity({**data, "tasks": [{"run_epoch": 8}, {"run_epoch": 9}]}, [{}], "host_build_graph")
