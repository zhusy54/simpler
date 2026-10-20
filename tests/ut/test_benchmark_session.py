# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent cases must retain run attribution and stop a failed device lane."""

from argparse import Namespace
from pathlib import Path

import pytest

from tests.st.scheduler_benchmark.run_cases import _timings_from_lines, make_manifest
from tests.st.scheduler_benchmark.session import BenchmarkSession


class FakeLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lines = []
        self.drops = 0

    def checkpoint(self):
        return {"offsets": {"fake": len(self.lines)}, "drops": self.drops}

    def read_since(self, before):
        return self.lines[before["offsets"]["fake"] :], self.checkpoint()

    def emit(self, inv):
        common = f"[STRACE] v=1 pid=77 tid=77 inv={inv} hid=abc"
        self.lines.extend(
            [
                f"{common} depth=0 name=chip.run ts={inv * 10000} dur=3000 run_id=0 run_epoch={inv}\n",
                f"{common} depth=2 name=chip.run.runner_run.device_wall ts=0 dur=2000 clk=dev\n",
            ]
        )


@pytest.fixture
def session(tmp_path, monkeypatch):
    import simpler.worker as worker_module  # noqa: PLC0415

    import tests.st.scheduler_benchmark.session as module  # noqa: PLC0415

    monkeypatch.setattr(module, "HostLog", FakeLog)
    workers = []
    active = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            self.inv = 0
            self.registrations = 0
            self.runs = []
            self.closed = False
            workers.append(self)

        def init(self):
            pass

        def register(self, chip):
            self.registrations += 1
            return (id(self), self.registrations)

        def run(self, handle, args, config=None):
            assert handle[0] == id(self), "stale handle crossed Worker generations"
            self.runs.append(args)
            if active.get("device_error") == args:
                raise RuntimeError("device lane failed")
            self.inv += 1
            active["session"].log.emit(self.inv)

        def close(self):
            self.closed = True

    class FakeScene:
        def _run_and_validate_l2(self, worker, chip, case, *, rounds, output_prefix, **kwargs):
            for _ in range(rounds):
                worker.run(type(self)._st_l2_handle, case["name"])
            if active.get("compare_error") == case["name"]:
                raise AssertionError("wrong output")
            if output_prefix:
                Path(output_prefix, "chip_swimlane_records.json").write_text("{}")

    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    instance = BenchmarkSession("a5", "host_build_graph", 0, 32, 64, tmp_path / "session")
    active["session"] = instance
    for cid in ("W1-01", "W2-01"):
        instance.adapters[cid] = {"class": FakeScene, "chip": object(), "case": {"name": cid}, "sha256": "same"}
    yield instance, workers, active
    instance.close()


def manifest(*, rounds=3, swimlane=0):
    return make_manifest(
        Namespace(
            platform="a5",
            runtime="host_build_graph",
            device=0,
            case=["W1-01", "W2-01"],
            scale=1,
            rounds=rounds,
            capture_deps=False,
            chip_swimlane=swimlane,
        ),
        counts=(32, 64, "test"),
        metadata={"commit": "test", "source_patch_sha256": "same"},
    )


def test_same_callable_cases_share_worker_without_mixing_rounds(session, tmp_path):
    instance, workers, _ = session
    result = instance.run_phase(manifest(), tmp_path / "timing.json", exclude_first=True)
    assert result["exit_code"] == 0
    assert len(workers) == 1 and workers[0].closed
    assert workers[0].registrations == 1
    assert workers[0].runs == ["W1-01"] * 3 + ["W2-01"] * 3
    a, b = result["case_results"].values()
    assert [r["run_identity"]["inv"] for r in a["rounds"]] == [2, 3]
    assert [r["run_identity"]["inv"] for r in b["rounds"]] == [5, 6]
    assert a["first_round"]["run_identity"]["inv"] == 1
    assert sum(result["phase_ns"].values()) == result["elapsed_ns"]
    assert all(sum(row["phase_ns"].values()) == row["elapsed_ns"] for row in (a, b))


def test_phase_change_rebuilds_worker_and_clears_class_handles(session, tmp_path):
    instance, workers, _ = session
    instance.run_phase(manifest(), tmp_path / "timing.json")
    result = instance.run_phase(manifest(rounds=1, swimlane=3), tmp_path / "events.json")
    assert len(workers) == 2 and all(w.closed for w in workers)
    assert [w.registrations for w in workers] == [1, 1]
    assert all(r["worker_generation"] == 2 for r in result["case_results"].values())
    assert all(r["rounds"] == [] and r["diagnostic_device_us"] == 2 for r in result["case_results"].values())
    cls = instance.adapters["W1-01"]["class"]
    assert "_st_l2_handle" not in cls.__dict__


def test_device_error_stops_remaining_cases_and_later_phases(session, tmp_path):
    instance, workers, active = session
    active["device_error"] = "W1-01"
    result = instance.run_phase(manifest(), tmp_path / "timing.json")
    assert result["case_results"]["W1-01"]["status"] == "failed"
    assert result["case_results"]["W2-01"]["status"] == "not_run"
    result = instance.run_phase(manifest(rounds=1, swimlane=3), tmp_path / "events.json")
    assert len(workers) == 1
    assert all(r["status"] == "not_run" for r in result["case_results"].values())


def test_compare_error_keeps_healthy_worker_usable(session, tmp_path):
    instance, workers, active = session
    active["compare_error"] = "W1-01"
    result = instance.run_phase(manifest(), tmp_path / "timing.json")
    assert result["case_results"]["W1-01"]["status"] == "failed"
    assert result["case_results"]["W2-01"]["status"] == "passed"
    assert len(workers) == 1 and instance.aborted is None


def test_missing_device_span_is_not_a_measurement():
    log = FakeLog("unused")
    log.emit(1)
    with pytest.raises(ValueError, match="device-wall"):
        _timings_from_lines(iter(log.lines[:1]), 1)


def test_strict_child_buffer_release_attempts_all_frees():
    from simpler_setup.scene_test import ChildMemoryTaskArgs  # noqa: PLC0415

    class BrokenFree:
        def __init__(self):
            self.attempted = []

        def free(self, buffer):
            self.attempted.append(buffer)
            raise RuntimeError("free failed")

    worker = BrokenFree()
    args = ChildMemoryTaskArgs(worker)
    args.buffers.update(a=1, b=2)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        args.release(strict=True)
    assert worker.attempted == [2, 1]


def test_single_phase_rerun_isolates_artifacts(tmp_path, monkeypatch):
    import tests.st.scheduler_benchmark.run_cases as runner  # noqa: PLC0415
    import tests.st.scheduler_benchmark.session as session_module  # noqa: PLC0415

    directories = []

    class StubSession:
        def __init__(self, *args, **kwargs):
            directories.append(args[5])

        def run_phase(self, manifest, output):
            return {"exit_code": 0}

        def close(self):
            pass

    monkeypatch.setattr(session_module, "BenchmarkSession", StubSession)
    monkeypatch.setattr(runner, "_hardware_counts", lambda *args: (32, 64, "test"))
    monkeypatch.setattr(runner, "_metadata", lambda: {})
    argv = ["--platform", "a5", "--runtime", "host_build_graph", "--manifest", str(tmp_path / "matrix.json")]
    assert runner.main(argv) == runner.main(argv) == 0
    assert directories[0] != directories[1]


def test_workflow_records_preparation_failure(tmp_path, monkeypatch):
    import json  # noqa: PLC0415

    import tests.st.scheduler_benchmark.run_benchmark as runner  # noqa: PLC0415

    def fail():
        raise RuntimeError("cannot read source metadata")

    monkeypatch.setattr(runner, "_metadata", fail)
    path = tmp_path / "workflow.json"
    assert runner.main(["--platform", "a5sim", "--manifest", str(path)]) == 1
    result = json.loads(path.read_text())
    assert result["status"] == "failed" and "source metadata" in result["error"]
    assert sum(result["wall_components_ns"].values()) == result["elapsed_ns"]
