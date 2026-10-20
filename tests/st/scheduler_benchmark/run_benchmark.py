# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect one timing batch, separate DAG/events, then the fixed matrix report."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
import uuid
from pathlib import Path
from time import perf_counter_ns

from tests.st.scheduler_benchmark.run_cases import ALL_CASE_IDS, _metadata, make_manifest, prepare_sim_environment
from tests.st.scheduler_benchmark.session import BenchmarkSession, utc_now, write_json

_RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")


def _serve(connection, platform, runtime, device, counts, directory, scale):
    session = None
    try:
        session = BenchmarkSession(platform, runtime, device, *counts[:2], directory, task_scale=scale)
        connection.send({"pid": os.getpid(), "session_id": session.session_id})
        while True:
            request = connection.recv()
            if request is None:
                break
            manifest = session.run_phase(request["manifest"], request["output"], exclude_first=request["exclude_first"])
            connection.send({"exit_code": manifest["exit_code"], "unsafe_device": manifest["unsafe_device"]})
    except BaseException:
        try:
            connection.send(
                {"fatal": traceback.format_exc(), "unsafe_device": session is not None and session.worker is not None}
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if session is not None:
            session.close()
        connection.close()


def _receive(connection, process):
    while not connection.poll(1):
        if not process.is_alive():
            raise RuntimeError(f"benchmark process exited unexpectedly: {process.exitcode}")
    result = connection.recv()
    if result.get("fatal"):
        raise RuntimeError(result["fatal"])
    return result


def _render_report(paths, output):
    from simpler_setup.tools import deps_viewer  # noqa: PLC0415
    from tests.st.scheduler_benchmark.report import (  # noqa: PLC0415
        build_report,
        report_markdown,
        write_paired_swimlanes,
    )

    manifests = [(path, json.loads(path.read_text())) for path in paths]
    for _, manifest in manifests:
        for row in manifest["case_results"].values():
            for name in row.get("diagnostic_paths", []):
                path = Path(name)
                if path.name == "deps.json":
                    rc = deps_viewer.main([str(path), "--format", "text"])
                    if rc:
                        raise ValueError(f"dependency view generation failed: {path}")
    report = build_report(manifests)
    rows = {(row["runtime"], row["case_id"], row["task_scale"]): row for row in report["results"]}
    for path, manifest in manifests:
        for case_id, result in manifest["case_results"].items():
            if not any(Path(p).name == "chip_swimlane_records.json" for p in result.get("diagnostic_paths", [])):
                continue
            row = rows[(manifest["runtime"], case_id, manifest["task_scale"])]
            result["analysis_paths"] = row["diagnostic_reports"]
            samples = row["diagnostic_summary"]
            if samples:
                result["scheduler_location"] = samples[0].get("scheduler_location")
                result["kernel_execution_us"] = samples[0].get("kernel_execution_us")
        write_json(path, manifest)
    report["paired_swimlanes"] = write_paired_swimlanes(report, output.parent)
    write_json(output, report)
    output.with_suffix(".md").write_text(report_markdown(report))
    return report


def main(argv=None):
    start, started = perf_counter_ns(), utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3", "a5", "a2a3sim", "a5sim"), required=True)
    parser.add_argument("--runtime", action="append", choices=_RUNTIMES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", choices=ALL_CASE_IDS)
    parser.add_argument("--rounds", type=int, default=101)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.rounds < 2:
        parser.error("full workflow requires at least two timing rounds (first round is separate)")
    runtimes = args.runtime or list(_RUNTIMES)
    if len(set(runtimes)) != len(runtimes) or (args.case and len(set(args.case)) != len(args.case)):
        parser.error("duplicate Runtime or case")
    if args.scale < 1 or (args.scale != 1 and any(cid.startswith("W") for cid in (args.case or ALL_CASE_IDS))):
        parser.error("--scale must be positive and applies only to C configurations")
    root = (
        args.manifest.with_suffix("")
        if args.manifest
        else (
            Path(__file__).resolve().parents[3]
            / "outputs"
            / f"runtime_benchmark_{args.platform}_{uuid.uuid4().hex[:12]}"
        )
    )
    root.mkdir(parents=True, exist_ok=False)
    summary_path = args.manifest or root / "workflow.json"
    summary = {"schema_version": 1, "started_utc": started, "status": "running", "runtimes": {}, "phases": []}
    write_json(summary_path, summary)
    processes = {}
    paths = []
    failed, unsafe = False, False
    phase_args = argparse.Namespace(**vars(args))
    try:
        metadata = _metadata()
        # Query once before the persistent processes start and share the expansion.
        from tests.st.scheduler_benchmark.run_cases import _hardware_counts  # noqa: PLC0415

        counts = (
            (8, 16, "simulated")
            if args.platform.endswith("sim")
            else _hardware_counts(args.device, "a5" if args.platform == "a5" else "a2a3")
        )
        prepare_sim_environment(args.platform)
        context = multiprocessing.get_context("spawn")
        for runtime in runtimes:
            parent, child = context.Pipe()
            process = context.Process(
                target=_serve, args=(child, args.platform, runtime, args.device, counts, root / runtime, args.scale)
            )
            before, process_started = perf_counter_ns(), utc_now()
            process.start()
            child.close()
            processes[runtime] = (process, parent)
            hello = _receive(parent, process)
            summary["runtimes"][runtime] = {
                **hello,
                "started_utc": process_started,
                "start_offset_ns": before - start,
                "process_start_ns": perf_counter_ns() - before,
            }
        schedule = [(runtime, "timing") for runtime in runtimes]
        schedule += [(runtime, phase) for runtime in runtimes for phase in ("deps", "events")]
        for runtime, phase in schedule:
            phase_args.runtime = runtime
            phase_args.rounds = args.rounds if phase == "timing" else 1
            phase_args.capture_deps = phase == "deps"
            phase_args.chip_swimlane = 3 if phase == "events" else 0
            manifest = make_manifest(phase_args, counts=counts, metadata=metadata)
            path = root / runtime / f"{phase}.json"
            paths.append(path)
            write_json(path, manifest)
            if unsafe:
                continue
            process, connection = processes[runtime]
            before = perf_counter_ns()
            entry = {"runtime": runtime, "phase": phase, "started_utc": utc_now(), "manifest": str(path)}
            connection.send({"manifest": manifest, "output": str(path), "exclude_first": phase == "timing"})
            result = _receive(connection, process)
            failed |= bool(result["exit_code"])
            unsafe |= result["unsafe_device"]
            entry.update(ended_utc=utc_now(), elapsed_ns=perf_counter_ns() - before, **result)
            summary["phases"].append(entry)
            write_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as exc:
        failed = True
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for process, connection in processes.values():
            if process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=65)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    failed = True
            failed |= process.exitcode != 0
            connection.close()
    analysis_start = perf_counter_ns()
    try:
        if paths:
            report = _render_report(paths, root / "report.json")
            summary["report_path"] = str(root / "report.json")
            summary["paired_swimlanes"] = report["paired_swimlanes"]
    except Exception as exc:
        failed = True
        summary["analysis_error"] = f"{type(exc).__name__}: {exc}"
    summary["analysis_ns"] = perf_counter_ns() - analysis_start
    summary["status"] = "failed" if failed else "completed"
    summary["ended_utc"] = utc_now()
    summary["elapsed_ns"] = perf_counter_ns() - start
    phases_ns = sum(p["elapsed_ns"] for p in summary["phases"])
    summary["wall_components_ns"] = {
        "device_phases_and_ipc": phases_ns,
        "offline_analysis": summary["analysis_ns"],
        "startup_shutdown_and_control": summary["elapsed_ns"] - phases_ns - summary["analysis_ns"],
    }
    for runtime, row in summary["runtimes"].items():
        row["ended_utc"] = summary["ended_utc"]
        row["elapsed_ns"] = summary["elapsed_ns"] - row["start_offset_ns"]
        row["own_phases_ns"] = sum(p["elapsed_ns"] for p in summary["phases"] if p["runtime"] == runtime)
        row["waiting_and_shared_processing_ns"] = row["elapsed_ns"] - row["own_phases_ns"]
    write_json(summary_path, summary)
    print(summary_path, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
