# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pair fixed benchmark manifests for read-only offline scheduler replay."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tests.st.scheduler_benchmark.matrix import CASE_IDS, CaseSpec, task_roles, validate_deps
from tests.st.scheduler_benchmark.workloads import WORKLOAD_BY_ID

from .scheduler_replay import UnsupportedReplay, replay

_IDENTITY = (
    "platform",
    "runtime",
    "device",
    "soc",
    "commit",
    "source_patch_sha256",
    "pto_isa_pin",
    "cann_version",
    "k_c",
    "k_v",
    "short_iterations",
    "long_iterations",
    "task_iterations",
    "task_scale",
)


def replay_manifests(paths, output, *, cases=(), config=None, build_directory=None):  # noqa: PLR0912
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    unknown = set(cases) - set(CASE_IDS) - set(WORKLOAD_BY_ID)
    if unknown:
        raise ValueError("unknown benchmark configuration IDs: " + ", ".join(sorted(unknown)))
    groups = defaultdict(list)
    environments = set()
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        environments.add((manifest["platform"], manifest["runtime"]))
        specs = {spec["case_id"]: spec for spec in manifest["cases"]}
        for case_id, result in manifest["case_results"].items():
            if not cases or case_id in cases:
                key = manifest["platform"], manifest["runtime"], case_id, manifest.get("task_scale", 1)
                groups[key].append((manifest, result, specs.get(case_id)))
    results = []
    for (platform, runtime, case_id, scale), inputs in sorted(groups.items()):
        row = {"platform": platform, "runtime": runtime, "case_id": case_id, "task_scale": scale, "samples": []}
        results.append(row)
        identities = {tuple(manifest.get(field) for field in _IDENTITY) for manifest, _, _ in inputs}
        if len(identities) != 1 or any(not m.get("source_patch_sha256") for m, _, _ in inputs):
            row.update(status="unsupported", reason="manifest source/environment identities are incompatible")
            continue
        deps = [
            p
            for _, result, _ in inputs
            if result["status"] == "passed"
            for p in result.get("diagnostic_paths", [])
            if Path(p).name == "deps.json"
        ]
        captures = [
            (m, r, spec, p)
            for m, r, spec in inputs
            if r["status"] == "passed"
            for p in r.get("diagnostic_paths", [])
            if Path(p).name == "chip_swimlane_records.json"
        ]
        if not captures or not deps:
            row.update(
                status="unsupported" if captures else "not_run", reason="paired DAG and execution capture required"
            )
            continue
        if case_id not in CASE_IDS:
            row.update(
                status="unsupported", reason="W/Graph workload needs verified execution-instance DAG association"
            )
            continue
        for manifest, result, spec, records in captures:
            sample_directory = (
                directory / f"{platform}_{runtime}_{case_id}_scale{scale}" / f"sample_{len(row['samples'])}"
            )
            try:
                if spec is None:
                    raise UnsupportedReplay("expanded benchmark configuration is absent")
                graph = json.loads(Path(deps[-1]).read_text())
                case_spec = CaseSpec(**spec)
                validate_deps(graph, case_spec)
                dummy_ids = [
                    task["task_id"] for task, role in zip(graph["tasks"], task_roles(case_spec)) if role == "dummy"
                ]
                replay_config = {"arch": "a5" if platform.startswith("a5") else "a2a3", **(config or {})}
                sample = replay(
                    records,
                    deps[-1],
                    sample_directory,
                    config=replay_config,
                    build_directory=build_directory,
                    verified_dummy_task_ids=dummy_ids,
                )
                sample["T_device_diagnostic_us"] = result.get("diagnostic_device_us")
                sample["capture_environment"] = {field: manifest.get(field) for field in _IDENTITY}
                sample["dag_association"] = "validated fixed-matrix oracle"
                sample["no_diagnostic_samples_us"] = [
                    timing["device_us"]
                    for _, r, _ in inputs
                    if r.get("measurement_mode") == "timing" and r["status"] == "passed"
                    for timing in r.get("rounds", [])
                ]
            except (UnsupportedReplay, AssertionError, ValueError, OSError) as error:
                sample = {"status": "unsupported", "reason": str(error)}
            except (RuntimeError, TimeoutError) as error:
                sample = {"status": "error", "reason": str(error)}
            sample_directory.mkdir(parents=True, exist_ok=True)
            (sample_directory / "replay_summary.json").write_text(json.dumps(sample, indent=2) + "\n")
            row["samples"].append(sample)
        row["status"] = "supported" if all(s["status"] == "supported" for s in row["samples"]) else "incomplete"
    for platform, runtime in sorted(environments):
        present = {row["case_id"] for row in results if (row["platform"], row["runtime"]) == (platform, runtime)}
        for case_id in (*CASE_IDS, *WORKLOAD_BY_ID):
            if case_id not in present:
                results.append(
                    {
                        "platform": platform,
                        "runtime": runtime,
                        "case_id": case_id,
                        "status": "not_run",
                        "samples": [],
                        "task_scale": 1,
                    }
                )
    results.sort(key=lambda r: (r["platform"], r["runtime"], r["case_id"], r["task_scale"]))
    report = {
        "schema_version": 1,
        "results": results,
        "case_ids": [*CASE_IDS, *WORKLOAD_BY_ID],
        "inputs": [str(path) for path in paths],
    }
    (directory / "replay_matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Scheduler replay matrix",
        "",
        "| Platform | Runtime | Case | Status | Ideal us | Swimlane |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in results:
        samples = row["samples"] or [{}]
        for sample in samples:
            trace = sample.get("comparison_swimlane")
            link = (
                f"[Perfetto]({Path(trace).relative_to(directory)})"
                if trace
                else sample.get("reason", row.get("reason", ""))
            )
            lines.append(
                f"| {row['platform']} | {row['runtime']} | {row['case_id']} | {sample.get('status', row['status'])} "
                f"| {sample.get('T_ideal_us', '')} | {link} |"
            )
    (directory / "replay_matrix.md").write_text("\n".join(lines) + "\n")
    return report
