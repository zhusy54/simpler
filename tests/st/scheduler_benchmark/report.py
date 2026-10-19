# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge timing, DAG and swimlane manifests into the fixed scheduler matrix report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from simpler_setup.tools.scheduler_metrics import (
    analyze,
    analyze_run,
    capture_overhead,
    distribution,
    markdown_report,
    scale_calibration,
)
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data
from tests.st.scheduler_benchmark.matrix import CASE_IDS, CaseSpec, is_long, task_roles, validate_deps
from tests.st.scheduler_benchmark.workloads import WORKLOADS

_ALL_IDS = (*CASE_IDS, *(w.case_id for w in WORKLOADS))
_ENV_KEYS = (
    "commit",
    "source_patch_sha256",
    "platform",
    "runtime",
    "soc",
    "device",
    "k_c",
    "k_v",
    "short_iterations",
    "long_iterations",
    "cann_version",
    "pto_isa_pin",
)


def _unmeasured(reason):
    return {"status": "unmeasured", "reason": reason}


def _per_task(data, epoch):
    tasks = {}
    for row in data["tasks"]:
        if row["run_epoch"] != epoch:
            continue
        task = tasks.setdefault(int(row["task_id"]), {"S": [], "E": [], "F": [], "R": [], "REG": []})
        for key, field in (
            ("S", "start_time_us"),
            ("E", "end_time_us"),
            ("F", "finish_time_us"),
            ("R", "ready_time_us"),
            ("REG", "registered_time_us"),
        ):
            if row.get(field) is not None:
                task[key].append(row[field])
    return tasks


def _case_metrics(data, deps, spec, run):
    tasks = _per_task(data, run["run_epoch"])
    ids = [int(t["task_id"]) for t in deps["tasks"]]
    spec = CaseSpec(**spec)
    if spec.family == 1:
        directions = defaultdict(list)
        roles = task_roles(spec)
        for index in range(1, len(ids)):
            source, target = tasks.get(ids[index - 1]), tasks.get(ids[index])
            if source and target:
                directions[f"{roles[index - 1]}_to_{roles[index]}"].append(min(target["S"]) - max(source["E"]))
        return {"family": "C1", "handoff_us": {key: distribution(values) for key, values in directions.items()}}
    if spec.family != 4:
        return {"family": "C2", "short_wait": run["phases"]["short_R_S"], "I_sched": run["I_sched"]}
    units = []
    stride = spec.producers + spec.consumers + 1
    for unit in range(spec.repeats):
        producers = [tasks.get(tid) for tid in ids[unit * stride : unit * stride + spec.producers]]
        consumers = [
            tasks.get(tid)
            for tid in ids[unit * stride + spec.producers : unit * stride + spec.producers + spec.consumers]
        ]
        result = {"unit": unit, "expected_consumers": spec.consumers}
        if (
            not all(producers)
            or not all(consumers)
            or not all(p["F"] for p in producers)
            or not all(c["R"] for c in consumers)
        ):
            units.append({**result, **_unmeasured("missing producer/consumer transitions")})
            continue
        last_f = max(max(p["F"]) for p in producers)
        first_e = min(max(p["E"]) for p in producers)
        first_s = min(min(p["S"]) for p in producers)
        pre_registered = data["runtime"] == "host_build_graph"
        registered = max(max(c["REG"]) for c in consumers) if all(c["REG"] for c in consumers) else None
        registration_valid = pre_registered or (registered is not None and registered < first_e)
        ready = [min(c["R"]) for c in consumers]
        valid = registration_valid and min(ready) >= last_f
        result.update(
            {
                "status": "measured" if valid else "unmeasured",
                "registration": "pre_registered_host_graph" if pre_registered else "device_registration",
                "registration_before_first_producer_end": registration_valid,
                "registration_before_first_producer_start": pre_registered
                or (registered is not None and registered < first_s),
                "actual_released_consumers": sum(r >= last_f for r in ready) if valid else None,
                "ready_spread_us": max(ready) - min(ready) if valid else None,
                "last_F_to_first_R_us": min(ready) - last_f if valid else None,
                "last_F_to_last_R_us": max(ready) - last_f if valid else None,
                "last_F_to_first_S_us": min(min(c["S"]) for c in consumers) - last_f if valid else None,
                "last_F_to_last_S_us": max(min(c["S"]) for c in consumers) - last_f if valid else None,
            }
        )
        units.append(result)
    trim = max(1, math.floor(spec.repeats * 0.1))
    middle = units[trim:-trim]
    eligible = [u for u in middle if u["status"] == "measured"]
    return {
        "family": "C3",
        "units": units,
        "middle_units": len(middle),
        "qualified_middle_units": len(eligible),
        "release_summary": {
            key: distribution([u[key] for u in eligible]) if len(eligible) == len(middle) else None
            for key in (
                "ready_spread_us",
                "last_F_to_first_R_us",
                "last_F_to_last_R_us",
                "last_F_to_first_S_us",
                "last_F_to_last_S_us",
            )
        },
    }


def _middle_metrics(data, deps, spec, run, labels):
    spec = CaseSpec(**spec)
    if spec.family == 3:
        return _unmeasured("independent roots use the full execution window")
    ids = [int(t["task_id"]) for t in deps["tasks"]]
    if spec.family in (1, 2):
        length = spec.length
        trim = max(1, math.floor(length * 0.1))
        width = spec.width if spec.family == 2 else 1
        selected = set(ids[trim * width : (length - trim) * width])
    else:
        trim = max(1, math.floor(spec.repeats * 0.1))
        stride = spec.producers + spec.consumers + 1
        selected = set(ids[trim * stride : (spec.repeats - trim) * stride - 1])
    fragments = [t for t in data["tasks"] if t["run_epoch"] == run["run_epoch"] and int(t["task_id"]) in selected]
    if not fragments:
        return _unmeasured("no middle-window execution events")
    window = (min(t["start_time_us"] for t in fragments), max(t["end_time_us"] for t in fragments))
    result = analyze_run(
        data,
        deps=deps,
        dag_verified=True,
        run_epoch=run["run_epoch"],
        labels=labels,
        window=window,
        selected_task_ids=selected,
    )
    result.pop("task_metrics", None)
    result.pop("critical_path", None)
    return result


def analyze_capture(records, *, case_id=None, spec=None, deps_path=None, device_us=None):
    records = Path(records)
    data = read_perf_data(records)
    deps_path = Path(deps_path) if deps_path else records.with_name("deps.json")
    deps = json.loads(deps_path.read_text()) if deps_path.exists() else None
    verified = False
    labels = None
    if spec is not None and deps is not None and case_id in CASE_IDS:
        case = CaseSpec(**spec)
        validate_deps(deps, case)
        verified = True
        roles = task_roles(case)
        labels = {
            int(task["task_id"]): "long" if is_long(case, index) else "short"
            for index, task in enumerate(deps["tasks"])
            if roles[index] != "dummy"
        }
    report = analyze(data, deps=deps, dag_verified=verified, device_us=device_us, labels=labels)
    report.update({"case_id": case_id, "records_path": str(records), "deps_path": str(deps_path) if deps else None})
    counts = data.get("raw_record_counts", {})
    report["record_volume"] = {"artifact_bytes": records.stat().st_size, "raw_record_counts": counts}
    if data.get("scheduler_task_producer") == "aicpu":
        baseline_bytes = 32 * (counts.get("scheduler", 0) + counts.get("executor", 0))
        added_bytes = 8 * counts.get("scheduler", 0)
        report["record_volume"].update(
            {
                "baseline_task_stream_bytes": baseline_bytes,
                "task_stream_bytes": baseline_bytes + added_bytes,
                "additional_task_stream_bytes": added_bytes,
                "additional_task_stream_fraction": added_bytes / baseline_bytes if baseline_bytes else None,
            }
        )
    else:
        report["record_volume"]["additional_native_trace_bytes"] = data.get("task_transition_metadata", {}).get(
            "additional_device_trace_bytes"
        )

    for run in report["runs"]:
        if verified:
            run["case_metrics"] = _case_metrics(data, deps, spec, run)
            run["middle"] = _middle_metrics(data, deps, spec, run, labels)
        durations = defaultdict(list)
        for task in data["tasks"]:
            if task["run_epoch"] == run["run_epoch"]:
                if labels is not None and int(task["task_id"]) not in labels:
                    durations[f"{task['core_type']}.control"].append(task["duration_us"])
                    continue
                durations[task["core_type"]].append(task["duration_us"])
                if labels and int(task["task_id"]) in labels:
                    durations[f"{task['core_type']}.{labels[int(task['task_id'])]}"].append(task["duration_us"])
        run["kernel_execution_us"] = {key: distribution(values) for key, values in durations.items()}
        for group, phases in run.get("phase_groups", {}).items():
            reference = run["kernel_execution_us"].get(group)
            if reference and reference["median"] > 0:
                for phase in phases.values():
                    stats = phase.get("distribution_us")
                    phase["median_over_kernel_median"] = stats["median"] / reference["median"] if stats else None
    trace_path = records.with_name("scheduler_stages.json")
    generate_chrome_trace_json(
        data["tasks"],
        trace_path,
        runtime_name=data["runtime"],
        scheduler_phases=data.get("scheduler_records"),
    )
    report["swimlane_path"] = str(trace_path)
    output = records.with_name("scheduler_metrics.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown_report(report))
    return output


def _paths(result, name):
    return [Path(path) for path in result.get("diagnostic_paths", []) if Path(path).name == name]


def build_report(manifests):
    grouped = defaultdict(list)
    for source, manifest in manifests:
        specs = {s["case_id"]: s for s in manifest["cases"]}
        for case_id, result in manifest["case_results"].items():
            key = (manifest["platform"], manifest["runtime"], case_id, manifest.get("task_scale", 1))
            grouped[key].append((source, manifest, specs.get(case_id), result))
    rows = []
    for (platform, runtime, case_id, scale), inputs in sorted(grouped.items()):
        environments = {tuple(m.get(k) for k in _ENV_KEYS) for _, m, _, _ in inputs}
        compatible = len(environments) == 1 and all(m.get("source_patch_sha256") for _, m, _, _ in inputs)
        row = {
            "platform": platform,
            "runtime": runtime,
            "case_id": case_id,
            "task_scale": scale,
            "spec": inputs[0][2],
            "sources": [str(source) for source, _, _, _ in inputs],
            "statuses": [r["status"] for _, _, _, r in inputs],
            "comparison_compatible": compatible,
        }
        timings = [
            t["device_us"]
            for _, _, _, r in inputs
            if r.get("measurement_mode") == "timing" and r["status"] == "passed"
            for t in r.get("rounds", [])
        ]
        row["environment"] = {key: inputs[0][1].get(key) for key in _ENV_KEYS} if compatible else None
        row["timing"] = (
            {"status": "measured", "device_us": distribution(timings), "samples_us": timings}
            if timings and compatible
            else _unmeasured("no compatible no-diagnostic timing samples")
        )
        if timings and compatible:
            row["timing"]["throughput_tasks_per_us"] = (
                sum(role != "dummy" for role in task_roles(CaseSpec(**row["spec"]))) / distribution(timings)["median"]
                if case_id in CASE_IDS
                else None
            )
        deps = [p for _, _, _, r in inputs if r["status"] == "passed" for p in _paths(r, "deps.json")]
        analyses = []
        diagnostic_summary = []
        for _, _, spec, result in inputs:
            if result["status"] != "passed":
                continue
            for records in _paths(result, "chip_swimlane_records.json"):
                try:
                    output = analyze_capture(
                        records,
                        case_id=case_id,
                        spec=spec if compatible else None,
                        deps_path=deps[-1] if deps and compatible else None,
                        device_us=result.get("diagnostic_device_us"),
                    )
                    analyses.append(str(output))
                    artifact = json.loads(output.read_text())
                    for run in artifact["runs"]:
                        diagnostic_summary.append(
                            {key: value for key, value in run.items() if key not in ("task_metrics", "phase_groups")}
                        )
                        critical = diagnostic_summary[-1].get("critical_path", {})
                        diagnostic_summary[-1]["critical_path"] = {
                            key: value
                            for key, value in critical.items()
                            if key not in ("observed_segments", "static_task_ids")
                        }
                        diagnostic_summary[-1]["record_volume"] = artifact.get("record_volume")
                        diagnostic_summary[-1]["swimlane_path"] = artifact.get("swimlane_path")
                except (ValueError, OSError, AssertionError) as exc:
                    analyses.append({"status": "unmeasured", "reason": str(exc), "records": str(records)})
        row["diagnostic_reports"] = analyses
        row["diagnostic_summary"] = diagnostic_summary
        rows.append(row)
    environments = {(r["platform"], r["runtime"]) for r in rows}
    for platform, runtime in sorted(environments):
        present = {r["case_id"] for r in rows if r["platform"] == platform and r["runtime"] == runtime}
        for case_id in _ALL_IDS:
            if case_id not in present:
                rows.append(
                    {
                        "platform": platform,
                        "runtime": runtime,
                        "case_id": case_id,
                        "task_scale": 1,
                        "spec": None,
                        "sources": [],
                        "statuses": ["not_run"],
                        "comparison_compatible": False,
                        "timing": _unmeasured("not run"),
                        "diagnostic_reports": [],
                    }
                )
    rows.sort(key=lambda row: (row["platform"], row["runtime"], row["case_id"], row["task_scale"]))
    calibration = []
    by_shape = defaultdict(list)
    for row in rows:
        if row["case_id"] not in CASE_IDS or row["timing"]["status"] != "measured":
            continue
        spec = row["spec"]
        x = (
            spec["repeats"]
            if spec["family"] == 4
            else spec["length"]
            if spec["family"] in (1, 2)
            else spec["task_count"]
        )
        by_shape[(row["platform"], row["runtime"], row["case_id"], tuple(row["environment"].values()))].append(
            (x, row["timing"]["device_us"]["median"])
        )
    for key, points in by_shape.items():
        calibration.append({"platform": key[0], "runtime": key[1], "case_id": key[2], **scale_calibration(points)})
    return {
        "schema_version": 1,
        "case_ids": list(_ALL_IDS),
        "missing_case_ids": sorted(set(_ALL_IDS) - {r["case_id"] for r in rows if r["sources"]}),
        "results": rows,
        "scale_calibration": calibration,
        "replay": {"status": "not_requested", "tool": "simpler_setup.tools.scheduler_replay"},
    }


def write_paired_swimlanes(report, directory):
    groups = defaultdict(list)
    for row in report["results"]:
        samples = row.get("diagnostic_summary", [])
        if row["comparison_compatible"] and samples and samples[0].get("swimlane_path"):
            groups[(row["case_id"], row["task_scale"])].append((row, samples[0]))
    paired = []
    for (case_id, scale), inputs in groups.items():
        if len(inputs) < 2:
            continue
        events = []
        sources = []
        for index, (row, sample) in enumerate(inputs):
            trace = json.loads(Path(sample["swimlane_path"]).read_text())
            timed = [event["ts"] for event in trace["traceEvents"] if "ts" in event]
            origin = min(timed, default=0)
            label = f"{row['platform']}/{row['runtime']}"
            for event in trace["traceEvents"]:
                event = dict(event)
                event["pid"] = (index + 1) * 100 + int(event.get("pid", 0))
                if "ts" in event:
                    event["ts"] -= origin
                if event.get("name") == "process_name":
                    event["args"] = {"name": label + ": " + event["args"]["name"]}
                for key in ("id", "bind_id"):
                    if key in event:
                        event[key] = f"{index}:{event[key]}"
                events.append(event)
            sources.append({"label": label, "source": sample["swimlane_path"], "rebase_us": origin})
            sources[-1]["environment"] = row.get("environment")
            sources[-1]["expanded_case"] = row.get("spec")
        path = directory / f"comparison_{case_id}_scale{scale}.json"
        path.write_text(
            json.dumps(
                {
                    "traceEvents": events,
                    "metadata": {"sources": sources, "alignment": "each capture's earliest recorded device event"},
                }
            )
            + "\n"
        )
        paired.append({"case_id": case_id, "task_scale": scale, "path": str(path), "sources": sources})
    return paired


def profiling_overhead_report(baselines, candidates):
    from tests.st.scheduler_benchmark.run_cases import _round_timings  # noqa: PLC0415

    groups = defaultdict(lambda: defaultdict(list))
    timings = defaultdict(lambda: defaultdict(list))
    builds = defaultdict(set)
    for kind, manifests in (("baseline", baselines), ("candidate", candidates)):
        for _, manifest in manifests:
            key_fields = tuple(
                manifest.get(k)
                for k in ("platform", "runtime", "soc", "device", "k_c", "k_v", "pto_isa_pin", "cann_version")
            )
            for case_id, result in manifest["case_results"].items():
                if result["status"] != "passed":
                    continue
                spec = next((case for case in manifest["cases"] if case["case_id"] == case_id), {})
                workload = (
                    manifest.get("task_scale", 1),
                    json.dumps(spec, sort_keys=True),
                    manifest.get("short_iterations"),
                    manifest.get("long_iterations"),
                )
                key = (key_fields, case_id, workload)
                builds[(key, kind)].add((manifest.get("commit"), manifest.get("source_patch_sha256")))
                if result.get("measurement_mode") == "timing":
                    timings[key][kind + "_timing"].extend(r["device_us"] for r in result.get("rounds", []))
                elif _paths(result, "chip_swimlane_records.json"):
                    level = manifest.get("swimlane_level")
                    if level is None:
                        try:
                            level = json.loads(_paths(result, "chip_swimlane_records.json")[0].read_text()).get(
                                "chip_swimlane_level"
                            )
                        except (OSError, ValueError):
                            pass
                    group = groups[(key, level)]
                    value = result.get("diagnostic_device_us")
                    if value is None and result.get("log_path"):
                        try:
                            value = _round_timings(Path(result.get("host_log_directory") or result["log_path"]), 1)[0][
                                "device_us"
                            ]
                        except (OSError, ValueError):
                            pass
                    if value is not None:
                        group[kind + "_diagnostic"].append(value)
    output = []
    for (key, level), samples in groups.items():
        identity, case_id, workload = key
        samples.update(timings[key])
        if not samples["baseline_timing"] and not samples["baseline_diagnostic"]:
            continue
        metric = capture_overhead(
            samples["candidate_timing"], samples["baseline_diagnostic"], samples["candidate_diagnostic"]
        )
        if level is None or any(len(builds[(key, kind)]) > 1 for kind in ("baseline", "candidate")):
            metric = _unmeasured("unknown capture level or mixed builds within a comparison arm")
        old = distribution(samples["baseline_timing"])
        new = distribution(samples["candidate_timing"])
        output.append(
            {
                "platform": identity[0],
                "runtime": identity[1],
                "device": identity[3],
                "case_id": case_id,
                "task_scale": workload[0],
                "swimlane_level": level,
                **metric,
                "disabled_baseline": old,
                "disabled_candidate": new,
                "disabled_change_fraction": (new["median"] - old["median"]) / old["median"] if old and new else None,
                "samples_us": dict(samples),
            }
        )
    return output


def report_markdown(report):
    lines = [
        "# Runtime scheduling benchmark",
        "",
        "| Platform | Runtime | Case | Scale | Timing samples | Median us | p95 us | Diagnostics |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        timing = row["timing"].get("device_us", {})
        lines.append(
            f"| {row['platform']} | {row['runtime']} | {row['case_id']} | {row['task_scale']} "
            f"| {timing.get('count', 0)} | {timing.get('median', 'unmeasured')} "
            f"| {timing.get('p95', 'unmeasured')} | {len(row['diagnostic_reports'])} |"
        )
    lines.extend(
        [
            "",
            "Diagnostic reference (first captured run per configuration; timing above uses no diagnostics):",
            "",
            "| Platform | Runtime | Case | Scheduler | U_AIC | U_AIV | I_sched | Ready p95 | C3 qualified |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["results"]:
        samples = row.get("diagnostic_summary", [])
        if not samples:
            continue
        sample = samples[0]
        utilization = sample.get("utilization", {})
        idle = sample.get("I_sched", {})
        c3 = sample.get("case_metrics", {})
        qualified = (
            f"{c3.get('qualified_middle_units')}/{c3.get('middle_units')}" if c3.get("family") == "C3" else "n/a"
        )
        lines.append(
            f"| {row['platform']} | {row['runtime']} | {row['case_id']} | {sample.get('scheduler_location')} "
            f"| {utilization.get('aic', {}).get('fraction', 'unmeasured')} "
            f"| {utilization.get('aiv', {}).get('fraction', 'unmeasured')} "
            f"| {idle.get('any_idle_ready_fraction', 'unmeasured')} "
            f"| {idle.get('ready_depth', {}).get('p95', 'unmeasured')} | {qualified} |"
        )
    if report.get("paired_swimlanes"):
        lines.extend(["", "Paired Perfetto swimlanes:", ""])
        lines.extend(
            f"- {item['case_id']} scale {item['task_scale']}: [{Path(item['path']).name}]({Path(item['path']).name})"
            for item in report["paired_swimlanes"]
        )
    if report.get("profiling_overhead"):
        lines.extend(
            [
                "",
                "Profiling overhead (baseline and candidate on the same device):",
                "",
                "| Runtime | Case | Level | Added us | Added / existing capture | Total / no capture | Samples |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["profiling_overhead"]:
            lines.append(
                f"| {row['runtime']} | {row['case_id']} | {row['swimlane_level']} "
                f"| {row.get('increment_us', 'unmeasured')} "
                f"| {row.get('increment_fraction_of_existing', 'unmeasured')} "
                f"| {row.get('total_fraction_of_baseline', 'unmeasured')} "
                f"| {row.get('extended', {}).get('count', 0)} |"
            )
    lines.extend(["", "Missing configuration IDs: " + ", ".join(report["missing_case_ids"]), ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overhead-baseline", type=Path, action="append", default=[])
    parser.add_argument("--replay-output", type=Path, help="also run the native host scheduler replay")
    args = parser.parse_args(argv)
    manifests = [(path, json.loads(path.read_text())) for path in args.manifests]
    report = build_report(manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["paired_swimlanes"] = write_paired_swimlanes(report, args.output.parent)
    if args.replay_output:
        from simpler_setup.tools.scheduler_replay_matrix import replay_manifests  # noqa: PLC0415

        report["replay"] = replay_manifests(args.manifests, args.replay_output)
    if args.overhead_baseline:
        baselines = [(path, json.loads(path.read_text())) for path in args.overhead_baseline]
        report["profiling_overhead"] = profiling_overhead_report(baselines, manifests)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(report_markdown(report))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
