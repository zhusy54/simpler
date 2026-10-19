# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host execution of runtime scheduling policies with a virtual execution clock."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from simpler_setup.environment import PROJECT_ROOT

from ._runtime_dispatch import normalize_task_id_int
from .swimlane_converter import generate_chrome_trace_json, read_perf_data


class UnsupportedReplay(ValueError):
    """Input cannot faithfully describe a supported scheduler backend."""


def _load(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_trace(path, trace):
    with Path(path).open("w") as stream:
        json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def _wait_edge(edge):
    return bool(_edge_flags(edge) & 1)


def _edge_flags(edge):
    if "flags" in edge:
        if not isinstance(edge["flags"], list) or set(edge["flags"]) - {"wait", "retain"}:
            raise UnsupportedReplay("unknown semantic dependency flags")
        return int("wait" in edge["flags"]) | (int("retain" in edge["flags"]) << 1)
    if "dep_flags" in edge:
        flags = int(edge["dep_flags"])
        if not 0 <= flags <= 3:
            raise UnsupportedReplay("unknown semantic dependency flags")
        return flags
    return 3


def normalize_input(records_path, deps_path, *, run_epoch=None, config=None):  # noqa: PLR0912, PLR0915
    """Retain exact device ticks and submission identities; never infer durations."""
    records_path, deps_path = Path(records_path), Path(deps_path)
    raw, deps = _load(records_path), _load(deps_path)
    data = read_perf_data(records_path)
    config = dict(config or {})
    allowed = {
        "arch",
        "scheduler_location",
        "scheduler_threads",
        "ring_task_window",
        "ring_dep_pool",
        "scheduler_order",
    }
    if set(config) - allowed:
        raise UnsupportedReplay("unknown replay configuration fields: " + ", ".join(sorted(set(config) - allowed)))
    if config.get("scheduler_order", "ascending") not in {"ascending", "descending"}:
        raise UnsupportedReplay("scheduler_order must be ascending or descending")
    runtime = data["runtime"]
    if deps.get("runtime") != runtime:
        raise UnsupportedReplay("DAG and execution runtime differ")
    capture = raw.get("task_transitions", {}).get("capture", {})
    if capture.get("complete") is False or capture.get("dropped", 0):
        raise UnsupportedReplay("capture is incomplete or has dropped fragments")
    if (
        capture.get("expected_fragments") is not None
        and capture.get("emitted_fragments") != capture["expected_fragments"]
    ):
        raise UnsupportedReplay("capture fragment counters disagree")
    if data.get("task_transition_errors") or data.get("task_transition_unmatched"):
        raise UnsupportedReplay("task transition association is invalid")
    epochs = sorted({int(row[6]) for row in raw["aicore_tasks"]})
    if run_epoch is None:
        if len(epochs) != 1:
            raise UnsupportedReplay("select one run_epoch from a multi-run capture")
        run_epoch = epochs[0]
    if run_epoch not in epochs:
        raise UnsupportedReplay("selected run_epoch has no execution fragments")
    producer = data.get("scheduler_task_producer")
    if not producer:
        producer = config.get("scheduler_location")
    arch = config.get("arch")
    frequency = int(raw["metadata"]["clock_freq_hz"])
    if not arch:
        if frequency == 1_000_000_000:
            arch = "a5"
        elif frequency == 50_000_000:
            arch = "a2a3"
    if (arch, runtime, producer) not in {
        ("a5", "host_build_graph", "aicore"),
        ("a5", "tensormap_and_ringbuffer", "aicpu"),
        ("a2a3", "tensormap_and_ringbuffer", "aicpu"),
        ("a2a3", "host_build_graph", "aicpu"),
        ("a5", "host_build_graph", "aicpu"),
    }:
        raise UnsupportedReplay(f"no native replay backend for {arch}/{runtime}/{producer}")
    core_types = data["core_types"]
    counts = Counter(core_types)
    clusters = counts["aic"]
    if not clusters or counts["aiv"] != 2 * clusters or core_types != ["aic"] * clusters + ["aiv"] * (2 * clusters):
        raise UnsupportedReplay("backend requires the native blocked 1 AIC / 2 AIV cluster topology")
    nodes = deps.get("tasks", [])
    ids = [normalize_task_id_int(node["task_id"]) for node in nodes]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise UnsupportedReplay("DAG has missing or duplicate task identities")
    index = {tid: position for position, tid in enumerate(ids)}
    predecessors = defaultdict(dict)
    for edge in deps.get("edges", []):
        pred, succ = (normalize_task_id_int(edge[key]) for key in ("pred", "succ"))
        if pred not in index or succ not in index:
            raise UnsupportedReplay("dependency references an unknown task")
        flags = _edge_flags(edge)
        if flags and (runtime == "tensormap_and_ringbuffer" or flags & 1):
            if index[pred] >= index[succ]:
                raise UnsupportedReplay("DAG is cyclic or inconsistent with submission order")
            predecessors[succ][index[pred]] = predecessors[succ].get(index[pred], 0) | flags
    ticks = {(int(row[6]), int(row[0]), int(row[2])): int(row[4]) - int(row[3]) for row in raw["aicore_tasks"]}
    fragments = defaultdict(list)
    for row in data["tasks"]:
        if row["run_epoch"] != run_epoch:
            continue
        tid = int(row["task_id"])
        if tid not in index:
            raise UnsupportedReplay("execution contains a task absent from the DAG")
        fragments[tid].append(row)
    tasks = []
    for tid, node in zip(ids, nodes):
        if int(node.get("block_num", 1)) != 1 or node.get("sync_start"):
            raise UnsupportedReplay("multi-block/sync-start task needs logical block identities and a matching backend")
        if node.get("early_dispatch"):
            raise UnsupportedReplay("early-dispatch attributes are not yet represented in the replay input")
        if node.get("resource_affinity"):
            raise UnsupportedReplay("custom resource affinity needs an explicit native topology adapter")
        if node.get("task_kind") in {"graph", "GRAPH", "async", "ASYNC"} or node.get("graph_definition"):
            raise UnsupportedReplay("Graph/async control work needs execution-instance and completion metadata")
        if runtime == "tensormap_and_ringbuffer" and any(arg.get("type") == "OUTPUT" for arg in node.get("args", [])):
            raise UnsupportedReplay("TMR output-buffer allocations require an allocation/lifetime event stream")
        kernels = node.get("kernel_ids")
        if not isinstance(kernels, list) or len(kernels) != 3:
            raise UnsupportedReplay("DAG is missing the three kernel subslots")
        active = [sub for sub, kernel in enumerate(kernels) if int(kernel) >= 0]
        observed = fragments[tid]
        durations = [0, 0, 0]
        for sub in active:
            ty = "aic" if sub == 0 else "aiv"
            candidates = [row for row in observed if row["core_type"] == ty]
            if len(candidates) != 1:
                raise UnsupportedReplay("fragment subslot identity is missing or ambiguous")
            row = candidates[0]
            if len([s for s in active if (s == 0) == (sub == 0)]) > 1:
                raise UnsupportedReplay("two AIV subslots require an explicit logical subslot mapping")
            duration = ticks[(run_epoch, row["core_id"], row["reg_task_id"])]
            if duration <= 0:
                raise UnsupportedReplay("executed fragment has no positive S-to-E duration")
            durations[sub] = duration
        if len(observed) != len(active):
            raise UnsupportedReplay("execution fragment count differs from the DAG resource shape")
        if not active and any(int(kernel) >= 0 for kernel in kernels):
            raise UnsupportedReplay("invalid control task")
        tasks.append(
            {
                "task_id": str(tid),
                "mask": sum(1 << s for s in active),
                "durations_ticks": durations,
                "predecessors": sorted(predecessors[tid]),
                "dependency_flags": [predecessors[tid][pred] for pred in sorted(predecessors[tid])],
                "scope": node.get("scope"),
            }
        )
    mapping = data.get("core_to_thread", [])
    owners = {int(owner) for owner in mapping if int(owner) >= 0}
    observed_threads = max(owners) + 1 if owners else None
    threads = int(config.get("scheduler_threads", observed_threads or min(clusters, 4 if arch == "a5" else 3)))
    if producer == "aicore":
        if "scheduler_threads" in config and threads != clusters:
            raise UnsupportedReplay("native HBG AICore scheduling requires one logical scheduler per cluster")
        threads = clusters
    if observed_threads and threads != observed_threads:
        raise UnsupportedReplay("scheduler thread configuration differs from capture topology")
    if not 1 <= threads <= clusters:
        raise UnsupportedReplay("invalid scheduler thread count")
    return (
        {
            "schema_version": 1,
            "arch": arch,
            "runtime": runtime,
            "scheduler_location": producer,
            "clock_freq_hz": frequency,
            "run_epoch": run_epoch,
            "clusters": clusters,
            "scheduler_threads": threads,
            "tasks": tasks,
            "scheduler_config_source": "native_cluster_topology"
            if producer == "aicore"
            else "explicit"
            if "scheduler_threads" in config
            else "capture"
            if observed_threads
            else "runtime_default_assumption",
            "config": config,
            "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (records_path, deps_path)},
            "assumptions": [
                "captured S-to-E durations remain fixed after rescheduling",
                "complete semantic DAG is supplied at zero host/device orchestration cost",
                "logical cores use the native blocked cluster topology",
                "scheduler transitions are instantaneous even on a computing AIV scheduler core",
                "TMR input models explicit dependency lifetimes; output-buffer allocations require additional metadata",
            ],
        },
        data,
        deps,
    )


def native_input(model):
    config = model.get("config", {})
    lines = [
        f"4 {model['clusters']} {model['scheduler_threads']} {len(model['tasks'])} "
        f"{int(config.get('ring_task_window', 16384))} {int(config.get('ring_dep_pool', 16384))} "
        f"{int(config.get('scheduler_order', 'ascending') == 'descending')}"
    ]
    for task in model["tasks"]:
        values = [
            task["task_id"],
            task["mask"],
            *task["durations_ticks"],
            len(task["predecessors"]),
        ]
        for pred, flags in zip(task["predecessors"], task.get("dependency_flags", [3] * len(task["predecessors"]))):
            values.extend((pred, flags))
        lines.append(" ".join(map(str, values)))
    return "\n".join(lines) + "\n"


def build_backend(model, build_directory=None):
    root = Path(PROJECT_ROOT)
    source = root / "src/common/scheduler_replay"
    build = Path(build_directory) if build_directory else root / "build/scheduler_replay"
    if model["runtime"] == "host_build_graph":
        target = "replay_a5_hbg" if model["scheduler_location"] == "aicore" else f"replay_{model['arch']}_hbg_aicpu"
    else:
        target = f"replay_{model['arch']}_tmr"
    for command in (
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--target", target, "--parallel", "2"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"native replay build failed:\n{result.stdout}\n{result.stderr}")
    return build / target


def execute(model, backend):
    try:
        result = subprocess.run(
            [str(backend)], input=native_input(model), capture_output=True, text=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("native scheduler replay exceeded the 120 second host execution limit") from error
    if result.returncode:
        raise RuntimeError(f"native scheduler replay failed: {result.stderr.strip()}")
    lines = result.stdout.splitlines()
    if not lines or not lines[0].startswith("time "):
        raise RuntimeError("native backend did not return a replay timeline")
    rows = []
    factor = 1_000_000 / model["clock_freq_hz"]
    for line in lines[1:]:
        tid, sub, core, start, end, owner, dispatch, admission = map(int, line.split())
        rows.append(
            {
                "task_id": str(tid),
                "subslot": sub,
                "core_id": core,
                "core_type": "aic" if sub == 0 else "aiv",
                "start_ticks": start,
                "end_ticks": end,
                "start_time_us": start * factor,
                "end_time_us": end * factor,
                "duration_us": (end - start) * factor,
                "scheduler_id": owner,
                "dispatch_ticks": dispatch,
                "dispatch_time_us": dispatch * factor,
                "admission_ticks": admission,
                "admission_time_us": admission * factor,
            }
        )
    validate_timeline(model, rows)
    total = int(lines[0].split()[1])
    if total != max((row["end_ticks"] for row in rows), default=0):
        raise RuntimeError("backend completion boundary differs from its execution timeline")
    return total * factor, rows


def validate_timeline(model, rows):
    tasks = {task["task_id"]: task for task in model["tasks"]}
    by_task, by_core = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["task_id"] not in tasks or not 0 <= row["core_id"] < model["clusters"] * 3:
            raise RuntimeError("backend returned an unknown task/core")
        by_task[row["task_id"]].append(row)
        if tasks[row["task_id"]]["mask"]:
            by_core[row["core_id"]].append(row)
        expected = "aic" if row["core_id"] < model["clusters"] else "aiv"
        if row["core_type"] != expected:
            raise RuntimeError("backend violated a core-type constraint")
    completion = {}
    for task in model["tasks"]:
        fragments = by_task[task["task_id"]]
        expected = {s for s in range(3) if task["mask"] & (1 << s)} or {1}
        if Counter(row["subslot"] for row in fragments) != Counter(expected):
            raise RuntimeError("backend omitted or repeated an execution fragment")
        for row in fragments:
            if row["end_ticks"] - row["start_ticks"] != task["durations_ticks"][row["subslot"]]:
                raise RuntimeError("backend changed an execution duration")
        earliest = min(row["start_ticks"] for row in fragments)
        flags = task.get("dependency_flags", [3] * len(task["predecessors"]))
        if any(
            completion[model["tasks"][p]["task_id"]] > earliest
            for p, flag in zip(task["predecessors"], flags)
            if flag & 1
        ):
            raise RuntimeError("backend violated a dependency")
        completion[task["task_id"]] = max(row["end_ticks"] for row in fragments)
    for fragments in by_core.values():
        end = 0
        for row in sorted(fragments, key=lambda r: (r["start_ticks"], r["end_ticks"])):
            if row["start_ticks"] < end:
                raise RuntimeError("backend oversubscribed an execution core")
            end = row["end_ticks"]


def _trace_rows(rows, model, deps):
    nodes = {normalize_task_id_int(t["task_id"]): t for t in deps["tasks"]}
    converted = []
    completion = defaultdict(float)
    for row in rows:
        completion[row["task_id"]] = max(completion[row["task_id"]], row["end_time_us"])
    ready = {}
    for task in model["tasks"]:
        flags = task.get("dependency_flags", [3] * len(task["predecessors"]))
        ready[task["task_id"]] = max(
            (completion[model["tasks"][p]["task_id"]] for p, flag in zip(task["predecessors"], flags) if flag & 1),
            default=0,
        )
    for row in rows:
        item = dict(row)
        node = nodes[int(row["task_id"])]
        item.update(
            task_id=int(row["task_id"]),
            func_id=node["kernel_ids"][row["subslot"]],
            run_epoch=model["run_epoch"],
            reg_task_id=len(converted),
            dispatch_time_us=row["dispatch_time_us"],
            acquisition_time_us=row["start_time_us"],
            ready_time_us=max(ready[row["task_id"]], row["admission_time_us"]),
            finish_time_us=row["end_time_us"],
        )
        converted.append(item)
    return converted


def write_traces(directory, model, data, deps, rows):
    edges = defaultdict(list)
    for edge in deps.get("edges", []):
        if _wait_edge(edge):
            edges[normalize_task_id_int(edge["pred"])].append(normalize_task_id_int(edge["succ"]))
    paths = []
    actual = [row for row in data["tasks"] if row["run_epoch"] == model["run_epoch"]]
    for name, tasks in (("observed", actual), ("replay_swimlane", _trace_rows(rows, model, deps))):
        path = directory / f"{name}.json"
        generate_chrome_trace_json(tasks, path, runtime_name=model["runtime"], deps_edges=edges)
        paths.append(path)
    events = []
    for group, path in enumerate(paths):
        trace = _load(path)
        if group:
            trace["metadata"] = {
                "synthetic": True,
                "runtime": model["runtime"],
                "scheduler_location": model["scheduler_location"],
                "assumptions": model["assumptions"],
            }
            for row in rows:
                if not row["duration_us"]:
                    trace["traceEvents"].append(
                        {
                            "ph": "i",
                            "s": "t",
                            "pid": 7,
                            "tid": 0,
                            "ts": row["start_time_us"],
                            "cat": "control",
                            "name": f"Control task {row['task_id']}",
                            "args": {"task_id": row["task_id"], "synthetic": True},
                        }
                    )
            for event in trace["traceEvents"]:
                event.setdefault("args", {})["synthetic"] = True
                if event.get("name") == "process_name":
                    event["args"]["name"] = event["args"]["name"].replace(
                        "Observed scheduling", "Theoretical scheduling"
                    )
        _write_trace(path, trace)
        origin = min((row["start_time_us"] for row in actual), default=0) if group == 0 else 0
        for event in trace["traceEvents"]:
            event["pid"] = int(event.get("pid", 0)) + 100 * (group + 1)
            if "ts" in event:
                event["ts"] -= origin
            if event.get("name") == "process_name":
                event["args"]["name"] = ("Observed: " if group == 0 else "Zero overhead: ") + event["args"]["name"]
            for key in ("id", "bind_id"):
                if key in event:
                    event[key] = f"{group}:{event[key]}"
            if group:
                event.setdefault("args", {})["synthetic"] = True
            events.append(event)
    _write_trace(
        directory / "comparison.json",
        {
            "traceEvents": events,
            "metadata": {
                "observed_alignment": "first AICore S",
                "replay_alignment": "model origin t=0",
                "unit": "microseconds",
                "synthetic_replay": True,
            },
        },
    )


def _policy_fingerprint(model):
    root = Path(PROJECT_ROOT)
    digest = hashlib.sha256()
    for relative in (
        f"src/{model['arch']}/runtime/{model['runtime']}",
        f"src/{model['arch']}/platform/include",
        f"src/common/{model['runtime']}",
        "src/common/platform/include",
        "src/common/task_interface",
        "src/common/scheduler_replay",
    ):
        for path in sorted((root / relative).rglob("*")):
            if path.suffix in {".h", ".cpp", ".txt", ".in"}:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _statistics(model, rows, total):
    factor = 1_000_000 / model["clock_freq_hz"]
    path_lengths = []
    work = [0, 0]
    for task in model["tasks"]:
        flags = task.get("dependency_flags", [3] * len(task["predecessors"]))
        path_lengths.append(
            max((path_lengths[p] for p, flag in zip(task["predecessors"], flags) if flag & 1), default=0)
            + max(task["durations_ticks"])
        )
        work[0] += task["durations_ticks"][0]
        work[1] += sum(task["durations_ticks"][1:])
    lower = max(*path_lengths, work[0] / model["clusters"], work[1] / (2 * model["clusters"])) * factor
    busy = Counter()
    for row in rows:
        busy[row["core_id"]] += row["end_ticks"] - row["start_ticks"]
    return {
        "lower_bound_us": lower,
        "gap_to_lower_bound_us": total - lower,
        "per_core_utilization": {
            str(core): busy[core] * factor / total if total else 0 for core in range(model["clusters"] * 3)
        },
        "aic_utilization": work[0] * factor / (model["clusters"] * total) if total else 0,
        "aiv_utilization": work[1] * factor / (model["clusters"] * 2 * total) if total else 0,
    }


def replay(records, deps, output, *, run_epoch=None, config=None, build_directory=None):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "replay_summary.json", {"status": "running"})
    model, data, graph = normalize_input(records, deps, run_epoch=run_epoch, config=config)
    _write(directory / "replay_input.json", model)
    backend = build_backend(model, build_directory)
    total, rows = execute(model, backend)
    _write(directory / "replay_tasks.json", rows)
    decisions = []
    for row in rows:
        for kind, field in (("dispatch", "dispatch_ticks"), ("start", "start_ticks"), ("complete", "end_ticks")):
            decisions.append(
                {
                    "kind": kind,
                    "time_ticks": row[field],
                    "task_id": row["task_id"],
                    "subslot": row["subslot"],
                    "core_id": row["core_id"],
                    "scheduler_id": row["scheduler_id"],
                }
            )
    _write(directory / "replay_decisions.json", sorted(decisions, key=lambda d: (d["time_ticks"], d["core_id"])))
    write_traces(directory, model, data, graph, rows)
    actual = [row for row in data["tasks"] if row["run_epoch"] == model["run_epoch"]]
    window = max(row["end_time_us"] for row in actual) - min(row["start_time_us"] for row in actual)
    summary = {
        "status": "supported",
        "runtime": model["runtime"],
        "scheduler_location": model["scheduler_location"],
        "T_ideal_us": total,
        "T_exec_observed_us": window,
        "execution_window_gap_us": window - total,
        "task_count": len(model["tasks"]),
        "clock_freq_hz": model["clock_freq_hz"],
        "scheduler_configuration": {
            "threads": model["scheduler_threads"],
            "source": model["scheduler_config_source"],
            **model["config"],
        },
        "arbitration": (
            "due completions visible together; serial scheduler actors; submission advances on explicit handoff"
        ),
        "policy_source_sha256": _policy_fingerprint(model),
        "native_binary_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
        "validation": {"exact_fragment_durations": True, "dependency_order": True, "core_capacity": True},
        "baseline_reproduction": {"status": "not_modeled", "reason": "ideal model has no measured scheduler delays"},
        **_statistics(model, rows, total),
        "assumptions": model["assumptions"],
        "interpretation": "runtime policy with zero scheduling cost; not a globally optimal schedule",
        "comparison_swimlane": str(directory / "comparison.json"),
    }
    _write(directory / "replay_summary.json", summary)
    (directory / "replay_summary.md").write_text(
        f"# Scheduler replay\n\nRuntime: `{model['runtime']}` / `{model['scheduler_location']}`.\n\n"
        f"Zero-overhead completion: **{total:.6f} us**. Observed execution window: {window:.6f} us.\n\n"
        "Durations are captured AICore S-to-E intervals. All scheduling operations take zero virtual time. "
        "This is a policy-specific reference, not a global optimum or a measured speedup.\n\n"
        "[Comparison swimlane](comparison.json)\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="one raw capture with --deps, or benchmark manifests")
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-epoch", type=int)
    parser.add_argument("--config", type=Path, help="explicit replay topology and scheduler configuration")
    parser.add_argument("--case", action="append", default=[], help="select configuration IDs in manifest mode")
    parser.add_argument("--build-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        config = _load(args.config) if args.config else None
        if args.deps:
            if len(args.inputs) != 1 or args.case:
                parser.error("--deps requires exactly one raw capture and no --case")
            summary = replay(
                args.inputs[0],
                args.deps,
                args.output,
                run_epoch=args.run_epoch,
                config=config,
                build_directory=args.build_directory,
            )
        else:
            from .scheduler_replay_matrix import replay_manifests  # noqa: PLC0415

            summary = replay_manifests(
                args.inputs, args.output, cases=args.case, config=config, build_directory=args.build_directory
            )
    except UnsupportedReplay as error:
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {"status": "unsupported", "reason": str(error)}
        _write(args.output / "replay_summary.json", summary)
        print(json.dumps(summary))
        return 2
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {"status": "error", "reason": str(error)}
        _write(args.output / "replay_summary.json", summary)
        print(json.dumps(summary))
        return 1
    if "results" in summary:
        print(
            json.dumps(
                {
                    "report": str(args.output / "replay_matrix.json"),
                    "statuses": dict(Counter(row["status"] for row in summary["results"])),
                }
            )
        )
    else:
        print(
            json.dumps({key: summary[key] for key in ("status", "T_ideal_us", "comparison_swimlane") if key in summary})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
