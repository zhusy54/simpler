# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observed scheduling metrics for the fixed runtime benchmark (no replay)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ._runtime_dispatch import normalize_task_id_int

_PHASES = ("E_F", "F_R", "R_D", "D_A", "R_A", "A_S")


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)],
        "min": values[0],
        "max": values[-1],
    }


def _missing(reason, **fields):
    return {"status": "unmeasured", "reason": reason, **fields}


def _samples(values, expected, *, trusted=True, reason="incomplete events"):
    complete = len(values) == expected and expected > 0 and trusted
    return {
        "status": "measured" if complete else "unmeasured",
        "reason": None if complete else reason,
        "expected_samples": expected,
        "valid_samples": len(values),
        "distribution_us": distribution(values) if complete else None,
        "available_samples_us": None if complete else distribution(values),
    }


def _weighted_distribution(segments):
    weights = Counter()
    for duration, value in segments:
        if duration > 0:
            weights[value] += duration
    total = sum(weights.values())
    if total <= 0:
        return None
    ordered = sorted(weights.items())

    def quantile(q):
        acc = 0
        for value, duration in ordered:
            acc += duration
            if acc >= q * total:
                return value
        return ordered[-1][0]

    return {
        "mean": sum(v * d for v, d in ordered) / total,
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "max": ordered[-1][0],
    }


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _logical_tasks(fragments, controls):
    grouped = defaultdict(list)
    for fragment in fragments:
        grouped[int(fragment["task_id"])].append(fragment)
    tasks = {}
    for task_id, rows in grouped.items():
        ready = {r.get("ready_time_us") for r in rows}
        registered = {r.get("registered_time_us") for r in rows}
        tasks[task_id] = {
            "task_id": task_id,
            "rows": rows,
            "S": min(r["start_time_us"] for r in rows),
            "E": max(r["end_time_us"] for r in rows),
            "F": max(r["finish_time_us"] for r in rows)
            if all(r.get("finish_time_us") is not None for r in rows)
            else None,
            "R": next(iter(ready)) if len(ready) == 1 else None,
            "registered": next(iter(registered)) if len(registered) == 1 else None,
            "control": False,
        }
    for control in controls:
        task_id = int(control["task_id"])
        if task_id not in tasks:
            tasks[task_id] = {
                "task_id": task_id,
                "rows": [],
                "S": None,
                "E": None,
                "F": control["end_time_us"],
                "R": control["start_time_us"],
                "registered": None,
                "control": True,
            }
    return tasks


def _graph(deps):
    nodes = {normalize_task_id_int(t["task_id"]): t for t in (deps or {}).get("tasks", [])}
    pred = defaultdict(set)
    succ = defaultdict(set)
    for edge in (deps or {}).get("edges", []):
        if edge.get("dep_flags") is not None and not int(edge["dep_flags"]) & 1:
            continue
        source, target = (normalize_task_id_int(edge[key]) for key in ("pred", "succ"))
        pred[target].add(source)
        succ[source].add(target)
    return nodes, pred, succ


def _resource_shape(node, rows):
    kernels = node.get("kernel_ids")
    if kernels is None:
        return tuple(Counter(r["core_type"] for r in rows).get(t, 0) for t in ("aic", "aiv"))
    blocks = int(node.get("block_num", 1)) if node.get("sync_start") else 1
    return (blocks * int(kernels[0] >= 0), blocks * sum(int(k >= 0) for k in kernels[1:]))


def _idle_and_depth(tasks, nodes, core_types, window, start_field):  # noqa: PLR0912
    start, end = window
    events = defaultdict(list)
    counts = Counter(core_types)
    resource_error = None
    # A queue entry survives dispatch/acquisition until the task actually starts.
    for task in tasks.values():
        if task["control"]:
            continue
        ready = task.get(start_field)
        if ready is None:
            return _missing(f"missing {start_field} for task {task['task_id']}")
        node = nodes.get(task["task_id"], {})
        if int(node.get("block_num", 1)) > 1 and "sync_start" not in node:
            resource_error = "multi-block launch/resource constraints are absent from DAG"
        shape = _resource_shape(node, task["rows"])
        observed_shape = Counter(row["core_type"] for row in task["rows"])
        multi_block = observed_shape["aic"] > 1 or observed_shape["aiv"] > (2 if observed_shape["aic"] else 1)
        if multi_block and "sync_start" not in node:
            resource_error = "multi-fragment launch/resource constraints are absent from DAG"
        placements = tuple(tuple(int(core) for core in placement) for placement in node.get("resource_affinity", []))
        if shape[0] and shape[1] and not placements:
            resource_error = "MIX core placement constraints are absent from DAG"
        if placements and any(
            len(placement) != sum(shape) or any(core < 0 or core >= len(core_types) for core in placement)
            for placement in placements
        ):
            resource_error = "invalid resource affinity"

        if not any(shape):
            # Runtime-executed dependency-only nodes still consume their observed engine.
            shape = tuple(max((int(r["core_type"] == ty) for r in task["rows"]), default=0) for ty in ("aic", "aiv"))
        left, right = max(start, ready), min(end, task["S"])
        if right > left:
            events[left].append(("ready", (shape, placements), 1))
            events[right].append(("ready", (shape, placements), -1))
        for row in task["rows"]:
            left, right = max(start, row["start_time_us"]), min(end, row["end_time_us"])
            if right > left:
                events[left].append(("running", (row["core_type"], row["core_id"]), 1))
                events[right].append(("running", (row["core_type"], row["core_id"]), -1))
    times = sorted({start, end, *events})
    running, ready, running_core = Counter(), Counter(), Counter()
    depth = []
    idle = Counter()
    any_idle = 0.0
    for left, right in zip(times, times[1:]):
        for kind, key, change in events[left]:
            if kind == "running":
                running[key[0]] += change
                running_core[key[1]] += change
            else:
                ready[key] += change
        width = sum(ready.values())
        depth.append((right - left, width))
        available = {ty: counts[ty] - running[ty] for ty in ("aic", "aiv")}
        blocked_types = set()
        for (shape, placements), number in ready.items():
            if (
                number > 0
                and (
                    not placements
                    or any(all(running_core[core] == 0 for core in placement) for placement in placements)
                )
                and all(available[ty] >= need for ty, need in zip(("aic", "aiv"), shape))
            ):
                blocked_types.update(ty for ty, need in zip(("aic", "aiv"), shape) if need)
        for ty in blocked_types:
            idle[ty] += right - left
        if blocked_types:
            any_idle += right - left
    return {
        "status": "unmeasured" if resource_error else "measured",
        "reason": resource_error,
        "window_us": end - start,
        "ready_depth": {"status": "measured", **(_weighted_distribution(depth) or {})},
        "idle_ready_us": None if resource_error else {ty: idle[ty] for ty in counts if ty in ("aic", "aiv")},
        "idle_ready_fraction": None
        if resource_error
        else {ty: idle[ty] / (end - start) for ty in counts if ty in ("aic", "aiv")},
        "any_idle_ready_fraction": None if resource_error else any_idle / (end - start),
    }


def _critical_path(tasks, nodes, pred, succ, window, trusted):  # noqa: PLR0912
    if not trusted:
        return _missing("DAG has not been verified against this workload")
    if set(nodes) - set(tasks):
        return _missing("DAG contains tasks without execution/control events")
    indegree = {tid: len(pred[tid]) for tid in nodes}
    queue = [tid for tid, degree in indegree.items() if not degree]
    weights, static_parent = {}, {}
    for tid in queue:
        task = tasks[tid]
        duration = (task["E"] - task["S"]) if not task["control"] else task["F"] - task["R"]
        parent = max(pred[tid], key=lambda t: weights[t], default=None)
        weights[tid] = duration + (weights[parent] if parent is not None else 0)
        static_parent[tid] = parent
        for following in succ[tid]:
            indegree[following] -= 1
            if indegree[following] == 0:
                queue.append(following)
    if len(weights) != len(nodes):
        return _missing("DAG is cyclic or contains unknown endpoints")
    static_sink = max(weights, key=lambda tid: weights[tid])
    static_path = []
    cursor = static_sink
    while cursor is not None:
        static_path.append(cursor)
        cursor = static_parent[cursor]
    # Expand logical tasks into physical execution slices for resource predecessors.
    slices = []
    by_task = defaultdict(list)
    by_core = defaultdict(list)
    for tid, task in tasks.items():
        rows = task["rows"] or [{"start_time_us": task["R"], "end_time_us": task["F"], "core_id": None}]
        for row in rows:
            index = len(slices)
            slices.append({"task": tid, "row": row, "S": row["start_time_us"], "E": row["end_time_us"]})
            by_task[tid].append(index)
            if row["core_id"] is not None:
                by_core[row["core_id"]].append(index)
    core_prev = {}
    for entries in by_core.values():
        entries.sort(key=lambda i: slices[i]["S"])
        for previous, current in zip(entries, entries[1:]):
            if slices[previous]["E"] > slices[current]["S"]:
                return _missing("overlapping execution intervals on one core")
            core_prev[current] = previous
    last_slice = {tid: max(indexes, key=lambda i: slices[i]["E"]) for tid, indexes in by_task.items()}
    cursor = max(range(len(slices)), key=lambda i: slices[i]["E"])
    walk = []
    seen = set()
    while cursor not in seen:
        seen.add(cursor)
        item = slices[cursor]
        candidates = [last_slice[p] for p in pred[item["task"]]]
        if cursor in core_prev:
            candidates.append(core_prev[cursor])
        if any(slices[i]["E"] > item["S"] + 1e-6 for i in candidates):
            return _missing("dependency completion follows successor start")
        parent = max(candidates, key=lambda i: slices[i]["E"], default=None)
        walk.append(cursor)
        if parent is None:
            break
        cursor = parent
    frontier = window[0]
    contribution = Counter()
    segments = []
    for index in reversed(walk):
        item = slices[index]
        task, row = tasks[item["task"]], item["row"]
        left, right = max(window[0], frontier), min(window[1], item["S"])
        classified = []
        if not task["control"]:
            parents = [tasks[p] for p in pred[item["task"]]]
            if parents and all(p["F"] is not None for p in parents):
                observed = max(p["F"] for p in parents)
                data_ready = max(p["E"] if p["E"] is not None else p["F"] for p in parents)
                classified.extend((("E_F", data_ready, observed), ("F_R", observed, task["R"])))
            r, d, a, g = task["R"], row.get("dispatch_time_us"), row.get("acquisition_time_us"), row.get("gate_time_us")
            if r is not None and d is not None and d >= r and (a is None or a >= r):
                classified.extend((("R_D", r, d), ("D_A", d, a), ("A_S", a, item["S"])))
            elif r is not None:
                classified.extend(
                    (
                        ("R_G", r, g),
                        ("gate_to_start", max(a, g) if a is not None and g is not None else None, item["S"]),
                    )
                )
        remaining = left
        for phase, begin, finish in classified:
            if begin is None or finish is None or finish < begin:
                continue
            lo, hi = max(remaining, begin), min(right, finish)
            if hi > lo:
                if lo > remaining:
                    contribution["unclassified"] += lo - remaining
                contribution[phase] += hi - lo
                remaining = hi
        contribution["unclassified"] += max(0, right - remaining)
        compute = max(0, min(window[1], item["E"]) - max(frontier, item["S"], window[0]))
        contribution["control" if task["control"] else "execution"] += compute
        segments.append(
            {"task_id": item["task"], "core_id": row["core_id"], "start_us": item["S"], "end_us": item["E"]}
        )
        frontier = max(frontier, item["E"])
    return {
        "status": "measured",
        "static_duration_us": weights[static_sink],
        "static_task_ids": static_path[::-1],
        "observed_segments": segments,
        "observed_contribution_us": dict(contribution),
        "observed_contribution_fraction": {k: v / (window[1] - window[0]) for k, v in contribution.items()},
        "interpretation": "observed timeline attribution; not removable device time",
    }


def _runtime_stats(data, epoch):
    stats = data.get("capture_metadata", {}).get("runtime_stats")
    if not stats or stats.get("run_epoch") != epoch:
        if data.get("runtime") == "host_build_graph":
            return {"status": "not_applicable", "reason": "HBG has no reclaiming task/dependency pool"}
        return _missing("no matching per-run resource statistics")
    frequency = data.get("capture_metadata", {}).get("clock_freq_hz", 0)
    if frequency <= 0:
        return _missing("missing device frequency")
    factor = 1e6 / frequency
    reasons = ("task_window", "heap", "dependency_pool", "fanin_spill")
    intervals = defaultdict(list)
    for stream in data.get("aicpu_orchestrator_phases", []):
        for record in stream:
            if record.get("phase") == "orch_wait" and record.get("run_epoch") == epoch:
                intervals[record["wait_reason"]].append(record["end_time_us"] - record["start_time_us"])
    waits = {}
    for entry in stats.get("waits", []):
        index = entry["reason"]
        count = entry["count"]
        samples = intervals[index]
        waits[reasons[index]] = {
            "count": count,
            "total_us": entry["total_cycles"] * factor,
            "max_us": entry["max_cycles"] * factor,
            "distribution_us": distribution(samples) if len(samples) == count else None,
            "distribution_status": "not_applicable"
            if count == 0
            else "measured"
            if len(samples) == count
            else "unmeasured",
        }
    return {"status": "measured", "waits": waits, "dependency_peak_per_ring": stats["dependency_peak"]}


def _group_stages(task_rows, phases):
    groups = defaultdict(lambda: defaultdict(list))
    for task in task_rows:
        for fragment in task["fragments"]:
            for key, value in fragment.items():
                if key.endswith("_us") and value is not None:
                    names = [fragment["core_type"], f"{fragment['core_type']}:{fragment['core_id']}"]
                    if task.get("work_class"):
                        names.append(f"{fragment['core_type']}.{task['work_class']}")
                    for group in names:
                        groups[group][key].append(value)
    return {
        group: {
            phase: {
                "status": phases[phase.removesuffix("_us")]["status"],
                "distribution_us": distribution(samples)
                if phases[phase.removesuffix("_us")]["status"] == "measured"
                else None,
            }
            for phase, samples in metrics.items()
        }
        for group, metrics in groups.items()
    }


def _capture_quality(data, nodes, tasks, fragments):
    missing_tasks = sorted(set(nodes) - set(tasks))
    incomplete = []
    for tid, node in nodes.items():
        if tid not in tasks:
            continue
        kernels = node.get("kernel_ids", [])
        expected = int(node.get("block_num", 1)) * sum(k >= 0 for k in kernels)
        if expected and len(tasks[tid]["rows"]) != expected:
            incomplete.append(tid)
    capture = data.get("capture_metadata", {}).get("collection", {})
    capture_ok = not any(
        capture.get(k, 0)
        for k in (
            "not_received_buffers",
            "unpublished_loss",
            "transport_retired_records",
            "unattributable_handoffs",
            "cut_failed_queues",
        )
    )
    capture_ok = capture_ok and capture.get("processing_complete", True) and capture.get("metadata_complete", True)
    capture_ok = capture_ok and not data.get("task_transition_errors") and not data.get("task_transition_unmatched")
    capture_ok = capture_ok and not incomplete
    capture_ok = capture_ok and data.get("task_transition_metadata", {}).get("capture", {}).get("complete", False)
    clock_errors = []
    for row in fragments:
        e, f = row["end_time_us"], row.get("finish_time_us")
        d, a = row.get("dispatch_time_us"), row.get("acquisition_time_us")
        if (f is not None and f < e) or (d is not None and a is not None and a < d):
            clock_errors.append(
                {
                    "task_id": row["task_id"],
                    "core_id": row["core_id"],
                    "reg_task_id": row.get("reg_task_id"),
                    "E_us": e,
                    "F_us": f,
                    "D_us": d,
                    "A_us": a,
                }
            )
    missing_executors = [tid for tid in missing_tasks if any(k >= 0 for k in nodes[tid].get("kernel_ids", []))]
    capture_ok = capture_ok and not missing_executors
    return capture_ok, clock_errors, incomplete, missing_tasks


def analyze_run(  # noqa: PLR0912
    data,
    *,
    deps=None,
    dag_verified=False,
    run_epoch=None,
    device_us=None,
    labels=None,
    window=None,
    selected_task_ids=None,
):
    fragments = [t for t in data.get("tasks", []) if run_epoch is None or t.get("run_epoch") == run_epoch]
    if not fragments:
        return _missing("no Executor execution events")
    start = min(t["start_time_us"] for t in fragments)
    end = max(t["end_time_us"] for t in fragments)
    if window is not None:
        start, end = max(start, window[0]), min(end, window[1])
    if end <= start:
        return _missing("empty execution window")
    controls = [
        r
        for stream in data.get("scheduler_records", [])
        for r in stream
        if r.get("phase") == "control_task" and (run_epoch is None or r.get("run_epoch") == run_epoch)
    ]
    tasks = _logical_tasks(fragments, controls)
    nodes, pred, succ = _graph(deps)
    valid_graph = dag_verified and bool(nodes) and set(tasks) <= set(nodes)
    capture_ok, clock_errors, incomplete, missing_tasks = _capture_quality(data, nodes, tasks, fragments)
    clock_ok = not clock_errors
    for tid, task in tasks.items():
        data_ready = None
        if valid_graph:
            if pred[tid]:
                parents = [tasks[p] for p in pred[tid] if p in tasks]
                if len(parents) == len(pred[tid]):
                    data_ready = max(p["E"] if p["E"] is not None else p["F"] for p in parents)
            else:
                data_ready = task["registered"]
                if data_ready is None:
                    data_ready = data.get("task_transition_metadata", {}).get("root_release_time_us")
            if data_ready is not None and task["registered"] is not None:
                data_ready = max(data_ready, task["registered"])
        task["data_ready"] = data_ready
    values, expected = defaultdict(list), Counter()
    task_rows = []
    for tid, task in tasks.items():
        if task["control"] or (selected_task_ids is not None and tid not in selected_task_ids):
            continue
        row_metrics = {
            "task_id": tid,
            "S_us": task["S"],
            "E_us": task["E"],
            "F_us": task["F"],
            "R_us": task["R"],
            "registered_us": task["registered"],
            "work_class": labels.get(tid) if labels else None,
            "fragments": [],
        }
        expected["E_F"] += 1
        if task["F"] is not None and task["F"] >= task["E"]:
            values["E_F"].append(task["F"] - task["E"])
        if pred[tid]:
            expected["F_R"] += 1
            parents = [tasks[p] for p in pred[tid] if p in tasks]
            if task["R"] is not None and len(parents) == len(pred[tid]) and all(p["F"] is not None for p in parents):
                latest = max(p["F"] for p in parents)
                if task["R"] >= latest:
                    values["F_R"].append(task["R"] - latest)
                    row_metrics["F_R_us"] = task["R"] - latest
                    row_metrics["registration_after_last_F"] = (
                        task["registered"] is not None and task["registered"] > latest
                    )
        elif valid_graph:
            expected["root_release_R"] += 1
            release = task["registered"]
            if release is None:
                release = data.get("task_transition_metadata", {}).get("root_release_time_us")
            if release is not None and task["R"] is not None and task["R"] >= release:
                values["root_release_R"].append(task["R"] - release)
        if task["R"] is not None and task["R"] <= task["S"]:
            values["R_S"].append(task["S"] - task["R"])
        expected["R_S"] += 1
        if labels and labels.get(tid) == "short":
            expected["short_R_S"] += 1
            if task["R"] is not None and task["R"] <= task["S"]:
                values["short_R_S"].append(task["S"] - task["R"])
        for row in task["rows"]:
            d, a, r, g, s = (
                row.get(k)
                for k in ("dispatch_time_us", "acquisition_time_us", "ready_time_us", "gate_time_us", "start_time_us")
            )
            early = (
                row.get("dispatch_mode") == "early"
                or (r is not None and ((d is not None and d < r) or (a is not None and a < r)))
                or g is not None
            )
            phase = {
                "core_type": row["core_type"],
                "core_id": row["core_id"],
                "reg_task_id": row.get("reg_task_id"),
                "dispatch_mode": "early" if early else "ordinary",
            }
            intervals = {"D_A": (d, a), "A_S": (a, s)}
            if not early:
                intervals.update({"R_D": (r, d), "R_A": (r, a)})
            else:
                expected["early_fragments"] += 1
                intervals.update(
                    {"R_G": (r, g), "gate_to_start": (max(a, g) if a is not None and g is not None else None, s)}
                )
            for name, (left, right) in intervals.items():
                expected[name] += 1
                value = right - left if left is not None and right is not None and right >= left else None
                phase[name + "_us"] = value
                if value is not None:
                    values[name].append(value)
            row_metrics["fragments"].append(phase)
        task_rows.append(row_metrics)
    phases = {
        name: _samples(
            values[name],
            expected[name],
            trusted=capture_ok
            and (valid_graph if name in ("F_R", "root_release_R") else True)
            and (clock_ok if name in ("E_F", "D_A", "R_A") else True),
            reason="missing/invalid events or unverified DAG",
        )
        for name in (*_PHASES, "R_S", "short_R_S", "R_G", "gate_to_start", "root_release_R")
    }
    grouped_phases = _group_stages(task_rows, phases)
    by_core = defaultdict(list)
    for row in fragments:
        by_core[(row["core_type"], row["core_id"])].append(
            (max(start, row["start_time_us"]), min(end, row["end_time_us"]))
        )
    core_types = data.get("core_types", [])
    capacity = Counter(core_types)
    utilization = {}
    for ty in ("aic", "aiv"):
        if not capacity[ty] or not capture_ok:
            utilization[ty] = _missing("missing core inventory or incomplete capture")
        else:
            per_core = {
                str(core): sum(right - left for left, right in _union(by_core[(ty, core)])) / (end - start)
                for core, core_ty in enumerate(core_types)
                if core_ty == ty
            }
            utilization[ty] = {
                "status": "measured",
                "executor_cores": capacity[ty],
                "fraction": sum(per_core.values()) / capacity[ty],
                "per_core": per_core,
            }
    idle_sched = (
        _idle_and_depth(tasks, nodes, core_types, (start, end), "R")
        if core_types and capture_ok and clock_ok
        else _missing("missing core inventory or invalid capture")
    )
    idle_data = (
        _idle_and_depth(tasks, nodes, core_types, (start, end), "data_ready")
        if valid_graph and core_types and capture_ok and clock_ok
        else _missing("DAG unverified or invalid capture")
    )
    if missing_tasks:
        idle_data = _missing("DAG has missing execution/control tasks")
    resources = _runtime_stats(data, run_epoch)
    if resources["status"] == "measured" and device_us:
        for wait in resources["waits"].values():
            wait["device_fraction"] = wait["total_us"] / device_us
    real_count = (
        sum(any(k >= 0 for k in nodes[t["task_id"]].get("kernel_ids", [])) for t in task_rows)
        if valid_graph
        else len(task_rows)
    )
    return {
        "status": "measured" if capture_ok and clock_ok else "unmeasured",
        "run_epoch": run_epoch,
        "scheduler_location": data.get("scheduler_task_producer"),
        "T_exec_us": end - start,
        "T_device_us": device_us,
        "throughput_tasks_per_us": real_count / device_us if device_us else None,
        "throughput_basis": "real_kernel_tasks" if valid_graph else "observed_executor_tasks",
        "window_throughput_tasks_per_us": real_count / (end - start),
        "coverage": {
            "observed_logical_tasks": len(tasks),
            "observed_fragments": len(fragments),
            "dag_verified": valid_graph,
            "missing_task_ids": missing_tasks,
            "incomplete_fragment_tasks": incomplete,
            "transition_errors": data.get("task_transition_errors", []),
            "clock_order_violations": clock_errors,
        },
        "phases": phases,
        "phase_groups": grouped_phases,
        "utilization": utilization,
        "I_sched": idle_sched,
        "I_data": idle_data,
        "critical_path": _critical_path(
            tasks, nodes, pred, succ, (start, end), valid_graph and capture_ok and clock_ok
        ),
        "late_registration_tasks": sum(t.get("registration_after_last_F", False) for t in task_rows),
        "runtime_resources": resources,
        "task_metrics": task_rows,
    }


def analyze(data, *, deps=None, dag_verified=False, device_us=None, labels=None):
    epochs = sorted({t.get("run_epoch") for t in data.get("tasks", [])}, key=lambda e: -1 if e is None else e)
    return {
        "schema_version": 1,
        "interpretation": "observed diagnostic metrics; no replay estimate",
        "runs": [
            analyze_run(data, deps=deps, dag_verified=dag_verified, run_epoch=e, device_us=device_us, labels=labels)
            for e in epochs
        ],
    }


def scale_calibration(points):
    """Fit fixed device cost against task count/depth/unit count."""
    if len(points) < 2 or len({p[0] for p in points}) < 2:
        return _missing("at least two distinct scales are required")
    x, y = zip(*points)
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a - xm) * (b - ym) for a, b in points) / sum((a - xm) ** 2 for a in x)
    intercept = ym - slope * xm
    if intercept < 0 or slope <= 0:
        return _missing("nonphysical fixed-cost fit; collect additional scales", intercept_us=intercept, slope_us=slope)
    residual = max(abs(b - (intercept + slope * a)) for a, b in points)
    return {
        "status": "measured",
        "fixed_us": intercept,
        "unit_us": slope,
        "max_residual_us": residual,
        "points": [
            {"scale": a, "device_us": b, "fixed_fraction": intercept / b, "below_5_percent": intercept / b <= 0.05}
            for a, b in points
        ],
    }


def capture_overhead(baseline, existing, extended):
    if not baseline or not existing or not extended:
        return _missing("all three repeated timing series are required")
    med = [statistics.median(series) for series in (baseline, existing, extended)]
    if min(med) <= 0:
        return _missing("non-positive timing sample")
    return {
        "status": "measured",
        "baseline": distribution(baseline),
        "existing": distribution(existing),
        "extended": distribution(extended),
        "increment_us": med[2] - med[1],
        "increment_fraction_of_existing": (med[2] - med[1]) / med[1],
        "total_fraction_of_baseline": (med[2] - med[0]) / med[0],
    }


def stage_events(tasks, *, pid=6):
    """Actual task transitions on separate Perfetto tracks; no inferred Ready."""
    events = []
    lanes = set()
    for task in tasks:
        ready, dispatch, acquire, start, end, finish, gate = (
            task.get(key)
            for key in (
                "ready_time_us",
                "dispatch_time_us",
                "acquisition_time_us",
                "start_time_us",
                "end_time_us",
                "finish_time_us",
                "gate_time_us",
            )
        )
        if ready is None:
            continue
        early = (
            task.get("dispatch_mode") == "early"
            or gate is not None
            or (dispatch is not None and dispatch < ready)
            or (acquire is not None and acquire < ready)
        )
        phases = [("E_F", end, finish), ("D_A", dispatch, acquire), ("A_S", acquire, start)]
        phases.append(("R_G", ready, gate) if early else ("R_D", ready, dispatch))
        for phase, left, right in phases:
            if left is None or right is None or right <= left:
                continue
            tid = int(task["core_id"]) * 8 + ("E_F", "D_A", "A_S", "R_D", "R_G").index(phase)
            if tid not in lanes:
                lanes.add(tid)
                events.append(
                    {
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "name": "thread_name",
                        "args": {"name": f"{task['core_type']} {task['core_id']} {phase}"},
                    }
                )
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "name": f"{phase} task {task['task_id']}",
                    "cat": "scheduler_stage",
                    "ts": left,
                    "dur": right - left,
                    "args": {
                        "task_id": str(task["task_id"]),
                        "run_epoch": task.get("run_epoch"),
                        "dispatch_mode": "early" if early else "ordinary",
                    },
                }
            )
    logical = defaultdict(list)
    for task in tasks:
        logical[(task.get("run_epoch"), task["task_id"])].append(task)
    depth_changes = Counter()
    for rows in logical.values():
        ready = {row.get("ready_time_us") for row in rows}
        if len(ready) == 1 and None not in ready:
            begin, end = next(iter(ready)), min(row["start_time_us"] for row in rows)
            if end > begin:
                depth_changes[begin] += 1
                depth_changes[end] -= 1
    if depth_changes:
        events.append(
            {"ph": "M", "pid": pid, "tid": 10000, "name": "thread_name", "args": {"name": "Actual Ready depth"}}
        )
        depth = 0
        for timestamp, change in sorted(depth_changes.items()):
            depth += change
            events.append(
                {"ph": "C", "pid": pid, "tid": 10000, "name": "Q_R", "ts": timestamp, "args": {"tasks": depth}}
            )
    if events:
        events.insert(
            0, {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": "Observed scheduling stages"}}
        )
    return events


def markdown_report(report):
    lines = ["# Observed scheduler metrics", "", report["interpretation"], ""]
    for run in report["runs"]:
        lines.extend(
            [
                f"## Run {run.get('run_epoch')}",
                "",
                f"Scheduler: {run.get('scheduler_location')}; T_exec: {run.get('T_exec_us')} us",
                "",
                "| Phase | Status | Samples | Median us | p95 us |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for phase, metric in run.get("phases", {}).items():
            dist = metric.get("distribution_us") or {}
            lines.append(
                f"| {phase} | {metric['status']} | {metric['valid_samples']}/{metric['expected_samples']} "
                f"| {dist.get('median', 'unmeasured')} | {dist.get('p95', 'unmeasured')} |"
            )
        lines.extend(
            [
                "",
                "```json",
                json.dumps({key: run.get(key) for key in ("coverage", "utilization", "I_sched", "I_data")}, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from .swimlane_converter import read_perf_data  # noqa: PLC0415

    data = read_perf_data(args.records)
    deps = json.loads(args.deps.read_text()) if args.deps else None
    report = analyze(data, deps=deps)
    output = args.output or args.records.with_name("scheduler_metrics.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown_report(report))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
