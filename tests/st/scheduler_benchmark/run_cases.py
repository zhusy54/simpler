# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expand and run the fixed 41-case scheduling matrix on one runtime and device."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tests.st.scheduler_benchmark.matrix import CASE_IDS, LONG_ITERS, SHORT_ITERS, expand_case, scale_case
from tests.st.scheduler_benchmark.workloads import WORKLOAD_BY_ID, WORKLOADS, supported_platforms, workload_case

_REPO = Path(__file__).resolve().parents[3]
_QUERY = _REPO / "tools/cann-examples/query/build/query"
_CAP = {"a2a3": 24, "a5": 36}
ALL_CASE_IDS = (*CASE_IDS, *(spec.case_id for spec in WORKLOADS))


def _metadata() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, check=True, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO, check=True, capture_output=True, text=True)
    pin = (_REPO / "pto_isa.pin").read_text().strip()
    cann_info = Path(os.environ.get("ASCEND_HOME_PATH", "")) / "compiler/version.info"
    cann = None
    if cann_info.is_file():
        match = re.search(r"^Version=(\S+)", cann_info.read_text(), re.MULTILINE)
        cann = match.group(1) if match else None
    source_paths = (
        "src",
        "simpler_setup",
        "python",
        "tests",
        "cmake",
        "CMakeLists.txt",
        "pyproject.toml",
        "pto_isa.pin",
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", *source_paths], cwd=_REPO, check=True, capture_output=True
    )
    digest = hashlib.sha256(diff.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=_REPO, check=True, capture_output=True
    )
    for name in sorted(untracked.stdout.split(b"\0")):
        if name and any(
            os.fsdecode(name) == prefix or os.fsdecode(name).startswith(prefix + "/") for prefix in source_paths
        ):
            path = _REPO / os.fsdecode(name)
            if path.is_file():
                digest.update(name)
                digest.update(path.read_bytes())
    return {
        "commit": commit.stdout.strip(),
        "source_patch_sha256": digest.hexdigest(),
        "dirty_worktree": bool(status.stdout.strip()),
        "pto_isa_pin": pin,
        "cann_version": cann,
    }


def _round_timings(log: Path, expected_rounds: int) -> list[dict]:
    from simpler_setup.tools.strace_timing import (  # noqa: PLC0415
        _round_metrics,
        bucket_by_hid,
        group_invocations,
        invocation_spans,
        parse_spans,
    )

    def lines():
        for path in sorted(log.glob("host.*.log")) if log.is_dir() else [log]:
            with path.open(encoding="utf-8", errors="replace") as stream:
                yield from stream

    buckets = bucket_by_hid(group_invocations(invocation_spans(parse_spans(lines()))))
    if not buckets:
        raise ValueError("no [STRACE] invocations")
    invocations = max(buckets.values(), key=len)
    if len(invocations) != expected_rounds:
        raise ValueError(f"expected {expected_rounds} rounds, found {len(invocations)}")
    rows = []
    for index, inv in enumerate(invocations):
        host, device, effective, orch, sched = _round_metrics(inv)
        if device <= 0:
            raise ValueError(f"round {index} has no device-wall marker")
        rows.append(
            {
                "round": index,
                "host_us": host,
                "device_us": device,
                "effective_us": effective or None,
                "orch_us": orch or None,
                "sched_us": sched or None,
            }
        )
    return rows


def _case_command(case_id: str, args: argparse.Namespace, arch: str) -> tuple[list[str], bool]:
    if case_id in CASE_IDS:
        scene = _REPO / "tests/st" / arch / args.runtime / "scheduler_benchmark/test_scheduler_benchmark.py"
        cmd = [
            sys.executable,
            str(scene),
            "--platform",
            args.platform,
            "--device",
            str(args.device),
            "--manual",
            "include",
            "--case",
            case_id,
        ]
    else:
        spec = WORKLOAD_BY_ID[case_id]
        test_file = spec.test_file(_REPO, arch, args.runtime)
        case_name = workload_case(spec, args.runtime)
        cmd = [
            sys.executable,
            str(test_file),
            "--platform",
            args.platform,
            "--device",
            str(args.device),
            "--manual",
            "include",
            "--case",
            case_name,
            "--skip-golden",
        ]
    if case_id in CASE_IDS and getattr(args, "scale", 1) != 1:
        module_name = f"tests.st.{arch}.{args.runtime}.scheduler_benchmark.test_scheduler_benchmark"
        expression = (
            "from tests.st.scheduler_benchmark.scene import execute_scaled_scene; "
            f"execute_scaled_scene({module_name!r}, {args.scale})"
        )
        cmd = [sys.executable, "-c", expression, *cmd[2:]]
    cmd.extend(("--rounds", str(args.rounds)))
    if args.capture_deps:
        cmd.append("--enable-dep-gen")
    if args.chip_swimlane:
        cmd.extend(("--enable-chip-swimlane", str(args.chip_swimlane)))
    supported = case_id in CASE_IDS or args.platform in supported_platforms(test_file, case_name)
    return cmd, supported


def _diagnostic_paths(case_id: str, runtime: str, before: set[Path]) -> list[str]:
    token = case_id if case_id in CASE_IDS else workload_case(WORKLOAD_BY_ID[case_id], runtime)
    if case_id == "W5-01":
        token = "Qwen314BDecode"
    roots = set((_REPO / "outputs").glob(f"*{token}*")) - before
    names = ("deps.json", "chip_swimlane_records.json")
    return sorted(str(path) for root in roots for name in names for path in root.rglob(name))


def _hardware_counts(device: int, arch: str) -> tuple[int, int, str]:
    if not _QUERY.exists():
        project = _QUERY.parent.parent
        subprocess.run(["cmake", "-S", str(project), "-B", str(_QUERY.parent)], check=True)
        subprocess.run(["cmake", "--build", str(_QUERY.parent)], check=True)
    result = subprocess.run([str(_QUERY), "device", str(device)], check=True, capture_output=True, text=True)
    cube = re.search(r"cube_cores \(AIC\):\s+(\d+)", result.stdout)
    vector = re.search(r"vector_cores \(AIV\):\s+(\d+)", result.stdout)
    if cube is None or vector is None:
        raise RuntimeError("CANN query did not return both runtime-visible core limits")
    k_c = min(int(cube.group(1)), int(vector.group(1)) // 2, _CAP[arch])
    if k_c < 1:
        raise RuntimeError(f"invalid runtime-visible core limits: {cube.group(1)}, {vector.group(1)}")
    soc = re.search(r"soc_name:\s*(\S+)", result.stdout)
    return k_c, 2 * k_c, soc.group(1) if soc else "unknown"


def _with_host_log_directory(command: list[str], directory: Path) -> list[str]:
    """Preserve diagnostic timing logs beyond the child session's lifetime."""
    directory.mkdir(parents=True, exist_ok=True)
    expression = (
        "from _task_interface import _set_host_log_directory; "
        f"_set_host_log_directory({str(directory.resolve())!r}); "
        "import sys, runpy; "
    )
    if command[1] == "-c":
        expression += f"sys.argv = {['-c', *command[3:]]!r}; exec({command[2]!r})"
    else:
        expression += f"sys.argv = {command[1:]!r}; runpy.run_path(sys.argv[0], run_name='__main__')"
    return [command[0], "-c", expression]


def _run_case(case_id: str, args: argparse.Namespace, arch: str, env: dict, output: Path) -> dict:
    diagnostic = bool(args.capture_deps or args.chip_swimlane)
    result_record = {
        "status": "pending",
        "exit_code": None,
        "measurement_mode": "diagnostic" if diagnostic else "timing",
        "scheduler_location": None,
        "kernel_execution_us": None,
        "rounds": [],
        "diagnostic_device_us": None,
        "diagnostic_paths": [],
    }
    log = output.with_name(f"{output.stem}_{case_id}.log")
    try:
        cmd, supported = _case_command(case_id, args, arch)
        if not supported:
            result_record["status"] = "unsupported_platform"
            return result_record
        timing_log = log
        if diagnostic:
            timing_log = Path(tempfile.mkdtemp(prefix=f"{output.stem}_{case_id}_host_logs_", dir=output.parent))
            cmd = _with_host_log_directory(cmd, timing_log)
            result_record["host_log_directory"] = str(timing_log)
        result_record["command"] = cmd
        result_record["log_path"] = str(log)
        before = set((_REPO / "outputs").iterdir()) if diagnostic else set()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(cmd, cwd=_REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
        result_record["exit_code"] = process.returncode
        if diagnostic:
            result_record["diagnostic_paths"] = _diagnostic_paths(case_id, args.runtime, before)
        if process.returncode:
            result_record["status"] = "failed"
        elif diagnostic:
            expected = ({"deps.json"} if args.capture_deps else set()) | (
                {"chip_swimlane_records.json"} if args.chip_swimlane else set()
            )
            found = {Path(path).name for path in result_record["diagnostic_paths"]}
            if missing := expected - found:
                result_record["status"] = "failed"
                result_record["error"] = f"missing diagnostic files: {sorted(missing)}"
            else:
                result_record["status"] = "passed"
                try:
                    result_record["diagnostic_device_us"] = _round_timings(timing_log, 1)[0]["device_us"]
                except ValueError as exc:
                    result_record["diagnostic_timing_error"] = str(exc)
                if args.chip_swimlane:
                    from tests.st.scheduler_benchmark.report import analyze_capture  # noqa: PLC0415

                    records = [
                        Path(path)
                        for path in result_record["diagnostic_paths"]
                        if Path(path).name == "chip_swimlane_records.json"
                    ]
                    summaries = [
                        analyze_capture(
                            path,
                            case_id=case_id,
                            device_us=result_record["diagnostic_device_us"],
                            spec=scale_case(
                                expand_case(
                                    case_id, int(env["SIMPLER_SCHED_BENCH_K_C"]), int(env["SIMPLER_SCHED_BENCH_K_V"])
                                ),
                                getattr(args, "scale", 1),
                            ).manifest()
                            if case_id in CASE_IDS
                            else None,
                        )
                        for path in records
                    ]
                    result_record["analysis_paths"] = [str(path) for path in summaries]
                    if len(summaries) == 1:
                        summary = json.loads(summaries[0].read_text())
                        producers = {run.get("scheduler_location") for run in summary["runs"]}
                        result_record["scheduler_location"] = producers.pop() if len(producers) == 1 else None
                        if len(summary["runs"]) == 1:
                            result_record["kernel_execution_us"] = summary["runs"][0].get("kernel_execution_us")
        else:
            result_record["rounds"] = _round_timings(log, args.rounds)
            result_record["status"] = "passed"
    except (FileNotFoundError, ValueError) as exc:
        result_record["status"] = "failed"
        result_record["error"] = str(exc)
    return result_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3sim", "a2a3", "a5sim", "a5"), required=True)
    parser.add_argument("--runtime", choices=("host_build_graph", "tensormap_and_ringbuffer"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", choices=ALL_CASE_IDS)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--scale", type=int, default=1, help="C-graph size multiplier for scale calibration")
    parser.add_argument("--capture-deps", action="store_true")
    parser.add_argument("--chip-swimlane", type=int, choices=(1, 2, 3, 4), default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.capture_deps and args.chip_swimlane:
        parser.error("capture the DAG and scheduling events in separate runs")
    if args.rounds > 1 and (args.capture_deps or args.chip_swimlane):
        parser.error("diagnostic capture requires --rounds 1; run timing separately")
    arch = "a5" if args.platform.startswith("a5") else "a2a3"
    if args.platform.endswith("sim"):
        k_c, k_v, soc = 8, 16, "simulated"
    else:
        k_c, k_v, soc = _hardware_counts(args.device, arch)
    cases = args.case or list(ALL_CASE_IDS)
    if args.scale < 1 or (args.scale != 1 and any(case_id not in CASE_IDS for case_id in cases)):
        parser.error("--scale must be positive and applies only to C configurations")
    if len(set(cases)) != len(cases):
        parser.error("duplicate --case ID")
    manifest = {
        "schema_version": 3,
        "task_scale": args.scale,
        "capture_deps": args.capture_deps,
        "swimlane_level": args.chip_swimlane,
        "platform": args.platform,
        "runtime": args.runtime,
        "device": args.device,
        "soc": soc,
        **_metadata(),
        "k_c": k_c,
        "k_v": k_v,
        "rounds_requested": args.rounds,
        "scheduler_policy": "runtime_default",
        "short_iterations": SHORT_ITERS,
        "long_iterations": LONG_ITERS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [
            scale_case(expand_case(case_id, k_c, k_v), args.scale).manifest()
            if case_id in CASE_IDS
            else {**WORKLOAD_BY_ID[case_id].manifest(), "case": workload_case(WORKLOAD_BY_ID[case_id], args.runtime)}
            for case_id in cases
        ],
        "case_results": {case_id: {"status": "not_run"} for case_id in cases},
    }
    output = args.manifest or (
        _REPO / "outputs" / f"scheduler_benchmark_{args.platform}_{args.runtime}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"matrix: {output}; K_C={k_c}, K_V={k_v}; cases={len(cases)}", flush=True)
    if args.list_only:
        return 0

    env = os.environ.copy()
    env["SIMPLER_SCHED_BENCH_K_C"] = str(k_c)
    env["SIMPLER_SCHED_BENCH_K_V"] = str(k_v)
    if args.platform.endswith("sim"):
        library = subprocess.run(
            ["g++-15", "-print-file-name=libstdc++.so.6"], check=True, capture_output=True, text=True
        ).stdout.strip()
        env["LD_LIBRARY_PATH"] = f"{Path(library).parent}:{env.get('LD_LIBRARY_PATH', '')}"
    failed = False
    for case_id in cases:
        print(f"running {case_id}", flush=True)
        case_result = _run_case(case_id, args, arch, env, output)
        manifest["case_results"][case_id] = case_result
        failed |= case_result["status"] == "failed"
        print(f"{case_id}: {case_result['status']}", flush=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["exit_code"] = int(failed)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
