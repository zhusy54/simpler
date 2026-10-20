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
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests.st.scheduler_benchmark.matrix import CASE_IDS, LONG_ITERS, SHORT_ITERS, expand_case, scale_case
from tests.st.scheduler_benchmark.workloads import WORKLOAD_BY_ID, WORKLOADS, workload_case

_REPO = Path(__file__).resolve().parents[3]
_QUERY = _REPO / "tools/cann-examples/query/build/query"
_CAP = {"a2a3": 24, "a5": 36}
ALL_CASE_IDS = (*CASE_IDS, *(spec.case_id for spec in WORKLOADS))


def prepare_sim_environment(platform):
    """Make spawned interpreters resolve the compiler's C++ runtime first."""
    if not platform.endswith("sim"):
        return False
    library = subprocess.run(
        ["g++-15", "-print-file-name=libstdc++.so.6"], check=True, capture_output=True, text=True
    ).stdout.strip()
    directory = str(Path(library).parent)
    previous = os.environ.get("LD_LIBRARY_PATH", "")
    if previous.split(":")[0] == directory:
        return False
    os.environ["LD_LIBRARY_PATH"] = directory + (":" + previous if previous else "")
    return True


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
        "examples",
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


def _timings_from_lines(lines, expected_rounds):
    from simpler_setup.tools.strace_timing import (  # noqa: PLC0415
        _round_metrics,
        group_invocations,
        invocation_spans,
        parse_spans,
    )

    invocations = group_invocations(invocation_spans(parse_spans(lines)))
    invocations = [inv for inv in invocations if inv.root() is not None]
    if len(invocations) != expected_rounds:
        raise ValueError(f"expected {expected_rounds} rounds, found {len(invocations)}")
    identities = set()
    rows = []
    for index, inv in enumerate(invocations):
        host, device, effective, orch, sched = _round_metrics(inv)
        if host <= 0 or device <= 0:
            raise ValueError(f"round {index} has no positive host/device-wall marker")
        identity = {"pid": inv.pid, "inv": inv.inv}
        attrs = dict(re.findall(r"(\w+)=(\S+)", inv.root().attrs))
        for key in ("run_id", "dispatch_id", "run_epoch", "slot_id", "generation"):
            if key in attrs:
                identity[key] = int(attrs[key])
        key = (inv.pid, inv.inv)
        if key in identities:
            raise ValueError(f"duplicate run identity: {key}")
        identities.add(key)
        rows.append(
            {
                "round": index,
                "host_us": host,
                "device_us": device,
                "effective_us": effective or None,
                "orch_us": orch or None,
                "sched_us": sched or None,
                "run_identity": identity,
            }
        )
    return rows


def _round_timings(log: Path, expected_rounds: int) -> list[dict]:
    def lines():
        for path in sorted(log.glob("host.*.log")) if log.is_dir() else [log]:
            with path.open(encoding="utf-8", errors="replace") as stream:
                yield from stream

    return _timings_from_lines(lines(), expected_rounds)


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


def make_manifest(args, *, counts=None, metadata=None):
    from tests.st.scheduler_benchmark.session import EXECUTION_MODEL  # noqa: PLC0415

    arch = "a5" if args.platform.startswith("a5") else "a2a3"
    k_c, k_v, soc = counts or (
        (8, 16, "simulated") if args.platform.endswith("sim") else _hardware_counts(args.device, arch)
    )
    cases = args.case or list(ALL_CASE_IDS)
    return {
        "schema_version": 4,
        "execution_model": EXECUTION_MODEL,
        "task_scale": args.scale,
        "capture_deps": args.capture_deps,
        "swimlane_level": args.chip_swimlane,
        "platform": args.platform,
        "runtime": args.runtime,
        "device": args.device,
        "soc": soc,
        **(_metadata() if metadata is None else metadata),
        "k_c": k_c,
        "k_v": k_v,
        "rounds_requested": args.rounds,
        "scheduler_policy": "runtime_default",
        "short_iterations": SHORT_ITERS,
        "long_iterations": LONG_ITERS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [
            scale_case(expand_case(cid, k_c, k_v), args.scale).manifest()
            if cid in CASE_IDS
            else {**WORKLOAD_BY_ID[cid].manifest(), "case": workload_case(WORKLOAD_BY_ID[cid], args.runtime)}
            for cid in cases
        ],
        "case_results": {cid: {"status": "not_run"} for cid in cases},
    }


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
    cases = args.case or list(ALL_CASE_IDS)
    if args.scale < 1 or (args.scale != 1 and any(cid not in CASE_IDS for cid in cases)):
        parser.error("--scale must be positive and applies only to C configurations")
    if len(set(cases)) != len(cases):
        parser.error("duplicate --case ID")
    if not args.list_only and prepare_sim_environment(args.platform):
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "tests.st.scheduler_benchmark.run_cases", *(sys.argv[1:] if argv is None else argv)],
            os.environ,
        )
    manifest = make_manifest(args)
    output = args.manifest or (
        _REPO / "outputs" / f"scheduler_benchmark_{args.platform}_{args.runtime}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    from tests.st.scheduler_benchmark.session import BenchmarkSession, write_json  # noqa: PLC0415

    write_json(output, manifest)
    print(f"matrix: {output}; K_C={manifest['k_c']}, K_V={manifest['k_v']}; cases={len(cases)}", flush=True)
    if args.list_only:
        return 0
    session = BenchmarkSession(
        args.platform,
        args.runtime,
        args.device,
        manifest["k_c"],
        manifest["k_v"],
        output.with_name(f"{output.stem}_{uuid.uuid4().hex[:12]}"),
        task_scale=args.scale,
    )
    try:
        result = session.run_phase(manifest, output)
    finally:
        session.close()
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
