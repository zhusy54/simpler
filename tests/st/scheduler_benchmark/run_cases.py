"""Expand and run the fixed C1–C3 matrix on one runtime and device."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tests.st.scheduler_benchmark.matrix import CASE_IDS, LONG_ITERS, SHORT_ITERS, expand_case

_REPO = Path(__file__).resolve().parents[3]
_QUERY = _REPO / "tools/cann-examples/query/build/query"
_CAP = {"a2a3": 24, "a5": 36}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3sim", "a2a3", "a5sim", "a5"), required=True)
    parser.add_argument("--runtime", choices=("host_build_graph", "tensormap_and_ringbuffer"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", action="append", choices=CASE_IDS)
    parser.add_argument("--capture-deps", action="store_true")
    parser.add_argument("--chip-swimlane", type=int, choices=(1, 2, 3, 4), default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)
    arch = "a5" if args.platform.startswith("a5") else "a2a3"
    if args.platform.endswith("sim"):
        k_c, k_v, soc = 8, 16, "simulated"
    else:
        k_c, k_v, soc = _hardware_counts(args.device, arch)
    cases = args.case or list(CASE_IDS)
    manifest = {
        "platform": args.platform,
        "runtime": args.runtime,
        "device": args.device,
        "soc": soc,
        "k_c": k_c,
        "k_v": k_v,
        "short_iterations": SHORT_ITERS,
        "long_iterations": LONG_ITERS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [expand_case(case_id, k_c, k_v).manifest() for case_id in cases],
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

    import os  # noqa: PLC0415

    env = os.environ.copy()
    env["SIMPLER_SCHED_BENCH_K_C"] = str(k_c)
    env["SIMPLER_SCHED_BENCH_K_V"] = str(k_v)
    if args.platform.endswith("sim"):
        library = subprocess.run(
            ["g++-15", "-print-file-name=libstdc++.so.6"], check=True, capture_output=True, text=True
        ).stdout.strip()
        env["LD_LIBRARY_PATH"] = f"{Path(library).parent}:{env.get('LD_LIBRARY_PATH', '')}"
    scene = _REPO / "tests/st" / arch / args.runtime / "scheduler_benchmark"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(scene),
        "--platform",
        args.platform,
        "--device",
        str(args.device),
        "--manual",
        "only",
    ]
    if args.capture_deps:
        cmd.append("--enable-dep-gen")
    if args.chip_swimlane:
        cmd.extend(("--enable-chip-swimlane", str(args.chip_swimlane)))
    manifest["case_results"] = {}
    for case_id in cases:
        print(f"running {case_id}", flush=True)
        result = subprocess.run([*cmd, "--case", case_id], cwd=_REPO, env=env, check=False)
        manifest["case_results"][case_id] = result.returncode
        output.write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            manifest["pytest_exit_code"] = result.returncode
            manifest["failed_case"] = case_id
            output.write_text(json.dumps(manifest, indent=2) + "\n")
            return result.returncode
    manifest["pytest_exit_code"] = 0
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
