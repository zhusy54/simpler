# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect A3 queue peaks with isolated runs; all occupancy values count queue slots."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")
LINE = re.compile(r"\b(QPEAK(?:_BEGIN|_END|_MISSING)?) (.*)")


def parse_records(text: str) -> tuple[list[dict], list[str]]:
    """Require complete, consistent per-process reports; deduplicate mirrored logs."""
    reports, errors = {}, []
    for line in text.splitlines():
        match = LINE.search(line)
        if not match:
            continue
        event, payload = match.groups()
        fields = dict(re.findall(r"(\w+)=([^\s]+)", payload))
        try:
            key = (fields["runtime"], int(fields["pid"]), int(fields["epoch"]))
            report = reports.setdefault(key, {"rows": {}, "bad": False})
            if event == "QPEAK_MISSING":
                report["bad"] = True
            elif event in ("QPEAK_BEGIN", "QPEAK_END"):
                count = int(fields["count"])
                if event in report and report[event] != count:
                    report["bad"] = True
                report[event] = count
                execution_rc = int(fields["rc"])
                if "execution_rc" in report and report["execution_rc"] != execution_rc:
                    report["bad"] = True
                report["execution_rc"] = execution_rc
            else:
                row = {name: int(fields[name]) for name in ("index", "cap", "peak", "spmd", "normal", "classified")}
                row.update(runtime=key[0], pid=key[1], epoch=key[2], queue=fields["queue"])
                row_key = (row["queue"], row["index"])
                if row_key in report["rows"] and report["rows"][row_key] != row:
                    report["bad"] = True
                if not 0 <= row["peak"] <= row["cap"] or row["classified"] not in (0, 1):
                    report["bad"] = True
                if (
                    row["classified"]
                    and not max(row["spmd"], row["normal"]) <= row["peak"] <= row["spmd"] + row["normal"]
                ):
                    report["bad"] = True
                if not row["classified"] and (row["spmd"] or row["normal"]):
                    report["bad"] = True
                report["rows"][row_key] = row
        except (ValueError, KeyError):
            errors.append(f"malformed: {match.group(0)}")
    rows = []
    for key, report in reports.items():
        count = len(report["rows"])
        if report["bad"] or not count or report.get("QPEAK_BEGIN") != count or report.get("QPEAK_END") != count:
            errors.append(f"incomplete or inconsistent report: {key}")
        else:
            rows.extend({**row, "execution_rc": report["execution_rc"]} for row in report["rows"].values())
    return rows, errors


def collect(runtime: str) -> list[dict]:
    import pytest  # noqa: PLC0415

    from simpler_setup.scene_test import SceneTestCase  # noqa: PLC0415

    class Collector:
        items = []

        def pytest_collection_finish(self, session):
            for item in session.items:
                cls = getattr(item, "cls", None)
                marker = item.get_closest_marker("device_count")
                common = {
                    "nodeid": item.nodeid,
                    "runtime": runtime,
                    "device_count": int(marker.args[0]) if marker else 1,
                    "fault": item.get_closest_marker("sdma_fault") is not None,
                    "batch_rounds": False,
                }
                if cls is not None and hasattr(cls, "CASES"):
                    if getattr(cls, "_st_runtime", None) != runtime:
                        continue
                    method = getattr(cls, "test_run", None)
                    common["batch_rounds"] = item.name.split("[")[0] == "test_run" and (
                        method is SceneTestCase.test_run
                        or (method is not None and "super().test_run(" in inspect.getsource(method))
                    )
                    for case in cls.CASES:
                        if "a2a3" not in case.get("platforms", []):
                            continue
                        selector = f"{cls.__name__}::{case['name']}"
                        self.items.append(
                            {
                                **common,
                                "id": f"{runtime}:{item.nodeid}::{case['name']}",
                                "selector": selector,
                                "device_count": int(case.get("config", {}).get("device_count", common["device_count"])),
                            }
                        )
                else:
                    self.items.append({**common, "id": f"{runtime}:{item.nodeid}", "selector": None})

    files = []
    for base in (ROOT / "examples", ROOT / "tests/st"):
        for path in base.rglob("test_*.py"):
            parts = path.relative_to(ROOT).parts
            if "a5" in parts or any(other in parts for other in RUNTIMES if other != runtime):
                continue
            files.append(str(path.relative_to(ROOT)))
    collector = Collector()
    with tempfile.TemporaryFile(mode="w+") as captured:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            rc = pytest.main(
                [
                    *sorted(files),
                    "--collect-only",
                    "-q",
                    "--platform",
                    "a2a3",
                    "--runtime",
                    runtime,
                    "--manual",
                    "include",
                ],
                plugins=[collector],
            )
        captured.seek(0)
        if rc:
            raise RuntimeError(captured.read()[-16000:])
    return sorted({row["id"]: row for row in collector.items}.values(), key=lambda row: row["id"])


def child(pytest_args: list[str], destination: Path) -> int:
    import pytest  # noqa: PLC0415

    class Capture:
        def pytest_sessionfinish(self, session):
            with destination.open("w") as output:
                paths = set()
                if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
                    paths.update(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]).glob("queue_peaks.*.log"))
                from _task_interface import _host_log_directory  # noqa: PLC0415

                bound = _host_log_directory()
                if bound:
                    paths.update(Path(bound).glob("queue_peaks.*.log"))
                for path in sorted(paths):
                    with path.open(errors="replace") as source:
                        for line in source:
                            if "QPEAK" in line:
                                output.write(line)

    return int(pytest.main(pytest_args, plugins=[Capture()]))


def run_case(spec: dict, repeat: int | str, rounds: int, output: Path) -> dict:
    slug = hashlib.sha256(spec["id"].encode()).hexdigest()[:16]
    run_dir = output / "runs" / slug / str(repeat)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ascend").mkdir(exist_ok=True)
    result = {
        "id": spec["id"],
        "repeat": repeat,
        "runtime": spec["runtime"],
        "rounds": rounds,
        "observations": [],
        "run_dir": str(run_dir.relative_to(output)),
    }
    if "network1" in spec["nodeid"] and not all(
        os.environ.get(key) for key in ("NETWORK1_REMOTE_ENDPOINT", "NETWORK1_REMOTE_DEVICES")
    ):
        return {**result, "status": "skip", "reason": "Network1 peer environment unavailable", "returncode": None}
    args = [
        spec["nodeid"],
        "--platform",
        "a2a3",
        "--runtime",
        spec["runtime"],
        "--manual",
        "include",
        "--skip-golden",
        "--require-pto-isa",
        "--pto-session-timeout",
        "1200",
        "--rounds",
        str(rounds),
        "-v",
        "-s",
    ]
    if spec["selector"]:
        args += ["--case", spec["selector"]]
    launcher = [
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--child",
        str(run_dir / "telemetry.log"),
    ]
    command = shlex.join([*launcher, "--", *args]) + ' --device "$TASK_DEVICE"'
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1", PYTHONPATH=f"{ROOT}:{ROOT / 'python'}", ASCEND_PROCESS_LOG_PATH=str(run_dir / "ascend")
    )
    with (run_dir / "pytest.log").open("w") as log:
        rc = subprocess.run(
            [
                "task-submit",
                "--timeout",
                "1800",
                "--max-time",
                "1800",
                "--device",
                "auto",
                "--device-num",
                str(spec["device_count"]),
                "--run",
                command,
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    selected_lines = []
    skipped = False
    with (run_dir / "pytest.log").open(errors="replace") as source:
        for line in source:
            if "QPEAK" in line:
                selected_lines.append(line)
            skipped |= " skipped" in line
    telemetry = run_dir / "telemetry.log"
    if telemetry.exists():
        selected_lines.append(telemetry.read_text(errors="replace"))
    rows, errors = parse_records("".join(selected_lines))
    status = "fail" if rc else "invalid" if errors else "pass" if rows else "skip" if skipped else "no_data"
    if status == "pass" and spec["fault"]:
        status = "expected_fault"
    result.update(status=status, returncode=rc, errors=errors, observations=rows, command=command)
    stdout = run_dir / "pytest.log"
    if status == "pass" and stdout.stat().st_size > 1024 * 1024:
        with stdout.open("rb") as source, gzip.open(stdout.with_suffix(".log.gz"), "wb") as compressed:
            shutil.copyfileobj(source, compressed)
        stdout.unlink()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(output: Path) -> None:
    path = output / "results.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    peaks = {}
    case_peaks = {}
    for result in results:
        if result["status"] != "pass":
            continue
        for row in result["observations"]:
            if row["execution_rc"] != 0:
                continue
            for metric in ["peak", "spmd", "normal"] if row["classified"] else ["peak"]:
                key = (row["runtime"], row["queue"], row["index"], metric)
                observation = {
                    "runtime": row["runtime"],
                    "queue": row["queue"],
                    "index": row["index"],
                    "metric": metric,
                    "value": row[metric],
                    "capacity": row["cap"],
                    "case": result["id"],
                    "repeat": result["repeat"],
                    "pid": row["pid"],
                    "epoch": row["epoch"],
                }
                if key not in peaks or row[metric] > peaks[key]["value"]:
                    peaks[key] = observation
                case_key = (result["id"], *key)
                if case_key not in case_peaks or row[metric] > case_peaks[case_key]["value"]:
                    case_peaks[case_key] = observation
    columns = ["runtime", "queue", "index", "metric", "value", "capacity", "case", "repeat", "pid", "epoch"]
    for filename, table in (("peaks.csv", peaks), ("case_peaks.csv", case_peaks)):
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(table[key] for key in sorted(table))
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("pass", "fail", "invalid", "no_data", "skip", "expected_fault")
    }
    inventory = output / "inventory.json"
    candidate_count = len(json.loads(inventory.read_text())) if inventory.exists() else None
    coverage = {
        "counts": counts,
        "attempts": len(results),
        "candidate_cases": candidate_count,
        "attempted_cases": len({r["id"] for r in results}),
        "attempted_rounds": sum(r.get("rounds", 1) for r in results),
        "cases_with_valid_data": len({r["case"] for r in case_peaks.values()}),
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    lines = [
        "# A3 queue peaks",
        "",
        f"Coverage: `{json.dumps(counts)}`",
        "",
        "Only successful runs with complete reports contribute. Missing data is not zero. "
        "Units are queue slots. Class peaks are independent maxima.",
        "Accounting precedes publication and slot reuse; counted unpublished entries are included.",
        "",
        "| Runtime | Queue | Index | Metric | Peak | Capacity |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for key in sorted(peaks):
        row = peaks[key]
        lines.append(
            f"| {row['runtime']} | {row['queue']} | {row['index']} | {row['metric']} "
            f"| {row['value']} | {row['capacity']} |"
        )
    lines += ["", "Peak provenance is in peaks.csv. Failed and missing reports are in results.jsonl."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/a3-queue-peaks")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--runtime", choices=RUNTIMES)
    parser.add_argument("--match", default="")
    parser.add_argument("--retry-from", type=Path, help="Retry unsuccessful attempts from another output directory")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--inventory-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inventory_child:
        print(json.dumps(collect(args.runtime)))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(output)
        return 0
    subprocess.run([str(ROOT / ".claude/skills/onboard-arch-precheck/check.sh"), "a2a3"], check=True)
    inventory = []
    for runtime in [args.runtime] if args.runtime else RUNTIMES:
        data = subprocess.check_output(
            [sys.executable, __file__, "--inventory-child", "--runtime", runtime], cwd=ROOT, text=True
        )
        inventory.extend(json.loads(data))
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Collected {len(inventory)} cases", flush=True)
    if args.collect_only:
        return 0
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=ROOT)
    digest = hashlib.sha256(head + diff)
    for path in sorted((ROOT / "src/common/utils").glob("queue_peaks*.h")):
        digest.update(path.read_bytes())
    identity = {"head": head.decode().strip(), "source_digest": digest.hexdigest(), "repeats": args.repeats}
    library_dir = ROOT / "build/lib"
    artifact_hashes = {}
    for path in sorted(library_dir.glob("a2a3/onboard/*/*.so")):
        artifact_hashes[str(path.relative_to(library_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity["artifacts"] = artifact_hashes
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("Source or repeat count changed: use a fresh output directory")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    results_path = output / "results.jsonl"
    previous = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    done = {(r["id"], r["repeat"]) for r in previous}
    retry_keys = None
    if args.retry_from:
        prior = [json.loads(line) for line in (args.retry_from / "results.jsonl").read_text().splitlines()]
        retry_keys = {(r["id"], r["repeat"]) for r in prior if r["status"] in ("fail", "invalid", "no_data")}
    jobs = []
    for spec in inventory:
        if args.match not in spec["id"]:
            continue
        repetitions = (
            [(f"1-{args.repeats}", args.repeats)]
            if spec["batch_rounds"]
            else [(n, 1) for n in range(1, args.repeats + 1)]
        )
        jobs.extend(
            (spec, repeat, rounds)
            for repeat, rounds in repetitions
            if (spec["id"], repeat) not in done and (retry_keys is None or (spec["id"], repeat) in retry_keys)
        )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool, results_path.open("a") as stream:
        futures = [pool.submit(run_case, *job, output) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"{result['status']}: {result['id']} repeat={result['repeat']}", flush=True)
    summarize(output)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        raise SystemExit(child(sys.argv[4:], Path(sys.argv[2])))
    raise SystemExit(main())
