#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare HBG and HBG-AICore ``[STRACE]`` benchmark samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from simpler_setup.tools.strace_timing import (
    _ROUNDS_TABLE_COLUMNS,
    _round_metrics,
    count_record_heads,
    group_invocations,
    parse_spans,
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_samples(path: Path) -> tuple[dict[str, list[float]], int, int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head_count = count_record_heads(lines)
    spans = list(parse_spans(lines))
    samples = {name: [] for name in _ROUNDS_TABLE_COLUMNS}
    for invocation in group_invocations(spans):
        row = _round_metrics(invocation)
        for index, name in enumerate(_ROUNDS_TABLE_COLUMNS):
            if index == 0 or row[index] > 0.0:
                samples[name].append(row[index])
    return samples, head_count, len(spans)


def _summarize(samples: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    summary = {}
    for metric, values in samples.items():
        if values:
            summary[metric] = {
                "count": len(values),
                "mean_us": sum(values) / len(values),
                "p50_us": _percentile(values, 0.50),
                "p95_us": _percentile(values, 0.95),
                "p99_us": _percentile(values, 0.99),
            }
    return summary


def compare_logs(hbg_log: Path, aicore_log: Path, regression_threshold: float = 2.0) -> dict:
    hbg_samples, hbg_heads, hbg_spans = _read_samples(hbg_log)
    aicore_samples, aicore_heads, aicore_spans = _read_samples(aicore_log)
    hbg_summary = _summarize(hbg_samples)
    aicore_summary = _summarize(aicore_samples)
    comparisons = {}
    for metric in _ROUNDS_TABLE_COLUMNS:
        if metric not in hbg_summary or metric not in aicore_summary:
            continue
        baseline = float(hbg_summary[metric]["p50_us"])
        current = float(aicore_summary[metric]["p50_us"])
        change = 0.0 if baseline == 0.0 else (current - baseline) * 100.0 / baseline
        comparisons[metric] = {
            "hbg_p50_us": baseline,
            "hbg_aicore_p50_us": current,
            "change_percent": change,
            "potential_regression": change > regression_threshold,
        }
    return {
        "schema_version": 1,
        "regression_threshold_percent": regression_threshold,
        "inputs": {
            "hbg": {"path": str(hbg_log), "record_heads": hbg_heads, "complete_spans": hbg_spans},
            "hbg_aicore": {
                "path": str(aicore_log),
                "record_heads": aicore_heads,
                "complete_spans": aicore_spans,
            },
        },
        "samples_us": {"hbg": hbg_samples, "hbg_aicore": aicore_samples},
        "summary": {"hbg": hbg_summary, "hbg_aicore": aicore_summary},
        "comparison": comparisons,
    }


def _print_report(result: dict) -> None:
    print("Metric      HBG p50 (us)  HBG-AICore p50 (us)  Change      Assessment")
    print("----------  ------------  -------------------  ----------  --------------------")
    for metric in _ROUNDS_TABLE_COLUMNS:
        row = result["comparison"].get(metric)
        if row is None:
            continue
        assessment = "POTENTIAL REGRESSION" if row["potential_regression"] else "review/noise band"
        print(
            f"{metric:<10}  {row['hbg_p50_us']:>12.1f}  {row['hbg_aicore_p50_us']:>19.1f}  "
            f"{row['change_percent']:>+9.2f}%  {assessment}"
        )
    for side in ("hbg", "hbg_aicore"):
        meta = result["inputs"][side]
        if meta["record_heads"] != meta["complete_spans"]:
            print(
                f"warning: {side} retained {meta['complete_spans']}/{meta['record_heads']} complete STRACE records",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbg-log", required=True, type=Path)
    parser.add_argument("--hbg-aicore-log", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--regression-threshold", type=float, default=2.0)
    args = parser.parse_args(argv)
    result = compare_logs(args.hbg_log, args.hbg_aicore_log, args.regression_threshold)
    if not result["comparison"]:
        print("error: no comparable complete [STRACE] samples", file=sys.stderr)
        return 2
    _print_report(result)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
