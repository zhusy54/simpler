# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Queue peak report integrity and aggregation contracts."""

import csv
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "a3_queue_peaks", Path(__file__).resolve().parents[3] / "tools/a3_queue_peaks.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def report(pid=1, peak=6, spmd=6, normal=6):
    key = f"runtime=host_build_graph pid={pid} epoch=1"
    return (
        f"QPEAK_BEGIN {key} count=1 rc=0\n"
        f"QPEAK {key} queue=ready index=0 cap=16 peak={peak} spmd={spmd} normal={normal} classified=1\n"
        f"QPEAK_END {key} count=1 rc=0\n"
    )


def test_deduplicate_logs_but_keep_processes():
    rows, errors = module.parse_records(report() * 2 + report(pid=2))
    assert not errors
    assert len(rows) == 2


def test_missing_end_is_not_zero():
    rows, errors = module.parse_records(report().split("QPEAK_END")[0])
    assert not rows
    assert errors
    assert module.parse_records("") == ([], [])


def test_independent_maxima_and_capacity():
    assert not module.parse_records(report())[1]
    assert module.parse_records(report(peak=17))[1]
    assert module.parse_records(report(peak=5, spmd=6))[1]


def test_failures_excluded_and_peak_provenance(tmp_path):
    records = []
    for pid, peak, spmd, normal, status in [(1, 9, 9, 0, "pass"), (2, 8, 0, 8, "pass"), (3, 15, 15, 0, "fail")]:
        rows, _ = module.parse_records(report(pid, peak, spmd, normal))
        records.append(dict(id=f"case{pid}", repeat=1, status=status, observations=rows))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    module.summarize(tmp_path)
    with (tmp_path / "peaks.csv").open() as stream:
        values = {r["metric"]: r for r in csv.DictReader(stream)}
    assert values["peak"]["value"] == "9"
    assert values["normal"]["case"] == "case2"
    assert values["spmd"]["case"] == "case1"


def test_expected_device_failure_is_not_successful_peak(tmp_path):
    rows, errors = module.parse_records(report().replace("rc=0", "rc=7"))
    assert not errors
    assert rows[0]["execution_rc"] == 7
    (tmp_path / "results.jsonl").write_text(json.dumps(dict(id="caught", repeat=1, status="pass", observations=rows)))
    module.summarize(tmp_path)
    with (tmp_path / "peaks.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
