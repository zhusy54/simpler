# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real WSE HBG coverage, using the unchanged A5 source scenes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.st.wse.host_build_graph.audit_capture import audit

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_original_scenes_with_complete_native_coverage(tmp_path, st_platform):
    root = Path(__file__).resolve().parents[4]
    destination = tmp_path / "capture"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.st.wse.host_build_graph.run_capture",
            "--platform",
            st_platform,
            "--out",
            str(destination),
        ],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = audit(destination)
    (destination / "audit.json").write_text(json.dumps(report, indent=2))
    assert report["status"] == "PASS"

    trace_path = destination / "mix-0/chip_swimlane_records.json"
    original = trace_path.read_text()
    try:
        trace = json.loads(original)
        trace["aicore_execution_identities"]["records"].pop()
        trace_path.write_text(json.dumps(trace))
        with pytest.raises(AssertionError):
            audit(destination)
        trace = json.loads(original)
        trace["metadata"]["host_capture"]["dropped_records"] = 1
        trace_path.write_text(json.dumps(trace))
        with pytest.raises(AssertionError):
            audit(destination)
    finally:
        trace_path.write_text(original)
