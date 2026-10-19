# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Disabled profiling must not inspect historical captures in the working directory."""

import json
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize("level", [0, 3])
def test_clock_log_hook_checks_only_enabled_captures(tmp_path, monkeypatch, level):
    scene = import_module("simpler_setup.scene_test")
    hook = import_module("tests.st.a5.host_build_graph.dfx.chip_swimlane.test_scheduler_phases")
    capture = tmp_path / "outputs" / "previous_capture"
    capture.mkdir(parents=True)
    (capture / "chip_swimlane_records.json").write_text(
        json.dumps(
            {
                "chip_swimlane_level": 3,
                "metadata": {"orchestrator_source": "host", "clock_alignment": {"status": "bounded"}},
            }
        )
    )
    (capture / "host_clock_alignment.1.log").write_text("")
    monkeypatch.chdir(tmp_path)
    token = object()
    monkeypatch.setattr(scene, "finalize_diagnostic_outputs", lambda *args, **kwargs: token)
    hook._verify_runtime_clock_logs(monkeypatch, "a5")
    if level:
        with pytest.raises(AssertionError):
            scene.finalize_diagnostic_outputs("capture", Path("."), chip_swimlane=level)
    else:
        assert scene.finalize_diagnostic_outputs("disabled", Path("."), chip_swimlane=level) is token
