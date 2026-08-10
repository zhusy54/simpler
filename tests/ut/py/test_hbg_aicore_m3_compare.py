# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from simpler_setup.tools.hbg_aicore_m3_compare import compare_logs


def _log(rounds):
    lines = []
    for inv, duration in enumerate(rounds, 1):
        lines.append(
            f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=abc depth=0 name=simpler_run ts={inv * 10000} dur={duration}"
        )
        lines.append(
            f"[STRACE] v=1 pid=1 tid=1 inv={inv} hid=abc depth=1 "
            f"name=simpler_run.runner_run.device_wall ts=0 dur={duration // 2} clk=dev"
        )
    return "\n".join(lines) + "\n"


def test_compare_preserves_samples_and_flags_more_than_two_percent(tmp_path):
    hbg = tmp_path / "hbg.log"
    aicore = tmp_path / "aicore.log"
    hbg.write_text(_log([100_000, 102_000, 104_000]))
    aicore.write_text(_log([104_000, 106_000, 108_000]))

    result = compare_logs(hbg, aicore)

    assert result["samples_us"]["hbg"]["Host"] == [100.0, 102.0, 104.0]
    assert result["summary"]["hbg"]["Host"]["p50_us"] == 102.0
    assert result["comparison"]["Host"]["change_percent"] > 2.0
    assert result["comparison"]["Host"]["potential_regression"] is True
