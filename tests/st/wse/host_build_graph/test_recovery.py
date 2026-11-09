# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE error containment through the same native probes used by A5."""

import pytest

from tests.st.a5.host_build_graph.resident_failure.test_resident_failure import (
    _exercise_failure,
)
from tests.st.a5.host_build_graph.resident_failure.test_resident_failure import (
    test_native_graph_failure_does_not_release_exit as _graph_failure,
)
from tests.ut.py.test_sim_startup_failure import (
    startup_faults as startup_faults,  # noqa: PLC0414 -- pytest fixture export
)
from tests.ut.py.test_sim_startup_failure import (
    test_sim_partial_startup_failure_is_drained as _partial_startup,
)

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@pytest.mark.parametrize(
    ("mode", "thread"), [("bind", "sim-aic-1"), ("bind", "sim-aicpu-1"), ("create", "sim-aicpu-0")]
)
def test_partial_startup_drains_created_threads(mode, thread, startup_faults):  # noqa: F811 -- fixture injection
    _partial_startup("wse", "host_build_graph", mode, thread, startup_faults)


def test_scheduler_timeout_poison_and_new_worker_recovery(st_platform, st_device_ids, monkeypatch):
    _exercise_failure(st_platform, st_device_ids, 7, 0, 2, monkeypatch, 2500)


def test_graph_failure_does_not_release_successor(st_platform, st_device_ids, monkeypatch):
    _graph_failure(st_platform, st_device_ids, 7, 0, monkeypatch)
