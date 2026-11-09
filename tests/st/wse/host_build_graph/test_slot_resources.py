# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real two-slot ownership through the installed native Python binding."""

import time

import pytest
import torch
from simpler.worker import Worker

from simpler_setup.scene_test import _build_chip_task_args, _compare_outputs
from tests.st.a5.host_build_graph.vector_example.test_vector_example import (
    TestVectorExampleHostBuildGraphA5 as _Vector,
)

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_two_slots_keep_distinct_arenas_and_retained_outputs(st_platform):
    case = _Vector()
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    pending = []
    try:
        worker.init()
        chip = worker._chip_worker
        assert chip.pipeline_depth == chip.runtime_slot_count == 2
        handle = worker.register(case.build_callable(st_platform))
        state = worker._resolve_handle(handle)
        config = case._build_config({})
        addresses = []
        for index, slot in enumerate((0, 1, 0, 1)):
            values = case.generate_args({})
            values.a.add_(index)
            values.b.add_(2 * index)
            sentinel = -1000.0 - index
            values.f.fill_(sentinel)
            expected = values.clone()
            case.compute_golden(expected, {})
            args, outputs = _build_chip_task_args(values, case.CALLABLE["orchestration"]["signature"])
            run = chip._prepare_native_run_with_pipeline_lease(
                state.slot_id, args, config=config, slot_id=slot, generation=index + 1
            )
            pending.append((run, args, values, expected, outputs, sentinel))
            storage, staging = chip.runtime_buffer_addrs[slot], chip.retained_temp_addr(slot)
            assert storage and staging
            assert all(storage != old_storage and staging != old_staging for old_storage, old_staging in addresses)
            addresses.append((storage, staging))
            chip._launch_native_run(run)
            deadline = time.monotonic() + 30
            while not chip._poll_native_run(run):
                assert time.monotonic() < deadline, "native execution did not retire"
                time.sleep(0)
            for _, _, retained, _, _, initial in pending:
                assert torch.all(retained.f == initial), "poll delivered an unobserved output"
        banks = [chip.arena_bank_gm_heap_base(slot) for slot in (0, 1)]
        assert all(banks) and banks[0] != banks[1], banks
        while pending:
            run, _, values, expected, outputs, _ = pending[-1]
            chip._wait_native_run(run)
            chip._finalize_native_run(run)
            pending.pop()
            _compare_outputs(values, expected, outputs, case.RTOL, case.ATOL)
        worker.unregister(handle)
    finally:
        # Raw native runs are not public RunHandles; retain their buffers until
        # their native owners are drained even when the ownership oracle fails.
        for run, *_ in reversed(pending):
            worker._chip_worker._wait_native_run(run)
            worker._chip_worker._finalize_native_run(run)
        worker.close()
