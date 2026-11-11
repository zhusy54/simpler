# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact block placement through ordinary/native Graphs and finite SRAM."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from tests.st.wse.host_build_graph.counter import test_counter as counter
from tests.st.wse.host_build_graph.counter.test_counter import config_for

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@lru_cache
def build_probe(platform, orchestration="probe_orch.cpp"):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent / "kernels"
    child = CoreCallable.build(
        signature=[D.INOUT] * 3 + [D.IN] * 2,
        binary=compiler.compile_incore(
            str(here / "probe.cpp"),
            core_type="aic",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        ),
    )
    return ChipCallable.build(
        signature=[D.INOUT] * 3 + [D.IN] * 4,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(here / orchestration)),
        children=[(0, child)],
    )


def probe_args(worker, blocks, mode, native, token, owner):
    tensors = [
        (torch.arange(1024).float() - 513).reshape(32, 32),
        torch.full((2, blocks, 32, 32), -999.0),
        torch.full((2, blocks, 8), 999, dtype=torch.uint64),
    ]
    buffers = []
    args = TaskArgs()
    for index, tensor in enumerate(tensors):
        buffer = worker.malloc(tensor.nbytes, core_id=owner if index < 2 else 0)
        worker.copy_to(buffer, tensor)
        args.add_tensor(
            buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.UINT64 if index == 2 else DataType.FLOAT32),
            TensorArgType.INOUT,
        )
        buffers.append(buffer)
    for value in (blocks, mode, int(native), token):
        args.add_scalar(value)
    return args, buffers, tensors


def expected_target(block, cores, mode, stage):
    if mode == 3:
        return cores - 1, 3 - stage
    core, pe = block % cores, 0
    return ((core + 1) % cores, (pe + 1 + stage) % 4) if mode == 2 else (core, pe)


def check_probe(worker, path, buffers, tensors, cores, mode, native, token, owner):
    for index in (1, 2):
        worker.copy_from(tensors[index], buffers[index])
    blocks = tensors[1].shape[1]
    golden = torch.relu(tensors[0]).expand_as(tensors[1])
    torch.testing.assert_close(tensors[1], golden, rtol=0, atol=0)
    events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
    expected_tasks = 4 if native else 2
    assert events[0]["expected_tasks"] == expected_tasks
    done = [event for event in events[1:] if event["kind"] == 6]
    assert sorted(event["value"] for event in done) == list(range(1, expected_tasks + 1))
    for stage in range(2):
        task = stage + int(native)
        for kind in (16, 17):
            emitted = [event for event in events[1:] if event["kind"] == kind and event["task"] == task]
            assert sorted(event["block"] for event in emitted) == list(range(blocks))
            for event in emitted:
                assert (event["core"], event["pe"]) == expected_target(event["block"], cores, mode, stage)
        for block in range(blocks):
            core, pe = expected_target(block, cores, mode, stage)
            assert tensors[2][stage, block].tolist() == [
                block,
                core,
                pe,
                owner,
                owner,
                buffers[0].base,
                buffers[1].base + (stage * blocks + block) * 4096,
                token,
            ]
    memory = json.loads((path / "wse_sram.json").read_text())
    queues = [section for section in memory["sections"] if section["kind"] == 9]
    assert sum(section["bytes"] for section in queues) == 2 * blocks * 56
    for section in memory["sections"]:
        core = memory["cores"][section["owner"]]
        assert core["base"] + core["capacity"] - core["reserve"] <= section["address"]
        assert section["address"] + section["bytes"] <= core["base"] + core["capacity"]
    assert all(worker.sram_info(core)["runtime_used_bytes"] == 0 for core in range(len(memory["cores"])))
    (path / "deployment-evidence.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cores": cores,
                "blocks": blocks,
                "mode": mode,
                "native": native,
                "owner": owner,
                "records": tensors[2].tolist(),
                "allocations": [
                    {"identity": str(buf.identity), "base": buf.base, "bytes": buf.nbytes} for buf in buffers
                ],
            },
            indent=2,
        )
    )


def configured_worker(platform, cores):
    worker = Worker(level=2, platform=platform, runtime="host_build_graph", device_id=0)
    worker.init()
    worker.configure_wse({"hardware": {"core_count": cores, "sram_bytes_per_core": 100 * 1024 * 1024}})
    return worker


@pytest.mark.parametrize("native", [False, True])
def test_deployment_block_counts_and_replay(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        for blocks in (39, 40, 41, 159, 160, 161):
            for iteration, mode in enumerate((0, 1, 2, 3, 2)):
                path = tmp_path / f"{blocks}-{iteration}"
                token = blocks * 10 + iteration
                args, buffers, tensors = probe_args(worker, blocks, mode, native, token, 3)
                worker.run(handle, args, config_for(path, 40))
                check_probe(worker, path, buffers, tensors, 40, mode, native, token, 3)
                for buffer in buffers:
                    worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()


@pytest.mark.parametrize("cores", [40])
def test_deployment_topology_and_remote_owner(tmp_path, st_platform, cores):
    device_cores = cores
    worker = configured_worker(st_platform, device_cores)
    try:
        handle = worker.register(build_probe(st_platform))
        for mode in (0, 2):
            args, buffers, tensors = probe_args(worker, cores * 4 + 1, mode, True, 123, device_cores - 1)
            path = tmp_path / str(mode)
            worker.run(handle, args, config_for(path, 0 if cores == 40 else cores))
            check_probe(worker, path, buffers, tensors, cores, mode, True, 123, device_cores - 1)
            for buffer in buffers:
                worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def assert_no_launch(path):
    events = path / "wse_events.jsonl"
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        assert not any(row.get("kind") in (3, 4) for row in rows)


@pytest.mark.parametrize("native", [False, True])
def test_deployment_rejects_invalid_and_recovers(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        for mode in (4, 5, 6, 7, 8, 10, 11, *([9] if native else [])):
            path = tmp_path / f"invalid-{mode}"
            args, buffers, _ = probe_args(worker, 5, mode, native, 0, 3)
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config_for(path, 40))
            assert_no_launch(path)
            assert all(worker.sram_info(core)["runtime_used_bytes"] == 0 for core in range(40))
            for buffer in buffers:
                worker.free(buffer)
            args, buffers, tensors = probe_args(worker, 5, 2, native, mode, 3)
            path = tmp_path / f"recovery-{mode}"
            worker.run(handle, args, config_for(path, 40))
            check_probe(worker, path, buffers, tensors, 40, 2, native, mode, 3)
            for buffer in buffers:
                worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_deployment_table_and_skew_queue_capacity(tmp_path, st_platform):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform))
        args, buffers, tensors = probe_args(worker, 19, 0, True, 1, 3)
        worker.run(handle, args, config_for(tmp_path / "sizing", 40))
        check_probe(worker, tmp_path / "sizing", buffers, tensors, 40, 0, True, 1, 3)
        memory = json.loads((tmp_path / "sizing/wse_sram.json").read_text())
        for buffer in buffers:
            worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()
    for constrained, mode in ((0, 12), (39, 3)):
        reserves = [256 * 1024] * 40
        reserves[constrained] = ((memory["cores"][constrained]["runtime_used"] + 127) // 128) * 128
        worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
        try:
            worker.init()
            worker.configure_wse(
                {
                    "hardware": {"core_count": 40, "sram_bytes_per_core": 100 * 1024 * 1024},
                    "runtime": {"sram_reserve_bytes_per_core": reserves},
                }
            )
            handle = worker.register(build_probe(st_platform))
            args, buffers, _ = probe_args(worker, 19, mode, True, 2, 3)
            path = tmp_path / f"reject-{constrained}"
            with pytest.raises(RuntimeError):
                worker.run(handle, args, config_for(path, 40))
            assert_no_launch(path)
            for buffer in buffers:
                worker.free(buffer)
            args, buffers, tensors = probe_args(worker, 1, 0, True, 3, 3)
            path = tmp_path / f"small-{constrained}"
            worker.run(handle, args, config_for(path, 40))
            check_probe(worker, path, buffers, tensors, 40, 0, True, 3, 3)
            for buffer in buffers:
                worker.free(buffer)
            worker.unregister(handle)
        finally:
            worker.close()


@lru_cache
def build_deployed_counter(platform):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent / "kernels"
    child = CoreCallable.build(
        signature=[D.INOUT] * 3 + [D.IN] * 5,
        binary=compiler.compile_incore(
            str(here / "counter.cpp"),
            core_type="aic",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph")
            + [str(here.parent.parent / "counter/kernels")],
        ),
    )
    return ChipCallable.build(
        signature=[D.INOUT] * 3 + [D.IN] * 6,
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(here / "counter_orch.cpp")),
        children=[(0, child)],
    )


class CounterSramWorker(Worker):
    def malloc(self, size, *, core_id, alignment=64):
        index = getattr(self, "_counter_allocation", 0)
        self._counter_allocation = index + 1
        return super().malloc(size, core_id=(3, 2, 0)[index % 3], alignment=alignment)


def test_deployment_sram_counter(tmp_path, st_platform):
    worker = CounterSramWorker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        worker.configure_wse({"hardware": {"core_count": 40}})
        handle = worker.register(build_deployed_counter(st_platform))
        for iteration, variant in enumerate((0, 1, 0)):
            base = iteration * 17
            args, buffers, tensors = counter.prepare(worker, base, variant)
            path = tmp_path / str(iteration)
            worker.run(handle, args, config_for(path, 40))
            placements = [[0, 0], [0, 1], [1, 2], [1, 3]]
            if variant:
                placements = [[1 - core, (pe + 1) % 4] for core, pe in placements]
            counter.check(worker, buffers, tensors, base, 40, path, False, placements=placements)
            ownership = []
            for buffer in buffers:
                owners = [
                    core
                    for core in range(40)
                    if worker.sram_info(core)["base"] <= buffer.base
                    and buffer.base + buffer.nbytes
                    <= worker.sram_info(core)["base"]
                    + worker.sram_info(core)["capacity_bytes"]
                    - worker.sram_info(core)["runtime_reserve_bytes"]
                ]
                assert len(owners) == 1
                ownership.append(owners[0])
            assert ownership == [3, 2, 0]
            (path / "counter-deployment.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "placements": placements,
                        "owners": ownership,
                        "allocations": [{"base": buffer.base, "bytes": buffer.nbytes} for buffer in buffers],
                    },
                    indent=2,
                )
            )
            for buffer in buffers:
                worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()


def test_deployment_timeout_ownership_and_close(tmp_path, st_platform):
    gate = counter.load_gate(tmp_path, "wse_counter_probe", 12)
    worker = CounterSramWorker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        worker.configure_wse({"hardware": {"core_count": 40}})
        handle = worker.register(build_deployed_counter(st_platform))
        args, buffers, tensors = counter.prepare(worker, 29, 1, gated=True)
        path = tmp_path / "wait"
        run = worker.submit(handle, args, config_for(path, 40))
        assert gate.wait_probe()
        with pytest.raises(TimeoutError):
            run.wait(timeout=0.01)
        for buffer in buffers:
            with pytest.raises(RuntimeError, match="in.flight"):
                worker.free(buffer)
        gate.release_probe()
        run.wait()
        placements = [[1, 1], [1, 2], [0, 3], [0, 0]]
        counter.check(worker, buffers, tensors, 29, 40, path, False, placements=placements)
        for buffer in buffers:
            worker.free(buffer)
        args2, _, _ = counter.prepare(worker, 41, 0)
        worker.submit(handle, args2, config_for(tmp_path / "close", 40))
        worker.close()
        events = [json.loads(line) for line in (tmp_path / "close/wse_events.jsonl").read_text().splitlines()][1:]
        assert sum(event["kind"] == 9 for event in events) == 1
        assert sum(event["kind"] == 8 for event in events) == 40
        assert sum(event["kind"] == 7 for event in events) == 160
    finally:
        gate.release_probe()
        worker.close()


@pytest.mark.parametrize("native", [False, True])
def test_deployment_cached_binding_and_predicate(tmp_path, st_platform, native):
    worker = configured_worker(st_platform, 40)
    try:
        handle = worker.register(build_probe(st_platform, "binding_orch.cpp"))
        for enabled in (False, True):
            args, buffers, tensors = probe_args(worker, 1, int(enabled), native, 91, 0)
            path = tmp_path / str(enabled)
            worker.run(handle, args, config_for(path, 40))
            if enabled:
                check_probe(worker, path, buffers, tensors, 40, 3, native, 91, 0)
                for buffer, tensor in zip(buffers, tensors, strict=True):
                    if tensor.dtype == torch.uint64:
                        tensor.fill_(999)
                    worker.copy_to(buffer, tensor)
                rejected = tmp_path / "shrink"
                with pytest.raises(RuntimeError):
                    worker.run(handle, args, config_for(rejected, 39))
                assert_no_launch(rejected)
                restored = tmp_path / "restore"
                worker.run(handle, args, config_for(restored, 40))
                check_probe(worker, restored, buffers, tensors, 40, 3, native, 91, 0)
            else:
                for buffer, tensor in zip(buffers[1:], tensors[1:], strict=True):
                    worker.copy_from(tensor, buffer)
                assert bool(torch.all(tensors[1] == -999))
                assert bool(torch.all(tensors[2] == 999))
                events = [json.loads(line) for line in (path / "wse_events.jsonl").read_text().splitlines()]
                skipped = sorted(
                    (event["task"], event["core"], event["pe"]) for event in events[1:] if event["kind"] == 11
                )
                assert skipped == [(int(native), 39, 0), (int(native) + 1, 39, 0)]
                expected = 4 if native else 2
                assert sorted(event["value"] for event in events[1:] if event["kind"] == 6) == list(
                    range(1, expected + 1)
                )
            for buffer in buffers:
                worker.free(buffer)
        worker.unregister(handle)
    finally:
        worker.close()
