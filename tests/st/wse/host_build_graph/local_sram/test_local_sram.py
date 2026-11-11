# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE SRAM ownership, runtime residency and counter publication."""

import json
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from tests.st.wse.host_build_graph.counter import test_counter as counter

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_sram_allocator_and_configuration(st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        spec = worker.wse_configuration()
        assert spec["hardware"]["core_count"] == 40
        assert spec["hardware"]["sram_bytes_per_core"] == [104857600] * 40
        assert spec["runtime"]["sram_reserve_bytes_per_core"] == [262144] * 40
        assert not hasattr(worker, "malloc_sram")
        with pytest.raises(ValueError, match="explicit core_id"):
            worker.malloc(64)
        with pytest.raises(RuntimeError):
            worker._chip_worker.malloc(64)
        for bad in (
            {"extra": 1},
            {"hardware": {"pe_per_core": 3}},
            {"hardware": {"core_count": True}},
            {"hardware": {"sram_bytes_per_core": [1024]}},
            {"hardware": {"core_count": 2**32}},
            {"runtime": {"unknown": 2}},
        ):
            with pytest.raises((ValueError, RuntimeError)):
                worker.configure_wse(bad)
        worker.configure_wse(
            {
                "hardware": {"core_count": 40, "sram_bytes_per_core": [4096, 8192] + [4096] * 38},
                "runtime": {"sram_reserve_bytes_per_core": 1024},
            }
        )
        assert worker.sram_info(0)["application_free_bytes"] == 3072
        buf = worker.malloc(3072, core_id=0, alignment=128)
        other = worker.malloc(64, core_id=1)
        data = torch.arange(768, dtype=torch.int32)
        worker.copy_to(buf, data)
        output = torch.empty_like(data)
        worker.copy_from(output, buf)
        assert torch.equal(output, data)
        with pytest.raises(RuntimeError, match="malloc failed"):
            worker.malloc(1, core_id=0)
        with pytest.raises(RuntimeError, match="frozen"):
            worker.configure_wse({})
        worker.free(buf)
        with pytest.raises((ValueError, RuntimeError), match="freed|released|stale|closed"):
            worker.copy_from(output, buf)
        again = worker.malloc(3072, core_id=0, alignment=128)
        assert worker.sram_info(0)["application_used_bytes"] == 3072
        worker.free(again)
        worker.free(other)
        assert worker.sram_info(0)["largest_free_bytes"] == 3072
        assert worker.sram_info(1)["application_used_bytes"] == 0
    finally:
        worker.close()


def test_sram_all_core_owners_and_physical_accounting(st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        buffers = [worker.malloc(128, core_id=core, alignment=128) for core in range(40)]
        regions = [worker.sram_info(core) for core in range(40)]
        assert len({region["base"] for region in regions}) == 40
        assert worker._chip_worker.committed_device_memory == 40 * 104857600
        for core, buffer in enumerate(buffers):
            value = torch.full((32,), core + 1, dtype=torch.int32)
            worker.copy_to(buffer, value)
        for core, buffer in enumerate(buffers):
            output = torch.empty(32, dtype=torch.int32)
            worker.copy_from(output, buffer)
            assert output.tolist() == [core + 1] * 32
            assert regions[core]["application_used_bytes"] == 128
            worker.free(buffer)
            assert worker.sram_info(core)["application_used_bytes"] == 0
        assert worker._chip_worker.committed_device_memory == 40 * 104857600
        host = torch.zeros(1, dtype=torch.int32)
        with pytest.raises(RuntimeError):
            worker._chip_worker.copy_to(host.data_ptr(), host.data_ptr(), host.nbytes)
    finally:
        worker.close()


@pytest.mark.parametrize("cores,split", [(40, True), (0, False)])
def test_sram_counter_replay_and_residency(tmp_path, st_platform, monkeypatch, cores, split):
    counter.test_counter_graph_rebind_and_reuse(tmp_path, st_platform, cores, split)
    records = list(tmp_path.rglob("wse_sram.json"))
    assert len(records) == 4
    for path in records:
        record = json.loads(path.read_text())
        assert record["configuration"][:2] == [40, 4]
        kinds = {section["kind"] for section in record["sections"]}
        assert {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} <= kinds
        for owner in range(40):
            core = record["cores"][owner]
            sections = sorted((v for v in record["sections"] if v["owner"] == owner), key=lambda v: v["address"])
            end = core["base"] + core["capacity"] - core["reserve"]
            for section in sections:
                assert section["address"] >= end
                end = section["address"] + section["bytes"]
                assert end <= core["base"] + core["capacity"]
            assert sum(v["bytes"] for v in sections) <= core["runtime_used"]
        active = cores or 40
        for c in record["cores"]:
            assert c["runtime_used"] <= c["reserve"]
            if c["owner"] < active:
                assert c["runtime_used"] > 0
                assert c["base"] + c["capacity"] - c["reserve"] <= c["control_address"] < c["base"] + c["capacity"]
            else:
                assert c["runtime_used"] == 0


def test_sram_wait_timeout_and_close(tmp_path, st_platform, monkeypatch):
    counter.test_counter_wait_timeout_and_close(tmp_path, st_platform)


def test_sram_pending_and_devices(tmp_path, st_platform, monkeypatch):
    counter.test_counter_pending_runs_and_device_isolation(tmp_path, st_platform)


def build_memory_probe(platform):
    compiler = KernelCompiler(platform)
    here = Path(__file__).parent / "kernels"
    child = CoreCallable.build(
        signature=[D.INOUT] * 3 + [D.IN],
        binary=compiler.compile_incore(
            str(here / "probe.cpp"),
            core_type="aic",
            extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        ),
    )
    return ChipCallable.build(
        signature=[D.INOUT] * 3 + [D.IN],
        func_name="aicpu_orchestration_entry",
        binary=compiler.compile_orchestration("host_build_graph", str(here / "orchestration.cpp")),
        children=[(0, child)],
    )


def memory_args(worker, blocks, owner, stride=1):
    args = TaskArgs()
    buffers = []
    tensors = []
    for index, size in enumerate((blocks * stride + 2, blocks * stride + 2, blocks * 8)):
        host = torch.full((size,), 999, dtype=torch.uint64)
        buffer = worker.malloc(host.nbytes, core_id=owner if index == 0 else 0)
        worker.copy_to(buffer, host)
        args.add_tensor(
            buffer.tensor(
                shapes=(blocks,) if index < 2 else (size,),
                dtype=DataType.UINT64,
                strides=(stride,) if index < 2 else (1,),
                byte_offset=8 if index < 2 else 0,
            ),
            TensorArgType.INOUT,
        )
        buffers.append(buffer)
        tensors.append(host)
    args.add_scalar(blocks)
    return args, buffers, tensors


@pytest.mark.parametrize("device_cores,active,blocks", [(40, 40, 5), (40, 40, 40), (40, 0, 160), (40, 40, 163)])
def test_sram_remote_owner_and_cross_kernel(tmp_path, st_platform, device_cores, active, blocks):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        config = {"hardware": {"core_count": device_cores, "sram_bytes_per_core": 100 * 1024 * 1024}}
        file = tmp_path / "hardware.json"
        file.write_text(json.dumps(config))
        worker.configure_wse(file)
        handle = worker.register(build_memory_probe(st_platform))
        stride = 2 if blocks == 160 else 1
        args, buffers, tensors = memory_args(worker, blocks, device_cores - 1, stride)
        before = worker.sram_info(device_cores - 1)["application_used_bytes"]
        worker.run(handle, args, counter.config_for(tmp_path / "run", active))
        for buffer, host in zip(buffers, tensors, strict=True):
            worker.copy_from(host, buffer)
        for index in range(2):
            expected = [999] * (blocks * stride + 2)
            for block in range(blocks):
                expected[1 + block * stride] = 100 + index + block
            assert tensors[index].tolist() == expected
        rows = tensors[2].reshape(blocks, 2, 4).tolist()
        cores = active or device_cores
        for b, row in enumerate(rows):
            assert row == [[device_cores - 1, 0, b % cores, (b // cores) % 4]] * 2
        assert worker.sram_info(device_cores - 1)["application_used_bytes"] == before
        assert worker.sram_info(0)["runtime_used_bytes"] == 0
        for buffer in buffers:
            worker.free(buffer)
    finally:
        worker.close()


@pytest.mark.parametrize("reserves,blocks", [([4096] * 40, 32768), ([65536, 256] + [65536] * 38, 5120)])
def test_sram_runtime_capacity_rejects_before_go_and_recovers(tmp_path, st_platform, reserves, blocks):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    try:
        worker.init()
        worker.configure_wse(
            {
                "hardware": {"core_count": 40, "sram_bytes_per_core": 104857600},
                "runtime": {"sram_reserve_bytes_per_core": reserves},
            }
        )
        handle = worker.register(build_memory_probe(st_platform))
        args, buffers, _ = memory_args(worker, blocks, 1)
        with pytest.raises(RuntimeError):
            worker.run(handle, args, counter.config_for(tmp_path / "rejected", 40))
        assert not (tmp_path / "rejected/wse_events.jsonl").exists()
        assert worker.sram_info(0)["runtime_used_bytes"] == 0
        for buffer in buffers:
            worker.free(buffer)
        args, buffers, tensors = memory_args(worker, 1, 1)
        worker.run(handle, args, counter.config_for(tmp_path / "valid", 40))
        worker.copy_from(tensors[1], buffers[1])
        assert tensors[1].tolist() == [999, 101, 999]
        for buffer in buffers:
            worker.free(buffer)
    finally:
        worker.close()
