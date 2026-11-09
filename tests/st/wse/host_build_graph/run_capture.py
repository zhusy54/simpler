# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture the fixed source scenarios through Worker without editing product code."""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

from simpler_setup.scene_test import _build_l2_ref_args


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene(root, name, relative, class_name):
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name), path


def capture(root, out, platform):
    root = Path(root).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    topology_header = root / "src/common/platform/sim/host/device_runner_base.h"
    blocks = int(re.search(r"constexpr int SIM_AUTO_BLOCKDIM = (\d+);", topology_header.read_text()).group(1))
    manifest = {
        "platform": platform,
        "runtime": "host_build_graph",
        "expected_blocks": blocks,
        "topology_source": str(topology_header.relative_to(root)),
        "topology_sha256": sha(topology_header),
        "captures": [],
        "status": "RUNNING",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    scenes = [
        (
            "vector",
            "tests/st/a5/host_build_graph/vector_example/test_vector_example.py",
            "TestVectorExampleHostBuildGraphA5",
        ),
        (
            "mix",
            "tests/st/a5/host_build_graph/graph_execution/test_graph_execution_mix_spmd.py",
            "TestGraphExecutionMixSpmdHostBuildGraphA5",
        ),
    ]
    torch.set_num_threads(1)
    for name, relative, class_name in scenes:
        cls, scene_path = load_scene(root, "step00_" + name, relative, class_name)
        case = cls()
        callable_obj = case.build_callable(platform)
        worker = cls._create_worker(platform)
        sources = [scene_path, Path(cls.CALLABLE["orchestration"]["source"])]
        sources.extend(Path(k["source"]) for k in cls.CALLABLE["incores"])
        try:
            handle = worker.register(callable_obj)
            for repeat in range(2):
                capture = out / f"{name}-{repeat}"
                capture.mkdir()
                args = case.generate_args({})
                if name == "vector" and repeat:
                    args.a.copy_((torch.arange(args.a.numel()) % 13).float() - 5)
                    args.b.fill_(3)
                elif name == "mix" and repeat:
                    for tensor_name in args.tensor_names():
                        getattr(args, tensor_name).fill_(-7)
                expected = args.clone()
                if name == "vector":
                    case.compute_golden(expected, {})
                else:
                    for tensor_name in expected.tensor_names():
                        tensor = getattr(expected, tensor_name)
                        for block in range(blocks):
                            for lane in range(3):
                                tensor[(block * 3 + lane) * 16] = float(block)
                before = {}
                for tensor_name in args.tensor_names():
                    tensor = getattr(args, tensor_name)
                    path = capture / f"input-{tensor_name}.bin"
                    path.write_bytes(tensor.numpy().tobytes())
                    before[tensor_name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": sha(path)}
                chip_args, output_names = _build_l2_ref_args(args, cls.CALLABLE["orchestration"]["signature"], worker)
                config = case._build_config({}, enable_chip_swimlane=4, enable_dep_gen=True, output_prefix=str(capture))
                published_before = worker.scheduler_mode_publish_count
                worker.run(handle, chip_args, config=config)
                published_after = worker.scheduler_mode_publish_count
                assert published_after > published_before
                assert worker.scheduler_runtime_mode in (1, 2), worker.scheduler_runtime_mode
                arrays = {}
                for tensor_name in output_names:
                    tensor = getattr(args, tensor_name)
                    golden = getattr(expected, tensor_name)
                    actual_path = capture / f"output-{tensor_name}.bin"
                    golden_path = capture / f"golden-{tensor_name}.bin"
                    actual_path.write_bytes(tensor.numpy().tobytes())
                    golden_path.write_bytes(golden.numpy().tobytes())
                    assert torch.equal(tensor, golden), (name, repeat, tensor_name)
                    arrays[tensor_name] = {
                        "elements": tensor.numel(),
                        "sha256": sha(actual_path),
                        "golden_sha256": sha(golden_path),
                    }
                result = {
                    "scene": name,
                    "repeat": repeat,
                    "inputs": before,
                    "outputs": arrays,
                    "scheduler_mode": worker.scheduler_runtime_mode,
                    "scheduler_publications": published_after - published_before,
                    "sources": {str(p.relative_to(root)): sha(p) for p in sources},
                    "status": "NUMERIC_PASS",
                }
                (capture / "numeric-result.json").write_text(json.dumps(result, indent=2) + "\n")
                manifest["captures"].append(str(capture.relative_to(out)))
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"{name}-{repeat}: numeric PASS, resident mode {worker.scheduler_runtime_mode}", flush=True)
            worker.unregister(handle)
        finally:
            worker.close()
    manifest["status"] = "NUMERIC_PASS_TRACE_AUDIT_PENDING"
    manifest["capture_script_sha256"] = sha(__file__)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()
    capture(Path(__file__).resolve().parents[4], ns.out, ns.platform)
