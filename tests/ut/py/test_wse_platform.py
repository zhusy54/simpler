# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE discovery and cache contracts stay independent of A5."""

import hashlib
import json
from pathlib import Path

import pytest
from simpler.callable_identity import build_chip_callable_descriptor
from simpler.task_interface import ChipCallable

from simpler_setup import runtime_builder
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.platform_info import (
    discover_runtimes,
    parse_platform,
    runtime_targets,
    to_platform,
)
from simpler_setup.pto_isa import read_pto_isa_pin
from simpler_setup.runtime_builder import RuntimeBuilder


def test_wse_supports_only_sim_hbg():
    assert parse_platform("wsesim") == ("wse", "sim")
    assert to_platform("wse", "sim") == "wsesim"
    assert discover_runtimes("wse") == ["host_build_graph"]
    with pytest.raises(ValueError):
        parse_platform("wse")
    builder = RuntimeBuilder("wsesim")
    with pytest.raises(ValueError, match="not available"):
        builder._validate_runtime("tensormap_and_ringbuffer")
    assert builder._resolve_build_pto_isa_commit() == read_pto_isa_pin()


def test_wse_orchestration_uses_own_headers_and_identity():
    wse = KernelCompiler("wsesim")
    a5 = KernelCompiler("a5sim")
    includes = wse.get_orchestration_include_dirs("host_build_graph")
    assert any("/src/wse/" in str(p) for p in includes)
    assert all("/src/a5/" not in str(p) for p in includes)
    assert wse.incore_compile_cache_token("aiv") != a5.incore_compile_cache_token("aiv")
    chip = ChipCallable.build(signature=[], func_name="empty", binary=b"identical-bytes", children=[])
    assert build_chip_callable_descriptor(
        target=chip, platform="wsesim", runtime="host_build_graph"
    ) != build_chip_callable_descriptor(target=chip, platform="a5sim", runtime="host_build_graph")


def test_source_manifest_matches_copied_files():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "src/wse/source-copy-manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["destination"]).read_bytes()).hexdigest() == entry["sha256"]


def test_real_scene_cache_keeps_wse_and_a5_separate(monkeypatch):
    from simpler_setup.scene_test import clear_compile_cache  # noqa: PLC0415
    from tests.st.a5.host_build_graph.vector_example.test_vector_example import (  # noqa: PLC0415
        TestVectorExampleHostBuildGraphA5,
    )

    case = TestVectorExampleHostBuildGraphA5()
    originals = {}
    for platform in ("a5sim", "wsesim"):
        originals[platform] = case.build_callable(platform)
        assert case.build_callable(platform) is originals[platform]
    clear_compile_cache()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("complete persistent cache entry was recompiled")

    monkeypatch.setattr(KernelCompiler, "compile_incore", unexpected_compile)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", unexpected_compile)
    descriptors = {}
    for platform, original in originals.items():
        cached = case.build_callable(platform)
        descriptors[platform] = build_chip_callable_descriptor(
            target=cached, platform=platform, runtime="host_build_graph"
        )
        assert descriptors[platform] == build_chip_callable_descriptor(
            target=original, platform=platform, runtime="host_build_graph"
        )
    assert descriptors["a5sim"] != descriptors["wsesim"]
    clear_compile_cache()


def test_runtime_components_are_declared_by_platform():
    assert runtime_targets("wsesim") == ("host", "aicore")
    for platform in ("a2a3", "a2a3sim", "a5", "a5sim"):
        assert runtime_targets(platform) == ("host", "aicpu", "aicore")


def test_wse_binary_lookup_uses_two_components(tmp_path, monkeypatch):
    builder = RuntimeBuilder("wsesim")
    monkeypatch.setattr(builder, "_requires_pto_isa_metadata_validation", lambda: False)
    monkeypatch.setattr(builder, "_resolve_sim_context_path", lambda: None)
    for target in ("host", "aicore"):
        path = tmp_path / getattr(builder._runtime_compiler, f"{target}_target").get_binary_name()
        path.write_bytes(b"lookup-only")
    bins = builder._lookup_binaries("host_build_graph", tmp_path)
    assert bins.aicpu_path is None
    assert bins.host_path.is_file() and bins.aicore_path.is_file()


def test_a5_binary_lookup_still_requires_aicpu(tmp_path, monkeypatch):
    builder = RuntimeBuilder("a5sim")
    monkeypatch.setattr(builder, "_requires_pto_isa_metadata_validation", lambda: False)
    for target in ("host", "aicore"):
        (tmp_path / getattr(builder._runtime_compiler, f"{target}_target").get_binary_name()).touch()
    with pytest.raises(FileNotFoundError, match="libaicpu_kernel"):
        builder._lookup_binaries("host_build_graph", tmp_path)


def test_wse_cache_keeps_runtime_and_pto_identity(monkeypatch):
    monkeypatch.setattr(runtime_builder, "_get_git_head", lambda _: "same-commit")
    builder = RuntimeBuilder("wsesim")
    stamp = builder._build_cache_stamp("fixed-pto")
    assert stamp == "same-commit:pto-isa=fixed-pto"
    assert stamp == builder._build_cache_stamp("fixed-pto")
