# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Normalize explicit WSE settings against the loaded runtime's effective specification."""

import json
import os
from pathlib import Path


def normalize_config(config, current):
    if isinstance(config, (str, os.PathLike)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, dict) or set(config) - {"hardware", "runtime"}:
        raise ValueError("WSE config accepts hardware and runtime sections")
    hardware, runtime = config.get("hardware", {}), config.get("runtime", {})
    if not isinstance(hardware, dict) or set(hardware) - {"core_count", "pe_per_core", "sram_bytes_per_core"}:
        raise ValueError("unknown WSE hardware setting")
    if not isinstance(runtime, dict) or set(runtime) - {"sram_reserve_bytes_per_core"}:
        raise ValueError("unknown WSE runtime setting")
    cores = hardware.get("core_count", current["hardware"]["core_count"])
    pes = hardware.get("pe_per_core", current["hardware"]["pe_per_core"])
    if type(cores) is not int or not 1 <= cores <= current["limits"]["max_core_count"]:
        raise ValueError("WSE core_count exceeds supported range")
    if type(pes) is not int or pes != current["hardware"]["pe_per_core"]:
        raise ValueError("unsupported WSE pe_per_core")

    def expand(value):
        if type(value) is int:
            value = [value] * cores
        if not isinstance(value, (list, tuple)) or len(value) != cores:
            raise ValueError("WSE per-core arrays must match core_count")
        if any(type(v) is not int or not 0 < v < 2**64 for v in value):
            raise ValueError("WSE capacities must be positive uint64 values")
        return list(value)

    def inherit(values):
        return values[0] if len(set(values)) == 1 else values

    capacities = expand(hardware.get("sram_bytes_per_core", inherit(current["hardware"]["sram_bytes_per_core"])))
    reserves = expand(
        runtime.get("sram_reserve_bytes_per_core", inherit(current["runtime"]["sram_reserve_bytes_per_core"]))
    )
    return cores, pes, capacities, reserves
