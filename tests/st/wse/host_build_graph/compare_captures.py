# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare independent A5 reference, candidate A5, and WSE captures."""

import argparse
import json
from pathlib import Path

from tests.st.wse.host_build_graph.audit_capture import audit


def compare(reference, candidate, wse):
    paths = [Path(reference), Path(candidate), Path(wse)]
    audits = [audit(path) for path in paths]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in paths]
    assert [item["platform"] for item in manifests] == ["a5sim", "a5sim", "wsesim"]
    assert len({item["expected_blocks"] for item in manifests}) == 1
    assert len({item["topology_sha256"] for item in manifests}) == 1
    for label in manifests[0]["captures"]:
        rows = [json.loads((path / label / "numeric-result.json").read_text()) for path in paths]
        for field in ("scene", "repeat", "inputs", "outputs", "sources", "scheduler_mode"):
            assert rows[0][field] == rows[1][field] == rows[2][field], (label, field)
        for original in (paths[0] / label).glob("*.bin"):
            assert (
                original.read_bytes()
                == (paths[1] / label / original.name).read_bytes()
                == (paths[2] / label / original.name).read_bytes()
            ), (label, original.name)
    return {
        "status": "PASS",
        "groups": dict(zip(("A0", "A1", "W0"), audits, strict=True)),
        "capabilities": {
            "phase": "A5-compatible baseline",
            "aicpu": True,
            "lanes": ["AIC", "AIV0", "AIV1"],
            "memory": "original GM",
            "tile_semantics": "pinned A5 CPU",
            "native_four_pe": False,
            "sram_capacity_model": False,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("wse", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.wse)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])
