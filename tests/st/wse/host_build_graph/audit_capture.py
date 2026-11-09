# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check actual identities, lifecycle and independent expected task sets."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(capture_root):
    capture_root = Path(capture_root)
    manifest = json.loads((capture_root / "manifest.json").read_text())
    assert manifest["captures"] == ["vector-0", "vector-1", "mix-0", "mix-1"]
    blocks = manifest["expected_blocks"]
    assert blocks > 0
    results = []
    epochs = []
    for label in manifest["captures"]:
        root = capture_root / label
        numeric = json.loads((root / "numeric-result.json").read_text())
        data = json.loads((root / "chip_swimlane_records.json").read_text())
        deps = json.loads((root / "deps.json").read_text())
        protocol = data["aicore_protocol"]
        metadata = data["metadata"]
        identities = data["aicore_execution_identities"]["records"]
        epoch = protocol["run_epoch"]
        epochs.append(epoch)
        assert protocol["producer"] == "aicore"
        assert protocol["scheduler_error"] == 0 and protocol["error_claimed"] == 0
        assert protocol["cluster_count"] == blocks
        assert metadata["num_cores"] == blocks * 3
        assert metadata["core_types"] == ["aic"] * blocks + ["aiv"] * (2 * blocks)
        assert metadata["host_capture"]["status"] == "complete"
        assert metadata["host_capture"]["dropped_records"] == 0
        assert len(protocol["workers"]) == blocks * 3
        assert all(w["finished"] == 1 and w["error_flags"] == 0 for w in protocol["workers"])
        assert all(r["executed"] and r["run_epoch"] == epoch for r in identities)
        assert len(identities) == len(data["aicore_tasks"])
        if numeric["scene"] == "vector":
            expected = {(task, 0, 1) for task in range(4)}
            actual = [(r["task_id"], r["block_idx"], r["subtask_slot"]) for r in identities]
            assert Counter(actual) == Counter(expected)
            edges = {(int(e["pred"]), int(e["succ"])) for e in deps["edges"]}
            assert edges == {(0, 1), (0, 2), (1, 3), (2, 3)}
            periods = {row[1]: (row[3], row[4]) for row in data["aicore_tasks"]}
            assert all(periods[a][1] <= periods[b][0] for a, b in edges)
        else:
            expected = {
                ((1 << 62) | (graph << 32), block, lane)
                for graph in range(3)
                for block in range(blocks)
                for lane in range(3)
            }
            actual = [(r["task_id"], r["block_idx"], r["subtask_slot"]) for r in identities]
            assert Counter(actual) == Counter(expected)
            assert all(r["cluster_lane"] == r["subtask_slot"] for r in identities)
            source = protocol["graph_source"]
            assert len(source["tasks"]) == 3 and len(source["definitions"]) == 1
            assert len({t["graph_key"] for t in source["tasks"]}) == 1
            child = source["definitions"][0]["children"]
            assert len(child) == 1 and child[0]["active_mask"] == 7 and child[0]["block_num"] == blocks
            assert deps["edges"] == []
            edges = set()
        life = data["aicpu_lifecycle_records"]
        assert life and len({r["aicpu_thread_id"] for r in life}) == len(life)
        stage_pairs = [
            ("handshake_start_cycles", "handshake_complete_cycles"),
            ("context_publish_start_cycles", "context_publish_complete_cycles"),
            ("bootstrap_wait_start_cycles", "bootstrap_complete_cycles"),
            ("register_release_start_cycles", "register_release_end_cycles"),
            ("exit_signal_start_cycles", "exit_signal_end_cycles"),
            ("exit_wait_start_cycles", "exit_wait_end_cycles"),
        ]
        for start, end in stage_pairs:
            assert any(r[start] > 0 for r in life), start
            for record in life:
                if record[start] or record[end]:
                    assert 0 < record[start] <= record[end], (start, record)
        leader = next(r for r in life if r["aicpu_thread_id"] == 0)
        assert 0 < leader["config_start_cycles"] <= leader["topology_complete_cycles"]
        assert max(r["context_publish_complete_cycles"] for r in life) <= min(t[3] for t in data["aicore_tasks"])
        assert max(t[4] for t in data["aicore_tasks"]) <= min(
            r["exit_signal_start_cycles"] for r in life if r["exit_signal_start_cycles"]
        )
        for name, record in numeric["outputs"].items():
            actual_bytes = (root / f"output-{name}.bin").read_bytes()
            golden_bytes = (root / f"golden-{name}.bin").read_bytes()
            assert actual_bytes == golden_bytes
            assert hashlib.sha256(actual_bytes).hexdigest() == record["sha256"]
        results.append(
            {
                "capture": label,
                "epoch": epoch,
                "kernel_entries": len(identities),
                "expected_entries": len(expected),
                "compute_edges": len(edges),
                "aicpu_threads_observed": len(life),
                "retired_workers": len(protocol["workers"]),
                "outputs": numeric["outputs"],
                "status": "PASS",
            }
        )
    assert len(epochs) == len(set(epochs)), epochs
    return {"status": "PASS", "expected_blocks": blocks, "captures": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    args = parser.parse_args()
    result = audit(args.capture_root)
    (args.capture_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
