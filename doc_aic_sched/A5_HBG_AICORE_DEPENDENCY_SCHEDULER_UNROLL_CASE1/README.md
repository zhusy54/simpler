# A5 HBG AICore Scheduler Swimlanes: Paged Attention Unroll Case1

This directory keeps the documentation and newly captured Level-1 chip-swimlane artifacts for the three optimized AICore dependency schedulers in one self-contained location. Each trace was produced from the exact benchmark commit named below, rather than from an intermediate historical revision.

## Open the swimlanes

Open [Perfetto UI](https://ui.perfetto.dev/) and drag in the corresponding `merged_swimlane_with_deps.json.gz` file. Perfetto accepts the compressed trace directly:

| Scheduler | Exact benchmark commit | Perfetto trace | Raw Level-1 records |
| --------- | ---------------------- | -------------- | ------------------- |
| Sharded event-workset bitmap | `c1de4652aeac460cbb5dd2c143852af26a6a4932` | [bitmap trace](bitmap/merged_swimlane_with_deps.json.gz) | [bitmap records](bitmap/chip_swimlane_records.json.gz) |
| Ticket cursor + worker-private pending | `4222fb154627439228eed18fa381e68e85a8479f` | [ticket trace](ticket_cursor/merged_swimlane_with_deps.json.gz) | [ticket records](ticket_cursor/chip_swimlane_records.json.gz) |
| Completion inbox + priority ticket streams | `8deaaadf3bddb280f89903320d1fa6fd1a746155` | [completion-inbox trace](completion_inbox/merged_swimlane_with_deps.json.gz) | [completion-inbox records](completion_inbox/chip_swimlane_records.json.gz) |

The profiling-off latency comparison and four-version design definitions are in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md).

## Capture summary

| Scheduler | Profiled device wall | AICore observed span | Captured scheduler detail |
| --------- | -------------------: | -------------------: | ------------------------- |
| Bitmap | 1.985 ms | 1,248.09 us | 1,024 kernel tasks and 1,024 resolve phases |
| Ticket cursor | 1.635 ms | 1,073.95 us | 1,024 kernel phases plus claim, pending, payload, completion, and drain phases |
| Completion inbox | 1.569 ms | 1,098.33 us | 1,024 kernel phases plus ready, claim, inbox completion, wake resolution, and lifecycle phases |

These device-wall values include chip-swimlane instrumentation and export overhead. They validate the captures but must not replace the profiling-off performance results in `PERFORMANCE_REPORT.md`.

## Workload and provenance

- Capture date: 2026-08-17, Asia/Shanghai.
- Hardware request: A5, device 0; runtime SoC fallback identifier `Ascend950PR_9579`.
- Architecture gate: `--force` was used under explicit user authorization because direct `npu-smi` silicon-name detection was unavailable.
- Queue task: `task_20260817_084132_87958722941`.
- Test: Paged Attention Unroll Case1, one golden-validated round per scheduler; all three passed.
- Workload: batch 256, 16 query heads, one KV head, head dimension 128, context length 8,192, and 1,024 graph tasks.
- Profiling: `--enable-chip-swimlane 1`; performance profiling, PMU, and dependency generation were not enabled in the timed capture.
- Test source blob: `9aac998b4758425ec771a772c911c0ceace058a8`.
- Orchestration source blob: `9cf1dc8ffe3f51f52acecd2181edbc7f64cf916b`.

Dependency arrows were added offline from the shared [deps.json.gz](deps.json.gz). It contains the same byte-identical Case1 graph source, 1,024 tasks, and 1,280 graph edges, so dependency generation did not perturb the timed swimlane captures. The converter emitted 1,024 visible task-to-task dependency flows in each trace.

The ticket trace reports 56 cross-core happens-before warnings when dependency arrows are joined. This is a known timestamp-ordering limitation also recorded by the historical completion-inbox analysis: the arrows remain useful for topology navigation but are not strict cross-core cycle-order proofs.

## Directory layout

```text
A5_HBG_AICORE_DEPENDENCY_SCHEDULER_UNROLL_CASE1/
  README.md
  PERFORMANCE_REPORT.md
  SHA256SUMS
  deps.json.gz
  bitmap/
    chip_swimlane_records.json.gz
    merged_swimlane_with_deps.json.gz
    name_map_TestPagedAttentionUnrollHostBuildGraphA5_Case1.json
  ticket_cursor/
    chip_swimlane_records.json.gz
    merged_swimlane_with_deps.json.gz
    name_map_TestPagedAttentionUnrollHostBuildGraphA5_Case1.json
  completion_inbox/
    chip_swimlane_records.json.gz
    merged_swimlane_with_deps.json.gz
    name_map_TestPagedAttentionUnrollHostBuildGraphA5_Case1.json
```

The `chip_swimlane_records.json.gz` files are compressed raw converter inputs. The `merged_swimlane_with_deps.json.gz` files are compressed, ready-to-open Perfetto artifacts. Use `gzip -dk <file>` when an uncompressed raw JSON is required by a local converter. The name-map files preserve function-ID mappings (`QK`, `SF`, `PV`, and `UP`).

Collector warnings about skipped `start_time=0` slots refer to unused tails of fixed-capacity per-core buffers. Artifact validation confirmed that each merged trace contains all 1,024 Case1 kernel tasks.
