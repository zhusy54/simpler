# A5 HBG AICore Dependency Scheduler Performance Report

## Technical summary

This report compares four AICore-side dependency-unblocking scheduler designs on Paged Attention Unroll Case1. The four measured source trees use byte-identical Case1 test and orchestration files and were rebuilt independently. All correctness runs and all performance rounds passed with profiling disabled.

The completion-inbox design at benchmark commit `8deaaadf3bddb280f89903320d1fa6fd1a746155` is the best overall result: its steady-state median device latency is **1.138 ms**, a **7.62x speedup** over the shared MPMC baseline and **4.05% lower latency** than the ticket-cursor design. It also has the tightest measured distribution.

The sharded event-workset design is the bitmap design. Its normal-path median is **1.556 ms**, but it has a confirmed severe tail-latency problem. The equal-sample comparison contains one 168.938 ms stall, and a separate 30-round robustness run contains two additional steady-state stalls of 410.993 ms and 434.754 ms. It should not be selected without resolving this instability.

## Key findings

### Equal-sample comparison

The primary statistic is the steady-state median. Each version ran in two independently started 10-round processes, once in forward order and once in reverse order. The first round of each process was excluded as cold start, leaving `n=18` per version. No other sample was removed.

| Rank | Scheduler | Benchmark commit | Median (ms) | Mean (ms) | P95 (ms) | Min–max (ms) | Versus MPMC median |
| ---: | --------- | ---------------- | ----------: | --------: | -------: | -----------: | -----------------: |
| 1 | Completion inbox + priority ticket streams | `8deaaadf3bddb280f89903320d1fa6fd1a746155` | **1.138** | **1.138** | **1.141** | 1.132–1.141 | **7.62x / -86.88%** |
| 2 | Ticket cursor + worker-private pending | `4222fb154627439228eed18fa381e68e85a8479f` | 1.186 | 1.185 | 1.204 | 1.157–1.205 | 7.32x / -86.33% |
| 3 | 64-shard event-workset bitmap | `c1de4652aeac460cbb5dd2c143852af26a6a4932` | 1.556 | 10.865 | 26.737 | 1.524–168.938 | 5.57x / -82.06% |
| 4 | Shared ReadyQ/CompletionQ MPMC | `183cea172f1b9f8752a0e4af29ccf8c91b9564be` | 8.674 | 8.677 | 8.895 | 8.329–8.928 | baseline |

Median-to-median comparisons among the three faster designs:

- Completion inbox is 4.05% faster than ticket cursor (`1.138 ms` versus `1.186 ms`).
- Ticket cursor is 23.80% faster than bitmap (`1.186 ms` versus `1.556 ms`).
- Completion inbox is 26.89% faster than bitmap (`1.138 ms` versus `1.556 ms`).

### Bitmap tail-latency sensitivity

The bitmap outliers are retained in the primary table. The following view separates its normal path from its tail behavior only to make the failure mode visible; it is not used to erase the stalls from the result.

| Bitmap sample set | Steady rounds | Stalls >10 ms | Normal-path median (ms) | Normal-path P95 (ms) | Worst stall (ms) |
| ----------------- | ------------: | ------------: | ----------------------: | -------------------: | ---------------: |
| Equal-sample comparison | 18 | 1 (5.56%) | 1.555 | 1.623 | 168.938 |
| Additional robustness run | 29 | 2 (6.90%) | 1.559 | 1.603 | 434.754 |

The robustness process also had an 875.669 ms first round, which is classified as cold start and excluded from the steady-state count. The two later stalls remain. All 30 rounds passed correctness, and the device log reports `profilingMode=0`.

## Version definitions

The benchmark commit is the exact tree used to build and measure each version. Where useful, the scheduler-origin commit is also listed to preserve design provenance.

| Version | Design | Scheduler-origin commit | Exact benchmark commit | Notes |
| ------- | ------ | ----------------------- | ---------------------- | ----- |
| V1 | Shared global ReadyQ/CompletionQ MPMC queues plus resolver workers | `000903a4f01c0458fb0353a43890486b3a60a358` | `183cea172f1b9f8752a0e4af29ccf8c91b9564be` | The benchmark commit adds only the missing Unroll Case1 test and orchestration on top of the scheduler commit. |
| V2 | Sharded event worksets | `a72d1f41a92b518a5ea9ca379a7d23aa110aaa08` | `c1de4652aeac460cbb5dd2c143852af26a6a4932` | This is the bitmap scheme: 64 shards, two-level ready/completion bitmaps, and topology-aware stealing. |
| V3 | Typed ticket cursors with worker-private pending slots and padded completion cells | `f8cd13454e21bd681078cd12d752ed573c04f12b` | `4222fb154627439228eed18fa381e68e85a8479f` | The measured tree includes later ticket-stream planning and reduced debug publication work. |
| V4 | Priority ticket streams and private pending slots with per-resolver completion inboxes | `6242e5d34ff6421e0ddfc3eac09cae92c1eff610` | `8deaaadf3bddb280f89903320d1fa6fd1a746155` | This retains ticket-based dispatch but changes completion routing and lifecycle handling; it is an evolved hybrid, not merely the same V3 binary with a local micro-optimization. |

The new V1 Case1 coverage is committed on local branch `bench/hbg-mpmc-unroll-case1`. The other benchmark commits are retained by their corresponding benchmark worktree branches or the current feature branch.

## Workload and metric definitions

Paged Attention Unroll Case1 uses:

- `batch=256`, `num_heads=16`, `kv_head_num=1`
- `head_dim=128`, `block_size=128`
- `context_len=8192`, `max_model_len=32768`
- `dtype=bfloat16`, output accumulation in FP32
- unroll group size of 64 blocks
- 1,024 graph tasks: 256 each of QK matmul, softmax prepare, PV matmul, and online update

The reported metric is `simpler_run.runner_run.device_wall`, measured by the device clock and converted from nanoseconds to milliseconds. It covers device execution of the already-bound graph run and excludes host-side bind and result-validation time.

All four benchmark commits contain identical workload sources:

- Python test blob: `9aac998b4758425ec771a772c911c0ceace058a8`
- Orchestration blob: `9cf1dc8ffe3f51f52acecd2181edbc7f64cf916b`

## Methodology

- Date: 2026-08-16, timezone Asia/Shanghai.
- Platform request: A5 onboard, device 0.
- Detected SoC fallback identifier: `Ascend950PR_9579`; CPU topology query was unavailable, so the runtime used the documented occupancy fallback mask `0x3e`.
- Architecture precheck: explicitly forced at the user's request because `npu-smi` did not expose a Chip/NPU Name. The actual runtime SoC identifier is recorded above.
- PTO-ISA revision: `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`.
- Builds: independent worktree and virtual environment per benchmark commit.
- Correctness: one golden run per version before performance sampling; all four passed.
- Primary performance sampling: two blocks of 10 rounds per version. Block 1 order was V1, V2, V3, V4; block 2 reversed the order to V4, V3, V2, V1. All 80 rounds passed.
- Warm-up policy: exclude only the first round of each newly started process. This produces 18 equal steady-state samples per version.
- Profiling controls: chip swimlane, dependency generation, PMU collection, and golden-reference generation were disabled during performance rounds. AICPU device logs for every version report `profilingMode=0`.
- Bitmap robustness: one additional profiling-off 30-round process; its first round was excluded and all later samples retained.

Primary task: `task_20260816_170956_18797794194`
Bitmap robustness task: `task_20260816_171405_18942933294`

Raw logs are retained under:

`tmp/a5_hbg_four_scheduler_unroll_case1_20260816_170848/`

The equal-sample inputs are `block1_<version>.log` and `block2_<version>.log`; the supplemental bitmap task output is `bitmap_robustness_task.log`.

## Interpretation

V1 exposes the cost of global multi-producer/multi-consumer queue contention and centralized completion resolution. V2 removes most normal-path contention by representing ready and completion work as sharded bitmaps, which explains its much lower median, but its repeated device-wall stalls make the design operationally unsafe in the measured form.

V3 replaces shared work discovery with monotonic typed ticket cursors and keeps dependency-pending state private to workers. This removes both the bitmap scan/steal work and its observed tail behavior in this sample. V4 keeps ticket-based dispatch, then moves completion closure and waiter routing through per-resolver inboxes and streamlines startup/shutdown publication. That reduces the median another 4.05% and cuts steady-state standard deviation from `0.0116 ms` to `0.0027 ms`.

For this workload, V4 is the recommended base. V3 remains a useful simpler reference implementation. V2 should remain an investigation target rather than a production candidate until its periodic stalls are root-caused.

## Limitations and next steps

- These results cover only Paged Attention Unroll Case1 on one A5-class device and one test session. They do not establish performance for Case2/Case3, other graphs, multiple devices, or sustained service traffic.
- The architecture precheck used `--force`; future runs should record a successful direct hardware-name detection when the host tooling supports it.
- P95 from 18 samples is descriptive, not a high-confidence tail estimate. The repeated bitmap stalls are nevertheless material because they reproduced in an independent 30-round run.
- No profiling was enabled, so the current data proves where the end-to-end device stalls occur but not which bitmap scheduler phase causes them.

Recommended follow-ups:

1. Reproduce V2 with scheduler DFX or a targeted swimlane, accepting profiling overhead for diagnosis rather than comparison.
2. Repeat V3 and V4 for at least 100 steady-state rounds to validate the 4.05% gap and long-run tail stability.
3. Run the same four exact commits on Unroll Case2 and Case3 before generalizing the recommendation beyond Case1.

## Open questions

- Do the V2 stalls correlate with bitmap shard rotation, full workset reconciliation, or a device/runtime event outside the scheduler?
- Does V4's inbox advantage persist when dependency fan-out, mixed AIC/AIV ratios, or context lengths change?
- Are the current task-stream priorities optimal for workloads with more than one unroll group per query?
