# A5 HBG Startup/Shutdown Protocol Case1 Performance Report

## Conclusion

The protocol is implemented and passes Case1 and the empty-graph lifecycle
test on A5 device 0. Relative to the staged snapshot baseline, steady-state
Case1 device wall time decreases from **1362.5 us** to **1134.8 us**, a
**227.7 us (-16.7%)** reduction. P95 decreases from **1395.2 us** to
**1141.6 us (-18.2%)**.

The Level-1 swimlane shows a 40.3 us (-7.0%) reduction in mean per-core startup
and a 67.5 us (-50.3%) reduction in the mean tail from executor drain to the
final local publication. It also exposes the centralized AICPU completion
decision and per-core EXIT/ACK independently: the last Kernel to the AICPU
completion decision takes 7.2 us, and the decision to the last ACK takes
24.5 us.

## Implementation summary

- Startup is now `handshake -> publish complete AICPU configuration -> store
  barrier -> partitioned DATA_MAIN_BASE=IDLE`. This removes
  `attached_count/startup_phase`, shared-GM startup polling, and redundant
  AICore identity writeback.
- AICore workers publish only `executed_task_count` and
  `executor_drained_worker_count`. AIV resolvers continue servicing inboxes
  after executor drain, while one AICPU supervisor waits for all executors and
  resolutions.
- AICPU threads broadcast `DATA_MAIN_BASE=EXIT` to their partitions in
  parallel. A global barrier ensures every signal is issued before ACK waits.
  AICore checks only its local DMB register and publishes final statistics
  before ACK.
- The duplicate GM exit handshake and
  `drained_worker_count/exit_requested/finished_count` are removed.
- Level-1 adds `aicpu_lifecycle_phases`. Perfetto renders a separate AICPU
  Lifecycle process with `WaitExecutors`, `WaitResolved`,
  `CompletionDecision`, `RegisterRelease`, and `ExitSignalToAck` lanes.

## Environment and method

- Platform: `--platform a5`, device 0. Runtime fallback logging identifies
  `Ascend950PR_9579`. The run uses 84 AICores (28 AIC + 56 AIV) at a 1 GHz
  device clock.
- Per explicit authorization, the architecture check used `--force` to skip
  silicon verification and run A5 directly. Every run retains the
  `silicon was not verified for --platform a5` warning.
- The baseline is `HEAD 401f954c + the staged snapshot` in an independent
  worktree. The candidate applies this protocol change on top of that same
  snapshot. Both variants were rebuilt independently.
- Performance uses a B-A-A-B order on the same device, with ten rounds and
  golden validation per block. Statistics discard the first round of each
  process, leaving 18 baseline and 18 candidate samples.
- Level-1 capture has observer overhead and is used for attribution only, not
  as the uninstrumented performance result.

## Case1 performance

### Steady-state results

Each ten-round process discards its first round.

- Device mean: 1362.502 us baseline, 1134.756 us candidate, a change of
  -227.746 us (**-16.715%**).
- Device SD: 17.908 us baseline, 4.296 us candidate, a change of -75.997%.
- Device P50: 1361.544 us baseline, 1134.591 us candidate, a change of
  -226.953 us (-16.668%).
- Device P95: 1395.208 us baseline, 1141.639 us candidate, a change of
  -253.569 us (**-18.174%**).
- Device range: 1330.084-1399.859 us baseline and 1129.596-1142.517 us
  candidate.
- Host `simpler_run` mean: 191200.439 us baseline, 204703.980 us candidate, a
  change of +13503.541 us (+7.063%).
- Host `simpler_run` P95: 197430.835 us baseline, 219078.338 us candidate, a
  change of +10.965%.

Device mean, P95, and variance all improve materially. Host `simpler_run`
includes graph binding, Python golden validation, runtime execution, and
validation; it is not the target metric for this device protocol. It increases
and remains noisy in this interleaved sample, so the device gain must not be
treated as an equivalent host end-to-end gain. If host latency is an acceptance
criterion, collect a larger sample and separate bind and validation first.

### Level-1 startup and shutdown phases

- Entry to descriptor-ready mean: 572.122 us baseline and 531.846 us
  candidate, a change of -40.276 us (-7.0%).
- Entry to descriptor-ready P95: 589.193 us baseline and 544.745 us candidate,
  a change of -44.448 us (-7.5%).
- 84-core SeedClaim start span: 38.396 us baseline and 37.342 us candidate, a
  change of -1.054 us (-2.7%).
- Executor drain to local final publication mean: 134.174 us baseline and
  66.707 us candidate, a change of -67.467 us (-50.3%).
- Executor drain to local final publication P95: 198.303 us baseline and
  139.364 us candidate, a change of -58.939 us (-29.7%).
- Last Kernel to AICPU completion decision: not observable in the baseline and
  7.160 us in the candidate.
- AICPU decision to last EXIT ACK: not observable in the baseline and 24.482 us
  in the candidate.
- Last Kernel to last EXIT ACK: not observable in the baseline and 31.642 us
  in the candidate.

The startup envelope improves, but cross-core SeedClaim dispersion improves by
only 2.7%. The new protocol removes shared-state round trips but not the
dispersion of per-core MMIO release. Shutdown improves more clearly: every core
stops duplicating global completion polls, and the path converges on one AICPU
decision followed by local-register EXIT/ACK.

The observed span of 1024 kernels decreases from 1117.07 us to 1096.13 us
(-1.87%). Baseline and candidate kernel core-times are 43004.78 us and
43522.59 us, respectively; the difference is independent Level-1 sampling
variation. Most device-wall improvement comes from startup, scheduling tail,
and shutdown rather than a kernel algorithm change.

## Correctness and regression results

- A5 Case1 golden: all 40 ABBA rounds pass.
  - Baseline A: `task_20260814_161754_441396268`
  - Candidate A: `task_20260814_161817_4444393157`
  - Candidate B: `task_20260814_161831_44566614656`
  - Baseline B: `task_20260814_161846_44729320319`
- Candidate Case1 Level-1: `task_20260814_160021_35284728973`, passed.
- Baseline Case1 Level-1: `task_20260814_161936_4501048715`, passed.
- Candidate Case1 dependency capture: `task_20260814_160114_35651019425`,
  passed.
- A5 empty-graph lifecycle: `task_20260814_162115_4566346814`, passed with a
  104.181 us device wall time.
- A5 simulation: Case1 mixed groups, empty graph, single AIC, single AIV, and
  single-AIC Level-1 all pass.
- Non-hardware C++ unit tests: 90/90 pass; A5 HBG ticket scheduler: 30/30 pass.
- Python swimlane-converter unit tests: 25/25 pass; `git diff --check` passes.

## Swimlanes and raw data

- **Recommended candidate Perfetto swimlane**:
  [merged_swimlane.json](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260814_160024/merged_swimlane.json)
- Candidate Level-1 raw capture:
  [chip_swimlane_records.json](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260814_160024/chip_swimlane_records.json)
- Staged-baseline Perfetto swimlane:
  [merged_swimlane.json](../outputs/hbg_startup_shutdown_protocol_20260814_154048/baseline_profile_artifacts/merged_swimlane.json)
- Staged-baseline Level-1 raw capture:
  [chip_swimlane_records.json](../outputs/hbg_startup_shutdown_protocol_20260814_154048/baseline_profile_artifacts/chip_swimlane_records.json)
- Dependency graph:
  [deps.json](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260814_160117/deps.json)
- ABBA raw logs:
  [baseline A](../outputs/hbg_startup_shutdown_protocol_20260814_154048/abba_baseline_a.log),
  [candidate A](../outputs/hbg_startup_shutdown_protocol_20260814_154048/abba_candidate_a.log),
  [candidate B](../outputs/hbg_startup_shutdown_protocol_20260814_154048/abba_candidate_b.log),
  [baseline B](../outputs/hbg_startup_shutdown_protocol_20260814_154048/abba_baseline_b.log).

Open the recommended `merged_swimlane.json` in Perfetto UI to inspect the
AICore Scheduler, Dependency, and new AICPU Lifecycle lanes.
