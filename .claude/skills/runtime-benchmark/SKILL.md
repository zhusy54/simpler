---
name: runtime-benchmark
description: Run or inspect the fixed 41-case C1–C3 and W1–W5 scheduling benchmark across simpler runtimes, chips, or revisions. Use for the scheduler benchmark matrix and its JSON manifests; use benchmark or perf-runtime-device for ordinary example latency sweeps.
---

# Runtime scheduling benchmark

Use `tests/st/scheduler_benchmark/run_cases.py` for the fixed matrix: 33 constructed DAG configurations from `matrix.py` and 8 existing workloads from `workloads.py`. The runner expands `configuration ID × platform × runtime` and executes cases sequentially in one process with one shared L2 Worker per measurement phase. Each case retains its own inputs, output validation, log slice and capture directory. If the sibling documentation checkout exists, read `../D200_docs/benchmark/README.md` from the `simpler` root for the matrix definitions and evaluation limits.

## Select and run

- Use `--list-only` to inspect all 41 IDs and the actual C configuration expansion. Use repeated `--case ID` to select a subset; omitting it runs the full matrix. `--rounds N` defaults to 10.
- The single-phase entry selects one runtime. The full-workflow entry below runs both runtimes by default, using one persistent process for each and the same Worker reuse rules. The runner uses each runtime's default Scheduler behavior. It does not select AICore or AICPU Scheduler.
- Use the project-local `.venv/bin/python`. Follow `testing` for build state and `onboard-arch-precheck` before every onboard invocation. On shared hardware, hold one `task-submit` device lock for the entire matrix, never one lock per case. Use `--device "$TASK_DEVICE"` inside the allocation.
- Keep no-diagnostic timing runs separate from diagnostic runs. `--capture-deps` or `--chip-swimlane N` requires `--rounds 1`; diagnostic runs do not provide comparable timing rounds.

```bash
.venv/bin/python -m tests.st.scheduler_benchmark.run_cases \
  --platform a5sim --runtime host_build_graph --list-only

.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 1 --max-time 7200 \
  --run '.venv/bin/python -m tests.st.scheduler_benchmark.run_cases --platform a5 --runtime host_build_graph --device "$TASK_DEVICE" --rounds 10'
```

Repeat the onboard command with `--runtime tensormap_and_ringbuffer` for the second runtime. On an A2/A3 host, use `a2a3` after its own precheck. The actual AIC/AIV executor counts are queried on hardware; do not substitute the illustrative counts in the design document. Existing W cases may be marked `unsupported_platform` on simulation; that is a recorded skip, not a passing measurement.

## Complete workflow with persistent processes

```bash
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device auto --device-num 1 --max-time 7200 \
  --run '.venv/bin/python -m tests.st.scheduler_benchmark.run_benchmark --platform a5 --device "$TASK_DEVICE" --manifest outputs/runtime_benchmark.json'
```

The full entry runs one 101-round timing batch (first round separate, 100 main samples), separate one-round DAG and level-3 event phases, offline metrics, paired swimlanes and the fixed 41-ID report. Repeated `--runtime` selects runtimes; omitting it selects both. Repeated `--case` narrows the run. Existing single-phase `run_cases` flags and its 10-round default remain valid. Use a fresh full-workflow output path; existing artifacts are not overwritten.

Repeating a single-phase manifest path uses a fresh capture directory. Native Host/device epochs must match when present. Without a native Host epoch, only an isolated one-round capture with one device epoch and a bounded one-round Host log can be associated; the native epoch check remains unmeasured. Never equate Host invocation numbers with device epochs.

`workflow.json` (or `--manifest`) records start/end UTC, actual wall duration, phase durations and shared-control/analysis time. Per-runtime spans include waits for the other runtime and final shared processing, so they overlap and must not be summed. Phase manifests include `execution_model`, process/session/Worker identity, callable fingerprints and Host-only phase accounting. Reports refuse mixed execution models, including legacy per-case-process timing combined with persistent timing. Unsupported memory queries remain null.

An execution/cleanup error stops that runtime, preserves its failure and leaves remaining IDs not_run. An unconfirmed close stops device collection. Golden failures and capture/parse failures are recorded without silently switching to per-case processes. Confirmed Runtime defects are diagnosed separately; this workflow does not change scheduler implementation. Respect an explicit user pause: implementation acceptance can use agreed subsets, but does not authorize a full hardware rerun.

## Read the manifest

Check every selected ID in `case_results`. In `measurement_mode=timing`, `passed` requires all requested positive `device_us` rounds. The full workflow stores round zero separately in `first_round` and leaves the other rounds in `rounds`; single-phase commands retain all requested rounds. In diagnostic mode, `passed` requires the requested capture files and has no timing rounds. `failed` has a case log or an error; `unsupported_platform` and `not_run` have no measurement. Record the commit, dirty-worktree flag, PTO-ISA pin, CANN version, SoC, runtime, and expanded executor counts with every comparison. Do not combine results from different build environments into a single runtime conclusion.

`device_us` is the full device run wall. TMR may also expose `effective_us`, `orch_us`, and `sched_us`; HBG normally does not. A missing field or `null` is unmeasured, never zero. `scheduler_location` and `kernel_execution_us` are filled from diagnostic events when available and remain `null` without independent observation. Diagnostic `diagnostic_device_us` is kept separate from timing rounds. The skill must not infer Scheduler location from the case ID or use W1 versus W5 device time as an AICore-versus-AICPU comparison.

Use paired configurations and repeated no-diagnostic rounds to compare runtimes or revisions. Keep correctness/diagnostic evidence, timing evidence, and any later per-task attribution distinct. The current manifest alone does not establish Scheduler bottleneck severity or replay-based improvement potential; for those claims, follow the event and replay requirements in the benchmark design document. Report failures and unsupported cases alongside timing results.

## Event diagnostics and matrix reports

At level 2, `task_transitions` adds actual R, dependency-registration/device-visibility timing and available early-dispatch gate timing. The AICPU record is 40 B and the Executor record remains 32 B. Native A5 HBG reuses existing AICore trace fields. Level 3 supplies dependency-only control markers; level 4 adds individual TMR backpressure intervals. Missing/overflowed events, drops, clock-order failures and incomplete multi-core resource constraints produce unmeasured metrics.

Diagnostic collection preserves raw captures. Offline reporting writes `scheduler_metrics.json/.md`, `scheduler_stages.json` and `merged_swimlane.json` once the independent DAG and event inputs are available. Merge the manifests to validate the C graph and calculate F-to-R, I_data and critical-path attribution:

```bash
.venv/bin/python -m tests.st.scheduler_benchmark.report outputs/deps_manifest.json outputs/events_manifest.json \
  outputs/timing_manifest.json --output outputs/scheduling_report.json
```

The report keeps all 41 IDs, including not_run entries. It checks source-patch fingerprints, environment, physical device and expanded configuration before combining data. Use `--scale 2` with selected C IDs for extra calibration runs; width and P/C stay fixed while chain length or repeated units grow. Calibration runs retain the same ID with a separate task_scale and do not enlarge the fixed matrix.

Inspect C3 qualification before quoting release costs: current TMR manual scope does not ensure live consumer edges are registered before producer completion. Unqualified units do not contribute to the target fan-in/out summary. W dependency metrics stay unmeasured until the actual DAG association is verified. Treat observed stage contributions and idle-with-ready intervals as diagnostic evidence; end-to-end removable-time estimates require the later replay phase.

Sessions prebind one persistent `host_logs/` directory before Worker initialization. Preserve `host_log_directory`, the per-case `host_log_slice`, run identities and Worker generation along with each case log: diagnostic `[STRACE]` records move to the process logs, and the default transient spool is deleted at process exit. Use `--overhead-baseline` on the report CLI with separate old-diagnostic, new-diagnostic and no-diagnostic repeated samples from the same device and configuration; absent device timing stays unmeasured.

## Zero-overhead host replay

Use `python -m simpler_setup.tools.scheduler_replay <deps-manifest> <events-manifest> --output outputs/replay`
to replay compatible C graphs with native runtime scheduling code on a virtual clock.
Direct capture mode takes `<chip_swimlane_records.json> --deps <deps.json>`.
See `simpler_setup/tools/README.md#scheduler_replay` for supported scheduler paths,
configuration, output artifacts and exclusions. No onboard command or NPU is needed.

The tool retains actual AICore S-to-E durations and native queue/core/slot policies,
but gives all scheduler and submission operations zero virtual cost, including E-to-F.
TMR uses native finite pools and scope reclamation. Inspect the configuration source:
absent capture ownership or an explicit setting, thread defaults are labelled assumptions.
Multi-block/MIX identity gaps, early-dispatch attributes, allocation-dependent W graphs
and unverified execution-instance associations must stay unsupported.

`comparison.json` contains paired observed and synthetic Perfetto timelines. The
model's `T_ideal_us` is policy-specific, not an optimum or a predicted measured speedup.
It holds kernel durations fixed after rescheduling and does not reproduce measured
scheduler delays. Keep the model gap separate from observed stage attribution and
from no-diagnostic end-to-end performance conclusions.
