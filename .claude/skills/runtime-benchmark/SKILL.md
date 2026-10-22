---
name: runtime-benchmark
description: Run or analyze the fixed 41-case scheduling benchmark, including offline replay, performance assessment and bottleneck reports across simpler runtimes, chips or revisions. Use benchmark or perf-runtime-device for ordinary example latency sweeps.
---

# Runtime scheduling benchmark

Use `tests/st/scheduler_benchmark/run_cases.py` for the fixed matrix: 33 constructed DAG configurations from `matrix.py` and 8 existing workloads from `workloads.py`. The runner expands `configuration ID × platform × runtime` and executes cases sequentially in one process with one shared L2 Worker per measurement phase. Each case retains its own inputs, log slice and capture directory. The unified runner skips golden generation and numerical output comparison for all C and W configurations in timing, DAG and event phases; input preparation, per-round reset and structural capture validation remain enabled. Standalone scene tests retain their own golden behavior. If the sibling documentation checkout exists, read `../D200_docs/benchmark/README.md` from the `simpler` root for the matrix definitions and evaluation limits.

For benchmark or performance-analysis requests, complete collection/inspection, offline replay and a written performance assessment with bottleneck findings. Preserve the user's selected subset and explicit collection-only or inspection-only scope. Read [the assessment methodology](../../../docs/dfx/runtime-benchmark-analysis.md) before interpreting results; it is the maintained reference for metrics, evidence requirements and report content.

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

The collection CLI does not execute replay or write the assessment narrative. Continue with the offline replay and assessment steps below after it exits. Existing compatible captures can be reused without another device run.

For a full run, use the enumerated 41 IDs as the coverage checklist for every selected Runtime and each timing, DAG and event phase. Reconcile the actual case IDs and sample counts, not just aggregate pass counts. Preserve every failed, skipped or missing entry and its reason; successful file collection alone does not establish usable diagnostic events or supported replay. Reuse valid captures and target any necessary recollection to the affected cases. Do not describe unsupported or incomplete analysis as successfully completed.

Repeating a single-phase manifest path uses a fresh capture directory. Native Host/device epochs must match when present. Without a native Host epoch, only an isolated one-round capture with one device epoch and a bounded one-round Host log can be associated; the native epoch check remains unmeasured. Never equate Host invocation numbers with device epochs.

`workflow.json` (or `--manifest`) records start/end UTC, actual wall duration, phase durations and shared-control/analysis time. Per-runtime spans include waits for the other runtime and final shared processing, so they overlap and must not be summed. Phase manifests include `execution_model`, process/session/Worker identity, callable fingerprints and Host-only phase accounting. Reports refuse mixed execution models, including legacy per-case-process timing combined with persistent timing. Unsupported memory queries remain null.

An execution/cleanup error stops that runtime, preserves its failure and leaves remaining IDs not_run. An unconfirmed close stops device collection. Capture/parse failures are recorded without silently switching to per-case processes. Confirmed Runtime defects are diagnosed separately; this workflow does not change scheduler implementation. Respect an explicit user pause: implementation acceptance can use agreed subsets, but does not authorize a full hardware rerun.

## Read the manifest

Check every selected ID in `case_results`. Executed cases record `golden_validation="skipped"`; a passing execution or capture does not establish numerical output correctness. Reports preserve absent validation metadata in older results as `unrecorded`, not passed or skipped. In `measurement_mode=timing`, `passed` requires all requested positive `device_us` rounds. The full workflow stores round zero separately in `first_round` and leaves the other rounds in `rounds`; single-phase commands retain all requested rounds. In diagnostic mode, `passed` requires the requested capture files and has no timing rounds. `failed` has a case log or an error; `unsupported_platform` and `not_run` have no measurement. Record the commit, dirty-worktree flag, PTO-ISA pin, CANN version, SoC, runtime, and expanded executor counts with every comparison. Do not combine results from different build environments into a single runtime conclusion.

`device_us` is the full device run wall. TMR may also expose `effective_us`, `orch_us`, and `sched_us`; HBG normally does not. A missing field or `null` is unmeasured, never zero. `scheduler_location` and `kernel_execution_us` are filled from diagnostic events when available and remain `null` without independent observation. Diagnostic `diagnostic_device_us` is kept separate from timing rounds. The skill must not infer Scheduler location from the case ID or use W1 versus W5 device time as an AICore-versus-AICPU comparison.

Use paired configurations and repeated no-diagnostic rounds to compare runtimes or revisions. Keep any independently obtained correctness evidence, diagnostic evidence, timing evidence, and any later per-task attribution distinct. The current manifest alone does not establish Scheduler bottleneck severity or replay-based improvement potential; for those claims, follow the event and replay requirements in the benchmark design document. Report failures and unsupported cases alongside timing results.

## Event diagnostics and matrix reports

At level 2, `task_transitions` adds actual R, dependency-registration/device-visibility timing and available early-dispatch gate timing. The AICPU record is 40 B and the Executor record remains 32 B. Native A5 HBG reuses existing AICore trace fields. Level 3 supplies dependency-only control markers; level 4 adds individual TMR backpressure intervals. Missing/overflowed events, drops, clock-order failures and incomplete multi-core resource constraints produce unmeasured metrics.

Diagnostic collection preserves raw captures. Offline reporting writes `scheduler_metrics.json/.md`, `scheduler_stages.json` and `merged_swimlane.json` once the independent DAG and event inputs are available. Merge the manifests to validate the C graph and calculate F-to-R, I_data and critical-path attribution:

```bash
.venv/bin/python -m tests.st.scheduler_benchmark.report outputs/deps_manifest.json outputs/events_manifest.json \
  outputs/timing_manifest.json --output outputs/scheduling_report.json
```

The report keeps all 41 IDs, including not_run entries. It checks source-patch fingerprints, environment, physical device and expanded configuration before combining data. Use `--scale 2` with selected C IDs for extra calibration runs; width and P/C stay fixed while chain length or repeated units grow. Calibration runs retain the same ID with a separate task_scale and do not enlarge the fixed matrix.

Inspect C3 qualification before quoting release costs: current TMR manual scope does not ensure live consumer edges are registered before producer completion. Unqualified units do not contribute to the target fan-in/out summary. W dependency metrics stay unmeasured until the actual DAG association is verified. Treat observed stage contributions and idle-with-ready intervals as diagnostic evidence; the available zero-overhead replay is a model reference, not a per-stage causal savings estimate.

Sessions prebind one persistent `host_logs/` directory before Worker initialization. Preserve `host_log_directory`, the per-case `host_log_slice`, run identities and Worker generation along with each case log: diagnostic `[STRACE]` records move to the process logs, and the default transient spool is deleted at process exit. Use `--overhead-baseline` on the report CLI with separate old-diagnostic, new-diagnostic and no-diagnostic repeated samples from the same device and configuration; absent device timing stays unmeasured.

## Offline replay

After collection, run replay for the collected configurations using their compatible DAG, event and timing manifests. For the default two-runtime invocation with `--manifest outputs/runtime_benchmark.json`, the phase files are under `outputs/runtime_benchmark/`:

```bash
.venv/bin/python -m simpler_setup.tools.scheduler_replay \
  outputs/runtime_benchmark/host_build_graph/deps.json \
  outputs/runtime_benchmark/host_build_graph/events.json \
  outputs/runtime_benchmark/host_build_graph/timing.json \
  outputs/runtime_benchmark/tensormap_and_ringbuffer/deps.json \
  outputs/runtime_benchmark/tensormap_and_ringbuffer/events.json \
  outputs/runtime_benchmark/tensormap_and_ringbuffer/timing.json \
  --output outputs/runtime_benchmark/replay
```

Use only the selected runtimes' actual manifest paths. A custom manifest path uses its suffix-stripped path as the collection directory. Keep all collected IDs visible, including unsupported W configurations; do not silently omit unsuccessful replay attempts. Use a fresh replay directory to keep derived artifacts separate from earlier assessments. This step needs no NPU or device lock.

When building a new matrix report from separate phase manifests, `tests.st.scheduler_benchmark.report --replay-output <directory>` is an alternative. After the complete collection entry has already analyzed captures, call `scheduler_replay` directly to avoid repeating that analysis. See [the replay guide](../../../docs/dfx/scheduler-replay.md) for direct-capture mode, `--config`, supported resources and output artifacts.

Inspect `replay_matrix.json` row and sample statuses even when the command exits zero. For supported inputs, retain `replay_summary.json`, observed/model `comparison.json`, configuration assumptions and source hashes. Record unsupported inputs and errors with their reasons. A replay error is an incomplete analysis; it is not a successful time estimate. Do not change the runtime scheduler or weaken input checks to obtain a result.

For unsupported results, distinguish the matrix entry's rejection from checks actually applied to the input. Inspect the relevant guard and the captured DAG/events: a blanket W exclusion does not prove missing profiling data. Group cases by verified cause, identify additional task-instance/resource limitations only where observed, and state whether an input-only check or a native replay was executed. Do not recommend recollection as the remedy for a software adapter limitation. Preserve the recorded statuses; diagnostic inspection does not promote a case to supported.

## Performance assessment and bottleneck report

Read [the assessment methodology](../../../docs/dfx/runtime-benchmark-analysis.md), then write `performance_assessment.md` alongside the run's report. Base it on both the benchmark/diagnostic report and replay outputs, not on command success or aggregate timing alone. Link the exact source manifests, JSON reports and relevant comparison swimlanes.

Write complete independent sections for each Runtime, each with its own 41-case coverage, timing, replay, diagnostics, first/second limiting factors and follow-up. Do not combine Runtime columns, count winners or publish cross-Runtime ratios/conclusions; that comparison is a separate later task. Rank observed path contributions using resource and registration evidence, distinguish stage observations from root-cause hypotheses, and explicitly mark factors that cannot be established. Explain C3 release qualification in a C3-only detail table. The maintained methodology specifies these reporting requirements.

C1, C2-01 through 10 and C3 computation tasks target approximately 5 us using `task_iterations`; C2-11 through 15 retain original short/long work, including the short tasks in mixed cases. Validate on hardware without changing graph shape or per-Runtime work. Preserve legacy captures separately from newly calibrated measurements.

Include scope and coverage; no-diagnostic timing distributions and valid comparisons; per-configuration replay status, reference completion time, observed-window gap in microseconds, actual/replay time ratio (`T_exec_observed_us / T_ideal_us`, expressed as a multiple), and scheduling-overhead share estimated by replay (`(T_exec_observed_us - T_ideal_us) / T_exec_observed_us`, expressed as a percentage). Keep the ratio and overhead share as separate columns, label the latter as a model estimate, and omit the model lower bound from the assessment. Include evidence-backed bottleneck findings, prioritized follow-up measurements, and missing-data/model limitations. Keep unrun IDs and unsupported cases visible. If only a subset was requested, label the assessment as a subset rather than a complete matrix result.

The main assessment must contain complete per-case coverage, timing, replay and diagnostic/bottleneck tables for the selected matrix: all 41 IDs for a full run, with every selected Runtime represented. Do not replace these tables with representative cases, top-N lists, plots or a link to an exhaustive appendix. Give each case its test objective, evidence-supported assessment or explicit inability to assess, and the next measurement if needed. Mark inapplicable family metrics with a reason and unavailable metrics with their actual status; neither is zero. Family-specific detail tables may accompany the complete chapter inventory, but their scope must be labeled. Before delivery, compare each complete table's ID/Runtime set with the collection checklist and disclose remaining gaps. Detailed requirements are maintained in the assessment methodology.

In the measured-performance chapter, report each case/Runtime's arithmetic mean task execution time from its valid diagnostic sample alongside the no-diagnostic graph timings. Follow the methodology's logical-task and execution-fragment definitions, exclude dummy/control tasks, and record counts and capture origin. Separate AIC/AIV fragment means when present. Do not substitute a kernel median or graph time divided by task count for the measured task mean; identify unsupported logical-task association or incomplete execution coverage explicitly.

Use kernel durations, critical-path stages, actual Ready depth, idle-with-ready intervals, utilization, submission/backpressure and C3 qualification together to identify limiting mechanisms. The replay model retains native scheduling/resource policies while assigning submission and scheduling zero virtual cost; its `T_ideal_us` and gap are references, not measured speedups or proof that the whole gap is removable Scheduler overhead. Distinguish observations, supported interpretations and hypotheses. If the inputs cannot establish a bottleneck or speedup, state what remains unknown and which evidence would resolve it.

Finish with the assessment link, timing/capture/replay coverage, principal findings and material limits. Do not present a passing collection or a replay table alone as the completed performance evaluation.
