# Runtime benchmark performance assessment and bottleneck analysis

This is the methodology reference for the [runtime-benchmark skill](../../.claude/skills/runtime-benchmark/SKILL.md). The skill organizes collection, replay and reporting; this document maintains metric definitions, comparison methods, bottleneck criteria and report requirements. Configuration definitions are in benchmark/README.md in the sibling D200_docs checkout. Replay inputs, resource constraints and commands are documented in the [replay guide](scheduler-replay.md).

## 1. Inputs, coverage and evidence limits

Analyze each configuration ID, physical device, Runtime, code/environment version and task scale separately. Record the selected configurations, available AIC/AIV counts, kernel parameters, execution model, round policy, revision/patch fingerprints, CANN and PTO-ISA. Revisions may differ in a version comparison, but timing, DAG, event and replay inputs must correspond within each revision. Do not mix captures from different revisions into one attribution.

| Input | What to inspect | Supported conclusions |
| ----- | --------------- | --------------------- |
| workflow.json and phase manifests | Execution status, environment, first round and main samples, log boundaries, capture paths, Host phase costs | Which configurations ran, whether comparisons are valid, and workflow wall-clock cost |
| No-diagnostic timing manifest | Per-round device_us and first_round | Device latency, throughput and observed revision/Runtime differences |
| report.json and scheduler_metrics.json | Actual events, stage distributions, core occupancy, Ready depth, I_data/I_sched, critical path, resources and C3 qualification | Limiting mechanisms supported by the captured diagnostic timeline |
| replay/replay_matrix.json and per-sample replay_summary.json | Status, T_ideal_us, observed window, modeled occupancy, configuration sources and assumptions | Reference performance under the selected Runtime's zero-scheduling-cost model |
| comparison.json, scheduler_stages.json and raw logs | Tasks, execution cores, critical path and stage intervals | Whether aggregate signals correspond to particular execution behavior |

Count passed, failed, not_run and unsupported_platform by collection phase. Separately count replay supported, unsupported, incomplete and error states, retaining their reasons. A zero replay CLI exit code does not mean every sample is supported. A full report retains all 41 IDs; subset acceptance makes conclusions only about selected configurations and marks the rest not_run. Report timing, DAG, event, attributable-metric and replay coverage separately rather than using one pass rate.

Reconcile the expected ID set with the actual entries for each selected Runtime and each collection phase. Check requested main sample counts and first-round handling per case. An aggregate count of 41 cannot detect duplicate or substituted IDs. Keep original failed diagnostic samples visible when a compatible recollection supplies a usable replacement, and identify the capture used for each derived metric. Reuse existing valid inputs rather than repeating completed device measurements to repair report presentation.

The unified runner skips golden generation and output comparison for all C/W cases. golden_validation=skipped records that policy; passed does not establish numerical correctness. Treat absent metadata in older results as unrecorded. Simulation validates execution and analysis workflows, not hardware latency or cross-chip performance.

Use only metrics with sufficient inputs when events are missing, overflowed or out of clock order, or when control-node or multi-core resource descriptions are incomplete. W/Graph inputs without verified DAG association do not support dependency-stage or critical-path attribution, or current fixed-matrix replay.

## 2. Comparing measured performance, diagnostics and replay

### 2.1 No-diagnostic performance

Report the main sample count, median, p95, min/max and individual rounds. Keep the first round separate according to the runner's contract. Inspect drift, multiple modes and outlier intervals; do not discard data just because trimming improves the result. One batch of 100 rounds describes that batch, not stability across days.

For a matching configuration, identify the reference and candidate:

- Device-time difference: candidate median minus reference median; negative means faster.
- Time reduction: (reference median minus candidate median) / reference median.
- Speedup: reference median / candidate median.
- C-graph throughput: actual kernel task count / T_device. Exclude C3 dummy gates from the kernel count, while their measured cost remains in device time.

Hold the device, equivalent graph, kernels, core counts and integer task parameters fixed when comparing HBG/TMR. Interleave reference/candidate runs on the same device when assessing an implementation change. Existing sequential captures describe the observed difference but cannot rule out batch drift. When variation is comparable to the revision difference, report that the alternatives cannot yet be distinguished reliably. For different chips, report absolute work and pressure normalized by available cores separately; do not combine them into one scheduling score.

C1-01 through 08, C2-01 through 10 and C3-01 through 10 use the common `task_iterations` workload, calibrated to approximately 5 us per computation task on A5. Increase work inside each kernel while retaining task counts, graph edges, widths and fan-in/out. C2-11 through 15 retain their original 1/2048 short/long iteration counts, including short tasks inside mixed cases; W workloads are unchanged. Record measured AIC/AIV task durations and integer iteration counts rather than promising the same duration on every chip. Legacy captures without `task_iterations` predate this calibration and must not be merged with calibrated captures.

### 2.2 Diagnostic timelines

T_device spans device-run entry through completion. T_exec spans the first Executor task start through the last task end; occupancy and current critical-path attribution use T_exec. Host bind, input preparation, golden/comparison, offline processing and queueing are outside T_device.

Use actual E/F/R/D/A/S events to interpret completion observation, dependency confirmation, dispatch, acquisition and start. F-to-R starts at the last predecessor completion observed at F; tasks without predecessors have no such interval. Analyze early dispatch using its actual doorbell/start semantics rather than ordinary event ordering.

R-to-A or R-to-S includes normal resource queueing while compatible execution cores are busy. Do not attribute the whole interval to software overhead. Per-task waits overlap and cannot be summed into device-time loss. Use the generated mutually exclusive critical-path contributions to locate stages that affect graph completion; these observed contributions are not directly removable time.

I_sched is the fraction of wall time with both a Ready task and a compatible idle core, not the fraction of total core time spent idle. High I_sched can coexist with high overall occupancy. Combine it with per-core occupancy, specific queues and path stages. F-to-R can also include waiting for a successor to be submitted or registered. Use late_registration_tasks and registration timestamps to distinguish upstream supply from dependency handling for already registered tasks; large F-to-R alone does not establish a slow dependency algorithm.

Diagnostics and no-diagnostic timing come from independent runs. Diagnostic T_device / no-diagnostic median may describe their observed difference, but it mixes variation with collection effects. Without repeated controls, do not report net collection overhead or extrapolate the diagnostic ratio into measured gains. Subtract T_exec from T_device only within one diagnostic run; the difference does not represent all fill and drain costs.

#### Mean task execution time

Include the arithmetic mean execution duration for every selected case/Runtime in the measured-performance chapter. Derive it from the corresponding valid diagnostic events, with their capture origin and sample counts. A stored median/p95/min/max distribution cannot recover the arithmetic mean; use the underlying execution events.

Group fragments by the verified run identity and logical task ID. For a computation task, its execution span is the last fragment end minus the first fragment start. Average these spans over observed computation tasks, counting a multi-core or MIX task once. This span includes start skew between that task's fragments, while excluding its wait before the first execution start. Exclude identified dummy/dependency controls and other tasks without computation. Preserve their observed costs in graph time; the replay's zero-time dummy policy does not supply measured kernel durations.

Also report AIC and AIV execution-fragment means separately: sum of observed S-to-E durations of that type divided by the number of its computation fragments. A multi-fragment task can contribute multiple fragment samples but only one logical-task sample. Give both denominators and label the granularity. With mixed short/long kernels, retain the configured mix and avoid interpreting the combined mean as the duration of either class.

Graph device time divided by task count is an amortized graph-time/throughput measure, not mean task execution time: parallel tasks overlap, and graph time includes scheduling. Keep it separately labeled if reported. Diagnostic task means are not 100 independent no-diagnostic timing samples and must not be presented as such. Missing execution events or ambiguous task identity prevent a complete logical-task mean; report the available fragment statistics and the limitation. A W input can have usable observed task execution spans without verified cross-phase DAG association, but those means do not establish complete DAG coverage, dependency attribution or supported replay.

### 2.3 Zero-scheduling-cost model

The model retains captured kernel S-to-E durations, native Runtime policy, core capacity and supported resource constraints. Submission, dependency processing, completion observation and dispatch take zero virtual time. Kernel durations remain fixed after rescheduling, so changes in cache behavior, contention or frequency caused by placement and overlap are outside the model.

Dummy dependency gates retain task identities, predecessors, successors and scopes. They take exactly zero modeled time and consume no execution-core capacity. The fixed-C adapter verifies the complete graph and dummy positions before treating HBG AICore-produced dummy AIV fragments as zero-time controls; unidentified extra fragments remain unsupported. Raw observations and the observed swimlane retain measured control durations. Replay uses Perfetto instant events with dependency arrows and adds no artificial 1 ns. Preserve dummy_controls and assumptions from replay_summary.json; the dummy policy means not every observed fragment retains its original duration in the model.

| Quantity | Definition and purpose | Interpretation limits |
| -------- | ---------------------- | --------------------- |
| T_ideal_us | Completion time under the current Runtime policy with zero scheduling cost | Neither a globally optimal schedule nor predicted measured device time |
| T_exec_observed_us | Observed execution window corresponding to this replay input | Diagnostic time, not no-diagnostic T_device |
| Execution-window model gap | T_exec_observed_us minus T_ideal_us | Guides investigation; not confirmed removable software cost |
| Actual/replay time ratio | T_exec_observed_us / T_ideal_us | How many times longer the actual window is: 1x means equal, 5x means five times as long; not predicted measured speedup |
| Scheduling-overhead share (replay estimate) | (T_exec_observed_us minus T_ideal_us) / T_exec_observed_us * 100% | Extra observed time relative to replay as a share of the actual window; a combined submission/dependency/dispatch control-path estimate, not causal attribution to one Scheduler stage |

Report the absolute gap, actual/replay ratio and overhead share together. They answer how many extra microseconds were observed, how many times longer execution took, and what share of the actual window the estimated overhead represents. For ratio M, overhead share is (1 - 1/M) * 100%; 5x corresponds to 80%. Mark both metrics unmeasured for invalid replay. Require positive T_ideal_us and T_exec_observed_us for calculation. Omit the model lower bound from the assessment table.

The overhead share comes from the model gap and includes the combined zero-cost assumptions for submission, dependency processing and dispatch, plus rescheduling effects. Retain the replay-estimate label. It is distinct from an overhead fraction of no-diagnostic T_device and does not directly predict gains from changing one scheduling stage.

Keep negative gaps rather than clipping them to zero. Investigate clocks, task association, resource settings, same-time event ordering and model assumptions. Preserve scheduler_configuration, assumptions, source/binary fingerprints and input paths. Disclose assumed default thread configurations.

Check every row and sample status before quoting time. unsupported has no valid estimate. For incomplete/error rows, list failed samples rather than treating the successful subset as a passing row. Supported whole-graph C3 replay does not replace validation of actual producer/consumer release conditions. The current model does not reproduce measured scheduling delays or causally estimate savings from optimizing individual stages.

#### Diagnosing unsupported inputs

Trace the recorded reason to the matrix-entry guard and then inspect the actual captured inputs. Separate a class-wide adapter exclusion, invalid/missing captures, execution-instance/DAG mismatch, and unsupported execution/resource semantics. A guard that rejects all W IDs before invoking the replay adapter does not establish which W inputs would otherwise pass. Likewise, collecting both DAG and events does not by itself prove that repeated Graph instances, control tasks and resource requirements are represented completely.

For each affected case/Runtime, retain the initial status and inspect DAG node/edge counts, observed logical tasks and fragments, task-ID coverage, event quality, early dispatch, allocations, multi-block/MIX shapes and available launch/placement metadata. Distinguish the first actual adapter rejection from other attributes found in the input; do not assert that removing one check resolves later checks. An input-only normalization probe can identify a more specific obstacle, but it executes no native replay and cannot validate cross-phase association or produce a supported time estimate. Never bypass checks or mutate the input to obtain a passing assessment.

State which remedy is required: an adapter/backend change, added capture metadata, or recollection of an invalid sample. Repeating a valid capture cannot by itself implement missing replay semantics. Keep run-specific counts and evidence in the assessment, with source links, rather than hard-coding them into this methodology.

## 3. Analysis by configuration

| Question | Configuration comparison | Method |
| -------- | ------------------------ | ------ |
| Do serial handoffs limit the critical path? | C1-01/02, 03/04, 05/06, 07/08 | Compare chain-length increments and middle-window successor S minus predecessor E; inspect stages and replay gaps. Time increments still include kernels and Runtime work that grows with scale. |
| Are cross-core-type handoffs asymmetric? | C1-05 through 08 | Both starting-type variants include AIC-to-AIV and AIV-to-AIC. Group individual edges by direction rather than subtracting whole-graph times to estimate directional cost. |
| When does added parallelism saturate execution resources? | C2-01 through 04 and 05 through 08 | Plot throughput against parallel chains / available cores, together with actual Ready depth, occupancy and I_sched. Task counts differ; do not rank configurations only by graph latency. |
| Can independent tasks supply enough work? | C2-09/10 | Analyze publication, startup, steady execution and drain. Independent tasks need not be published or Ready simultaneously; comparing them with parallel chains does not hold everything except dependency edges fixed. |
| Does scheduling matter less for larger compute granularity? | C2-03/11 and 07/12 | Hold graph shape fixed and compare short/long kernels, stage latency relative to kernel duration, occupancy and replay gaps. |
| Do long tasks delay short tasks or the tail? | C2-13 through 15 | Compare short-task Ready-to-first-start median/p95, per-core load and drain. Changing the mix also changes compute work; slower graph completion alone does not establish unfair scheduling. |
| Does fan-out cause batch release cost? | C3-01 through 04 | Fix one producer and consumer counts of 1, 4, 8 and 32 across chips; compare last predecessor F to first/last consumer R, Ready spread and post-release fill. |
| Does fan-in increase dependency-processing cost? | C3-01 and 05 through 08 | Fix one consumer across chips and vary producer counts of 1, 8, 16, 32 and 64. Last F-to-R does not represent all earlier dependency updates. |
| Does high fan-in/fan-out scale well? | C3-07/09, 08/10 and 09/10 | The first two pairs fix producer count; the last fixes consumer count. Compare release/start per unit, dummy gates, peak live dependencies and reclamation backpressure. |
| Do microbenchmark effects matter for real workloads? | Matching configurations/implementations within W1 through W5 | Validate actual differences with no-diagnostic device time and sufficiently supported event metrics. PA automatic/manual variants use different kernels, so their time difference is not directly attributable to scope. |

For C3, report qualified middle units / total middle units first. Only units with dependency registration before producer completion and qualified release ordering support the intended fan-in/fan-out conclusion. The current reporter requires all middle units to qualify before generating release_summary. Repeat counts differ across configurations: analyze release per unit while retaining gate and reclamation costs in whole-graph time.

A single width, chain length or long/short mix supports conclusions about that point, not a saturation curve, scaling stability or fairness. List missing comparisons as follow-up measurements.

## 4. Bottleneck criteria and strength of conclusions

Ask whether executable work exists, whether compatible resources are available, which stage prevents start, whether the interval affects the critical path, and whether it changes no-diagnostic device time. Model gaps guide interval selection; events, resources and swimlanes establish possible mechanisms.

| Candidate limitation | Supporting signals | Alternatives or further checks |
| -------------------- | ------------------ | ------------------------------ |
| Graph parallelism | Low occupancy, Ready depth limited by graph width; replay also has little executable work | Low C1 occupancy follows from its serial graph, while handoff latency may still limit its critical path. |
| Kernel compute/resource capacity | Ready tasks persist while compatible cores are busy; replay also queues for resources; execution dominates the path | Busy cores do not prove efficient compute units; investigate kernel internals separately. |
| Delayed completion observation | Significant critical-path E-to-F while data is ready and actual R has not arrived | Clock calibration, task publication/visibility and completion notification. |
| Dependency confirmation/batch release | Qualified F-to-R or release spread grows with fan-in/fan-out and lies on path gaps | Late submission/registration, C3 qualification, already-completed producers, scope/gates and edge reclamation. |
| Dispatch, acquisition or start waits | Actual R has occurred, compatible cores are idle, and I_sched agrees with path R-to-D/D-to-A/A-to-S | Multi-core/MIX resource requirements, affinity, early-dispatch doorbells, already acquired tasks and normal queueing. |
| Upstream submission or reclamation backpressure | Late publication/registration, insufficient Ready supply, pool high-water marks or backpressure intervals coinciding with idle time | Total edges are not peak live edges. Level-2 totals do not establish temporal correlation; complete wait intervals usually require level 4. |
| Fill/drain and fixed cost | Head/tail or out-of-window costs are prominent for small cases; scale comparisons suggest a large fixed term | T_device minus T_exec omits fill/drain inside the execution window; additional scale points are needed. |
| Host/test workflow cost | Long workflow wall time with a low fraction spent executing on device | Separate input preparation, transfers, reset, compilation, analysis, queueing and device execution. |

Distinguish three levels in each finding:

- Observation: cite sample counts, windows, configurations, metrics and swimlane intervals.
- Supported interpretation: identify the combined signals and alternatives already ruled out.
- Hypothesis: identify missing comparisons/events and explain how confirming or rejecting them changes the conclusion.

Prioritize by affected workloads, critical-path impact, repeatability and evidence completeness. Do not use fixed severity percentages without calibration or promise speedup. A single diagnostic sample supports interpretation of that timeline; measured gains require no-diagnostic comparisons between revisions on the same device.

For every case within each Runtime, explicitly identify the first and second limiting factors. Start from the largest mutually exclusive observed critical-path contributions in microseconds and as a fraction of T_exec, then interpret them using Ready supply, registration, compatible-core availability, utilization, placement and qualified release evidence. Include graph/compute/resource limits as well as control-path waits. A stage ranking is an observation, not a proven root-cause ranking or a per-stage speedup estimate. In particular, distinguish late registration from dependency processing, and normal resource queueing from dispatch stalls. Record each factor's evidence, interpretation strength, alternatives and next verification. If a first or second factor cannot be established, say so rather than filling the rank with a small residual or an unsupported hypothesis. Missing W DAG association prevents path-based ranking even when timing and utilization are measured.

## 5. Assessment report and delivery checks

Write performance_assessment.md beside the run's report and link the source manifests, report.json, replay matrix, sample summaries and swimlanes. Keep run-specific findings in that report rather than this methodology document. Save reports separately and preserve raw captures.

Organize the assessment into complete, independent Runtime sections: first host_build_graph, then tensormap_and_ringbuffer when both are selected. Each section contains its own scope/coverage, measured performance, replay, diagnostics, first/second limiting factors and follow-up priorities. Do not use paired Runtime columns, winner counts, cross-Runtime ratios or comparative conclusions in this assessment. Cross-Runtime comparison is a separate later task. Observed-versus-replay comparisons within one Runtime remain part of its analysis.

Include at least:

1. Main findings and scope: device, Runtime, revision, full matrix or subset, supported conclusions and principal limits.
2. Coverage: timing/DAG/event/attribution/replay counts, missing or failed inputs and reasons, golden policy and first-round rules.
3. Measured performance: configuration, cores/task scale, sample count, median, p95 and min/max; add differences and ratios only with a valid reference.
4. Replay: status, observed T_exec, T_ideal, gap in microseconds, actual/replay ratio, scheduling-overhead share (replay estimate), assumptions and comparison swimlanes; retain unsupported cases.
5. Bottleneck findings: observations, path/resource/model evidence, interpretation strength, alternative explanations and affected scope; not just stage p95 values.
6. Follow-up priorities: concrete comparisons or collection needs that resolve the main uncertainties. Do not infer model-level gains when relevant real workloads are missing.
7. Limits and artifact index: collection effects, variation, non-interleaved runs, missing DAGs, C3 qualification, resource/clock/model assumptions and links to all evidence.

### Complete per-case tables

Each Runtime section's coverage, measured-performance, replay and diagnostic/bottleneck chapters need an exhaustive case table. A full run means all 41 unique IDs in each chapter of each selected Runtime section. For an explicitly requested subset, cover the selected IDs and label the subset. Representative examples, plots and appendix links supplement these tables rather than replacing them.

- Coverage: identify timing sample counts, DAG/event collection status, diagnostic usability and replay status for each case/Runtime. Distinguish collection success, input quality and supported analysis.
- Timing: include main sample count, median, p95 and min/max for the current Runtime, plus diagnostic mean task execution time with counts, granularity and capture origin as defined above. Preserve cases without measurements with their status and reason.
- Replay: include every case/Runtime with status, capture origin, observed execution window, ideal time, absolute gap, ratio and overhead share. Unsupported cases retain their reasons and have no model estimate; show neither zero nor a fabricated ratio. Keep original failure states visible when using supplementary captures.
- Diagnostics and bottlenecks: provide each case's objective, first and second limiting factors with available path/resource evidence, evidence-supported interpretation or explicit assessment limit, and needed follow-up. Complete coverage does not require inventing a bottleneck. Explain why an unavailable metric cannot support attribution.

Family-specific tables can use their applicable cases when explicitly labeled, but the chapter still needs the exhaustive inventory. When a family-specific metric appears in a full-matrix table, mark other families not_applicable and explain why. This differs from unmeasured, unsupported, failed and not_run. For example, C3 release qualification is not applicable to a serial C1 graph; an unqualified C3 capture still has a measured qualification count but no valid target-release estimate.

Present C3 qualification/release in a C3-only detail table, with an explanation of last predecessor F, first/last consumer R, first S and the registration-before-producer-completion condition. C1 measures serial handoffs, C2 measures sustained supply/throughput, and unverified W graphs do not establish C3 release units. Keep those families in the complete diagnostic inventory without repeating rows of inapplicable C3 metrics.

Before delivery, verify each exhaustive table against the expected ID/Runtime set, including duplicates and missing pairs, and reconcile numeric cells with their source samples. Report collection and analysis gaps separately: a case may have valid device timing while its replay remains unsupported. A full measurement run with analysis gaps must not be described as fully supported analysis.

Before delivery, check that counts and times come from the corresponding inputs, denominators are consistent, unsupported/unmeasured values are not treated as zero, and model gaps are not described as measured gains. Every bottleneck conclusion must trace to actual evidence; explicitly identify conclusions that cannot yet be established. Report workflow wall time separately as workflow cost.
