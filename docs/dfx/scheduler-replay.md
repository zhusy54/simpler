# Host-side zero-overhead scheduler replay

For interpreting replay alongside measured benchmark results and writing a bottleneck report, use the [runtime benchmark assessment methodology](runtime-benchmark-analysis.md).

Replay a captured DAG on the host using the original runtime's Ready queues,
dispatch selection, execution slots and completion processing. The C++ backend
replaces AICore execution with events whose durations are the captured **AICore
S-to-E ticks**. Dependency processing, submission, dispatch, ACK/FIN, gates and
completion observation take zero virtual time; host CPU wall time is irrelevant.
Scheduler actions remain instantaneous when an AIV scheduler also executes a
kernel. Computation still occupies its execution core for its measured duration.

```bash
# One capture and its independently verified semantic DAG.
python -m simpler_setup.tools.scheduler_replay outputs/events/chip_swimlane_records.json \
  --deps outputs/deps/deps.json --output outputs/replay

# Fixed benchmark manifests: validate the C graph against the matrix oracle.
python -m simpler_setup.tools.scheduler_replay outputs/deps_manifest.json \
  outputs/events_manifest.json outputs/timing_manifest.json --case C1-01 \
  --case C2-03 --case C3-10 --output outputs/replay_matrix

# Existing benchmark report plus replay artifacts.
python -m tests.st.scheduler_benchmark.report outputs/deps_manifest.json \
  outputs/events_manifest.json --output outputs/report.json --replay-output outputs/replay_matrix
```

The offline backend requires CMake 3.20+ and a C++17 host compiler. It builds
incrementally under `build/scheduler_replay`; `--build-directory` selects a
writable alternative. No NPU, CANN installation or device lock is needed.
The source and CMake files are included in wheel assets. Manifest mode needs
the source checkout's `tests.st.scheduler_benchmark` definitions.

Supported backend paths are A5 HBG AICore, A5 HBG AICPU, A2/A3 HBG AICPU,
and the AICPU TMR path on either architecture. The initial input contract
supports single-block AIC/AIV tasks, unambiguous partial MIX shapes, and
dependency-only controls on the native blocked 1-AIC/2-AIV topology. HBG
reconstructs its whole-graph image; TMR calls the original submit, scope,
dependency-reduction and reclamation implementation. WAIT and RETAIN edge
flags remain distinct. TMR supplies tasks in submission order until finite
native resources require execution to advance. Its scope reconstruction covers
the benchmark's outer auto scope and manual units separated by outer dummy gates.
The host submission actor pauses inside the original allocation wait and resumes
after scheduler progress; CPU thread handoffs consume no virtual time. Resource
pool sizes remain effective, including native reclamation granularity. A stalled
native configuration produces an error rather than an unconstrained schedule.
The dependency-only TMR adapter does not allocate output tensors; workloads
with captured OUTPUT allocations are rejected until an allocation event stream
is available.

Dummy dependency gates retain their task IDs, edges and scopes but take exactly
zero virtual time and consume no execution-core capacity. The matrix adapter
identifies them using the validated fixed-C graph; direct captures with an
inline HBG AICore-produced AIV fragment must explicitly mark the DAG node
`task_kind: "dummy"`. Only an empty three-kernel shape and a single AIV fragment
with `func_id=-1` are accepted for that inline form. Graph/async nodes and
unidentified extra fragments remain unsupported. Original observations retain
the captured control duration; only the replay model sets it to zero.
`replay_summary.json` records the dummy count, zero-duration policy and captured
inline task IDs. Perfetto shows replay dummies as thread-scoped instant events
on their Worker lanes, retaining the incoming/outgoing dependency anchors;
no artificial 1 ns is added to the model or display.

Provide explicit configuration when the capture lacks effective settings:

```json
{
  "arch": "a5",
  "scheduler_location": "aicpu",
  "scheduler_threads": 4,
  "ring_task_window": 16384,
  "ring_dep_pool": 16384,
  "scheduler_order": "ascending"
}
```

Pass this file with `--config`. Captured core ownership takes precedence over
an incompatible thread count. Without ownership or an explicit setting, the
report labels its runtime-default thread configuration as an assumption.
`scheduler_order: "descending"` provides a second deterministic same-time
interleaving. Each thread's internal queue/dispatch ordering is unchanged.
`--run-epoch` selects one instance from a multi-run capture.

Outputs include `replay_input.json`, `replay_tasks.json`,
`replay_decisions.json`, `replay_summary.json/.md`, `observed.json`,
`replay_swimlane.json` and `comparison.json`. Open the latter in Perfetto for
the existing Worker/Scheduler views, stages and dependency arrows in separate
Observed and Zero-overhead groups. Preserve pending-slot dispatch timestamps:
zero dispatch cost does not eliminate waiting for a busy execution core.
The comparison currently aligns observed first AICore S with model t=0; it
does not infer a missing device-entry or graph-release timestamp.

`T_ideal_us` is the selected runtime policy's completion time, not a global
optimum. The report includes compute/resource lower bounds, per-core occupancy,
source/binary hashes, assumptions and legality checks. Captured kernel times
are held fixed despite changes in overlap or placement. The gap from an observed
execution window or diagnostic device wall is a model comparison, not a measured
speedup or an attribution of the entire gap to Scheduler software. This tool
does not reproduce measured scheduler delays or publish per-stage causal savings.

Manifest mode retains all 41 configuration IDs and rejects incompatible
source/environment fingerprints. W/Graph instances remain unsupported until
their DAG/execution association is verified. Missing/ambiguous fragment
identities, multi-block/sync-start, early-dispatch attributes, custom affinity
and unsupported control/resource descriptions produce explicit unsupported
results. They are never replaced with zero durations. Raw inputs are read-only;
replay artifacts are written only under the requested output directory.
