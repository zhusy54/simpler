# A5 HBG AICore scheduler profiling

A5 `host_build_graph` supports chip-swimlane levels 0 and 1 only:

| `enable_chip_swimlane` | Behavior |
| ---------------------- | -------- |
| `0` | Disabled |
| `1` | Kernel records plus Ready/Completion scheduler phases |
| `2`–`4` | Rejected before device launch |

Use an explicit level; the pytest flag without a value selects the global
default level 4.

```bash
pytest <case> --platform a5sim --enable-chip-swimlane 1
```

## Output

A successful level-1 run writes
`<output_prefix>/chip_swimlane_records.json` and appends
`aicore_scheduler_phases` plus `aicpu_lifecycle_phases`.

The AICore phases are:

- startup: `AICoreEntryToHandshake`, `HandshakeToRegisterRelease`,
  `RegisterReleaseToDescriptorReady`, `DescriptorReadyToReadyClaim`,
  `ExecutionStartToFirstReady`, `FirstReadyToReadyClaim`,
  `BootstrapGraphScan`, `BootstrapBarrier`, and the
  `Bootstrap{AIC,AIV}{ReadyClaim,SlotFill,FreeAdvertise,Other}` breakdown;
- scheduling: `ReadyPop`, `ReadySteal`, `ReadyScan`, `ReadyPublish`,
  `SlotRefill`, and the
  `InterTaskCompletion{Scan,Consume,Resolve,ReadyPublish,Refill,Finalize}`,
  `InterTaskGangService`,
  `InterTaskDispatch{AIC,AIV}{Probe,Claim,Prepare,Materialize,Publish}`, and
  `InterTask{ReadyPoll,Backoff,Other}`
  breakdown;
- execution: `Payload`, `Kernel`, `CompletionEnqueue`, `PostCompletion`;
- resolution: `CompletionBatchPrepare`, `CompletionBatchClaim`, `WakeResolve`;
- teardown: `TraceCommit`, `WaitForExit`, `FinalStatsPublish`,
  `ExitAckPublish`, `Drain`.

`ReadyPop` and `ReadySteal` execute on the resolver that claimed the unbound
task; `Kernel` executes on the target slot's worker. Only AIV resolvers emit
completion-resolution phases. Payload construction remains part of the
resolver refill path, while `Payload` on the executor measures observing the
published dispatch payload before kernel entry.

All `DescriptorReadyToReadyClaim` spans end at the earliest first-task claim,
which is the global execution start. `ExecutionStartToFirstReady` measures
dependency/bootstrap delay until that worker's first task enters Ready state;
`FirstReadyToReadyClaim` measures the subsequent Ready-inbox and placement
delay. Resolver bootstrap phases independently show graph scan, the global
barrier, and target-slot initialization. Per-target component durations are
accumulated on device and rendered contiguously, so their widths are exact but
their display order does not represent individual slot-operation order.

Long inter-task gaps, including the interval before each worker's first task,
are split by accumulated time in completion scan/consume/resolve/refill,
Gang service, AIC/AIV probe/claim/prepare/materialize/publish, unsuccessful
slot polling, and idle backoff. The residual `InterTaskOther` contains loop
control and instrumentation overhead.
Old traces without these counters retain the aggregate `InterTaskSchedule`.

`CompletionEnqueue` records include `completion_id` and `inbox_index`. The ID
is based on per-worker execution order rather than graph task ID, so Ready
partitioning cannot force the corresponding completions back into the same
resolver shard.

AICPU lifecycle phases are global `WaitBootstrap`, `WaitResolved`, and
`CompletionDecision`, plus per-core `RegisterRelease` and `ExitSignalToAck`.
The global wait records include `poll_count`, `poll_cycles`, and
`error_poll_count`.

Worker counters include bootstrap tasks; Ready enqueue/batch/pop/steal, CAS
retry, and link-publication waits; free-slot claims/stale probes; completion
enqueue/batch/resolve/steal and link waits; wake registration/migration; and
READY-to-kernel/completion-resolution lag. These are diagnostic measurements,
not performance merge gates.

Each task has one 256-byte trace cell indexed by task ID. Trace writes occur
only at level 1. The common collector publishes its JSON first; HBG inserts the
scheduler arrays through a temporary file and atomic rename.
