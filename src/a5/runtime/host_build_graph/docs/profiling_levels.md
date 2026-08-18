# A5 HBG AICore scheduler profiling

A5 `host_build_graph` supports chip-swimlane levels 0 and 1 only:

| `enable_chip_swimlane` | Behavior |
| --- | --- |
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
  `RegisterReleaseToDescriptorReady`, `DescriptorReadyToReadyClaim`;
- scheduling: `ReadyPop`, `ReadySteal`, `InterTaskSchedule`, `ReadyScan`,
  `ReadyPublish`, `SlotRefill`;
- execution: `Payload`, `Kernel`, `CompletionEnqueue`, `PostCompletion`;
- resolution: `CompletionBatchPrepare`, `CompletionBatchClaim`, `WakeResolve`;
- teardown: `TraceCommit`, `WaitForExit`, `FinalStatsPublish`,
  `ExitAckPublish`, `Drain`.

`ReadyPop` and `ReadySteal` execute on the resolver that claimed the unbound
task; `Kernel` executes on the target slot's worker. Only AIV resolvers emit
completion-resolution phases. Payload construction remains part of the
resolver refill path, while `Payload` on the executor measures observing the
published dispatch payload before kernel entry.

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
