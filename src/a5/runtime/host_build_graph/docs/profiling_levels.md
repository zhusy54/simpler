# A5 HBG AICore scheduler profiling

A5 `host_build_graph` supports chip-swimlane levels 0 and 1 only:

| `enable_chip_swimlane` | Behavior |
| ---------------------- | -------- |
| `0` | Disabled |
| `1` | AICore kernel records plus ticket/pending/completion scheduler phases |
| `2`–`4` | Rejected before device launch |

Use an explicit level because the pytest flag without a value selects the
global default level 4:

```bash
pytest <case> --platform a5sim --enable-chip-swimlane 1
```

## Output

A successful level-1 run writes
`<output_prefix>/chip_swimlane_records.json`. In addition to the common
metadata and `aicore_tasks`, HBG appends:

```json
{
  "aicore_scheduler_phases": [
    {
      "worker_id": 0,
      "core_type": 0,
      "task_id": 7,
      "phase": "TicketClaim",
      "start_cycles": 100,
      "end_cycles": 110
    }
  ],
  "aicpu_lifecycle_phases": [
    {
      "aicpu_thread_id": 0,
      "worker_id": 0,
      "core_type": 0,
      "phase": "ExitSignalToAck",
      "start_cycles": 200,
      "end_cycles": 215
    }
  ]
}
```

Supported AICore phase names are `AICoreEntryToHandshake`,
`HandshakeToRegisterRelease`, `RegisterReleaseToDescriptorReady`,
`DescriptorReadyToSeedClaim`, `SeedClaim`,
`TicketClaim`, `PendingWait`, `Payload`, `Kernel`, `CompletionEnqueue`, `PostCompletion`,
`CompletionService`, `TraceCommit`, `SchedulerToClaim`, `TaskInitialize`, `TaskRoute`,
`SchedulerToReadyScan`, `InterTaskSchedule`, `ReadyScan`,
`CompletionBatchClaim`, `WakeResolve`, `ReadyPublish`, `ExecutorDrainPublish`,
`CompletionBatchPrepare`, `SlotRefill`,
`WaitForExit`, `FinalStatsPublish`, `ExitAckPublish`, and `Drain`. The swimlane converter renders
them on per-worker `Scheduler` lanes. Each `Kernel` phase also carries
the resolved function name and serves as the task/dependency-flow anchor; the
converter does not emit a duplicate `AIC_N` or `AIV_N` lane for those tasks.
Task classification is validated on the host and encoded in each typed-stream
ticket. Claim initialization copies that metadata into the pending slot;
callable lookup and argument materialization remain in `Payload`.
`aicpu_tasks` is empty because AICPU does not dispatch steady-state work.
The converter renders `aicpu_lifecycle_phases` on separate per-thread AICPU
lanes. Supported names are global `WaitExecutors`, `WaitResolved`, and
`CompletionDecision`, plus per-core `RegisterRelease` and `ExitSignalToAck`.

For consecutive tasks on one worker, the interval from the previous task's
`CompletionEnqueue` end to the current task's `Payload` start is partitioned
into `PostCompletion`, optional `CompletionService`, `TraceCommit`, scheduler
work, `ReadyScan`, and `Payload`. `PostCompletion` covers the common
profiling commit, statistics, pending-slot clear, and fairness bookkeeping.
`CompletionService` is present every fourth completion on AIV workers and
covers completion inbox service. `TraceCommit` isolates publication of the
previous task's Level-1 sidecar trace.

When the next task is claimed inside this gap, scheduler work is split into
`SchedulerToClaim`, `TicketClaim`, `TaskInitialize`, `TaskRoute`, and
`SchedulerToReadyScan`.
`TaskInitialize` copies the host-validated ticket into a pending slot;
`TaskRoute` covers dependency registration and READY/BLOCKED classification.
Otherwise `InterTaskSchedule` covers the scheduler work between trace
publication and the final pending-slot scan.
`ReadyScan` covers that scan through observing READY, and `Payload` covers
local selection, payload cache observation, setup, and materialization. A task without a
`PendingWait` phase was ready on its first observed scan.

Only AIV workers emit `CompletionService`, `CompletionBatchClaim`, and
`WakeResolve`: completion inbox service and dependency resolution are not
performed by AIC workers.

Completion refill tracing follows a successfully claimed inbox head through
slot reuse. `CompletionBatchPrepare` covers successful inbox observation and
detach plus entry into and link observation within the detached completion
chain. `SlotRefill` covers binding/slot observation, prefetched-ticket
consumption, dependency routing, payload/READY publication, and replenishing
the next prefetched ticket. Refill phases use the newly assigned task ID, while
completion preparation uses the completed task ID.

The termination tail is split across AICore and AICPU lanes.
`ExecutorDrainPublish` covers the shared `executed_task_count` and
`executor_drained_worker_count` atomic increments. A drained AIV worker keeps
servicing inboxes during `WaitForExit`; other workers only poll their local DMB
register. The AICPU supervisor owns the global `WaitExecutors` and
`WaitResolved` polling, then records `CompletionDecision`. All AICPU threads
broadcast DMB `EXIT` before any waits for acknowledgement. `FinalStatsPublish`
and `ExitAckPublish` expose the AICore publication/ack tail, while per-core
`ExitSignalToAck` measures the AICPU-observed register round trip. The global
wait records carry raw `poll_count`, `poll_cycles`, and `error_poll_count`; the
converter exposes cycle fields as `poll_time_us` and `poll_avg_time_us`.
`poll_time_us` is the timestamp-bracketed cache-invalidate and load window, not
an end-to-end GM atomic latency. Poll timing uses timestamp reads at Level 1, so
compare phase envelopes, poll counts, and workload variants rather than
treating the instrumented tail as an unobserved Level-0 latency.

The common AICore profiling buffer retains one kernel anchor per worker in the
raw capture. The sidecar trace cell is indexed by task ID and therefore captures
every executed task without contending for a shared append cursor. When the
sidecar contains `Kernel` phases, the converter treats them as the authoritative
task timing source and discards the sparse common anchors. Tracing is
conditional on level 1; normal execution does not write trace cells.

The Level-1 sidecar keeps coarse scheduler boundaries to limit observer cost:
each task uses a 256-byte trace cell, payload and refill internals use one phase
each, and idle scheduler polling is reported as `InterTaskSchedule` without
per-poll timestamp sampling.

Worker statistics are published after local DMB `EXIT` observation. They include completion
enqueue, batch, resolve and steal counts; unpublished-link wait total/max;
enqueue-to-resolve lag total/max; READY-to-kernel lag total/max; and the
existing wake registration/migration counters. With profiling disabled, the
per-task timing fields are not written.

## Publication guarantees

The common collector validates counters and atomically publishes its JSON.
After device state is copied back, HBG inserts the scheduler phase array through
a temporary file and atomic rename. A missing, malformed, or unwritable trace
file fails profiling validation instead of leaving a partial result.
