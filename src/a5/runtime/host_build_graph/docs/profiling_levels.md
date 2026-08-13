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
  ]
}
```

Supported phase names are `SeedClaim`, `TicketClaim`, `PendingWait`,
`Payload`, `Kernel`, `CompletionEnqueue`, `CompletionBatchClaim`,
`WakeResolve`, `ReadyPublish`, and `Drain`. The swimlane converter
renders them on per-worker `Scheduler` lanes. Task classification is cached
during claim initialization and is not repeated in `Payload`; callable lookup
and argument materialization remain in `Payload`. `aicpu_tasks` is empty
because AICPU does not dispatch steady-state work.

The common AICore profiling buffer retains one kernel anchor per worker. The
sidecar trace cell is indexed by task ID and therefore captures every executed
task without contending for a shared append cursor. Tracing is conditional on
level 1; normal execution does not write trace cells.

Worker statistics are published only at resolver drain. They include completion
enqueue, batch, resolve and steal counts; unpublished-link wait total/max;
enqueue-to-resolve lag total/max; READY-to-kernel lag total/max; and the
existing wake registration/migration counters. With profiling disabled, the
per-task timing fields are not written.

## Publication guarantees

The common collector validates counters and atomically publishes its JSON.
After device state is copied back, HBG inserts the scheduler phase array through
a temporary file and atomic rename. A missing, malformed, or unwritable trace
file fails profiling validation instead of leaving a partial result.
