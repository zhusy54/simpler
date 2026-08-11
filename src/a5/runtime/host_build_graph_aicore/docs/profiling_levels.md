# HBG-AICore profiling

`host_build_graph_aicore` supports chip-swimlane levels 0 and 1 only:

| `enable_chip_swimlane` | Behavior |
| ---------------------- | -------- |
| `0` | Disabled |
| `1` | AICore task receive/start/end timestamps |
| `2`–`4` | Rejected before device launch |

Use an explicit level because the pytest flag without a value selects the
global default level 4:

```bash
pytest <case> --platform a5sim --enable-chip-swimlane 1
```

## Ownership

AICore owns the steady-state path:

1. pop a ready task;
2. materialize its payload;
3. reserve a profiling record;
4. execute the kernel;
5. commit the record and make its timestamp writes visible;
6. publish the completion queue entry; and
7. resolve/wake dependent tasks.

Committing before completion publication guarantees that a successor cannot be
released before its predecessor's profiling record is complete.

AICPU performs lifecycle work only: initialize profiling buffers, handshake
workers, wait for the AICore run to finish, publish final counters, flush, and
deinitialize register windows. It does not dispatch tasks, poll task completion,
or run dependency scheduling during normal execution.

## Output

Successful level-1 capture writes `<output_prefix>/chip_swimlane_records.json`
with the existing schema:

```json
{
  "chip_swimlane_level": 1,
  "metadata": {},
  "aicore_tasks": [],
  "aicpu_tasks": []
}
```

`aicore_tasks` contains one complete record per captured task.
`aicpu_tasks` is empty. Scheduler and orchestrator phase arrays are omitted.

Each core has `PLATFORM_AICORE_BUFFER_SIZE` (1024) record slots. Task execution
continues correctly after that capacity is reached, but every excess attempt is
counted as dropped. Host reconciliation then fails the profiling result, removes
any temporary/final JSON, and returns `SIMPLER_PROFILING_VALIDATION_ERROR`.
That error is diagnostic-only: output tensors are still copied back before the
run reports failure.

## Publication guarantees

The host exports to `chip_swimlane_records.json.tmp`, checks stream accounting
and record timestamps, flushes the file, then atomically renames it to the final
path. Strict failure removes both paths, so a stale or partial final JSON cannot
be mistaken for a successful capture.
