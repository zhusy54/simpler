# A5 `host_build_graph` AICore scheduling design

## Execution model

```text
host: load orchestration SO and materialize the complete graph
  → host: relocate/upload graph image and AICore execution sidecar
  → AICPU init: initialize profiling and handshake AICore workers
  → AICore: classify roots, resolve dependencies, execute tasks, wake successors
  → AICPU teardown: wait, flush diagnostics, close register windows
  → host: validate sidecar and copy output tensors back
```

The device has no orchestration thread. AICPU has no steady-state task scheduler.
One AICPU supervisor publishes the resolver set and run phase, then waits for
the AICore completion counters; its peers wait for teardown.

## Graph image and sidecar

The host graph uses `PTO2TaskDescriptor` and `PTO2TaskPayload` as its read-only
device view. Fanins are producer IDs, not host pointers. Before upload, task and
payload references are relocated to their final device addresses.

Per-run mutable state is separate in the aligned AICore sidecar:

- one task control per task for completion and intrusive wake-list state;
- AIC/AIV ready queues and a completion queue;
- run-wide attach/classify/run/exit counters; and
- one worker context per AICore, including execution/profiling counters and a
  private dispatch-payload slot.

## Dependency resolution

Selected resolver cores partition the initial task scan. A task with no unmet
fanin enters the ready queue for its core type. Otherwise it registers in an
unmet predecessor's intrusive wake list.

After kernel execution, the worker commits profiling (when enabled) and pushes
the task ID to the completion queue. A resolver then:

1. marks the task complete;
2. atomically closes and detaches its wake list;
3. reclassifies each waiter; and
4. pushes newly ready tasks to the appropriate AIC/AIV queue.

Level-1 profiling records the full CompletionQ-pop-to-wake interval on the
resolver core and exports it as an AICore resolve phase.

Completion is monotonic. Publication barriers order kernel outputs and profiling
records before successor release.

## AICPU lifecycle boundary

AICPU is limited to:

- profiling-buffer initialization;
- parallel worker handshake and register-window setup;
- topology/resolver configuration;
- waiting for AICore completion/error/exit counters;
- final profiling-counter publication and buffer flush; and
- PMU/register teardown.

Legacy `SchedulerContext` dispatch, completion polling, and cold-path scheduler
sources are not part of this runtime target.

## Validation

Host validation checks task controls, ready/completion accounting, worker task
counts, resolver counts, and final run state. A device execution failure skips
output copy-back. A profiling-only validation failure is distinct: outputs are
copied back, profiling JSON is withheld, and the profiling error is returned
after cleanup.

Chip-swimlane behavior is documented in
[profiling_levels.md](profiling_levels.md). Scalar access during host graph
construction is documented in [SCALAR_DATA_ACCESS.md](SCALAR_DATA_ACCESS.md).
