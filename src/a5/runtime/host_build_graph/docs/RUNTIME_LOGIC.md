# A5 `host_build_graph` AICore scheduling design

## Execution model

```text
host: build, validate, relocate, and upload the complete graph
  → host: order AIC/AIV task streams by remaining critical-path length
          and build the execution sidecar
  → AICPU: handshake workers, discover topology, publish configuration,
           then open each worker's DMB register gate
  → AICore: seed by type rank, claim tickets, route private pending tasks,
            execute kernels, enqueue completions, and resolve completion batches
  → AICPU: centrally wait for executors and resolutions, broadcast DMB EXIT,
           wait for every acknowledgement, and tear down
  → host: validate completions/counters and copy output tensors back
```

The device has no orchestration thread. AICPU owns launch and teardown only;
it does not schedule tasks in the steady-state path.

## Graph image and sidecar

The graph is a read-only view of `PTO2TaskDescriptor` and `PTO2TaskPayload`.
Fanin entries are producer task IDs and must be strictly smaller than the
consumer task ID.

The 128-byte-aligned v1 sidecar contains:

- one padded task control per task with state, wake-list, and intrusive
  completion link;
- one cache-line-isolated completion inbox per possible worker;
- dependency-priority-ordered AIC and AIV task-ticket streams, each with an
  isolated cursor; every ticket carries the host-validated task ID, kernel ID,
  subtask slot, and a zero/nonzero-fanin flag;
- split configuration, lifecycle, and first-error run-control cache lines;
- two private dispatch payloads (one per pending slot), one claim-binding cache
  line per task, and one debug/statistics context per worker; and
- per-task scheduler phase cells and per-core AICPU lifecycle cells written
  only during level-1 capture.

Inline-completed tasks start as `DONE` and do not enter either typed stream.

The host computes `bottom_level(task)` as the longest remaining edge distance
to any sink. Each typed stream is ordered by
`(bottom_level descending, task_id ascending)`. Because every graph edge has a
strictly decreasing bottom level, every producer has higher stream priority
than its consumer even when the resulting task IDs are non-monotonic.

## Ticket and pending scheduling

For each core type, AICPU activates
`min(available cores, executable tasks)` workers. Rank `r` seeds stream entry
`r`; the shared cursor starts after the seeded prefix. Later ownership uses one
`atomicAdd(next_index, 1)` per ticket attempt.

Each active worker maintains two owner-only pending slots. Claim initialization
copies the host-validated ticket metadata into a slot without re-reading or
revalidating graph descriptors. Before routing, the owner publishes a binding
from the task to its worker, pending slot, payload address, callable, kernel ID,
and subtask slot. Routing scans from the remembered fanin index
and links a blocked consumer into the first incomplete producer's wake list.
The consumer stays in its owner's pending slot; there is no ReadyQ and no
execution ownership migration.

Ready polling observes only the task-state cache line. The second control
cache line contains diagnostic dependency progress and is refreshed when a
task first waits, on the periodic scheduler diagnostic sample, and when the
ticket cursor is exhausted.

Routing does not publish `READY` when the final dependency is resolved. For a
claim-ready task, the owner materializes its slot's payload and then publishes
`READY`. For a blocked task, the AIV worker resolving the completion observes
the binding, materializes that owner's payload, publishes its exact eight cache
lines, and only then publishes `READY`. The owner invalidates those same eight
lines after observing remotely published readiness and executes immediately;
the old execution-time descriptor, function-table, and whole-data-cache
invalidations are not needed. A payload slot is cleaned before its next claim
so a stale owner cache line cannot overwrite a remote materialization.

Each worker atomically snapshots the immutable `aiv_active_worker_count` once
after startup and reuses it for enqueue routing and inbox service. After a
kernel returns, its unique owner stores `DONE` and pushes the task onto
`task_id % aiv_active_worker_count` with one atomic exchange of the inbox head.
The exchange returns the previous head, which the owner stores in the per-task
`completion_next` link and publishes. An AIV resolver may detach the new head
before that link is visible; in that case it waits on the initialized
`UNPUBLISHED` sentinel and observes the link cache line until publication. This
keeps retry and contention handling out of the producer hot path. LIFO order is
valid because completion events have no FIFO semantic; dependency order is
enforced by task state and wake-list routing. Kernel code owns publication of
cacheable GM output, matching the `tensormap_and_ringbuffer` execution contract.
Tensor data moved through the Tensor/MTE path needs no scheduler-wide D-cache
operation. A kernel that uses scalar LSU loads/stores for cross-core GM
communication must invalidate the producer cache lines before reading and clean
them after writing, with a `dsb` after each batch; writers must not share a
cache line.

Before idle backoff, an AIV worker detaches its whole local inbox or rotates
across one AIV victim inbox. AIC workers never service completion inboxes. The
AIV resolver closes each completed producer's wake list and reroutes all
waiters. Registration and close are race-safe: a registration CAS that wins
before close is in the detached wake chain; a CAS that observes `CLOSED`
rescans and sees the producer as `DONE`. Every fourth executed kernel also
provides a low-priority resolver service point so sustained ticket traffic
cannot starve completions.

For a graph with no AIV tasks, one AIV worker is activated as a resolver-only
worker and skips seed/ticket execution. Cursor exhaustion plus empty pending
slots publishes `executor_drained_worker_count`, but an AIV worker continues
resolving. One AICPU supervisor polls the isolated lifecycle cache line until
every executor has drained and the batch-updated resolved count equals the
executable task count. AICore workers no longer duplicate those global polls.
There is no peer-core wake primitive, so an idle resolver still performs
bounded GM inbox checks followed by the existing local exponential backoff.

Startup uses two AICPU barriers. All parallel handshake slices finish before
the leader derives topology, active prefixes, inbox ownership, stream cursors,
and PMU state. After those cache lines are published, every AICPU thread opens
its disjoint DMB register slice; the second barrier prevents init from returning
before all cores are released. An initialization failure uses the same slices
to send DMB `EXIT` and wait for acknowledgement.

After central completion, every AICPU thread first broadcasts `EXIT` to its
slice. A barrier guarantees that all signals are issued before any thread waits
for acknowledgements. Each AICore observes only its local DMB register in the
scheduler loop, publishes final worker statistics, acknowledges exit, and
returns. DFX and PMU finalization happen only after the corresponding slice has
acknowledged.

## Validation and diagnostics

Host validation checks:

- every graph edge has strictly decreasing bottom-level priority;
- both typed streams are complete, unique, core-type-correct, and priority ordered;
- every task is `DONE` and every wake list is `CLOSED`;
- every active completion inbox is empty and enqueue/resolve totals match the
  executable task count;
- worker and run-control execution totals match both typed streams;
- inline plus executable counts match the graph task count;
- the active and executor-drained worker counts agree, and the resolved count
  matches the executable task count;
- each typed cursor reached or passed its stream length.

The first scheduler error records task, worker, graph-address, and window
metadata before publishing the nonzero error code. Worker snapshots are
diagnostic breadcrumbs rather than a live mirror: they are published when a
blocked slot changes producer, the typed cursor is exhausted, and the worker
drains. Seed, successful ticket claim, and task completion do not publish a
snapshot on the steady-state path.

Level-1 chip-swimlane capture exports `AICoreEntryToHandshake`,
`HandshakeToRegisterRelease`, `RegisterReleaseToDescriptorReady`,
`DescriptorReadyToSeedClaim`, `SeedClaim`,
`TicketClaim`, `PendingWait`, `Payload`, `Kernel`, `CompletionEnqueue`,
`PostCompletion`, `CompletionService`, `TraceCommit`, scheduler-loop detail,
`ReadyScan`, `ReadyToPayload`, `CompletionBatchClaim`, `WakeResolve`,
`ReadyPublish`, `ExecutorDrainPublish`, `WaitForExit`, `FinalStatsPublish`,
`ExitAckPublish`, and `Drain` AICore phases. Separate AICPU lifecycle lanes
export global `WaitExecutors`, `WaitResolved`, and `CompletionDecision`, plus
per-core `RegisterRelease` and `ExitSignalToAck`. These phases partition the same-worker
inter-task scheduling gap from `CompletionEnqueue` end to `Payload` start.
The termination phases expose executor-drain publication, the AICPU's central
executor and completion-resolution waits, DMB-exit latency, poll counts,
atomic-poll instruction-window cycles, and local backoff cycles.
Result-dependent A5 pipeline retirement can appear in the wait phase's residual
time rather than the bracketed atomic-poll cycles. Final statistics include own/steal
batches, link publication waits, enqueue-to-resolve lag, and READY-to-kernel
lag.
The host STRACE tree also exposes `simpler_run.bind.ticket_stream_plan` for
the stream-planning cost.
See [profiling_levels.md](profiling_levels.md).
