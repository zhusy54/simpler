# A5 `host_build_graph` AICore scheduling design

## Execution model

```text
host: build, validate, relocate, and upload the complete graph
  → host: order AIC/AIV task streams by remaining critical-path length
          and build the execution sidecar
  → AICPU: handshake workers, discover topology, publish active prefixes and RUN
  → AICore: seed by type rank, claim tickets, route private pending tasks,
            execute kernels, enqueue completions, and resolve completion batches
  → AICPU: wait for active workers to drain, publish EXIT, and tear down
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
- dependency-priority-ordered AIC and AIV task-ID streams, each with an
  isolated ticket cursor;
- split configuration, lifecycle, and first-error run-control cache lines;
- one private dispatch payload and one debug/statistics context per worker; and
- per-task scheduler phase cells written only during level-1 capture.

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
classifies the task once and caches its kernel ID and subtask slot. Routing
scans from the remembered fanin index and links a blocked consumer into the
first incomplete producer's wake list. The consumer stays in its owner's
pending slot; there is no ReadyQ and no execution ownership migration.

After a kernel publishes its output, its owner changes `READY` to `DONE` and
pushes the task onto `task_id % active_worker_count` using one `atomicExch`.
The per-task `completion_next` link is published after the exchange, so a
resolver treats `UNPUBLISHED` as a short protocol window rather than the end of
the list. LIFO order is valid because completion events have no FIFO semantic;
dependency order is enforced by task state and wake-list routing.

Before idle backoff, a worker detaches its whole local inbox or rotates across
one victim inbox. It closes each completed producer's wake list and reroutes
all waiters. Registration and close are race-safe: a registration CAS that wins
before close is in the detached wake chain; a CAS that observes `CLOSED`
rescans and sees the producer as `DONE`. Every fourth executed kernel also
provides a low-priority resolver service point so sustained ticket traffic
cannot starve completions.

Cursor exhaustion plus empty pending slots marks `EXECUTOR_DRAINED`, but that
worker continues resolving. Final `drained_worker_count` is published only
after every executor drained and the batch-updated resolved count equals the
executable task count. There is no peer-core wake primitive, so an idle worker
still performs bounded GM inbox checks followed by the existing local
exponential backoff.

## Validation and diagnostics

Host validation checks:

- every graph edge has strictly decreasing bottom-level priority;
- both typed streams are complete, unique, core-type-correct, and priority ordered;
- every task is `DONE` and every wake list is `CLOSED`;
- every active completion inbox is empty and enqueue/resolve totals match the
  executable task count;
- worker and run-control execution totals match both typed streams;
- inline plus executable counts match the graph task count;
- active, executor-drained, resolver-drained, and finished worker counts agree;
- each typed cursor reached or passed its stream length.

The first scheduler error records task, worker, graph-address, and window
metadata before publishing the nonzero error code. Worker snapshots are
diagnostic breadcrumbs rather than a live mirror: they are published when a
blocked slot changes producer, the typed cursor is exhausted, and the worker
drains. Seed, successful ticket claim, and task completion do not publish a
snapshot on the steady-state path.

Level-1 chip-swimlane capture exports `SeedClaim`, `TicketClaim`,
`PendingWait`, `Payload`, `Kernel`, `CompletionEnqueue`,
`CompletionBatchClaim`, `WakeResolve`, `ReadyPublish`, and `Drain` phases.
Drain-time counters include own/steal batches, link publication waits,
enqueue-to-resolve lag, and READY-to-kernel lag.
The host STRACE tree also exposes `simpler_run.bind.ticket_stream_plan` for
the stream-planning cost.
See [profiling_levels.md](profiling_levels.md).
