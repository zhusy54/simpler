# A5 `host_build_graph` AICore scheduling design

## Execution model

```text
host: build, validate, relocate, and upload the complete graph
  → host: build sorted AIC/AIV task streams and the execution sidecar
  → AICPU: handshake workers, discover topology, publish active prefixes and RUN
  → AICore: seed by type rank, claim tickets, poll private pending fanins,
            execute kernels, and publish per-task completion flags
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

- one padded monotonic completion cell per task;
- sorted AIC and AIV task-ID streams, each with an isolated ticket cursor;
- split configuration, lifecycle, and first-error run-control cache lines;
- one private dispatch payload and one debug/statistics context per worker; and
- per-task scheduler phase cells written only during level-1 capture.

Inline-completed tasks start as `DONE` and do not enter either typed stream.

## Ticket and pending scheduling

For each core type, AICPU activates
`min(available cores, executable tasks)` workers. Rank `r` seeds stream entry
`r`; the shared cursor starts after the seeded prefix. Later ownership uses one
`atomicAdd(next_index, 1)` per ticket attempt.

Each active worker maintains two owner-only pending slots. It scans them in
round-robin order, remembers the completed fanin prefix, and polls at most the
first unresolved producer per slot. Ready work is executed before another
ticket is claimed. When all pending work is blocked, a compiler-preserved local
exponential backoff reduces repeated GM loads without adding shared traffic.

After a kernel publishes its output, the worker atomically changes that task's
completion cell from `NOT_DONE` to `DONE`. There is no ReadyQ, CompletionQ,
wake list, or resolver role. Cross-type dependencies use the same global
completion namespace.

A worker publishes its local statistics and increments `drained_worker_count`
once its typed cursor is exhausted and both pending slots are empty. AICPU then
publishes EXIT after every active worker has drained.

## Validation and diagnostics

Host validation checks:

- every task completion cell is `DONE`;
- worker and run-control execution totals match both typed streams;
- inline plus executable counts match the graph task count;
- active, drained, and finished worker counts agree; and
- each typed cursor reached or passed its stream length.

The first scheduler error records task, worker, graph-address, and window
metadata before publishing the nonzero error code. Worker snapshots expose both
pending slots and their current producer without adding writes to the polling
loop.

Level-1 chip-swimlane capture exports `SeedClaim`, `TicketClaim`,
`PendingWait`, `Payload`, `Kernel`, `CompletionPublish`, and `Drain` phases.
See [profiling_levels.md](profiling_levels.md).
