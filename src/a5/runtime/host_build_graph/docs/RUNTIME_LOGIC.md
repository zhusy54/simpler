# A5 `host_build_graph` Ready scheduler

## Execution flow

```text
Host: validate/upload graph and task-indexed metadata
  -> AICPU: discover topology, select active workers, release AICores
  -> AIV resolvers: full static bootstrap and wake-list registration
  -> Ready inbox -> two-slot dispatch -> AIC/AIV kernel
  -> Completion inbox -> WakeResolve -> Ready inbox/refill
  -> AICPU: wait for bootstrap + all resolutions, signal EXIT
  -> Host: validate terminal scheduler state
```

AICPU owns launch and teardown only. AIV AICores are resolvers; AIC and AIV
AICores execute tasks published to their two private dispatch slots.

## Bootstrap and dependencies

The host publishes immutable metadata indexed by task ID: kernel ID, subtask
slot, executable flag, and zero/nonzero-fanin flag. It does not construct or
order task streams.

With `R` active AIV resolvers, resolver `r` scans task IDs
`r, r + R, r + 2R, ...`. It uses the existing wake-list registration protocol:
a blocked consumer registers on its first incomplete producer; a dependency-free
task is appended to the resolver's local typed Ready batch. Execution starts
only after all `R` scans finish. This preserves a simple full-bootstrap barrier;
overlapping bootstrap with execution is a later optimization.

## Ready inboxes and stealing

Each resolver owns one AIC and one AIV Overflow Ready Inbox. Producers only
push to their own inbox, normally as a batch. Consumers pop one task at a time
using CAS on the inbox head:

1. Try the resolver-local typed inbox.
2. If empty, inspect the typed Ready bitmask and steal from a marked victim.
3. Advance a resolver-local round-robin cursor seeded from resolver ID.

The Ready bitmask has one bit per resolver inbox and core type. Push sets the
bit when it may make an inbox nonempty. Last-pop clears the bit, rechecks the
head, and sets it again if a concurrent push won the race. Stale set bits are
allowed; a nonempty inbox must not remain unmarked.

`AicoreTaskControlV1::next_waiter` links the dependency wake list while a task
is blocked and its Ready inbox after routing. `inbox_next` is reserved for the
Completion inbox. A successful Ready pop leaves its link immutable, so losing
consumers can safely finish a stale-head CAS attempt.

Both links share a cache line with task timing fields. Writers therefore use
one DCCI-coherent read/modify/publish path for the whole line; raw-GM link
stores must not be mixed with later whole-line writeback.

## Dispatch slots

Every active executor has two slots (`K=2`) so payload and callable preparation
can overlap the currently running kernel. A slot cycles through:

```text
FREE(g) -> FILLING(g) -> READY(g+1) -> FREE(g+1)
```

A resolver first refills the slot released by the completion it is processing.
If no Ready task exists, it advertises that exact typed slot in the free-slot
bitmask. A resolver with surplus Ready work may clear a free-slot bit, validate
the target worker/type/generation, and CAS `FREE` to `FILLING`. Thus stealing
moves only unbound Ready tasks; it never takes a task already prepared in
another executor's slot.

Payload construction reuses the previous materialization path. The resolver
copies immutable metadata, resolves the callable, writes the per-task binding,
materializes and publishes the payload, then publishes the slot as `READY`.

## Completion and WakeResolve

After a kernel returns, the executor publishes task state `DONE` and pushes the
task to `task_id % resolver_count` in the Completion Inbox. The resolver:

1. confirms the binding and releases the exact slot to private `FREE`;
2. closes the producer wake list;
3. reroutes waiters and batches newly ready tasks into its local typed inboxes;
4. tries local pop, then stealing, to refill the released slot;
5. advertises the slot if no task is available.

Completion-service fairness and throughput tuning are intentionally deferred.

## Termination and validation

The AICPU supervisor exits only when full bootstrap is published and
`resolved_task_count` equals the executable-task count. An empty or all-inline
graph publishes bootstrap completion directly from AICPU.

After EXIT, host validation requires:

- all tasks `DONE` and all wake lists `CLOSED`;
- Completion and Ready inboxes empty, and the Ready bitmasks clear;
- all active slots `FREE`, with the free-slot bitmask matching them exactly;
- bootstrap, Ready enqueue/pop, execution, completion enqueue/resolve, and
  wake registration/migration counts to agree.

Level-1 profiling records `ReadyPop`/`ReadySteal`, payload, kernel, completion,
WakeResolve, refill, bootstrap wait, resolution wait, and exit phases. Normal
execution does not write per-task trace cells.
