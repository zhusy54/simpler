# A5 `host_build_graph` Ready scheduler

## Execution flow

```text
Host: validate/upload graph and task-indexed metadata
  -> AICPU: discover topology, select active workers, release AICores
  -> AIV resolvers: full static bootstrap and wake-list registration
  -> Ready inbox or Gang cohort -> two-slot dispatch -> AIC/AIV kernel
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

With `R` active AIV resolvers, the graph is divided into `R` contiguous ranges.
This keeps immutable metadata and payload reads local and avoids aliasing a
periodic task pattern with the resolver count. A bootstrap-only route registers
each blocked consumer on its first executable producer with one wake-head
exchange. Inline-completed producers are skipped from immutable metadata.

Each resolver links dependency-free tasks into private typed FIFO Ready batches,
writes back all waiter links, and executes one cache barrier before directly
publishing its initially empty inbox heads. Resolver-local Ready-type flags are
stored contiguously beside the Ready directory. The last resolver aggregates
those flags into the directory and releases the bootstrap barrier. Execution
cannot start before that release, which makes the bootstrap-specific exchange
and batched publication safe; steady-state routing retains the full CAS-based
close-race protocol.

## Ready inboxes and stealing

Each resolver owns one AIC and one AIV Ready inbox. Each typed inbox is an
owner-banked FIFO: the shared out bank is visible to consumers through its
head, while a Resolver-owned GM scheduler state bank tracks both pending endpoints in
one packed 64-bit word. The single-word snapshot prevents the owner from
observing a head and tail from different GM updates.
The pending metadata has a dedicated cache line per core type and is never
read or written by thieves. Producers only append batches to their own inbox.
Consumers pop one task at a time using CAS on the shared head:

1. Try the resolver-local typed inbox.
2. If empty, inspect the typed Ready bitmask and steal from a marked victim.
3. Advance a resolver-local round-robin cursor seeded from resolver ID.

The published out-bank links are immutable. When it drains, only its Resolver
owner may publish the older pending bank; thieves never observe or promote
pending work. This preserves FIFO order within each typed Resolver inbox
without a contended shared tail or producer-side CAS. Local-first claiming and
sharded stealing still mean this is not a global FIFO across Resolvers or core
types.

The Ready bitmask has one bit per resolver inbox and core type. The owner sets
the bit when an empty inbox becomes nonempty and clears it only after both the
shared out bank and owner-only pending bank are empty. Last-pop does not modify the
bit. Stale set bits are allowed until the owner next services the inbox; a
published nonempty inbox must not remain unmarked.

`SchedulerTaskControl::next_waiter` links the dependency wake list while a task
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

## MIX, SPMD, and sync-start cohorts

A task enters the Gang scheduler when it has multiple active lanes or more than
one logical block. Resolver 0 admits one active cohort at a time, prioritizing
sync-start work, then MIX, then single-lane SPMD. It partitions logical blocks
across Resolver participants: a single-AIV task can use both AIV lanes per
cluster, while AIC and MIX tasks advance in cluster-sized strides.

Non-sync cohorts fill and publish their assigned slots as capacity becomes
available. MIX publication reserves every active lane in a physical cluster or
publishes none of them, so a partial cluster cannot observe the task.

A sync-start cohort follows a generation-tagged tree protocol:

```text
DRAINING -> STAGING -> RELEASING -> DISPATCHING
```

Every participant first drains the required local lanes, materializes all of
its dispatches as `GATED`, and publishes a subtree token. Resolver 0 releases
the cohort only after the root token proves that every participant staged its
share. The same tree aggregates completion and retirement, preventing a stale
generation from satisfying a later cohort.

Predicates remain on the ordinary Ready path and therefore require one active
lane and one logical block. Dependency-only and host-completed tasks do not
enter a Gang cohort.

## Completion and WakeResolve

After a kernel returns, the executor publishes its completed slot generation to
the Completion inbox using a per-executor completion ID:

```text
completion_id = local_completion_index * runtime_worker_count + worker_id
inbox_index = completion_id % resolver_count
```

The local completion index is the executor's existing successful-enqueue
counter. This keeps the ID unique without a contended global atomic and avoids
correlating Ready ownership with graph task IDs. For an ordinary task, the
resolver:

1. confirms the binding and releases the exact slot to private `FREE`;
2. marks the task `DONE` and closes its wake list;
3. reroutes waiters and appends newly ready batches to its local typed FIFO inboxes;
4. tries local pop, then stealing, to refill the released slot;
5. advertises the slot if no task is available.

For a Gang task, each completion advances its participant's local completion
count. The task becomes `DONE` and runs WakeResolve only after the root
completion token covers every participant; all participants must then observe
retirement before the cohort is reusable.

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
