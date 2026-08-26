# Cluster-Aware Submission in `host_build_graph`

## Contract

One call to `rt_submit_task` creates one mixed graph task. The task may use any
nonempty combination of the physical cluster's `AIC`, `AIV0`, and `AIV1`
lanes; all active lanes share one argument payload, dependency set, logical
block count, and completion record.

```cpp
MixedKernels kernels;
kernels.aic_kernel_id = aic_func_id;
kernels.aiv0_kernel_id = aiv0_func_id;
kernels.aiv1_kernel_id = aiv1_func_id;

CoreTaskArgs args;
args.add_inout(output);
args.launch_spec.set_block_num(block_num);
TaskOutputTensors result = rt_submit_task(kernels, args);
```

`rt_submit_aic_task` and `rt_submit_aiv_task` are convenience wrappers over the
same mixed-task contract.

## Task Representation

The shared graph image separates stable task identity from scheduling state:

- `TaskDescriptor` contains the task ID, kernel IDs, and packed-buffer
  addresses.
- `TaskPayload` contains the argument counts, predicate, and a delta naming
  each of its tensor, scalar and position-independent fanin-id regions. The
  arguments live in the pool segments, not in the fixed two-cache-line payload.
- `ChipTaskSlotState` contains host-side classification metadata: the active
  mask, task attributes, logical block count, required-subtask total,
  orchestration-time task state, and descriptor/payload bindings. Device
  scheduling counters live in the scheduler state.

This image is built on the host and copied to the device verbatim. It contains
no fanout adjacency or dependency-pool pointers, and its descriptor/payload
bindings are deltas from the slot state's own address rather than pointers.

## Resource Shapes

`ActiveMask::to_shape()` maps a task to one of four shapes:

| Shape | Meaning |
| ----- | ------- |
| `AIC` | One AIC lane per logical block |
| `AIV` | One AIV lane per logical block |
| `MIX` | Two or three active lanes in one physical cluster |
| `DUMMY` | Dependency-only task with no AICore dispatch |

A single-AIV task is normalized to the AIV0 submit slot. MIX dispatch still
uses the full active mask so unused lanes do not block placement.

## Logical Blocks and Validation

`block_num` is the number of logical SPMD blocks, not the number of physical
clusters. Non-sync tasks may be wider than the device and dispatch in waves.

Submission validates before allocating or publishing a slot:

```text
block_num >= 1
block_num * popcount(active_mask) <= INT16_MAX
```

The product is stored as the scheduler state's 16-bit required-subtask total. Invalid
values latch `SIMPLER_ERROR_INVALID_ARGS`; non-sync work is not capped at the
device's physical cluster count.

`require_sync_start` adds a separate residency constraint because every block
must launch as one cohort:

- AIV-only: `block_num <= rt_available_aiv_count()`
- AIC or MIX: `block_num <= rt_available_cluster_count()`

## Dependency and Readiness Flow on the A5 HBG AICore Scheduler

1. Host orchestration allocates a task slot and builds its payload.
2. TensorMap and explicit dependencies append producer local IDs to
   `fanin_local_ids[]`.
3. Submit publishes the finished graph and task-indexed execution metadata.
4. AIV resolvers statically partition a full bootstrap scan, register blocked
   tasks on producer wake lists, and push dependency-free tasks to local typed
   Ready inboxes.
5. Resolvers pop locally first, then steal unbound tasks from another resolver's
   typed Ready inbox. A compact bitmask identifies inboxes worth probing.
6. Each AIC/AIV executor owns two dispatch slots. Resolvers refill a private
   free slot after completion or claim an advertised free slot by typed bitmask.
7. Kernel completion publishes the slot generation to a Completion inbox. An
   AIV resolver marks an ordinary task `DONE`, or advances a Gang participant;
   once the whole task completes it closes the wake list, routes newly ready
   consumers, and refills the released capacity.

## Dispatch and Completion

- A single-lane, single-block task uses the ordinary typed Ready inbox and is
  late-bound to one advertised executor slot.
- MIX or multi-block work uses the Gang scheduler. Resolver 0 admits a cohort
  and partitions its blocks across participating Resolver clusters; later
  blocks fill slots released by earlier completions.
- MIX placement publishes all active lanes of one physical cluster together.
  If any required lane is unavailable, that cluster publishes none of the
  task.
- `require_sync_start` drains the required lanes, stages every dispatch as
  `GATED`, and releases the cohort only after a generation-tagged Resolver tree
  reports all participants ready.
- Each completed Gang lane increments its participant's
  `local_completed_subtasks`. Resolver 0 resolves the graph task exactly once
  after the completion tree covers every participant and the total reaches
  `block_num * popcount(active_mask)`.

## Executor Model

The host loads and executes the orchestration shared object synchronously. The
device has no orchestration thread. One AICPU supervisor publishes the active
worker prefixes and waits for full bootstrap plus resolved-task completion;
other AICPU threads wait for teardown. Persistent AICore workers own dependency
resolution, Ready scheduling, kernel execution, and completion publication.

The ordinary Ready path handles one AIC or AIV lane with one logical block; a
dispatch predicate is supported only on this shape. The Gang path handles MIX,
single-lane SPMD, and sync-start cohorts. Dependency-only and host-completed
tasks are resolved without launching a user kernel.

## Capacity

`host_build_graph` is whole-graph-resident. Task slots, heap bytes, fanin IDs,
and TensorMap entries are not reclaimed while the graph is executing. The graph
must fit the configured task count (`runtime_env.ring_task_window`) and the
TensorMap pool before launch; the heap is not a third capacity, since its device
region is committed after orchestration at the size the graph turned out to need.

## Validation

Run the HBG runtime simulation sweep after runtime changes:

```bash
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim --runtime host_build_graph
```

The A5 `mix_spmd_sync_start` scene covers normal MIX, single-lane SPMD, and
sync-start cohort execution. `graph_execution_mix_spmd` covers the same shape
metadata through Graph Execution, while `predicated_dispatch` covers the
ordinary single-lane predicate path.
