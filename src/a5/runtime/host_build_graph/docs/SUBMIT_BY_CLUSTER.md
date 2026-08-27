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
  arguments live in the pool segments, not in the payload.
- `ChipTaskSlotState` contains the active mask, task attributes, logical block
  count, subtask counters, completion state, and descriptor/payload bindings.

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

The product is the number stored in the 16-bit completion counter. Invalid
values latch `SIMPLER_ERROR_INVALID_ARGS`; they are not capped at the device's
physical cluster count.

`require_sync_start` adds a separate residency constraint because every block
must launch as one cohort:

- AIV-only: `block_num <= rt_available_aiv_count()`
- AIC or MIX: `block_num <= rt_available_cluster_count()`

## Dependency and Readiness Flow

1. Host orchestration allocates a task slot and builds its payload.
2. TensorMap and explicit dependencies append producer local IDs to the payload's
   fanin region.
3. Submit publishes only the finished graph data; it does not push ready tasks.
4. After H2D, AIV schedulers collectively scan every submitted task exactly once.
5. A task with no executable fanins is routed to its scheduler-owner Ready inbox
   or Gang admission bit; otherwise it registers on the first executable
   producer in stored fanin order.
6. Producer completion resumes each detached consumer from its saved
   `next_fanin_index`, skips producers already in `DONE`, and registers on the
   next unfinished producer or routes the task Ready.

`SchedulerTaskControl::state` is the resident readiness truth. `DONE` is
terminal, and the closed wake-list sentinel prevents a racing registration from
missing completion, so a task needs no periodic fanin polling.

## Dispatch and Completion

- Single-lane tasks use two generation-tagged dispatch slots per worker and may
  directly refill a slot while resolving its previous completion.
- MIX placement reserves the same pending slot on every active lane of a
  physical cluster before any lane is published READY.
- MIX, SPMD, and `require_sync_start` tasks enter the Gang scheduler. Admission
  order is sync-start, MIX, then single-lane SPMD.
- Sync-start cohorts drain their required lanes, stage all slots as GATED, and
  release only after the generation-tagged participant tree converges.
- Scheduler-local participant records aggregate completed subtasks, and the
  cohort retires the graph task exactly once. Generation tags prevent stale
  drain, stage, dispatch, or completion tokens from satisfying a later cohort.

## Executor Model

The host loads and executes the orchestration shared object synchronously. The
device has no orchestration thread. AICPU manages the AICore lifecycle; resident
AIV schedulers own dependency resolution, Ready routing, Gang coordination, and
dispatch. Cluster ownership is assigned during the AICore handshake and remains
stable for the run. Graph replay uses a separately selected AICPU compatibility
executor.

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

The A5 `mix_spmd_sync_start` scene covers resident MIX, SPMD, and sync-start
execution. C++ scheduler tests cover admission priority, capacity rejection,
whole-cluster reservation, generation safety, and multi-participant lifecycle
convergence.
