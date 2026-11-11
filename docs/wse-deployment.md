# WSE Block deployment and execution

A Kernel invocation is one original Graph node. Its Blocks are assigned to AICores;
each Block expands into 1–4 logical Threads mapped to distinct PEs of that Core.
A Core runs one active Block and advances its static queue after all participating
Threads return and operation state is released. A Buffer's owner and address do
not change when execution placement changes.

## Submission and identity

```cpp
CoreTaskArgs task;
task.add_input(input);
task.add_output(output);
task.launch_spec.set_block_num(2);
WseLaunchSpec launch;
launch.thread_num = 2;
launch.placement = {{1, {3, 0}}, {0, {1, 2}}};
rt_submit_wse_task(kernel_id, task, launch);
```

Without a placement table, Block `b` uses Core `b % active_core_count`, and logical
Thread `t` uses PE `t`. The default Thread count is one. Blocks beyond the number
of active Cores enter those Cores' queues; there is no rotation of Blocks over
PEs. Each explicit table entry selects one Core and a list of 1–4 distinct local
PEs. Repeated Core destinations are legal. Blocks support counts 1–32767.

`get_block_idx/num`, `get_thread_idx/num`, `get_core_id` and `get_pe_id` expose
separate logical and physical identities. All Threads of a Block call the same
entry. PE0 coordinates dispatch and can also execute a Thread. There is no
instruction-lockstep promise, Mix/lane scheduling, work stealing or automatic
placement. A Block waiting for a later Block on its own Core cannot progress;
applications must place cooperating roles within a Block or on concurrent Cores.

Placement belongs to `WseLaunchSpec`, not common `LaunchSpec`. A3/A5 do not carry
WSE containers, Definition fields or payload extensions. The WSE orchestration
header and its WSE-owned operation table provide the submission entry.

## Per-Block arguments and Graph ownership

For separate shards, set `launch.block_args` to exactly `block_num` pointers to
`CoreTaskArgs`. Those argument objects and their Tensor references must remain
valid through the synchronous submission call. The invocation's common Tensor
and scalar lists must be empty in this form; its launch count, dependencies and
predicate remain invocation attributes. Each Block supplies the same Tensor/scalar
signature and uses existing SRAM Tensor views. Allocation requests and per-Block
predicates/dependency lists are rejected. Threads share their Block's bindings.

The runtime captures owned Tensor geometry, scalar values/inheritance and placement
before submission returns. Every Block's reads and writes enter the original
TensorMap dependency analysis, including aliases and creator retention. There is
no raw-address scalar table or artificial contiguous Tensor spanning allocations.

Graph recording retains WSE records alongside the common Definition for the
lifetime of the bind and any retained preparation plan. Records name boundary
parameters or Graph heap offsets, not current-run pointers. Native binding
resolves them against each invocation's current Buffers and scalar parameters.
Graph structural choices belong in an owned `ConfigBase` with value-based hash
and equality. Unsupported common materialization of WSE companion records is
explicitly rejected; native execution is supported and is not disabled.

## Static queues, completion and SRAM

Host preparation builds each Core's immutable Block queue in deterministic Graph
order before GO. At execution, PE0 checks only its Core's head: the previous Block
must have drained and every original Graph predecessor must be complete. It waits
in place and does not bypass the head. The predicate is evaluated once before
Thread dispatch; the existing single-Block predicate restriction remains.

Graph enter/exit and other control nodes execute once when their predecessors
complete. They have no Thread expansion and do not occupy a Core's Block queue.
The simulator's synchronization lock claims each ready control node once, so a
waiting Graph exit does not hold an independent Graph entry behind it.

Threads return, operation state drains, the Block scope is invalidated, then the
Block completes. The last Block publishes its Kernel node's completion and adds
one to the whole-Graph atomic completed count. Control nodes also count once.
Every Core knows the expected whole-Graph count. Runtime execution never traverses
successors, decrements remaining dependency counters or enqueues newly ready work.
Normal STOP, quiescence, Core retirement and Host join retain their ownership rules.

Queues, Block descriptors, Thread mappings and bound arguments live in the
executing Core's finite Runtime SRAM. Core0 retains whole-Graph coordination,
predecessor references and node completion/remaining-Block state. `wse_sram.json`
reports actual sections and alignment-inclusive `runtime_used`; argument sections
are kinds 5/6 and queue storage is kind 9. Capacity failure rejects before GO with
no Host shadow queue, automatic growth or changed data owner.

## SharedTile scope and operation boundary

Construct SharedTile with `get_wse_block_scope(args)` and optionally
`get_wse_operation_trace(args)`. SharedTile data, readiness, configuration,
consumer acknowledgements and generations belong to that Kernel Block. All
participants drain before invalidation. The next Block can reuse the slot with a
new shape and start generation one; a handle from the ended Block is rejected.
Ending one Core's Block does not clear another Core's state. X/W/H/A/Y application
Buffers persist independently until explicitly released or overwritten.

The temporary operation backend uses the existing CPU simulator pipe storage.
Its lifecycle adapter owns the temporary slot-key convention; Runtime supplies
the Block boundary and scoped release primitive. No Core/scope/object manager or
new hardware shared-memory allocation is introduced. Future PTO-ISA replacement
remains operation-level; scheduling, Graphs, Buffer ownership and SRAM allocation
remain in simpler. Runtime admits 1–4 Threads, while the FFN collective uses four.

## FFN and acceptance

The deployment FFN contains ten independent groups: X is 128×32, W1 is 32×128,
W2 is 128×128, and H=XW1, A=ReLU(H), Y=AW2. Each of its three Kernels has 40 Blocks
and four Threads per Block. `group=b/4`, `column=b%4`, and `row=thread_idx`.
X row `r` belongs to Core `4*group+r`; weight and H/A/Y column `c` belongs to
Core `4*group+c`, including when execution Core/PE maps are permuted.

Each Core owns one 72 KiB FP32 main Buffer with these byte ranges:

| View | Start | Bytes |
| ---- | ----- | ----- |
| X | 0x00000 | 4096 |
| W1 | 0x01000 | 4096 |
| W2 | 0x02000 | 16384 |
| H | 0x06000 | 16384 |
| A | 0x0A000 | 16384 |
| Y | 0x0E000 | 16384 |

A separate 2 KiB UINT64 Buffer holds 1920 bytes of diagnostics. One instance uses
80 Buffers and 74 KiB/Core; two use 160 Buffers and 148 KiB/Core. These figures
exclude Runtime metadata and padding. Graph boundaries contain 80 Tensors, below
the existing limit of 128; the Chip limit stays 256. Per Block, the first and last
Kernels bind seven Tensors and ReLU binds three. Logical Thread zero publishes
weights from its mapped physical PE. Each new Block starts its own generations.

Tests compare H/A/Y with independent PyTorch goldens for two seeds, default and
permuted maps, repeated instances, ordinary/native execution and reverse waits.
They also check exact Thread/Block/node completion, finite SRAM admission,
Block-scoped SharedTile isolation, typed argument rebinding, timeout ownership
and normal close. The default remains 40 Cores, four PEs and 100 MiB/Core with
256 KiB reserved for Runtime. A separate 1 MiB/Core test verifies the constrained
budget without increasing it. Host copies occur before execution and after drain.

```bash
.venv/bin/python -m pytest tests/st/wse/host_build_graph/deployment \
  --platform wsesim --require-pto-isa --rounds 1 --manual include \
  --pto-session-timeout 1200
```

This is CPU functional simulation. Program/stack and physical Tile capacity,
hardware timing, communication engines and execution-fault recovery remain
outside this implementation.
