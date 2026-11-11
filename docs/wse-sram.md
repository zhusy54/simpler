# WSE SRAM and hardware configuration

WSEsim provides finite per-core SRAM for direct L2 Workers using
`host_build_graph`. Application buffers persist across kernels and runs until
freed or the Worker closes. Resident execution metadata, dependency state and
PE queues occupy the Runtime partitions while a run is active.

## Configure a device

The default specification lives in
`src/wse/platform/config/default_simulation.json`. Runtime builds derive their
defaults from this file; it is also included in installed packages. Hardware
capacity and Runtime reservation policy are separate configuration sections.

```python
worker.init()
worker.configure_wse({
    "hardware": {
        "core_count": 40,
        "pe_per_core": 4,
        "sram_bytes_per_core": 100 * 1024 * 1024,
    },
    "runtime": {
        "sram_reserve_bytes_per_core": 256 * 1024,
    },
})
print(worker.wse_configuration())
```

`configure_wse` also accepts a path to a JSON file. Explicit fields override
the loaded runtime's effective configuration. Capacity and reservation accept
either a scalar or an array whose length equals `core_count`. Unknown fields,
invalid sizes and unsupported topology are rejected.

The default is 40 cores with four PEs each, 100 MiB SRAM per core and 256 KiB
reserved for Runtime storage. The implementation supports 1 through 64 device
cores and four PEs per core. These are functional simulator capabilities, not
a hardware specification. Other PE counts are rejected. Capacity and reserve
must be multiples of 128 bytes, with `0 < reserve < capacity`.

Configuration freezes at the first SRAM allocation or execution preparation.
Changing it requires closing and recreating the Worker. The default JSON is read
at build time; edit it and rebuild, or pass a JSON path to `configure_wse` before
first use. `CallConfig.core_count`
selects the active cores for a run: zero uses all configured cores, and a
positive value cannot exceed the device's core count. It does not resize SRAM.
A core outside the active execution set can still own application data.

## Allocate and use application storage

```python
buffer = worker.malloc(4096, core_id=3, alignment=128)
worker.copy_to(buffer, host_input)
tensor = buffer.tensor(shapes=(1024,), dtype=DataType.FLOAT32)
# Submit tasks using tensor, then wait for their completion.
worker.copy_from(host_output, buffer)
worker.free(buffer)
```

The allocation is an ordinary `Buffer`: identity, tensor views, direction
checks and in-flight borrowing retain their existing meaning. `core_id` names
the storage owner, independently of which PE executes a kernel. Different PEs
using the same address access the same bytes; remote access does not introduce
a Host copy. The simulator does not model interconnect latency or bandwidth.

Application storage occupies the low-address portion of each core. Runtime
uses the reserved high-address portion, and unused reservation cannot be
borrowed by applications. Allocation uses aligned, coalescing free intervals;
padding consumes capacity until the allocation is freed. An allocation never
spans cores and does not fall back to GM. New contents are uninitialized.

WSE has one device allocation API: `worker.malloc(size, core_id=..., alignment=64)`.
The core owner is required, and there is no separate SRAM allocation method or
GM allocation path. `free` and `copy_to/copy_from` use the allocation's existing
ownership. Other platforms retain their device-wide allocation semantics.

`committed_device_memory` reports the total physical SRAM backing committed by
the Worker: zero before initialization of the backing, then the sum of all core
capacities. Application suballocations and Runtime occupancy are not added again.
`worker.sram_info(core_id)` reports base address, capacity, reservation, application
usage, free bytes, largest free interval and current Runtime usage. Freeing an
application buffer returns space to that core; physical backing lasts until close.

Internal tensor temporaries, staged Host tensor arguments and graph-created data
use Core0's application partition and count against its capacity. They follow
the same SRAM allocation and release implementation as caller buffers. Host graph
preparation templates have separate Host ownership and are never legal kernel
tensor backing; the executor copies the reachable execution metadata into the
reserved Runtime partitions before launch.

Views share the original allocation's ownership and budget. `free` refuses
while a submitted run can still access the allocation. A wait timeout does
not authorize freeing, resetting or overwriting it. Host initialization and
readback must not race kernel accesses, including counter atomic operations.

## Kernel access and resident control data

Kernels can include `sram_args.h` and call
`wse::make_sram_view(args, tensor, view, expected_owner)` to validate a live
application range. `expected_owner` is optional; the resolved view reports
address, byte span and owner. Validation includes strided extents and rejects
private Runtime storage. Existing Tile and counter operations can use these
addresses. This is checked API access, not a sandbox for arbitrary C++ pointers.

Runtime preparation constructs a compact execution image from the bound graph.
PEs execute its SRAM-backed task records, tensor/scalar arguments, predicates
and successor lists. Per-core controls and PE queue payloads reside in their
own core's Runtime partition. Shared graph control and execution metadata
currently reside on Core0. All PEs may update shared state through the existing
CPU synchronization; Core0 is not a dedicated per-task dispatcher.

Queue capacity conservatively covers all blocks assigned to that PE in the
graph. A graph that exceeds a core's Runtime reservation is rejected before
GO, and the same Worker can subsequently execute a smaller legal graph.
No unbounded Host queue or shadow graph supplies overflow storage.

With `CallConfig.output_prefix`, `wse_sram.json` accompanies PE events. It
records the effective numeric configuration, per-core usage and actual
resident section addresses/sizes. Section kinds identify run control (1),
task records (2), core offsets (3), region descriptors (4), tensors (5),
scalars (6), successors (7), core controls (8), queues (9) and application
range descriptors (10). Padding is included in Runtime usage. Observations
are emitted after execution and do not provide kernel synchronization.

Host threads, locks and diagnostic containers remain simulator machinery.
Device program/stack storage and physical Tile capacity are not modeled.
CPU atomics do not certify hardware atomic visibility, and execution-fault
recovery or forced kernel interruption is not provided. Core0 capacity and
remote shared-state traffic remain limitations of the centralized placement.

The primary FFN qualification starts all 40 cores and executes 160 blocks in
each compute stage, covering every core and PE. Its 40 column shards and four
row shards are checked against independent H/A/Y golden tensors, including
replay and pending runs. Boundary cases use the same 40-core topology.
