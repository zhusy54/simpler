# Developer Guide

## Directory Structure

```text
pto-runtime/
├── src/
│   ├── common/task_interface/            # Cross-architecture shared headers (data_type.h, tensor.h, task_args.h)
│   └── {arch}/                         # Architecture-specific code (a2a3, a5)
│       ├── platform/                   # Platform-specific implementations
│       │   ├── include/                # Shared headers (host/, aicpu/, aicore/, common/)
│       │   ├── shared/                 # Sources shared between onboard and sim backends (compiled into both)
│       │   ├── onboard/               # Real hardware backend
│       │   │   ├── host/              # Host runtime (.so)
│       │   │   ├── aicpu/             # AICPU kernel (.so)
│       │   │   └── aicore/            # AICore kernel (.o)
│       │   └── sim/                   # Thread-based simulation backend
│       │       ├── host/
│       │       ├── aicpu/
│       │       └── aicore/
│       │
│       └── runtime/                   # Runtime implementations
│           ├── common/                # Shared components across runtimes
│           ├── host_build_graph/      # Host-built graph runtime
│           └── tensormap_and_ringbuffer/  # Advanced production runtime
│
├── python/                            # Language bindings
│   ├── bindings/                      # nanobind extension module (_task_interface)
│   │   ├── CMakeLists.txt
│   │   ├── task_interface.cpp
│   │   └── worker_bind.h
│   └── simpler/                       # Stable user-facing API (packaged in wheel)
│       ├── worker.py                  # Unified Worker (L2 single-chip, L3 distributed)
│       ├── task_interface.py          # Python re-exports of nanobind types + helpers
│       ├── env_manager.py             # Environment variable management
│       ├── kernel_compiler.py         # Transitional, ships in wheel; use simpler_setup version
│       ├── runtime_compiler.py        # Transitional, ships in wheel; use simpler_setup version
│       ├── elf_parser.py              # Transitional, ships in wheel; use simpler_setup version
│       └── toolchain.py               # Transitional, ships in wheel; use simpler_setup version
│
├── simpler_setup/                     # Test framework + authoritative compilers (packaged in wheel)
│   ├── scene_test.py                  # SceneTestCase + @scene_test decorator
│   ├── runtime_builder.py             # RuntimeBuilder (pre-built lookup or compile)
│   ├── runtime_compiler.py            # Authoritative copy of runtime cmake driver
│   ├── kernel_compiler.py             # Authoritative copy of kernel compiler
│   ├── toolchain.py                   # Authoritative copy
│   ├── elf_parser.py                  # Authoritative copy
│   ├── platform_info.py               # Platform/runtime discovery
│   ├── environment.py                 # PROJECT_ROOT resolver (wheel vs source tree)
│   ├── pto_isa.py                     # pinned PTO-ISA checkout management
│   ├── build_runtimes.py              # Pre-build all runtime variants (invoked by pip install)
│   └── _assets/                       # (wheel only) src/ + build/lib/ shipped with wheel
│
├── examples/                          # Working examples
│   └── {arch}/                        # Architecture-specific examples
│       ├── host_build_graph/
│       └── tensormap_and_ringbuffer/
│
├── tests/                             # Test suite
│   ├── ut/                           # Unit tests
│   │   ├── py/                       # Python unit tests (pytest)
│   │   └── cpp/                      # C++ unit tests (GoogleTest)
│   └── st/                           # Device scene tests (hardware-only)
│
└── docs/                              # Documentation
```

## Role-Based Directory Ownership

| Role | Directory | Responsibility |
| ---- | --------- | -------------- |
| **Platform Developer** | `src/{arch}/platform/` | Platform-specific logic and abstractions |
| **Runtime Developer** | `src/{arch}/runtime/` | Runtime logic (host, aicpu, aicore, common) |
| **Codegen Developer** | `examples/` | Code generation examples and kernel implementations |

**Rules:**

- Stay within your assigned directory unless explicitly requested otherwise
- Create new subdirectories under your assigned directory as needed
- When in doubt, ask before making changes to other areas

## Compilation Pipeline

The build has two layers: **runtime binaries** (platform-dependent, user-code-independent) and **user code** (orchestration + kernels, compiled per-example).

### Runtime binaries

Runtime binaries (host `.so`, aicpu `.so`, aicore `.o`) are pre-built during `pip install .` and cached in `build/lib/{arch}/{variant}/{runtime}/`. After wheel install they are shipped under `simpler_setup/_assets/build/lib/...`; `simpler_setup/environment.py::PROJECT_ROOT` resolves the right location automatically (see [Path resolution](#path-resolution)). The pipeline:

1. `simpler_setup/build_runtimes.py` — detects available toolchains, iterates all (platform, runtime) combinations
2. `simpler_setup/runtime_builder.py` — orchestrates per-runtime build (lookup pre-built or compile)
3. `simpler_setup/runtime_compiler.py` — invokes cmake for each target (host, aicpu, aicore)

Persistent cmake build directories under `build/cache/` enable incremental compilation — only changed files are recompiled.

**Architecture note:** binaries are built separately for each architecture and
backend. The onboard AICore targets are `dav-c220-cube/vec` for a2a3 and
`dav-c310-cube/vec` for a5, so their artifacts are not interchangeable.
`pip install .` selects architectures from the available toolchains. A5 additionally
requires the HCCL channel-config API used by the pinned URMA provider (including
`HcclChannelAcquireWithConfig`). If A5 is selected and these declarations are
absent, the build fails with an SDK requirement error; it does not silently omit
A5 or delete existing artifacts. Set `ASCEND_HOME_PATH` to a compatible SDK.
For an intentionally A2/A3-only install, use
`pip install --no-build-isolation . --config-settings=build.targets=build_package_a2a3`;
use `build_package_sim` for simulation only. Build availability does not imply
that the host has both kinds of NPU.
Toolchain detection (`build_runtimes.py`):

- **sim** (a2a3sim, a5sim): requires `gcc` + `g++` in `PATH`
- **onboard** (a2a3, a5): requires `ccec` in `PATH` + cross-compiler under `ASCEND_HOME_PATH`

### User code (per-example)

1. `simpler_setup/kernel_compiler.py` — compiles user-written kernel `.cpp` files (one per `func_id`)
2. `python/bindings/` — nanobind extension providing ChipWorker, task types, and distributed types to Python

### Path resolution

`simpler_setup/environment.py::PROJECT_ROOT` returns one of two layouts depending on how the package was installed:

| Install mode | `PROJECT_ROOT` | Where `src/` lives | Where `build/lib/` lives |
| ------------ | -------------- | ------------------ | ------------------------ |
| `pip install .` (wheel) | `<site-packages>/simpler_setup/_assets/` | `_assets/src/` | `_assets/build/lib/` |
| `pip install -e .` (editable) | repo root | `<repo>/src/` | `<repo>/build/lib/` |
| Source tree without install | repo root | `<repo>/src/` | `<repo>/build/lib/` |

The resolver uses `importlib.resources.files("simpler_setup") / "_assets"`. If `_assets/src/` exists there, it picks that; otherwise it falls back to `Path(__file__).parent.parent` (repo root). All compilers (`runtime_compiler`, `kernel_compiler`, `runtime_builder`) consume `PROJECT_ROOT` and stay agnostic to install mode.

### Python package layout

| Package | Where on disk | What it ships in wheel |
| ------- | ------------- | ---------------------- |
| `simpler` | `python/simpler/` | Every module, including the runtime API and the four transitional compiler/toolchain modules |
| `simpler_setup` | `simpler_setup/` | Test framework + authoritative copies of the four transitional files |
| `_task_interface` | built from `python/bindings/` | Top-level nanobind extension (.so) |

**Migration direction:** existing `from simpler.{kernel_compiler,runtime_compiler,toolchain,elf_parser} import ...` should move to `from simpler_setup.{kernel_compiler,...} import ...`. The `simpler_setup` copies are authoritative. Both copies ship because `wheel.packages` includes both packages and there is no `wheel.exclude`; deletion of the transitional modules requires migrating their callers.

## Cross-Platform Preprocessor Convention

When preprocessor guards are used to isolate platform code paths, the `__aarch64__` block must be placed first:

```cpp
#if defined(__aarch64__)
// aarch64 path (must be first)
#elif defined(__x86_64__)
// x86_64 host simulation path
#else
// other platforms
#endif
```

## Example / Test Layout

Every example and device test follows this structure:

```text
my_example/
  test_my_example.py     # @scene_test class (CALLABLE + CASES + generate_args + compute_golden)
  kernels/
    aic/                 # AICore kernel sources (optional)
    aiv/                 # AIV kernel sources (optional)
    orchestration/       # Orchestration C++ source
```

Run via pytest (`pytest examples tests/st --platform <platform>`) or standalone
(`python test_my_example.py -p <platform>`).

For the kernel-author contract (SPMD execution context, accessor functions,
and the gotcha around CCE topology intrinsics that breaks ports from native
CANN code), see [aicore-kernel-programming.md](aicore-kernel-programming.md).

## Build Workflow

### Initial setup

```bash
pip install --no-build-isolation -e .
```

`--no-build-isolation` is required: scikit-build-core needs the system `nanobind` and `cmake` (and any other build dep already in the venv); build isolation would hide them. The flag is also faster — no temporary build venv per install.

This builds the nanobind `_task_interface` extension **and** pre-builds runtime binaries for available toolchains into `build/lib/`. Sim platforms (a2a3sim, a5sim) are built when `gcc`/`g++` are available; onboard platforms (a2a3, a5) are built when `ccec` and the cross-compiler under `ASCEND_HOME_PATH` are available. Each architecture has its own sources, compiler target, and output directory.

### No rebuild on import

`pyproject.toml` sets `editable.rebuild = false`, so importing the package does **not** trigger a `cmake --build`. (Enabling it breaks under build isolation: the cmake path baked into `build.ninja` points into pip's deleted ephemeral env, so the next import fails with `cmake: No such file or directory` — see [Python packaging](python-packaging.md).) Therefore **any** C++ change — nanobind bindings or runtime/platform sources — only takes effect after re-running `pip install --no-build-isolation -e .`. The rebuild is incremental via the persistent cmake caches under `build/cache/` (~1-2s for a one-file change). CI should never use editable installs.

### When to rebuild

| What changed | Action |
| ------------ | ------ |
| First time / clean checkout | `pip install --no-build-isolation -e .` |
| Runtime C++ source (`src/{arch}/runtime/`, `src/{arch}/platform/`) | Re-run `pip install --no-build-isolation -e .` (or `.` for a non-editable install). Incremental via the cmake caches under `build/cache/` (~1-2s). |
| Nanobind bindings (`python/bindings/`) | Re-run `pip install --no-build-isolation -e .` (no rebuild-on-import; `editable.rebuild = false`) |
| Nothing — but `HEAD` moved (branch switch, rebase, pull, **or your own commit**) | Re-run `pip install ...`. You may have changed no compiled file, yet the tree moved out from under a binary frozen at install time. `simpler.task_interface` refuses to import on this skew (see below) rather than let struct fields read as 0. |
| Python-only code (`python/*.py`, `simpler_setup/*.py`) | Nothing to recompile (editable install) — but see the row above: once you *commit* it, the import guard keys on `HEAD`, so reinstall before the next import. |
| Examples / kernels (`examples/{arch}/`, `tests/st/`) | Nothing to recompile, just re-run — same commit caveat as the row above |
| `pto_isa.pin` changed | Re-run `pip install`. The cmake cache stamp and the `host_runtime` ccache key include the pinned PTO-ISA commit for all a2a3 onboard and a5 onboard builds, so a pin bump invalidates stale runtime objects automatically. |

`_task_interface` records the commit it was compiled from, and
`simpler.task_interface` compares it against the working tree at import,
raising when they differ. It is git-based for the same reason
`RuntimeBuilder._build_cache_stamp` is — mtimes do not survive a branch switch —
and deliberately keyed on the whole HEAD rather than on the binding sources
alone: a root `CMakeLists.txt`, `pyproject.toml` or nanobind change alters the
extension without touching `python/bindings/`, and under-detecting here means
silently wrong values. The cost of over-detecting is one ~20 s reinstall, and
120 of the last 200 commits touched the ABI surface anyway — so most of those
reinstalls are ones you owed regardless, merely made visible.

Verifying that `import simpler` resolves into your worktree does **not** cover
this: under an editable install that only proves the *Python* is live, while the
compiled extension stays at whatever `pip install` produced.

The check is inert outside a source tree — a wheel has no `.git` to compare
against, and a build made without git carries an empty stamp.

A `pto_isa.pin` bump changes the SDMA headers embedded by
`host_runtime.so`. Install-time runtime builds and run-time kernel compilation
both read `pto_isa.pin`; use a different ISA revision by updating that file.

The a5 RDMA workspace is an explicit opt-in overlay enabled with
`SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=ON`. It requires a `pto_isa.pin` revision
that provides the PTO-ISA RDMA/HNS1825 headers. SDMA and URMA are always compiled
on A5; their resources are initialized only when requested by a domain. All three providers
can be compiled together. A5 domains select resources with `workspace_engines`;
RDMA uses a separate domain because its window has no mapped peer VAs. See
[communication domains](comm-domain.md#4-backends).

### Runtime binary lookup

Scene tests load pre-built runtime binaries from `build/lib/`. These are produced
by `build_runtimes.py` during `pip install`, using the persistent cmake cache in
`build/cache/` to recompile only what changed. If the binaries are missing, the
runtime loader raises with a hint to run `pip install --no-build-isolation .`.

### Disk layout

```text
build/
  cache/{arch}/{variant}/{runtime}/   # runtime cmake intermediate files (persistent)
    host/                             # cmake build dir for host target
    aicpu/                            # cmake build dir for aicpu target
    aicore/                           # cmake build dir for aicore target
  lib/{arch}/{variant}/{runtime}/     # runtime final binaries (stable lookup paths)
    libhost_runtime.so
    libaicpu_kernel.so
    aicore_kernel.o                   # or .so for sim
  {wheel_tag}/                        # scikit-build-core's top-level cmake (e.g. cp39-cp39-linux-x86_64)
                                      #   builds the _task_interface nanobind module
```

All three subdirs are siblings under `build/` and ignored by `.gitignore`. `rm -rf build/` clears everything.

## Dynamic Kernel Compilation

Kernels are compiled externally by `KernelCompiler` and uploaded to the device at runtime:

```python
from simpler_setup.kernel_compiler import KernelCompiler

compiler = KernelCompiler(platform="a2a3sim")
kernel_binary = compiler.compile_incore("path/to/kernel.cpp", core_type="aiv")
```

CPU simulator kernels select their PTO instruction family from the platform:
`a2a3sim` defines `PTO_CPU_SIM_TARGET_A2A3`, and `a5sim` and the WSE compatibility target `wsesim` define
`PTO_CPU_SIM_TARGET_A5`. Both AIC and AIV compilation force-include
`pto/cpu/target_config.hpp` from the supplied `pto_isa_root` checkout, or from
the managed pinned checkout when `pto_isa_root` is omitted. An explicit root
selects that checkout's headers and ISA revision. The selected family
controls instruction semantics and the simulator memory profile, without a
`PTO_CPU_SIM_ARCH` override or an initialization call in the kernel.
An ISA checkout without this interface fails compilation. Compiler flags,
compiler logic, and the ISA pin participate in the managed kernel cache keys;
previous artifacts are rebuilt after the interface or pin changes.
Each compiled kernel also exports `simpler_cpu_sim_target()` through
`cpu_sim_target.h`, returning the platform name (`a2a3sim` or `a5sim`).
WSE uses `wse_cpu_sim_target.h` to export `wsesim` while retaining A5 CPU semantics.
The simulator loader checks this marker before resolving
the kernel entry or registering hooks. It rejects prebuilt kernels from the
other family, unknown families, and older kernels without a marker; rebuild
those kernels for the destination platform.

The compiled binaries are packed into a `ChipCallable` (orch SO + each
child `CoreCallable`) and registered once via
`DeviceRunner::upload_chip_callable_buffer(callable)`, which fixes up each
child's `resolved_addr_` and returns the address of the ChipCallable header.
The same call derives each child's address from that header plus the child's
recorded offset and builds the callable's func_id-indexed function tables, so
every run that binds the callable references them instead of rebuilding them.

Where those tables live is backend-specific. **Onboard** places them in the
`CALLABLE_ALIGN`-aligned tail of the same device allocation and delivers them in
the same single H2D as the code, so registration still makes one allocation and
one copy. **Sim** issues no H2D at all — its "upload" is a host scratch — and
keeps the object and entry tables in separate host vectors that the pooled
registration owns; a run references those vectors' own addresses.

## Features

- **Three programs compile independently** with clear API boundaries
- **Full Python API** via nanobind with torch integration
- **Modular design** enables parallel component development
- **Runtime linking** via binary loading

### WSE functional simulation

`wsesim` provides a Host/AICore runtime under `src/wse`, with no AICPU
component or three-lane dispatch. It defaults to 40 AICores with four homogeneous
PEs each. A block runs one Kernel entry on one PE; that entry can contain matrix
and vector operations. Kernel block counts are independent of the Core count,
and blocks beyond the available PEs execute in waves.

`CallConfig.core_count` selects 1–40 Cores for a run; 0 selects the default 40.
This option is WSE-only and is not transported by the remote L3 protocol.
The packed configuration uses native ABI version 2. Rebuild the Python binding
and runtime components together, and update handwritten C ABI mirrors. The
existing contract check rejects version-1 components before initialization.
`rt_submit_wse_task` submits a single entry, `rt_available_core_count` reports the
configured topology, and Kernel `get_core_id`/`get_pe_id` report physical identity.
PE0 initializes its Core before computing; Core0 coordinates startup and normal
retirement. Device tasks publish completion once after all their blocks finish.
Host wait timeouts retain resources; normal close drains and joins the PE threads
on the thread that called `init()`.

Kernels including `<wse/tile_ops.h>` use the isolated `wse-temporary-fp32`
CPU operation backend: static FP32 Tiles up to 32x32, row/column-major logical
storage, bounded strided views, TLOAD/TSTORE, TMATMUL/TMATMUL_ACC and TRELU.
This does not implement physical CUBE packing. Legacy `<pto/pto-inst.hpp>`
fixtures explicitly retain pinned A5 CPU operations; there is no implicit fallback
from an unsupported WSE operation to A5. WSE device data uses finite per-core SRAM with explicit allocation ownership;
Graph preparation templates remain on the Host.

SharedTile uses Core-local slots S0..S63 in the existing CPU simulator context.
The temporary backend resolves `pto_sim_get_pipe_shared_state`; the PE executor
already binds the device/Core identity and clears shared state before each run.
Fixed, trivially destructible FP32 slot state needs no Runtime object registry,
layout IDs or construction/destruction callbacks. `publish` waits for prior
consumers; `read` copies a Tile value and marks that consumer finished before
returning. The matmul overloads consume the shared operand this way. There is no
public storage lease. Slots keep one shape/participant configuration for a run;
tasks deliberately sharing a slot continue its generation sequence.

`get_core_id`, `get_pe_id` and block intrinsics supply execution identity.
`get_wse_operation_trace(args)` provides an optional observation callback only.
The producer must be runnable on an independent PE; arbitrary cross-block waits
are not guaranteed to make progress. Shared operations do not group four PEs.
Shared state is freed by the existing simulator reset between joined runs or
at device teardown. It is not an addressable hardware shared-memory region.

Temporary implementation details live under `src/wse/functional/temporary`,
behind the public facade under `functional/include/wse`. The executor does not
include backend-private Tile definitions. A future PTO WSE adapter can replace
operation semantics while keeping scheduling, graph completion and resource
ownership in simpler. PR #418 is reference-only and is not incorporated.
Kernel cache identity hashes the backend headers and retains the target family and PTO pin.

Kernels including `<wse/counter_ops.h>` can signal and poll application-owned
contiguous `UINT64` buffers. Include `counter_args.h` to adapt a runtime Tensor
with `wse::make_counter_view(tensor, view)`; it returns `CounterStatus::OK` or
`INVALID_ARGUMENT` and checks dtype, one-dimensional shape, unit stride,
element offset, byte extent and atomic alignment without dereferencing memory.
The view borrows the allocation; it cannot validate forged or dangling pointers.

`counter_signal(view, index, delta=1)` atomically adds a positive delta and
returns `status`, `old_value` and `new_value`. Zero delta and invalid views return
`INVALID_ARGUMENT`; overflow returns `OVERFLOW` without wrapping or modifying
the word. `counter_poll(view, index, target, max_reads=1)` performs at most the
specified number of acquire reads and returns `READY` when `observed >= target`,
otherwise `NOT_READY`. A zero read budget is invalid. This is a read-count budget,
not an elapsed timeout or a cancellation mechanism; no operation throws.

Initialize live uint64_t objects through existing Host copies before submission.
During concurrent use, access each counter only through the atomic operations.
Producers write disjoint payloads before signal; consumers read after READY.
The CPU backend uses acq_rel CAS and acquire loads. For repeated rounds use
checked cumulative targets, and a separate application ack counter before
reusing payload storage. Keep buffers alive until execution drains; a Host wait
timeout does not permit reset or free. Concurrent runs use disjoint counter
ranges; reset a reused range only after its previous users have exited.

Counters do not update Runtime task completion or SharedTile generations.
Cross-Kernel concurrent protocols must declare actual read/write arguments and
explicit manual dependencies; the caller must ensure producers can run on
independent PEs. The examples in `tests/st/wse/host_build_graph/counter` exercise
this with producer/consumer acknowledgement before either task can finish.
There is no counter allocation service or scheduler change. The CPU semantics
apply within one simulated device/process and do not certify remote hardware
atomics. Explicit SRAM buffers use the same counter operations and lifetime protection.
See [WSE SRAM and hardware configuration](wse-sram.md) for allocation and capacity contracts.

The implementation accounts for application SRAM and resident execution/control data.
Device program/stack storage, physical Tile capacity, hardware pipeline timing,
and execution-fault recovery are not modeled. Only `host_build_graph`
is supported; there is no WSE onboard or TMR target and no SuperScalarModel/Linx
dependency.

Build it explicitly in the existing project environment. Use GCC 15 for the
native Python binding and simulator components (`CC=gcc-15 CXX=g++-15`),
and make unversioned `gcc`/`g++` resolve to the same toolchain for orchestration.
When changing compilers, configure a fresh build cache rather than reusing
CMake caches that select an older compiler.

```bash
CC=gcc-15 CXX=g++-15 python -m pip install --no-build-isolation --no-deps \
  --config-settings=build.targets=build_package_wsesim -e .
```

The default `build_package_sim` still builds A3/A5. WSE runtime outputs live
under `build/lib/wse/sim/host_build_graph`, with build caches under
`build/cache/wse/sim/host_build_graph`. WSE kernels have distinct target and
cache identities; an A5 kernel is rejected by WSE, and vice versa.
`src/wse/source-copy-manifest.json` records the fixed source provenance and
per-file hashes. Shared `src/common` dependencies remain shared.

The WSE suite also needs A5 binaries for cross-target rejection checks:

```bash
CC=gcc-15 CXX=g++-15 python -m pip install --no-build-isolation --no-deps \
  '--config-settings=build.targets=build_package_wsesim;build_package_a5sim' -e .
python -m pytest tests/st/wse/host_build_graph --platform wsesim \
  --require-pto-isa --manual include --rounds 1 --pto-session-timeout 1200
```

The suite retains the original vector math and uses WSE single-entry Graph
fixtures for the former Mix/SPMD output contract. It checks real PE execution,
mixed matrix/vector operations, configurable blocks, Graph replay, completion,
startup rollback, wait/close and buffer ownership. With `output_prefix` set,
`wse_events.jsonl` records Core/PE lifecycle and expanded task/block identities;
its sequence numbers express functional ordering, not hardware timing.
Host dependency capture (`enable_dep_gen`) remains available. Legacy device
swimlane, argument dump, PMU and scope-stat collector requests are rejected at
admission; use the PE event trace for this execution model.
WSE has no separate entry/startup/component version negotiation or protocol JSON.
Rebuild the binding, Runtime and Kernels together after interface changes. Target
family and required-entry checks remain, as does the shared PipelineContract ABI.
The JSON event trace records lifecycle and operation events 12–15 (publication,
wait, read, consumed), with slot/generation fields. It does not report a separate
Runtime allocation service. Packaging contains only Host/AICore WSE binaries.
Unused AICPU/three-lane executors are removed; common Host arena layout declarations
remain because shared HBG initialization consumes them, and do not schedule PEs.
Historical sources, binaries and capture tools are retained in migration evidence.
A GCC 15 kernel requires a compatible `libstdc++` at runtime; use the same
compiler runtime for reference and candidate runs.
