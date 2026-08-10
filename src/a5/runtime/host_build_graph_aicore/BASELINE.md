# HBG-AICore Frozen Baseline

The initial `host_build_graph_aicore` runtime is a mechanical copy of the A5
`host_build_graph` runtime at the following repository identity:

- source commit: `ba4062c1bc872b068e85ccf2187ed3bbb0e10bf1`
- source path: `src/a5/runtime/host_build_graph`
- source tree: `61cf8e8831b061a1cbc7430fcc4678db3c3900b0`

The import commit permits the sibling directory name, this baseline record,
and removal of the source tree's unused orchestration `execution` metadata to
differ. Orchestration toolchains are selected centrally by runtime name:
`host_build_graph` and `host_build_graph_aicore` use the host compiler, while
`tensormap_and_ringbuffer` uses the host compiler in simulation and the
AArch64 compiler onboard. Runtime-specific code is independent after the
import and must not include or link private sources from `host_build_graph`.

## Frozen behavior matrix

The source runtime's A5 tests are the behavioral oracle:

| Capability | Baseline test |
| ---------- | ------------- |
| Basic graph construction and AIC/AIV execution | `tests/st/a5/host_build_graph/vector_example` |
| Real paged-attention graph and manual scale case | `tests/st/a5/host_build_graph/paged_attention` |
| Callable prepare, reuse, and lifecycle | `tests/st/a5/host_build_graph/prepared_callable` |
| Argument dump diagnostics | `tests/st/a5/host_build_graph/dump_args` |
| Queue, graph, scheduler, and layout behavior | `tests/ut/cpp/a5` |

The frozen comparison uses the repository-pinned PTO-ISA and records the
simpler commit, dirty state, compiler/CANN versions, platform, build flags, and
raw per-run results. Later HBG changes do not update this baseline implicitly.

## Import verification

The import review verifies the recorded source tree hash, then treats the
runtime directory rename, this file, and the removed `execution` metadata as
the complete allowed import-time difference. Later scheduler changes are
reviewed independently. The name-based compiler mapping is covered by:

```bash
pytest tests/ut/py/test_runtime_builder.py -q
```
