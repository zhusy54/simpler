# Resolver paired loads: keep packed completions, drop metadata/gang LDP

**Date**: 2026-08-25
**Verdict**: keep the packed `uint32_t[2]` completion publication; drop the metadata and gang-command `LDP.B64` changes

## Question

Can the A5 HBG AICore Resolver reduce GM transactions after `DCCI` by:

1. changing the two completion generations from `uint64_t[2]` to `uint32_t[2]` and reading both with one 64-bit `LD_DEV`;
2. reading the 16-byte `AicoreTaskMetadataV1` with one `LDP.B64`;
3. reading the 32-byte active portion of `AicoreGangCommandV1` with two `LDP.B64` instructions?

The dispatch slot generation is already `uint32_t`, so item 1 does not narrow the source value. The completion inbox stays 128 bytes and the publication line stays at offset 64.

## What was tried

Each item was compiled and measured independently, followed by a combined prototype. All onboard runs used A5 with the explicitly authorized architecture-precheck override and avoided device 0.

Representative commands:

```bash
python tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
  -p a5 -d 4 --case Case1 --rounds 20 --skip-golden

python tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
  -p a5 -d 2 --case Case1 --rounds 1 --skip-golden --enable-chip-swimlane 1

python tests/st/a5/host_build_graph/mix_spmd_sync_start/test_mix_spmd_sync_start.py \
  -p a5 -d 2 --case mixed_priorities --rounds 1 --skip-golden --enable-chip-swimlane 1
```

For endpoint timing, round 1 was treated as warm-up and the remaining 19 device-wall samples were averaged. Resolver phase comparisons aggregate repeated chip-swimlane runs on the same device and normalize by the number of phase events.

The C310 compiler accepted the output-constrained inline assembly:

```cpp
asm volatile("ldp.b64 %0, %1, [%2], 0\n"
             : "=l"(first), "=l"(second)
             : "l"(address)
             : "memory");
```

LLVM IR confirmed that the packed completion implementation emits a 64-bit `llvm.hivm.LD.DEV.u64.GM` read and a 32-bit `llvm.hivm.ST.DEV.u32` clear. The discarded prototypes emitted one `ldp.b64` for metadata and two for the gang command.

## Result

### Endpoint timing: Page Attention Case1 on device 4

| Variant | Warm 19 device wall | Difference from baseline | Mean absolute deviation | Task ID |
| ------- | ------------------: | -----------------------: | ----------------------: | ------- |
| Baseline | 1349.0 us | — | 13.9 us | `task_20260825_093916_38092081088` |
| Packed completions, first run | 1344.5 us | -0.33% | 16.0 us | `task_20260825_094726_38532228332` |
| Packed completions, repeat | 1354.0 us | +0.37% | 15.6 us | `task_20260825_100733_40417914713` |
| Metadata LDP only | 1348.6 us | -0.03% | 11.0 us | `task_20260825_094831_386284629279` |
| All three combined | 1348.2 us | -0.06% | 16.8 us | `task_20260825_094550_383981427439` |

There is no statistically useful endpoint signal: the packed-completion repeats fall on opposite sides of the baseline and all differences are much smaller than ordinary run-to-run variation.

Before commit, the retained optimization was rebuilt against an independent
`HEAD` worktree and compared on the same device for 100 rounds per side. Round
1 was excluded as warm-up:

| Variant | Warm samples | Mean device wall | Median | Median abs. dev. | Task ID |
| ------- | -----------: | ---------------: | -----: | ---------------: | ------- |
| Baseline (`HEAD`) | 99 | 1352.652 us | 1349.872 us | 14.613 us | `task_20260825_102011_416071421146` |
| Retained packed completions | 99 | 1351.226 us | 1349.695 us | 15.357 us | `task_20260825_102057_416641227893` |

The point estimate is a 1.426 us, or 0.105%, improvement. A 20,000-sample
bootstrap gives a 95% interval of -7.406 to +4.560 us for optimized minus
baseline (-0.548% to +0.337%). The interval crosses zero, so this larger run
also finds no measurable Case1 endpoint benefit.

### Resolver phase timing: Page Attention Case1 on device 2

| Item | Baseline cycles/event | Prototype cycles/event | Change | Interpretation |
| ---- | --------------------: | ---------------------: | -----: | -------------- |
| Completion scan, packed `uint32_t[2]` | 38,282 | 32,068 | **-16.2%** | Repeatable local win; two 64-bit completion loads become one |
| Dispatch prepare, metadata LDP, AIC | 835 | 958 | **+14.7%** | Regression |
| Dispatch prepare, metadata LDP, AIV | 2,679 | 2,530 | -5.6% | Improvement |
| Dispatch prepare, metadata LDP, combined AIC+AIV | 2,257 | 2,193 | -2.8% | Small mixed result, hidden at endpoint |
| Gang service, gang LDP while no command is active | 1,388 | 971 | -30.0% | Faster idle polling only |

The packed-completion numbers aggregate 432 baseline and 300 prototype completion-scan events. Metadata numbers aggregate 349 baseline and 238 prototype dispatch-prepare events.

### Active gang workload: MIX/SPMD/sync-start on device 2

| Item | Baseline cycles/event | Two-LDP prototype | Change |
| ---- | --------------------: | ----------------: | -----: |
| Resolver gang service | 29,677 | 32,798 | **+10.5% regression** |

Both sides aggregate 60 gang-service events. This active-command measurement overrides the apparent idle-polling win.

## Decision

- Keep item 1. It removes one `LD_DEV` from every two-slot completion-line scan and improves the target Resolver phase by 16.2%. It is correctly characterized as a local Resolver optimization; no measurable end-to-end Case1 speedup is claimed.
- Drop item 2. The result is core-type dependent, only 2.8% better in the combined local phase, and has no endpoint signal. The unpack shifts and extra register pressure likely consume most of the saved scalar-load issue work.
- Drop item 3. It speeds up the inactive polling path but regresses the real gang-command path by 10.5%.

Final validation for item 1:

- C310 AIC/AIV build passed.
- `test_a5_hbg_ready_scheduler`: 28/28 tests passed, including a new packed-order/32-bit-clear test.
- Onboard `mixed_priorities` correctness run passed on device 4 (`task_20260825_100759_40440969241`).
- LLVM IR contains the expected 64-bit device read and `ST.DEV.u32` clear, with no `ldp.b64` retained.

## When to reconsider

- Reconsider metadata LDP only if a workload shows dispatch prepare as an endpoint-critical phase and both AIC and AIV improve in repeated same-card measurements.
- Reconsider gang-command LDP only after explaining the active-command regression, preferably with an instruction-level trace that includes register pressure and dependency stalls.
- Revisit the packed completion change if a future dispatch generation becomes wider than 32 bits or if the two publication slots stop being independent SPSC cells.

## References

- `src/a5/runtime/host_build_graph/runtime/aicore_execution_sidecar_v1.h`
- `src/a5/runtime/host_build_graph/runtime/aicore_gang_scheduler_v1.h`
- `src/a5/runtime/host_build_graph/runtime/aicore_gm_atomic.h`
- `.claude/rules/discipline.md`
