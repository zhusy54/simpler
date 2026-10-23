# A3 queue peak measurements

Baseline: upstream main `b24bfc1875ae04248edea14824239f4ac39de66d`.
Hardware: Ascend 910_9392 (A3). PTO-ISA pin:
`b1d460ee8c001585ede36b68ba3678dc5a226bbf`.

The [CSV](a3-runtime-queue-peaks-results.csv) contains independent observed
maxima and the invocation that produced each value. Units are queue entries.
See the [measurement contract](a3-runtime-queue-peaks.md) before interpreting
the numbers as capacity guidance.

The initial corpus covered 370 candidate cases across both runtimes: 700 test
process starts and 1,110 requested rounds, with golden comparison disabled.
There were 476 starts with valid reports, 79 skips, 136 without queue data,
6 with incomplete reports and 3 interrupted by a task-queue daemon restart.
293 distinct cases produced valid data in that campaign. Some no-data cases
exercise direct L2 kernels, rejected prepares or host-only paths, rather than
these runtime queues; others needed a more reliable report transport.

High-volume traces filled the existing asynchronous host-log queue during
several concurrency tests. The final implementation writes a bounded report
synchronously to the run's isolated log directory after completion. Its A3
smoke recovered all eight calls (136 queue records) in an outstanding-close
case. The two builds have identical AICPU `.text` hashes for each runtime:

- HBG: `159305ea7c7988899ccb90d3ea4d484355795b8c79cc80fc9cba29ff1350d877`
- TMR: `2809febdffa16e1867372f972cfaea76725f9634f9d7ed874a1bd55a80173dff`

The CSV also includes valid observations from diagnostic retries. Failed and
incomplete reports never contribute; missing observations are not zero.
A further complete retry of unsuccessful attempts and a same-device ABBA
latency confirmation are pending hardware availability at this snapshot.
The shared host's 16 devices are occupied by another allocation.

## Shared-queue observations

These are independent maxima across the completed runs, so the class maxima
must not be added. HBG logical capacity varies with the graph; consult each
CSV row for the capacity of the invocation that produced that maximum.

| Runtime | Queue          | Shape | Total peak | SPMD peak | Normal peak |
| ------- | -------------- | ----- | ---------: | --------: | ----------: |
| HBG     | Ready          | AIC   | 16384      | 7         | 16384       |
| HBG     | Ready          | AIV   | 512        | 3         | 512         |
| HBG     | Ready          | MIX   | 1          | 1         | 0           |
| HBG     | Early dispatch | AIC   | 8          | 8         | 0           |
| HBG     | Early dispatch | AIV   | 3          | 1         | 3           |
| HBG     | Early dispatch | MIX   | 1          | 1         | 0           |
| TMR     | Ready          | AIC   | 94         | 12        | 88          |
| TMR     | Ready          | AIV   | 46         | 3         | 46          |
| TMR     | Ready          | MIX   | 12         | 1         | 12          |
| TMR     | Early dispatch | AIC   | 8          | 8         | 0           |
| TMR     | Early dispatch | AIV   | 5          | 2         | 5           |
| TMR     | Early dispatch | MIX   | 1          | 1         | 0           |

## Initial overhead comparison

Both arms ran sequentially under one device allocation. Each benchmark ran
six rounds; these deltas use the median of rounds 1-5, excluding round 0.
Eight HBG configurations and nine TMR configurations completed in both arms.
This comparison used the initial asynchronous host exporter; the final
synchronous exporter runs after device completion and still needs the queued
ABBA confirmation for host latency.

TMR effective-time changes ranged from -0.51% to +3.55%. The largest increase
was `benchmark_bgemm Case0` (734.9 to 761.0 us). Device-wall changes were
larger in two short configurations: +16.50% for `benchmark_bgemm Case0` and
+12.20% for manual-scope paged attention Case1. Do not interpret the compact
counter layout as proof of zero runtime overhead.

HBG device time decreased by 0.84%-16.95% in this comparison. This is the net
experiment change, which also removes the baseline's device-side queue-print
call; it does not isolate the cost of the new counters. Shared-host noise and
run order also limit a single sequential comparison.

## Verification

- Four C++ targets passed, including both actual queue implementations,
  counter arithmetic and the existing HBG ready-queue seed tests.
- Both queue targets passed with scheduler profiling enabled, including the
  profiling overloads.
- Five parser/aggregation tests passed.
- The default non-SDMA simulation corpus passed; A3 hardware smoke passed for
  both runtimes. The final reliable exporter also passed a three-round HBG
  simulation smoke (51 complete queue records).
- A5 simulation runtime builds passed after the shared descriptor change.
- Repository pre-commit checks passed, including clang-tidy and pyright.

Detailed artifacts are under `outputs/a3-runtime-queue-peaks-report/`, with
per-campaign evidence in `outputs/a3-queue-peaks-*` and timing comparisons in
`outputs/a3-queue-overhead*`. These directories are intentionally not committed.
