# A3 runtime queue peak experiment

This experiment is based on upstream main
`b24bfc1875ae04248edea14824239f4ac39de66d`. It instruments the A2/A3
`host_build_graph` and `tensormap_and_ringbuffer` runtimes. Rebuild the runtime
before collecting data. Golden comparisons can be skipped.

## Measurement

The unit is **queue entries**, not SPMD blocks or live tasks. A task occupying
one slot contributes one even when it launches many blocks. Queue operations
account an accepted entry before publication and remove it before slot reuse.
The measurement includes counted entries awaiting publication, and excludes
reserved entries that have not reached the accounting operation. It is the
high-water mark of this accounting interval, not a sampled cursor difference.

Both runtimes report ready, sync-ready, early-dispatch, dummy, early-sync-start
and completion-mailbox peaks. HBG additionally reports graph-prepare,
early-dispatch publish-drain and each active scheduler's completion queue.
Resource-shape indexes are 0=AIC, 1=AIV, 2=MIX; -1 denotes an unsharded queue.

Ready and early-dispatch queues also report independent peaks for normal and
non-sync SPMD entries (`logical_block_num > 1`). A one-block entry is normal.
The two class peaks may occur at different times: **do not add them to derive
the total peak**. TMR preserves the enqueue-time class across tagged early
queue re-enqueues, including stale entries whose task storage has been reused.

## Instrumentation cost

- Queue slots remain 24 bytes and the ready-queue header remains 192 bytes,
  guarded by static assertions. Task descriptors acquire no profiling fields.
- Queue headers use existing padding for two pointers. Only the six classified
  queues allocate a one-byte-per-slot side array. At maximum configured
  capacities this is 98,496 bytes for HBG and 197,376 bytes for TMR.
- Each queue has one separate 64-byte counter block. Ordinary queues update
  current and total peak; classified queues additionally update two class
  peaks. Their current counts share one 64-bit atomic word.
- Each successful batch uses one atomic current-count update. Peak updates use
  a CAS only when a new maximum is observed. Failed operations add no counts.
  No per-entry events, timestamps, push/pop totals or hot-path logs are added.
- After all AICPU participants have stopped, one 1,600-byte POD report is
  published to a per-call slice of the existing retained temporary arena.
  Host readback synchronously appends a bounded set of records to
  `queue_peaks.<pid>.log` under `ASCEND_PROCESS_LOG_PATH` (falling back to
  the existing process log directory), including
  successful calls. This bypasses the bounded async trace queue.
  Epoch validation rejects missing or stale reports. The output header carries
  the existing execution return code so caught device failures cannot become
  successful occupancy samples.

These operations still perturb scheduling. Use the matched baseline timing
comparison to quantify their cost; the small data layout alone does not prove
negligible runtime overhead. Peak values are observations under instrumentation,
not guaranteed capacity requirements for every possible scheduling order.

## Collect on A3

Use a worktree-local virtual environment and the repository build instructions.
The runner performs the architecture precheck and uses `task-submit` for every
hardware invocation. Source the CANN environment before running it.

```bash
python tools/a3_queue_peaks.py --collect-only
python tools/a3_queue_peaks.py --jobs 6 --output outputs/a3-queue-peaks
python tools/a3_queue_peaks.py --summarize --output outputs/a3-queue-peaks
```

Collection covers applicable `examples/` and `tests/st/` entries for both
runtimes, including manual cases. Use `--runtime` or `--match` for a subset.
By default each case runs three rounds. Standard scene-test methods batch them
in one process; other tests get three independent starts. Golden comparisons
are disabled. A Network1 case without peer environment variables is recorded
as skipped. Device demands are taken from the case or pytest marker.

`inventory.json` retains the candidate set. `manifest.json` identifies the
baseline and instrumented source. Resume requires an identical source digest
and repeat count; use a new directory after changing instrumentation.
`results.jsonl` contains per-attempt status, per-invocation values and errors.
`case_peaks.csv` reports per-case maxima. `peaks.csv` records each global independent maximum with its capacity, case, repeat,
PID and epoch. `coverage.json` and `report.md` summarize the completed attempts.
Per-attempt logs remain under `runs/` for diagnosis; large successful stdout
logs are gzip-compressed. Use `--retry-from <prior-output>` with a new output
directory to rerun failed, incomplete or missing-data attempts without losing
the original evidence. Output directories are
ignored by git.

Only complete reports from successful test attempts and successful device
calls contribute to `peaks.csv`. Expected SDMA fault cases are kept separately.
A missing report is **not zero**. Failed runs retain partial observations for
diagnosis. The CSV capacity belongs to the same invocation as its maximum;
HBG logical capacity can vary across graphs and differs from reserved backing
storage. A single scalar maximum does not describe every capacity configuration.

## Validation

C++ tests exercise queue fullness, wraparound, mixed concurrent batches,
independent class peaks, stale tagged re-enqueues, reset, mailbox message kinds
and HBG completion queues. Python tests check incomplete reports, duplicate
logs, PID isolation, independent peak provenance and caught device failures.
Simulation and hardware smoke tests exercise report publication and readback
through both actual runtimes. Hardware corpus results and paired baseline
measurements belong in the experiment's output directory.
