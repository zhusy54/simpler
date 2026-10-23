# Per-case A3 queue peak data

This directory publishes the per-case detail behind the
[measurement report](../a3-runtime-queue-peaks-results.md). The snapshot has
**7,759 rows across 302 cases with valid observations**, partitioned by runtime
and queue. HBG means `host_build_graph`; TMR means
`tensormap_and_ringbuffer`.

It is the same snapshot as the report's global-maxima CSV, including valid
diagnostic retries. It does not incorporate the pending final retry campaign.
Cases without valid reports are absent, not represented by zero-valued rows.
A recorded zero means that metric was zero in the valid observations available
for that case; it does not establish that the queue can never be used.

## Files

| Runtime | Queue / CSV | Rows |
| ------- | ----------- | ---- |
| HBG | [completion](hbg-completion.csv) | 301 |
| HBG | [dummy](hbg-dummy.csv) | 105 |
| HBG | [early_dispatch](hbg-early_dispatch.csv) | 945 |
| HBG | [early_sync_start](hbg-early_sync_start.csv) | 105 |
| HBG | [ed_publish_drain](hbg-ed_publish_drain.csv) | 105 |
| HBG | [graph_prepare](hbg-graph_prepare.csv) | 105 |
| HBG | [mailbox](hbg-mailbox.csv) | 105 |
| HBG | [ready](hbg-ready.csv) | 945 |
| HBG | [ready_sync](hbg-ready_sync.csv) | 315 |
| TMR | [dummy](tmr-dummy.csv) | 197 |
| TMR | [early_dispatch](tmr-early_dispatch.csv) | 1773 |
| TMR | [early_sync_start](tmr-early_sync_start.csv) | 197 |
| TMR | [mailbox](tmr-mailbox.csv) | 197 |
| TMR | [ready](tmr-ready.csv) | 1773 |
| TMR | [ready_sync](tmr-ready_sync.csv) | 591 |

## Row definition

Each row is the maximum for one `(runtime, case, queue, index, metric)` group
across its successful, complete observations. The remaining fields identify
an invocation that attained that maximum.

- `runtime`: runtime implementation.
- `case`: runtime-prefixed pytest node ID and, where applicable, scene case name.
- `queue`: queue name.
- `index`: AIC=0, AIV=1, MIX=2 for shape queues; scheduler thread index for
  completion queues; -1 for an unsharded queue.
- `metric`: `peak` is total occupancy; `spmd` and `normal` are independent
  class occupancy maxima in ready and early-dispatch queues.
- `value`: maximum number of queue entries, not number of SPMD blocks.
- `capacity`: logical capacity in the invocation that attained this maximum.
  HBG capacities may differ across cases.
- `repeat`: independent process repeat, or `1-3` for three batched rounds.
- `pid`, `epoch`: process and per-call report identifiers within that attempt.

The three metrics can peak in different invocations: **do not add class peaks
to derive the total peak**. An SPMD entry has `logical_block_num > 1`; a
one-block entry counts as normal. Refer to the
[measurement contract](../a3-runtime-queue-peaks.md) for accounting boundaries.

The files retain identical columns and can be concatenated while keeping one
header. The partition only makes the data easier to inspect; it does not
change the aggregation.
