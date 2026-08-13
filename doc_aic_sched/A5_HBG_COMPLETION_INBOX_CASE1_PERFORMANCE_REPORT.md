# A5 HBG Completion Inbox — Paged Attention Unroll Case1 Performance

## 结论

当前 `feat/hbg-completion-inbox` 在 A5 `PagedAttentionUnrollHostBuildGraphA5::Case1` 上完成同卡顺序 A/B。缓存 resolver 数量后，20 次无 profiling 的 `device_wall` 为：

- 均值 **1.364 ms**，P50 **1.355 ms**，P95 **1.418 ms**；
- 样本标准差 **0.042 ms**，95% 均值置信区间 **[1.344, 1.384] ms**；
- 最小/最大 **1.323/1.516 ms**。

同一实现、同一 device 0 的优化前 20 轮均值是 **1.706 ms**，优化后降低 **342.514 µs（-20.08%）**。Level-1 中 `CompletionEnqueue` 从 **21.983 µs** 降到 **0.558 µs（-97.46%，39.42x）**。根因是 A5 的 64-bit GM load 通过 `atomicAdd(addr, 0)` 实现；原实现每次 enqueue 都让 84 个 worker 竞争读取同一个只读 `aiv_active_worker_count`。现在每个 worker 在 startup 结束后只快照一次，并在 enqueue routing 和 inbox service 中复用。

producer 侧仍使用单次 `atomicExch` 入栈，由 AIV Scheduler 等待尚未发布的 `completion_next`。功能正确性由优化后的同一 Case1 golden run 验证通过。

## 测试信息

| 项目 | 值 |
|---|---|
| 分支 / commit | `feat/hbg-completion-inbox` / 基于 `2ebfae1c` 的当前 worktree |
| 用例 | `TestPagedAttentionUnrollHostBuildGraphA5::Case1` |
| 设备 | task-submit 分配 device 0；运行时识别 `Ascend950PR_9579` |
| 软件 | CANN `9.2.0`；PTOAS `0.24` |
| 缓存优化前任务 | `task_20260813_201115_4844626888`，20 轮 + Level-1 PASSED |
| 缓存优化后任务 | `task_20260813_201619_7917622428`，20 轮 + Level-1 + golden PASSED |
| 正确性任务 | 缓存优化后任务中的独立 golden run，PASSED |
| 原始热点泳道任务 | `task_20260813_200324_1990327526`，PASSED |
| Scheduler 计数任务 | `task_20260813_195304_415802111207`，golden PASSED |
| 采样协议 | 同一卡锁内独立进程预热 1 次，随后新进程 20 轮；`--skip-golden`；profiling 关闭 |
| 架构预检 | 按用户授权使用 `--force`；保留 task-submit 卡锁。实际日志识别到 A5 SoC |

性能命令的核心参数：

```bash
python tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
  -p a5 -d "$TASK_DEVICE" --case Case1 --rounds 20 --skip-golden
```

## 缓存优化同卡 A/B

单位均为微秒。P95 使用线性插值；95% CI 使用 t 分布（df=19）。

| `device_wall` | Mean | Std | Min | P50 | P95 | Max | Mean 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 优化前 | 1706.143 | 44.371 | 1653.791 | 1700.454 | 1753.453 | 1861.581 | [1685.377, 1726.909] |
| 优化后 | **1363.629** | 42.480 | 1322.724 | 1355.372 | 1417.945 | 1515.855 | [1343.748, 1383.510] |

两组都在同一 device 0 上按“1 次 warmup、随后新进程 20 轮”的协议采集，profiling 关闭。优化后均值降低 **342.514 µs（-20.075%）**。这是顺序 A/B，不是交替配对采样；差值大于两组均值置信区间，足以支持本热点优化有效，但不替代长期性能基线。

## `CompletionEnqueue` 具体构成

临时 cycle 打点采集 5 次 Case1、共 5120 次 enqueue。打点后的总时长均值为 22.904 µs，存在轻微观察者效应；内部累计分布如下：

| 内部步骤 | Avg | 占打点总时长 |
|---|---:|---:|
| 读取 `run_control->aiv_active_worker_count`（64-bit GM atomic load） | **22.197 µs** | **96.91%** |
| 写 `task_control.state = DONE` 及地址计算 | 0.166 µs | 0.72% |
| 计算 inbox / metadata | 0.008 µs | 0.03% |
| `atomicExch(inbox->head, task_id)` | 0.002 µs | 0.01% |
| 写 `completion_next` | 0.309 µs | 1.35% |
| `dcci + dsb` 发布 cache line | 0.220 µs | 0.96% |
| task id 检查及未归类 | 0.002 µs | 0.01% |

热点并不在 inbox 的 `atomicExch`，而是所有 worker 对同一地址的共享原子读。缓存该只读值以后，独立 Level-1 capture 的 `CompletionEnqueue` 分布如下：

| 版本 | Count | Mean | P50 | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 优化前 | 1024 | 21.983 µs | 21.949 µs | 34.261 µs | 1.006 µs | 51.333 µs |
| 优化后 | 1024 | **0.558 µs** | 0.468 µs | 1.094 µs | 0.276 µs | 4.435 µs |

## 优化前 Level-1 芯片泳道

泳道 capture 与性能采样分开执行，避免 profiling 开销污染上述 20 轮结果。依赖图与泳道来自同一次 replay，离线转换时合入 1024 条任务依赖 flow。

```text
Dependency: QK (AIC) ─▶ SF (AIV) ─┬─▶ PV (AIC)
                                  └─▶ UP (AIV)

AIC worker: Claim → Kernel → CompletionEnqueue
AIV worker: Claim → Kernel → CompletionEnqueue → CompletionBatchClaim → WakeResolve → ReadyPublish
```

采集包含 **1024 个 AICore task**、**84 个物理核**和 **1024 条可渲染依赖 flow**。kernel 阶段跨度为 **1.416 ms**，转换器观测到的 AICore span 为 **1.424 ms**，所有 kernel 执行时间之和为 39.761 ms。全部 **1024 个 `CompletionBatchClaim` 和 1024 个 `WakeResolve` 均位于 56 个 AIV Scheduler lane**；AIC resolver 数为 0。

| 核类型 | 核数 | Task 数 | 执行时间总和 | lane 利用率（观测窗口内） |
|---|---:|---:|---:|---:|
| AIC | 28 | 512 | 24.526 ms | 61.84% |
| AIV | 56 | 512 | 15.235 ms | 19.21% |

| Kernel | Count | Avg kernel exec | Avg local setup |
|---|---:|---:|---:|
| QK | 256 | 55.40 µs | 0.50 µs |
| SF | 256 | 56.98 µs | 27.88 µs |
| PV | 256 | 40.41 µs | 0.00 µs |
| UP | 256 | 2.53 µs | 0.00 µs |

泳道中的 completion-inbox 相关 AICore marker：

| Marker | Count | Avg | P95 | Max |
|---|---:|---:|---:|---:|
| `CompletionEnqueue` | 1024 | 21.386 µs | 32.228 µs | 39.520 µs |
| `CompletionBatchClaim` | 1024 | 0.000 µs | 0.000 µs | 0.000 µs |
| `WakeResolve` | 1024 | 1.690 µs | 3.386 µs | 5.921 µs |
| `ReadyPublish` | 1024 | 0.000 µs | 0.000 µs | 0.000 µs |
| `TicketClaim` | 940 | 0.758 µs | 1.634 µs | 20.539 µs |
| `SeedClaim` | 84 | 1.346 µs | 1.742 µs | 2.371 µs |
| `Drain` | 84 | 0.196 µs | 0.229 µs | 0.526 µs |

这些 marker 来自 Level-1 诊断 capture，包含 profiler 的观察者效应，适合定位相对热点，不应直接加总为无 profiling 的 `device_wall`。
`CompletionBatchClaim` 和 `ReadyPublish` 是状态转换时间点，当前仅记录单个 timestamp，因此显示为零时长 marker。
`completion_next` 的等待发生在 resolver 摘取 inbox head 之后、写入 `completion_resolve_start_cycles` 之前，因此不会显示为 `CompletionBatchClaim` 区间。独立 info 诊断 run 的 runtime 计数为 `completion_link_waits=0`、`completion_link_wait_max=0`。

## 泳道文件

- [Perfetto 完整泳道](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_195157/merged_swimlane.json)：在 `ui.perfetto.dev` 中拖入即可查看 84 条物理核 lane、task bar、AICore scheduler marker 和依赖箭头。
- [Level-1 原始记录](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_195157/chip_swimlane_records.json)
- [dep-gen 依赖图](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_195157/deps.json)
- [缓存优化前 Level-1](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_201134/chip_swimlane_records.json)
- [缓存优化后 Level-1](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_201852/chip_swimlane_records.json)
- [缓存优化前 20 轮日志](../tmp/completion-cache-ab/before_rounds20.log)
- [缓存优化后 20 轮日志](../tmp/completion-cache-ab/after_rounds20.log)
- [5 轮内部打点日志](../tmp/completion-enqueue-breakdown/fine_rounds5.log)

## 限制与风险

- 缓存优化前后是同卡顺序 A/B，不是交替配对采样；结论限定为本次 Case1 和当前 worktree，不外推为仓库级基线。
- Level-1 capture 的 `device_wall` 为 2.016 ms，高于无 profiling 均值 1.697 ms；这是预期的采集开销，不能与无 profiling 结果直接混用。
- 转换器对 1024 条依赖 flow 标记了 56 条 happens-before violation；依赖箭头可用于拓扑定位，但这些跨核时间关系不应作为严格的周期顺序证明。
