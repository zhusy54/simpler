# A5 HBG Completion Inbox — Paged Attention Unroll Case1 Performance

## 结论

当前 `feat/hbg-completion-inbox` 在 A5 `PagedAttentionUnrollHostBuildGraphA5::Case1` 上完成 1 次预热和 20 次无 profiling 测量，全部通过。核心指标 `device_wall` 为：

- 均值 **1.777 ms**，P50 **1.776 ms**，P95 **1.814 ms**；
- 标准差 **0.032 ms**，95% 均值置信区间 **[1.762, 1.792] ms**；
- 最小/最大 **1.732/1.872 ms**。

按用户要求，本报告跳过基线，只给出当前分支的绝对性能，不计算提升比例。功能正确性已由同一 Case1 的 golden run 验证通过。

## 测试信息

| 项目 | 值 |
|---|---|
| 分支 / commit | `feat/hbg-completion-inbox` / `d09e7137370fd6e083b5154e0029fff40bced764` |
| 用例 | `TestPagedAttentionUnrollHostBuildGraphA5::Case1` |
| 设备 | task-submit device 2；运行时识别 `Ascend950PR_9579` |
| 软件 | CANN `9.2.0`；PTOAS `0.24` |
| 性能任务 | `task_20260813_171939_222001325862` |
| 正确性任务 | `task_20260813_171325_193145516515`，golden PASSED |
| 采样协议 | 同一卡锁内预热 1 次，随后 20 轮；`--skip-golden`；profiling 关闭 |
| 架构预检 | 按用户授权使用 `--force`；保留 task-submit 卡锁。实际日志识别到 A5 SoC |

性能命令的核心参数：

```bash
python tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
  -p a5 -d "$TASK_DEVICE" --case Case1 --rounds 20 --skip-golden
```

## 20 轮性能

单位均为微秒。P95 使用线性插值；95% CI 使用 t 分布（df=19）。

| 指标 | Mean | Std | Min | P50 | P95 | Max | Mean 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| `device_wall` | **1777.136** | 31.741 | 1731.812 | 1775.500 | 1814.139 | 1871.597 | [1762.281, 1791.991] |
| `runner_run` | 2090.660 | 428.868 | 1944.048 | 2002.356 | 2141.899 | 3907.084 | [1889.946, 2291.375] |
| `bind.ticket_stream_plan` | 30.499 | 7.633 | 24.687 | 27.732 | 46.067 | 56.124 | [26.927, 34.071] |
| `bind` | 193104.651 | 20375.378 | 164660.813 | 192648.955 | 222955.493 | 233031.401 | [183568.790, 202640.512] |
| `simpler_run` | 205964.756 | 21984.906 | 172572.938 | 206788.708 | 241271.638 | 241328.124 | [195675.622, 216253.890] |

`simpler_run` 每轮包含 host bind，因而约 206 ms 的端到端时间主要由约 193 ms 的 bind 决定。completion-inbox 调度热路径应以 `device_wall` 为主；20 轮中该指标的变异系数为 1.79%。

## Level-1 芯片泳道

泳道 capture 与性能采样分开执行，避免 profiling 开销污染上述 20 轮结果。依赖图也通过独立 dep-gen run 采集，再在离线转换时合入。

```text
Dependency: QK (AIC) ─▶ SF (AIV) ─┬─▶ PV (AIC)
                                  └─▶ UP (AIV)

AICore worker: Claim → Kernel → CompletionEnqueue → WakeResolve → ReadyPublish
```

采集包含 **1024 个 AICore task**、**84 个物理核**和 **1024 条可渲染依赖 flow**。原始 task 执行窗口为 **1.555 ms**（18.401–1572.976 µs），所有 kernel 执行时间之和为 40.318 ms。

| 核类型 | 核数 | Task 数 | 执行时间总和 | lane 利用率（观测窗口内） |
|---|---:|---:|---:|---:|
| AIC | 28 | 512 | 25.081 ms | 57.62% |
| AIV | 56 | 512 | 15.237 ms | 17.50% |

| Kernel | Count | Avg kernel exec | Avg local setup |
|---|---:|---:|---:|
| QK | 256 | 55.86 µs | 0.61 µs |
| SF | 256 | 57.02 µs | 42.93 µs |
| PV | 256 | 42.11 µs | 0.00 µs |
| UP | 256 | 2.50 µs | 0.00 µs |

泳道中的 completion-inbox 相关 AICore marker：

| Marker | Count | Avg | P95 | Max |
|---|---:|---:|---:|---:|
| `CompletionEnqueue` | 1024 | 21.483 µs | 31.828 µs | 43.922 µs |
| `WakeResolve` | 1024 | 1.628 µs | 3.189 µs | 6.788 µs |
| `TicketClaim` | 940 | 0.663 µs | 1.432 µs | 13.876 µs |
| `SeedClaim` | 84 | 1.571 µs | 2.504 µs | 2.712 µs |
| `Drain` | 84 | 0.274 µs | 0.326 µs | 4.354 µs |

这些 marker 来自 Level-1 诊断 capture，包含 profiler 的观察者效应，适合定位相对热点，不应直接加总为无 profiling 的 `device_wall`。

## 泳道文件

- [Perfetto 完整泳道](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172008/merged_swimlane_completion_inbox.json)：在 `ui.perfetto.dev` 中拖入即可查看 84 条物理核 lane、task bar、AICore scheduler marker 和依赖箭头。
- [Level-1 原始记录](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172008/chip_swimlane_records.json)
- [dep-gen 依赖图](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172001/deps.json)
- [Observed critical-path trace](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172008/CPM_observed.json)
- [关键路径工具报告](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172008/critical_path_report.md)
- [20 轮原始日志](../tmp/completion-inbox-perf/current_rounds20.log)

## 限制与风险

- 本次按要求没有执行基线，不能据此声明相对加速或回归。
- dep-gen 与泳道为两次独立 replay。转换器保留全部 1024 个 task bar，但标记了 56 条跨 replay 的 happens-before 时间次序告警。
- `critical_path` 后处理因上述跨 replay 次序过滤只纳入 84 个 task / 22 条边；其 CPM 数字仅供诊断，不作为正式性能结论。
- Level-1 capture 的 `device_wall` 为 2.188 ms，高于无 profiling 均值 1.777 ms；这是预期的采集开销，不能与无 profiling 结果直接混用。
