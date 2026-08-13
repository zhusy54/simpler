# A5 HBG Completion Inbox — Paged Attention Unroll Case1 Performance

## 结论

当前 `feat/hbg-completion-inbox` 在 A5 `PagedAttentionUnrollHostBuildGraphA5::Case1` 上完成 1 次预热和 20 次无 profiling 测量，全部通过。核心指标 `device_wall` 为：

- 均值 **1.703 ms**，P50 **1.696 ms**，P95 **1.738 ms**；
- 标准差 **0.035 ms**，95% 均值置信区间 **[1.686, 1.719] ms**；
- 最小/最大 **1.666/1.818 ms**。

按用户要求，本报告跳过基线，只给出当前分支的绝对性能，不计算提升比例。功能正确性已由同一 Case1 的 golden run 验证通过。

## 测试信息

| 项目 | 值 |
|---|---|
| 分支 / commit | `feat/hbg-completion-inbox` / `e443734f01fc53127b3b853e843beecf6df9724a` |
| 用例 | `TestPagedAttentionUnrollHostBuildGraphA5::Case1` |
| 设备 | 性能：device 2；泳道/golden：device 1；运行时识别 `Ascend950PR_9579` |
| 软件 | CANN `9.2.0`；PTOAS `0.24` |
| 性能任务 | `task_20260813_174420_250966220481` |
| 正确性/泳道任务 | `task_20260813_174339_24973392516`，golden PASSED |
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
| `device_wall` | **1702.883** | 35.338 | 1666.331 | 1696.027 | 1737.893 | 1818.052 | [1686.345, 1719.422] |
| `runner_run` | 1957.207 | 276.491 | 1830.698 | 1894.929 | 1997.628 | 3126.593 | [1827.806, 2086.607] |
| `bind.ticket_stream_plan` | 27.261 | 1.776 | 25.478 | 26.890 | 29.553 | 32.949 | [26.430, 28.093] |
| `bind` | 190553.955 | 17996.483 | 167220.895 | 190294.576 | 217590.212 | 240696.527 | [182131.438, 198976.472] |
| `simpler_run` | 199310.017 | 19203.935 | 175385.429 | 198634.264 | 226339.180 | 256796.540 | [190322.402, 208297.631] |

`simpler_run` 每轮包含 host bind，因而约 199 ms 的端到端时间主要由约 191 ms 的 bind 决定。completion-inbox 调度热路径应以 `device_wall` 为主；20 轮中该指标的变异系数为 2.08%。

## Level-1 芯片泳道

泳道 capture 与性能采样分开执行，避免 profiling 开销污染上述 20 轮结果。依赖图也通过独立 dep-gen run 采集，再在离线转换时合入。

```text
Dependency: QK (AIC) ─▶ SF (AIV) ─┬─▶ PV (AIC)
                                  └─▶ UP (AIV)

AIC worker: Claim → Kernel → CompletionEnqueue
AIV worker: Claim → Kernel → CompletionEnqueue → CompletionBatchClaim → WakeResolve → ReadyPublish
```

采集包含 **1024 个 AICore task**、**84 个物理核**和 **1024 条可渲染依赖 flow**。原始 task 执行窗口为 **1.465 ms**（10.596–1475.770 µs），所有 kernel 执行时间之和为 39.502 ms。全部 **1024 个 `CompletionBatchClaim` 和 1024 个 `WakeResolve` 均位于 56 个 AIV Scheduler lane**；AIC resolver 数为 0。

| 核类型 | 核数 | Task 数 | 执行时间总和 | lane 利用率（观测窗口内） |
|---|---:|---:|---:|---:|
| AIC | 28 | 512 | 24.298 ms | 59.23% |
| AIV | 56 | 512 | 15.205 ms | 18.53% |

| Kernel | Count | Avg kernel exec | Avg local setup |
|---|---:|---:|---:|
| QK | 256 | 56.25 µs | 0.53 µs |
| SF | 256 | 56.89 µs | 31.89 µs |
| PV | 256 | 38.66 µs | 0.00 µs |
| UP | 256 | 2.50 µs | 0.00 µs |

泳道中的 completion-inbox 相关 AICore marker：

| Marker | Count | Avg | P95 | Max |
|---|---:|---:|---:|---:|
| `CompletionEnqueue` | 1024 | 23.236 µs | 35.085 µs | 73.932 µs |
| `CompletionBatchClaim` | 1024 | 0.000 µs | 0.000 µs | 0.000 µs |
| `WakeResolve` | 1024 | 1.664 µs | 3.110 µs | 7.199 µs |
| `ReadyPublish` | 1024 | 0.000 µs | 0.000 µs | 0.000 µs |
| `TicketClaim` | 940 | 0.473 µs | 0.969 µs | 6.517 µs |
| `SeedClaim` | 84 | 1.382 µs | 1.813 µs | 2.242 µs |
| `Drain` | 84 | 0.217 µs | 0.387 µs | 0.671 µs |

这些 marker 来自 Level-1 诊断 capture，包含 profiler 的观察者效应，适合定位相对热点，不应直接加总为无 profiling 的 `device_wall`。
`CompletionBatchClaim` 和 `ReadyPublish` 是状态转换时间点，当前仅记录单个 timestamp，因此显示为零时长 marker。

## 泳道文件

- [Perfetto 完整泳道](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_174346/merged_swimlane_aiv_resolver.json)：在 `ui.perfetto.dev` 中拖入即可查看 84 条物理核 lane、task bar、AICore scheduler marker 和依赖箭头。
- [Level-1 原始记录](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_174346/chip_swimlane_records.json)
- [dep-gen 依赖图](../outputs/TestPagedAttentionUnrollHostBuildGraphA5_Case1_20260813_172001/deps.json)
- [20 轮原始日志](../tmp/aiv-only-resolver-hw/perf_rounds20.log)

## 限制与风险

- 本次按要求没有执行基线，不能据此声明相对加速或回归。
- dep-gen 与泳道为两次独立 replay。转换器保留全部 1024 个 task bar，但标记了 56 条跨 replay 的 happens-before 时间次序告警。
- Level-1 capture 的 `device_wall` 为 2.063 ms，高于无 profiling 均值 1.703 ms；这是预期的采集开销，不能与无 profiling 结果直接混用。
