# A5 Paged Attention Case1 性能基线与优化复测报告

**测试日期**：2026-08-11（基线诊断）、2026-08-12（ticket/pending 与 wake-list + cursor 复测）

**测试目的**：量化 HBG AICore scheduler 各代实现的 Case1 性能和可运行性差异，并建立后续调优的可重复基线。

**当前结论**：wake-list + cursor scheduler 的 profiling-off Device wall 为 **35.045 ± 0.069 ms**，10/10 个样本全部成功。它消除了最新 `main` 上固定 8,192 项 ReadyQ 导致的 Case1 容量失败，但相对直接改动前的 ticket cursor + worker-private pending scheduler（29.350 ± 0.088 ms）增加 **5.695 ms / 19.404%**。两组数据分别来自 device 2 和 device 0，因此 19.404% 是方向性回退信号，最终归因仍需在同一卡上复测改动前提交。相对更早的 ReadyQ/CompletionQ + resolver 实现（522.411 ms），当前 Device wall 仍降低 **93.292%**，加速 **14.91x**。

## 0. Wake-list + cursor scheduler 复测（2026-08-12）

本轮验证旧版 wake list 与无 ReadyQ/CompletionQ 的 cursor 调度组合。被测工作树为 `884ec07f985ef35a9b5047562748e2c0e3c8a5fd` 上的未提交 wake-list + cursor 改动，PTO-ISA pin 为 `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`。Case1 使用 512 MiB ring heap 和 131,072 task window，在用户确认的 A5 上通过 `task-submit` 独占 device 2 运行 10 轮；运行时拓扑回退实际识别到 `soc=Ascend950PR_9579`。

### 0.1 运行结果

单位为 ms，`±` 后为样本标准差，均为 `n=10`。

| 指标 | 均值 ± sd | p50 | p95 | 最小值 | 最大值 |
| ---- | --------: | --: | --: | -----: | -----: |
| Device wall | **35.045 ± 0.069** | 35.067 | 35.148 | 34.912 | 35.148 |
| Runner | 35.364 ± 0.354 | 35.289 | 36.352 | 35.110 | 36.352 |
| Bind | 453.503 ± 49.432 | 451.567 | 575.105 | 408.913 | 575.105 |
| 整个 `simpler_run` | 499.520 ± 52.634 | 496.271 | 631.596 | 453.464 | 631.596 |

Device wall 的 10 个原始样本为：35.148、35.024、34.956、35.081、34.912、35.079、35.023、35.065、35.069、35.095 ms。范围仅为 0.236 ms，约占均值的 0.67%，设备执行结果稳定。Host 总时间主要由 bind 阶段构成且波动明显，不适合作为 scheduler 性能主指标。本次 HBG marker 没有产生 Effective/Orch/Sched 分项，因此不对这些指标作推断。

### 0.2 与直接改动前实现对比

直接改动前实现是提交 `f8cd1345` 对应的 ticket cursor + worker-private pending scheduler。其 10 轮正式样本来自本报告前一轮测试；两种实现的 Case1 参数、ring 配置和 PTO-ISA pin 相同。

| 指标 | 改动前 ticket/pending | Wake-list + cursor | 差值 | 变化 |
| ---- | --------------------: | -----------------: | ---: | ---: |
| 成功率 | 10/10 | 10/10 | 0 | 相同 |
| Device wall mean ± sd | **29.350 ± 0.088** | 35.045 ± 0.069 | +5.695 ms | **+19.404%** |
| Device wall p50 | **29.322** | 35.067 | +5.745 ms | +19.593% |
| Device wall p95 | **29.565** | 35.148 | +5.583 ms | +18.884% |
| Runner mean ± sd | **30.540 ± 0.201** | 35.364 ± 0.354 | +4.824 ms | +15.796% |

该对比显示 wake-list + cursor 保持了前序方案的稳定 forward progress，但设备执行延迟回退约 19.4%。改动前样本使用 device 0，当前样本使用 device 2，且当前工作树包含后续依赖图导出改动，所以这组数据不能把全部差异严格归因于 wake-list。由于差异远大于通常采用的 ±2% 噪声区间，应先视为需要处理的性能回退，并在同一 device 2 上检出 `f8cd1345` 后按相同命令复测。

### 0.3 与最新 `main` 的可运行性对比

最新 `main` `50c0660646217f1b56172a81990279bd066ed971` 在同一 device 2 上首轮失败，没有可用性能样本。设备分类错误为 `sched_error_code=104 READY_QUEUE_OVERFLOW`；该实现的 ReadyQ 固定容量为 8,192，而 Case1 的 ready task 峰值超过此容量。调整 task window 不能在不改变主线实现的前提下同时容纳完整 host-built graph 并解决固定 ReadyQ 容量，因此不应把失败前的 Host 或 Runner span 当作性能数据。

| 版本 | PTO-ISA pin | 设备 | 结果 | Device wall |
| ---- | ----------- | ---: | ---- | ----------: |
| 最新 `main` `50c06606` | `0cefc9a5a1c24c62655cc345d408559595a8af32` | 2 | 首轮 ReadyQ overflow | 无有效样本 |
| Wake-list + cursor | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` | 2 | 10/10 通过 | **35.045 ± 0.069 ms** |

两者 PTO-ISA pin 不同，因此即使主线能够完成 Case1，也需要统一 pin 后才能做严格延迟对比。本轮可以确认的是容量和可运行性改善，不能计算相对最新 `main` 的性能提升百分比。

## 1. Ticket/pending scheduler 优化复测（2026-08-12）

本轮专门比较共享 ReadyQ/CompletionQ + resolver 与 ticket cursor + worker-private pending 两种调度器。两者都基于 Git `000903a4f01c0458fb0353a43890486b3a60a358`；旧实现从该提交的独立 worktree 构建，当前实现为同一提交上的未提交工作树改动。两边使用相同 PTO-ISA pin `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`、Case1 参数和 512 MiB ring heap。

硬件为用户确认的 A5，`task-submit` 在整个对比期间独占 device 0。每个版本预热 1 次，随后按交错顺序各运行 10 个 profiling-off 正式样本；20/20 个正式样本全部通过，均完成 65,536 个 AICore task。

### 1.1 正式性能结果

单位为 ms，`±` 后为样本标准差，均为 `n=10`。

| 指标 | 旧 ReadyQ/CompletionQ | Ticket/pending | 差值 | 改善 |
| ---- | --------------------: | -------------: | ---: | ---: |
| Device wall mean ± sd | 522.411 ± 18.045 | **29.350 ± 0.088** | -493.062 | **-94.382% / 17.80x** |
| Device wall p50 | 523.573 | **29.322** | -494.251 | -94.396% |
| Device wall p95 | 552.928 | **29.565** | -523.363 | -94.653% |
| Runner mean ± sd | 523.692 ± 18.099 | **30.540 ± 0.201** | -493.152 | -94.168% |
| Bind mean ± sd | 466.655 ± 24.604 | 474.196 ± 36.445 | +7.541 | +1.616% |
| 整个 `simpler_run` mean ± sd | 1009.688 ± 33.159 | **522.612 ± 36.459** | -487.076 | -48.240% |

当前 Device wall 的 10 个原始样本为：29.565、29.395、29.313、29.319、29.314、29.369、29.230、29.325、29.364、29.302 ms。逐轮配对改善均位于 94.09% 至 94.65%，未出现 stall 或长尾回退。

### 1.2 Level-1 trace 验证

另运行 1 次启用 level-1 profiling 的当前实现，Case1 通过，Device wall 为 29.930 ms。Runner 为 145.921 ms，明显高于 profiling-off；增加部分来自执行后的 profiling 记录收集与 JSON 导出，不应计入设备执行改善。

| Scheduler phase | 记录数 | 累计 worker core-time | 单次均值 |
| --------------- | -----: | --------------------: | -------: |
| SeedClaim | 84 | 0.108 ms | 1.281 us |
| TicketClaim | 65,452 | 18.454 ms | 0.282 us |
| PendingWait | 32,854 | 4,467.074 ms | 135.967 us |
| Payload | 65,536 | 72.038 ms | 1.099 us |
| Kernel | 65,536 | 93.315 ms | 1.424 us |
| CompletionPublish | 65,536 | 18.237 ms | 0.278 us |
| Drain | 84 | 0.015 ms | 0.181 us |

全部 scheduler phase 的墙钟观测跨度为 29.464 ms。`PendingWait` 是 84 个 worker 的累计 core-time，存在并行重叠，不能与 Device wall 直接相加。28 个 AIC worker 合计执行 32,768 个 AIC 任务，平均每 worker 1,170.3 个；56 个 AIV worker 合计执行 32,768 个 AIV 任务，平均每 worker 585.1 个。Trace 中没有旧的 `aicore_resolve_phases`，65,536 个唯一 task id 均有 Kernel 和 CompletionPublish 记录。

## 2. 历史集成分支基线版本

第 2 至第 7 节记录 2026-08-11 的历史基线，其中 `main` 指 SHA `4f417732`，不是第 0.3 节测试的最新 `main` `50c06606`。

| 项目 | `main` 基线 | HBG AICore 集成分支 |
| ---- | ----------- | ------------------- |
| Git SHA | `4f4177324f9afbe5fdd811a667015cd375131395` | `2fb5f613`（运行时代码截至 `43b79c36`） |
| 分支 | `upstream/main` | `feat/hbg-aicore-dependency-integration` |
| PTO-ISA pin | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` | 相同 |
| 测试文件 | `tests/st/a5/host_build_graph/paged_attention/test_paged_attention.py` | 相同 |
| Case | `Case1` | `Case1` |

`2fb5f613` 只把 Case1 所需 ring 配置写入 Python case，没有修改运行时代码：

```text
ring_task_window = 131072
ring_heap = 512 * 1024 * 1024
```

## 3. 测试条件与指标口径

- 硬件：A5，固定使用 device 2。
- Case1 参数：batch 256、16 heads、1 KV head、head dimension 128、block size 128、context length 8192、max model length 32768、BF16。
- Golden：`--skip-golden`，避免参考计算影响计时。
- Profiling：关闭，或 `--enable-chip-swimlane 1`。
- 主线：预热 1 次；profiling-off 5 次、profiling-on 5 次，采用交错顺序并在同一次 `task-submit` 设备锁内执行。
- 集成分支：由于间歇性 scheduler stall 和 profiling 导出失败，当前只有 profiling-off/on 各 1 个有效成功样本。
- `Device wall`：`simpler_run.runner_run.device_wall`，表示 NPU 上 AICPU run 的设备墙钟时间，是本报告的主性能指标。
- `Runner`：主机侧 `simpler_run.runner_run`，包含设备运行及 profiling 记录收集/导出。
- `simpler_run`：包含 bind、runner 和 validate 的整个调用时间，容易受进程初始化与 bind 波动影响。

为避免普通超时掩盖真实执行时间，测试还使用了 120 s op timeout、130 s stream sync timeout，并开启 `SIMPLER_HOST_STRACE=1`。所有硬件运行均经 `task-submit` 独占设备。

## 4. 历史 `main` 重复测试结果

单位均为 ms，`±` 后为样本标准差，`n=5`。

| 指标 | Profiling off | Profiling level 1 | Profiling 增量 | 变化率 |
| ---- | ------------: | ----------------: | -------------: | -----: |
| Device wall | 91.982 ± 1.902 | 88.294 ± 0.809 | -3.688 | -4.01% |
| Runner | 93.046 ± 1.923 | 218.889 ± 5.326 | +125.843 | +135.25% |
| 整个 `simpler_run` | 654.591 ± 39.948 | 800.772 ± 33.660 | +146.181 | +22.33% |

Device wall 的 -4.01% 应视为运行波动，不能解释为 profiling 提升了 kernel 或运行时性能。主线没有出现 profiling 导致设备执行变慢的信号；可观测开销主要位于设备执行后的记录收集和导出：

```text
(Runner - Device wall) profiling on
- (Runner - Device wall) profiling off
= 129.531 ms
```

### 4.1 主线原始样本

| 样本 | Profiling | Device wall | Runner | `simpler_run` |
| ---- | --------- | ----------: | -----: | ------------: |
| 1 | off | 88.629 | 89.664 | 680.574 |
| 2 | off | 92.825 | 94.107 | 625.273 |
| 3 | off | 92.571 | 93.560 | 603.849 |
| 4 | off | 92.537 | 93.548 | 701.777 |
| 5 | off | 93.347 | 94.353 | 661.481 |
| 1 | level 1 | 88.612 | 215.555 | 819.289 |
| 2 | level 1 | 87.164 | 215.057 | 833.608 |
| 3 | level 1 | 89.361 | 218.293 | 750.268 |
| 4 | level 1 | 88.349 | 217.423 | 817.082 |
| 5 | level 1 | 87.984 | 228.118 | 783.615 |

主线有效运行 11/11 成功，包括 1 次预热和 10 次正式样本。

## 5. 集成分支结果

| 指标 | Profiling off | Profiling level 1 | Profiling 增量 | 变化率 |
| ---- | ------------: | ----------------: | -------------: | -----: |
| Device wall | 515.690 | 547.072 | +31.382 | +6.09% |
| Runner | 516.855 | 687.030 | +170.175 | +32.93% |
| 整个 `simpler_run` | 1155.791 | 1421.757 | +265.966 | +23.01% |

这些数值各来自一个成功样本。不能用它们建立稳定的 profiling 开销置信区间，原因如下：

- 一个相同的 profiling-off 运行停在 65,778/65,792 个 runtime task，最终触发 120 s op timeout。
- 一个 profiling level 1 运行完成设备执行，但丢失 45,078 条 AICore profiling record，严格校验拒绝导出并返回 `-10001`。
- 使用提交后 512 MiB embedded ring 配置进行验证时，scheduler 停在 44,806/65,792 个 runtime task。

成功的集成分支性能样本使用了外部设置的 4 GiB ring heap；主线重复测试使用与当前提交相同的 512 MiB ring heap。因此当前主线与分支对比还不是严格的配置对称实验。Device wall 和 AICore span 的差距很大，足以构成强回退信号，但最终结论仍需在集成分支稳定后用相同 512 MiB 配置复测。

## 6. 历史 `main` 与集成分支性能差异

### 6.1 Profiling off

| 指标 | `main` 均值 | 集成分支样本 | 集成分支增量 | `main` 加速比 |
| ---- | ----------: | -----------: | -----------: | ------------: |
| Device wall | 91.982 | 515.690 | +423.708 ms / +460.64% | 5.61x |
| Runner | 93.046 | 516.855 | +423.809 ms / +455.49% | 5.55x |
| 整个 `simpler_run` | 654.591 | 1155.791 | +501.200 ms / +76.57% | 1.77x |

### 6.2 Profiling level 1

| 指标 | `main` 均值 | 集成分支样本 | 集成分支增量 | `main` 加速比 |
| ---- | ----------: | -----------: | -----------: | ------------: |
| Device wall | 88.294 | 547.072 | +458.778 ms / +519.60% | 6.20x |
| Runner | 218.889 | 687.030 | +468.141 ms / +213.87% | 3.14x |
| 整个 `simpler_run` | 800.772 | 1421.757 | +620.985 ms / +77.55% | 1.78x |

整体 `simpler_run` 被约 0.5 s 的 bind 阶段稀释，因此调优时应优先比较 Device wall，而不是进程总时间。

## 7. AICore swimlane 对比

| Trace 指标 | `main` | 集成分支 | 差异 |
| ---------- | -----: | -------: | ---: |
| AICore task 数 | 65,536 | 65,536 | 相同 |
| 单 task 平均执行时间 | 1.69 us | 1.50 us | 集成分支约低 11.2% |
| task 执行时间之和 | 110.521 ms | 98.352 ms | 集成分支约低 11.0% |
| AICore 观测跨度 | 73.953 ms | 404.186 ms | 集成分支高 446.5%，跨度为 5.47x |
| 执行时间之和 / 观测跨度 | 1.494 | 0.243 | 集成分支仅为主线的 16.3% |

最后一项只能作为任务重叠/活跃程度的近似指标，不等同于物理 AICore 利用率。单 task 执行时间并未变长，而整体完成跨度显著增加，因此当前最强的调优假设是：新 AICore 调度路径存在较大的任务间空隙、低并发推进或 ready task 供给不足。该结论是由 trace 推断得出，尚未定位到具体代码根因。

### 7.1 共享队列与 resolver 竞争打点

为验证上述假设，在 chip-swimlane level 1 下增加了按 worker 本地累积、退出时一次写回的 AICore 打点。打点覆盖：

- ready/completion MPMC queue 的完整调用周期、CAS 尝试/失败和 sequence retry；
- resolver 成功处理、completion 空轮询、executor 成功处理和 ready 空轮询；
- dependency wake-list 注册的 CAS 尝试/失败。

以下结果来自当前 512 MiB ring 配置在 device 0 上的两个完整样本。两次运行均完成全部 65,536 个 AICore task；由于 level-1 profiling ring 各丢失 45,056 条记录，runner 最终以 strict profiling error `-10001` 返回，但 sidecar 竞争统计已完整发布并通过 final-state 校验。

| 指标 | 样本 1 | 样本 2 | 聚合占比/比率 |
| ---- | -----: | -----: | ------------: |
| Runner | 626.919 ms | 660.195 ms | 643.557 ms 均值 |
| queue core-time / 全 worker core-time | 73.821% | 74.806% | 74.331% |
| 发生竞争的 queue 调用 core-time / 全 worker core-time | 72.172% | 73.235% | 72.723% |
| executor 成功路径 / 全 worker RUN core-time | 48.844% | 46.547% | 47.645% |
| resolver core-time / 全 worker RUN core-time | 20.499% | 24.350% | 22.509% |
| resolver core-time / resolver worker RUN core-time | 30.749% | 36.526% | 33.764% |
| ready queue CAS 失败率 | 94.164% | 94.936% | 94.578% |
| completion queue CAS 失败率 | 85.755% | 85.371% | 85.566% |

#### AICore 执行时间口径

“AICore 执行时间”有两个不同口径，不能混用：

| 口径 | 数值 | 占比 | 含义 |
| ---- | ---: | ---: | ---- |
| 纯 kernel 累计执行时间 | 98.352 ms | Device wall 的 19.072% | 来自成功 swimlane 中 65,536 个 task 的 kernel duration 之和 |
| 纯 kernel 累计执行时间 / AICore 观测跨度 | 98.352 / 404.186 ms | 24.333% | 任务重叠和活跃程度的近似指标，不是物理利用率 |
| executor 成功路径 | 31,102,568,136 cycles | worker RUN core-time 的 47.645% | 包含 ready pop、payload 准备、kernel 和 completion push |
| executor 成功路径 / 含 classification 的总计时 core-time | 31,102,568,136 / 80,266,667,870 cycles | 38.749% | 与 queue 使用相同的总计时分母，但仍包含 queue 操作 |

纯 kernel 的 19.072% 是“跨任务累计 kernel duration / Device wall”的诊断比率。由于 65,536 个 kernel 可在多个 AICore 上并行，它不能被解释为严格的墙钟时间分解，也不能据此直接断言其余 80.928% 全部是调度开销。这里更可靠的结论来自对比关系：单 task kernel 更快、kernel 累计时间更短，但 Device wall 和 AICore 观测跨度分别显著增加，说明主要损失发生在 task 之间。

executor 成功路径的 47.645% 也不是纯 kernel 占比。它从 ready queue pop 前开始计时，到 completion queue push 后结束，因此与下面的 ready/completion queue 和 resolver 指标存在重叠，不能相加。

聚合后，ready queue 调用占全部 worker core-time 的 55.344%，completion queue 调用占 18.987%。其中 72.723% 的全部 worker core-time 落在至少经历一次 CAS 失败或 sequence retry 的队列调用内。这是“发生竞争的调用完整延迟”，包含该次调用的正常操作时间，不等同于可直接消除的纯等待时间。

resolver 的 22.509% 与 queue 时间存在嵌套，不能相加。进一步拆分显示：

- resolver 成功处理占全部 worker RUN core-time 的 22.156%；
- completion 空轮询只占 0.353%；
- executor 的 ready 空轮询占 1.754%；
- wake-list 注册只占 0.217%，81,664 次注册均无 CAS 失败。

因此第一瓶颈已确认是全局 ready/completion queue 的原子竞争。第二点中，56 个 AIV resolver 的主要影响是同时争抢 completion queue，并将共享队列竞争放大；纯空轮询和 dependency wake-list 注册本身不是主要耗时。优先优化方向应是减少共享队列参与者、拆分或分片队列、降低无效 CAS，而不是先优化单 kernel 或 wake-list 注册。

打点会在每次队列调用前后读取 cycle counter，因此本次 Runner 均值比原报告的 level-1 样本高，数据适合判断 core-time 构成，不应替代 profiling-off Device wall 作为最终性能基线。

## 8. 调优时的测量准则

后续每次改动建议使用以下准则，确保结果可以与本报告直接比较：

1. 固定 A5 device 2、PTO-ISA pin、Case1 参数和 512 MiB ring heap。
2. 先跑 profiling-off；预热 1 次，至少记录 5 个成功样本，推荐 10 个。
3. 主线与分支在同一设备上按 ABBA 顺序交错运行，避免时段和温度漂移。
4. 以 Device wall 为首要指标，同时报告均值、样本标准差、成功率及任务完成数。
5. 只有 profiling-off 稳定后才采集 level 1 trace；profiling 结果必须通过严格 reconciliation，不能把丢记录的 trace 当作完整证据。
6. 同时检查 AICore task 数、观测跨度、task 间 gap、每核任务分布和 scheduler forward progress。
7. Wake-list + cursor 以直接前序 ticket/pending 的 29.350 ms 为基线，暂定落入 ±2% 为“性能差距关闭”，即 Device wall 不高于约 29.937 ms；该值在用户给出正式性能目标前只作为比较基准。

每轮调优至少保留以下表格：

| 版本/SHA | 成功率 | Device wall mean ± sd | 对比基线 | AICore span | Profiling record 状态 |
| -------- | -----: | --------------------: | -------: | ----------: | --------------------- |
| 当前集成分支 | 有效样本不足 | 515.690 ms（n=1） | 历史 main +460.64% | 404.186 ms | 一个成功 trace；另一次丢 45,078 条 |
| Ticket/pending scheduler | 10/10 | 29.350 ± 0.088 ms | 历史 main -68.09% | 29.464 ms | 65,536 个 Kernel/Completion 均完整 |
| Wake-list + cursor scheduler | 10/10 | 35.045 ± 0.069 ms | Ticket/pending +19.404% | 未采集 | 未采集 level-1 |

## 9. 原始数据与产物

以下路径均相对本报告所在的工作树根目录；`tmp/` 和 `outputs/` 为 git ignored，本地清理这些目录前应先归档需要保留的产物。

- 主线 profiling-off 日志：[`tmp/main_case1_comparison_4f417732_20260811/main_logs/off_1.log`](tmp/main_case1_comparison_4f417732_20260811/main_logs/off_1.log)，同目录下共 5 个样本。
- 主线 profiling-on 日志：[`tmp/main_case1_comparison_4f417732_20260811/main_logs/on_1.log`](tmp/main_case1_comparison_4f417732_20260811/main_logs/on_1.log)，同目录下共 5 个样本。
- 主线 Perfetto trace：[`tmp/main_case1_comparison_4f417732_20260811/main_profile_on1/merged_swimlane.json`](tmp/main_case1_comparison_4f417732_20260811/main_profile_on1/merged_swimlane.json)。
- 集成分支 profiling-off 成功日志：[`tmp/paged_attention_profiling_l1_20260811_132553/warmup.log`](tmp/paged_attention_profiling_l1_20260811_132553/warmup.log)。
- 集成分支 profiling-off stall 日志：[`tmp/paged_attention_profiling_l1_20260811_132553/off_1.log`](tmp/paged_attention_profiling_l1_20260811_132553/off_1.log)。
- 集成分支 profiling 丢记录日志：[`tmp/paged_attention_profile_single_20260811_132854/profile.log`](tmp/paged_attention_profile_single_20260811_132854/profile.log)。
- 集成分支成功 Perfetto trace：[`outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260811_115615/merged_swimlane.json`](outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260811_115615/merged_swimlane.json)。
- 竞争打点完整样本 1：[`tmp/aicore_contention_case1.MohFA4/run.log`](tmp/aicore_contention_case1.MohFA4/run.log)。
- 竞争打点完整样本 2：[`tmp/aicore_contention_case1_repeat.3pqf88/run_2.log`](tmp/aicore_contention_case1_repeat.3pqf88/run_2.log)。
- Ticket/pending 对比测试状态：[`tmp/hbg_ticket_perf_20260812_083702/status.txt`](tmp/hbg_ticket_perf_20260812_083702/status.txt)；`baseline/` 和 `current/` 子目录分别保存 10 个正式样本及预热日志。
- Ticket/pending level-1 日志：[`tmp/hbg_ticket_perf_20260812_083702/current/profile_level1.log`](tmp/hbg_ticket_perf_20260812_083702/current/profile_level1.log)。
- Ticket/pending level-1 原始 trace：[`outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/chip_swimlane_records.json`](outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/chip_swimlane_records.json)。
- Ticket/pending level-1 Perfetto trace：[`outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/merged_swimlane.json`](outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/merged_swimlane.json)；本次未采集 `deps.json`，因此包含完整 scheduler phase 轨道但不包含依赖箭头。
- 最新 `main` Case1 失败日志：[`tmp/paged_attention_case1_20260812/main_latest.log`](tmp/paged_attention_case1_20260812/main_latest.log)。
- Wake-list + cursor 10 轮日志：[`tmp/paged_attention_case1_20260812/wake_cursor.log`](tmp/paged_attention_case1_20260812/wake_cursor.log)。

## 10. 当前状态与剩余验证

- Ticket/pending scheduler 的 Case1 对比中未再出现 forward-progress stall，20/20 个正式样本全部成功。
- 当前 level-1 样本完整记录全部 65,536 个任务的调度 phase，未再出现旧实现的记录丢失或 resolver 导出问题。
- Wake-list + cursor scheduler 在 device 2 上完成 10/10 轮，Device wall 稳定在 35.045 ± 0.069 ms，并消除了最新 `main` 的 ReadyQ 容量失败。
- Wake-list + cursor 相对直接前序 ticket/pending scheduler 回退 19.404%；由于两轮使用不同设备，需要在同一 device 2 上复测 `f8cd1345` 才能完成严格归因。
- Ticket/pending 复测已经完成与 ReadyQ/CompletionQ scheduler 的同基点、同配置重复对比；wake-list + cursor 尚缺同 device 的前序实现复测，也尚未在其他大规模 DAG workload 上建立硬件性能样本。
- 当前 marker 格式仍没有提供可用的 Effective/Orch/Sched 分项，不能据此进一步拆分设备时间。
- 本机芯片自动探测被用户明确要求跳过；测试依据用户对 A5 的确认继续运行，且所有硬件任务均通过 `task-submit` 独占 device 0 或 device 2。
