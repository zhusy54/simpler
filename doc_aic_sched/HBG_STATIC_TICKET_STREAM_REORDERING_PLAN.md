# A5 HBG 静态 Ticket Stream 重排开发计划

```text
状态：已实现；PA 正确性与性能验收通过；通用 DAG 硬件门槛受基线既有失败阻塞
目标分支基线：884ec07f985ef35a9b5047562748e2c0e3c8a5fd
关键前序提交：f8cd13454e21bd681078cd12d752ed573c04f12b
日期：2026-08-12
```

## 1. 结论

本方案保留 `f8cd1345` 引入的 ticket cursor、worker-private pending slots 和 per-task completion cells，不恢复 ReadyQ、CompletionQ、wake-list resolver，也不要求 Paged Attention 用例改变任务依赖。

改动发生在 Host 完整构图之后、AICore sidecar 上传之前：Host 根据完整 DAG 为每个任务计算静态优先级，然后只重排 `aic_task_ids[]` 和 `aiv_task_ids[]`。任务 ID、descriptor、payload、fanin 和 completion-cell 索引全部保持不变。

第一版候选算法使用“剩余关键路径长度”排序，而不是简单的正向依赖层级排序：

```text
priority(task) = 从 task 到任意 sink 的最长路径长度，越大越优先
tie-break       = 原 task_id 递增
```

该优先级使 PA 各 batch 的浅层关键任务先被领取，减少当前 batch-major stream 把大量 worker 填入同一 batch 深层 UP 链的概率，同时不增加 device 侧共享队列或轮询热点。

生产实现只保留上述新顺序，不提供 task-ID/critical-path 双模式。收益验证使用独立 worktree 中的
`884ec07f` 基线构建与当前构建做临时 A/B；验证代码不进入 Runtime 配置、环境变量或 ABI。

## 2. 问题背景

### 2.1 已提交版本

本计划以 `884ec07f` 为诊断基线。它已经包含 `f8cd1345` 的 ticket scheduler 改造：

- AIC/AIV 各一个 ticket cursor；
- 每个 worker 两个 private pending slots；
- task 完成后直接发布独立 completion cell；
- 无 ReadyQ、CompletionQ 和 steady-state resolver。

`f8cd1345` 的改造不能回退。旧 shared ReadyQ/CompletionQ 方案在 A5 上存在全局 `enqueue_pos`/`dequeue_pos` CAS 和队头轮询热点，Case1 的历史 Device wall 从约 `522.411 ms` 降至 ticket/pending 的约 `29.350 ms`。恢复动态 ReadyQ 会重新引入已经证实的性能瓶颈。

### 2.2 当前 stream 如何产生

Paged Attention 用例按下面的循环顺序创建 task：

```text
batch -> q-loop -> block -> QK/SF/PV/UP
```

对应代码：

- `tests/st/a5/host_build_graph/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
- 外层 `b_idx` 循环约在第 147 行；
- 内层 `bn` 循环约在第 171 行。

Host 的 `create_aicore_sidecar_v1()` 当前按递增 task ID 扫描整张图，并直接追加到 typed stream：

```cpp
if (task.core_type == AicoreRootCoreTypeV0::AIC) {
    aic_task_ids.push_back(static_cast<uint32_t>(task_id));
} else {
    aiv_task_ids.push_back(static_cast<uint32_t>(task_id));
}
```

对应代码：

- `src/a5/runtime/host_build_graph/host/runtime_maker.cpp`
- `create_aicore_sidecar_v1()` 约在第 573 行；
- typed stream 收集约在第 589～635 行；
- stream 复制到 sidecar 约在第 678～681 行。

因此 typed stream 继承了用例的 batch-major 创建顺序。

### 2.3 Case1 的具体后果

Case1 包含：

```text
batch = 256
block per batch = 8192 / 128 = 64
每个 block = QK -> SF -> PV -> UP
UP(b, n) -> UP(b, n + 1)，由 INOUT 累积状态形成
总 executable task = 256 * 64 * 4 = 65,536
```

按 core type 过滤后，当前 stream 近似为：

```text
AIC: B0-QK0, B0-PV0, B0-QK1, B0-PV1, ... B1-QK0, ...
AIV: B0-SF0, B0-UP0, B0-SF1, B0-UP1, ... B1-SF0, ...
```

启动时 28 个 AIC worker 和 56 个 AIV worker 按 `type_rank` seed stream 前缀。大量初始 ownership 因此集中在 batch 0，并覆盖尚未满足依赖的深层任务。两个 pending slot 被占满后，worker 只能等待前驱 completion。

基线 level-1 trace 的关键事实：

- 65,536 个 kernel，平均 kernel 约 `1.424 us`；
- 32,854 条 PendingWait；
- PendingWait 累计 worker core-time 约 `4,467 ms`，平均约 `135.967 us`；
- PendingWait 在 worker 间大量并行重叠，不能直接累加到约 `29 ms` Device wall；
- PA 中 PV/UP 是主要长等待任务，等待主体是“任务已被 owner 占用但前驱未完成”。

## 3. 目标与非目标

### 3.1 目标

1. 保留 ticket scheduler 的低共享争用特性。
2. 降低 seed 和后续 ticket 对远期、未 ready task 的提前 ownership。
3. 将初始任务分布从“同一 batch 的深链”改为“多个独立 batch 的关键前沿”。
4. 使用 Host 已有的完整 DAG 信息生成通用静态顺序。
5. 保持 task 唯一执行、依赖可见性、完成计数和 sidecar ABI 语义不变。
6. 生产路径只生成 critical-path 顺序；通过独立基线构建完成 A/B 和回退判定。

### 3.2 非目标

1. 不恢复 ReadyQ、CompletionQ 或集中式 ready-only 调度。
2. 不在 AICore 热路径扫描全图或全局 ready bitmap。
3. 不在生产 Runtime 中硬编码 `batch=256`、`block=64` 或 `QK/SF/PV/UP`。
4. 不修改 PA kernel 的数值算法和 tensor 依赖。
5. 第一阶段不实现 work stealing、任务迁移或动态 fanout publication。
6. 不承诺完全消除 PendingWait；目标是减少不必要的提前 claim。

## 4. 方案概览

### 4.1 只重排索引，不重写图

当前：

```text
Graph task IDs:       0, 1, 2, 3, 4, 5, ...
AIC ticket stream:    0, 2, 4, 6, ...（示意）
AIV ticket stream:    1, 3, 5, 7, ...（示意）
```

修改后：

```text
Graph task IDs:       完全不变
descriptor/payload:   完全不变
fanin IDs:            完全不变
completion[task_id]:  完全不变

AIC ticket stream:    按静态 DAG priority 重排后的 AIC task ID
AIV ticket stream:    按静态 DAG priority 重排后的 AIV task ID
```

AICore 仍通过 `atomicAdd(next_index, 1)` 获得唯一 stream index，再从数组读取原 task ID。device 侧领取和执行协议不增加新的共享数据结构。

### 4.2 为什么不使用简单 dependency level

不能直接使用：

```text
top_level(task) = root 到 task 的最长距离，按 top_level 升序
```

PA 的所有 QK 都是 root，`top_level=0`。简单排序会把全部 16,384 个 QK 放到前面，形成近似全局阶段屏障，严重推迟 PV/UP，破坏跨类型流水。

第一版候选改用 reverse criticality：

```text
bottom_level(task) =
    0                                      task 没有 successor
    1 + max(bottom_level(successor))       其他情况
```

然后按：

```text
(bottom_level 降序, task_id 升序)
```

分别稳定排序 AIC 和 AIV stream。

对任意边 `producer -> consumer`：

```text
bottom_level(producer) > bottom_level(consumer)
```

因此 producer 在优先级上严格早于 consumer。该性质是更新 forward-progress 证明和测试模型的基础。

### 4.3 PA 上预期形成的近似 wavefront

在每个 batch 都有相同 64-block 链时，相同 block 位置的任务具有相同或接近的 bottom level；task-ID tie-break 会先覆盖多个 batch 的同一关键位置。

预期初始前缀近似为：

```text
AIC 前缀：B0-QK0, B1-QK0, B2-QK0, ...
AIV 前缀：B0-SF0, B1-SF0, B2-SF0, ...
```

后续不是硬编码的严格 `(block, stage, batch)` 顺序，而是根据 DAG criticality 自动形成近似 wavefront。需要通过离线打印和真实 trace 验证实际顺序，不能只依赖推导。

## 5. Forward progress 与正确性约束

现有设计文档中的无死锁说明依赖“typed stream 按 task ID 递增”。重排后必须更新为“stream priority 对所有依赖边单调”。不能只删除测试中的 `std::is_sorted(task_id)` 检查而不补充新的证明和约束。

建议维持以下不变量：

1. 每个 executable task 在 AIC/AIV 两个 stream 中恰好出现一次。
2. stream 中的 task core type 与 descriptor 一致。
3. 原图继续满足 `producer_task_id < consumer_task_id`，因此 graph payload 校验不变。
4. 对每条边，`bottom_level(producer) > bottom_level(consumer)`。
5. 每个 typed stream 按 `(bottom_level desc, task_id asc)` 排序。
6. inline-completed task 不进入 stream，但其 completion cell 在 RUN 前设置为 DONE。

进展性论证：取全局 priority 最高的未完成 executable task `T`。`T` 的所有 executable fanin priority 都严格高于 `T`；如果仍未完成，它们才是更高 priority 的未完成任务，与 `T` 的选择矛盾。因此 `T` 的 fanin 已完成或 inline-completed。由于对应 typed stream 也按同一 priority 排序，cursor 不可能永久跳过 `T` 去领取更低 priority task；当 owner 扫描到 `T` 时它可执行。有限 pending slots 不会形成只包含相互等待任务的闭环。

实现和评审时必须用模型测试覆盖该论证，尤其是 AIC/AIV 交叉依赖图。

## 6. 分阶段开发计划

### 阶段 0：冻结基线并离线验证假设

目标：先证明排序确实改变 Case1 的 seed/frontier 分布，再碰生产路径。

1. 新会话开始时检查：

   ```bash
   git status --short --branch
   git rev-parse HEAD
   git diff -- src/a5/runtime/host_build_graph
   ```

2. 基线必须记录为 `884ec07f`，性能结论不得混入现有未提交修改。
3. 保留并引用输入 trace：

   ```text
   outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/
   merged_swimlane_with_deps_20260812_144355.json
   ```

4. 用 Host 侧小工具或单元测试读取 synthetic PA DAG，打印：

   - baseline stream 前 128 个 task；
   - criticality stream 前 128 个 task；
   - AIC/AIV seed task；
   - seed 覆盖的不同依赖链/根任务数量；
   - seed 中 blocked task 数；
   - 每个 task 的 `top_level`、`bottom_level`、core type。

5. 阶段 0 通过条件：

   - 新 AIC seed 主要覆盖不同 batch 的 QK/frontier；
   - 新 AIV seed 不再集中覆盖同一 batch 的连续 UP 深链；
   - stream 覆盖、唯一性和 core type 全部正确。

若 criticality 排序没有形成预期分布，停止生产改动，先调整优先级定义。不要退回 ReadyQ。

### 阶段 1：实现 Host-only stream planner

建议新增：

```text
src/a5/runtime/host_build_graph/host/aicore_ticket_stream_planner.h
```

如果实现较大，再增加同名 `.cpp` 并接入对应构建目标。planner 只在 Host 构图阶段运行，不进入 AICore image。

实际接口：

```cpp
struct AicoreTicketStreams {
    std::vector<uint32_t> aic;
    std::vector<uint32_t> aiv;
    std::vector<uint32_t> bottom_level;
};

bool build_aicore_ticket_streams(
    const AicoreReadonlyGraphV0 &graph,
    const std::vector<uint8_t> &inline_completed,
    AicoreTicketStreams *result
);
```

实现步骤：

1. 按 task ID 扫描并复用现有 `aicore_classify_task_v0()` 校验。
2. 按 task ID 逆序遍历 consumer，直接以 consumer 的 bottom level 松弛其 fanin producer；
   现有 graph 保证 fanin task ID 小于 consumer，因此无需 successor CSR 或通用拓扑排序。
3. 计算完整图（包括 inline-completed task）的 `bottom_level`。
4. executable task 按 core type 放入 typed vector。
5. 按 `(bottom_level desc, task_id asc)` 排序。
6. 校验覆盖、唯一性、core type 和 priority 单调性。

复杂度目标：

```text
构图：O(V + E)
排序：O(V log V)
临时内存：O(V)
device 热路径新增成本：0
```

### 阶段 2：接入 sidecar，生产路径只保留新顺序

修改：

```text
src/a5/runtime/host_build_graph/host/runtime_maker.cpp
```

将当前“分类并直接按 task ID 生成 stream”的逻辑收敛到 planner。sidecar layout、stream ABI 和 AICore claim 代码不变。

接入原则：

1. `create_aicore_sidecar_v1()` 始终调用 planner 生成 critical-path stream。
2. 不增加 `AicoreTicketOrder`、CallConfig 字段、环境变量、宏或 Runtime 分支。
3. 不允许通过读取 PA 参数或 func ID 在 Runtime 中选择特殊排序。
4. A/B 只使用基线 SHA 独立 worktree 与当前 worktree 的两个临时构建。

### 阶段 3：更新调度模型与设计文档

修改：

```text
src/a5/runtime/host_build_graph/runtime/aicore_ticket_model_v1.h
tests/ut/cpp/a5/test_aicore_ticket_scheduler_v1.cpp
src/a5/runtime/host_build_graph/docs/RUNTIME_LOGIC.md
doc_aic_sched/a5_hbg_ticket_cursor_pending_scheduler_design.md
```

模型当前在 `add_workers()` 中要求 task vector 按 task ID 递增。需要：

1. 移除“task ID 必须递增”这一过时条件。
2. 保留 task 唯一性、合法 ID、core type 覆盖校验。
3. 给模型传入或计算新的 priority，并校验每条边 priority 单调。
4. 增加非单调 task-ID stream 的 forward-progress 测试。
5. 更新无死锁证明，明确它依赖 priority 拓扑单调，而不是 task-ID stream 单调。

### 阶段 4：功能测试

至少增加以下单元测试：

1. `TaskIdOrderIsByteCompatibleWithBaseline`
2. `CriticalPathOrderPreservesCoverageAndCoreType`
3. `ProducerPriorityStrictlyPrecedesConsumer`
4. `CriticalPathOrderAcceptsNonMonotonicTaskIds`
5. `CrossTypeDiamondCompletesExactlyOnce`
6. `CrossTypeAlternatingChainDoesNotDeadlock`
7. `InlineCompletedTasksAreNotStreamed`
8. `UnevenIndependentChainsPrioritizeLongerRemainingPath`
9. `PagedAttentionSyntheticSeedsSpanIndependentBatches`
10. `RandomTopologicalDagCompletesExactlyOnceWithTwoPendingSlots`

补充边界：

- 空图；
- 只有 AIC 或只有 AIV；
- task 数小于 worker 数；
- 多 fanin；
- 大量同 priority task；
- inline-completed predecessor；
- 最大合法 fanin。

然后运行 A5 simulation 中所有受影响 HBG scene，确认输出和退出计数一致。实际命令必须按仓库 `testing` skill 和平台规则确定。

### 阶段 5：A5 硬件性能验证

运行 onboard 前必须先执行 `onboard-arch-precheck`，并通过 `task-submit` 独占单个 A5 device。禁止 baseline/current 并发占用同一设备。

对比条件必须一致：

```text
同一物理 device
同一 PTO-ISA pin
同一编译选项
同一 Case 参数
同一 profiling 状态
同一 task-submit 资源约束
baseline/current 交错运行
```

建议样本：

1. profiling-off：每个版本预热 1 次，正式样本至少 10 次，交错运行。
2. level-1：每个版本至少 1 次，用于 PendingWait 和 ownership 分布诊断。
3. Case1 之外至少选择：

   - 短链宽 DAG；
   - 纯 AIC；
   - 纯 AIV；
   - AIC/AIV 交替链；
   - task 数少于 core 数的小图。

必须报告：

- Device wall mean、p50、p95、标准差；
- 总 task 数、AIC/AIV worker 数；
- SeedClaim/TicketClaim 数量和耗时；
- PendingWait 数量、累计 worker core-time、均值、p50/p95；
- 按 kernel/function 分组的 PendingWait；
- 每 worker executed task 分布；
- pending peak、fanin load/wait、idle/backoff；
- Host sidecar/planner 构建耗时。

注意：PendingWait 是并行 worker core-time，不能直接与 Device wall 相加。最终判断以 profiling-off Device wall 为主，trace 用于解释原因。

### 阶段 5 实测结果（2026-08-12）

硬件由仓库 `query device` 工具确认：`Ascend950PR_9579`、
`Short_SoC_version=Ascend950`，28 AIC + 56 AIV。仓库预检脚本使用旧版
`npu-smi -c` 参数，在本机新版驱动上无法解析，因此使用同一 CANN ini 映射的
仓库查询工具完成等价核验。所有执行均通过 `task-submit` 独占 device 0。

Case1 profiling-off 使用独立 `884ec07f` worktree 和当前 worktree，在同一个设备锁内按
ABBA 顺序交错，各 20 个正式样本（另有各 1 次预热）：

| 构建 | mean (ms) | p50 (ms) | p95 (ms) | stddev (ms) | mean 95% CI (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `884ec07f` baseline | 29.336 | 29.365 | 29.478 | 0.096 | [29.294, 29.378] |
| critical-path current | 7.382 | 7.358 | 7.470 | 0.078 | [7.348, 7.416] |

Device wall 改善 `74.835%`，加速 `3.974x`。新 planner 平均 `4.038 ms`、p95
`4.230 ms`、最大 `4.273 ms`，占当前平均 bind `489.089 ms` 的约 `0.83%`，
满足绝对 `<10 ms` 和相对 `<5%` 的门槛。

level-1 DFX（每个构建 1 次，65,536 kernel）显示：

| 指标 | baseline | current |
| --- | ---: | ---: |
| PendingWait count | 32,857 | 32,752 |
| PendingWait core-time | 4,464.420 ms | 456.107 ms |
| PendingWait p50 / p95 | 91.236 / 322.387 us | 14.633 / 18.767 us |
| PV PendingWait count | 16,382 | 0 |
| UP PendingWait p95 | 342.740 us | 19.071 us |
| global pending overlap peak | 163 | 90 |
| per-worker pending peak | 2 | 2 |
| AIC seed | 14 QK + 14 PV，1 batch | 28 QK，28 batches |
| AIV seed | 28 SF + 28 UP，1 batch | 56 SF，56 batches |

`TicketClaim` 平均 phase 从 `0.283 us` 增至 `2.063 us`，但 cursor/claim 代码和共享
结构未改变；总 Device wall 仍显著下降。该项保留为后续观察指标，不能把并行 worker
core-time 直接加到 Device wall。

非主场景同样采用 baseline/current ABBA block，每个构建丢弃每 block 首轮后保留 20 个样本：

| 场景 | baseline mean (ms) | current mean (ms) | delta |
| --- | ---: | ---: | ---: |
| pure AIC `aic_chain_64` | 0.548 | 0.548 | -0.038% |
| pure AIV `aiv_chain_64` | 0.552 | 0.538 | -2.447% |
| mixed `mixed_chain_1024` | 6.506 | 6.555 | +0.751% |
| task-count-below-cores `SmallCase1` | 0.178 | 0.175 | -2.107% |

PA 的 `SmallCase1/2/VarSeq2/VarSeq4` 与 Case1 在 baseline/current 上均通过 golden；
Case1 另有 20 个 current profiling-off 样本全部完成。

通用 DAG hardware suite 仍存在基线既有正确性失败：当前构建首次全量运行结果为
`15 passed, 1 skipped, 2 failed`，失败为 `single_core_dag` 和 `multi_core_dag` 的
`task_state` mismatch；对失败 case 做同卡基线复测时，`884ec07f` 也出现相同类型且
具有波动性的 mismatch（包括无 fanin 的 `mixed_multi_root_4096`）。因此不能将该失败
归因于本次 stream 重排，但在该既有问题解决前，也不能宣称“所有目标硬件 ST 通过”。

## 7. 验收标准

### 7.1 正确性硬门槛

- 所有目标 UT/ST 通过；
- 每个 executable task 恰好执行一次；
- 所有 completion cell 最终为 DONE；
- 无 scheduler error、stall、timeout；
- 数值输出与 baseline 一致；
- 20 次 Case1 压力运行无偶现失败；
- stream 覆盖和 priority 单调校验通过。

任何正确性失败都直接否决该排序，不允许用性能收益豁免。

### 7.2 性能门槛

保留该生产排序的门槛定为：

- Case1 profiling-off Device wall 相对 `884ec07f` 同卡基线改善至少 `2%`，且置信区间/样本分布明确超过设备噪声；
- Case1 的 PV/UP PendingWait p50/p95 和累计 worker core-time显著下降；
- 其他代表性 HBG DAG 的 Device wall 无超过 `2%` 的稳定回退；
- Host planner 额外耗时不超过 Host build/bind 的 `5%`，且绝对增量建议控制在 `10 ms` 内；
- device 侧 TicketClaim、CompletionPublish 不增加新的共享热点。

如果 PendingWait 明显下降但 Device wall 不改善，不能仅凭 trace 指标保留实现。可能原因包括流水被静态排序过度约束、AIC/AIV 利用率下降或 Host 排序收益被其他阶段抵消。

### 7.3 本次验收判定

- PA 数值、Case1 压力、Case1 性能、PendingWait、Host planner 开销和代表性场景性能均通过。
- C++ 非硬件测试 `90/90` 通过（包含最大合法 128 fanin 边界）。
- A5 HBG simulation（排除用户未跟踪的 `paged_attention_unroll`）`18 passed, 1 deselected`。
- 通用 DAG onboard correctness 未达到“全部通过”，但同样失败可在冻结基线上复现；
  这是合入前需要单独跟踪的基线硬件问题，而不是本次 A/B 的性能回退。

因此本次算法的收益门槛通过，生产实现保持单一 critical-path 顺序；同时不得把通用 DAG
硬件 suite 描述为通过，也不得用本次改动掩盖或修改其 golden。

## 8. 风险与应对

### 8.1 静态优先级不等于动态 ready

任务实际完成时间依赖 kernel 长度和 AIC/AIV 相对速度。静态 criticality 只能降低错误提前 claim 的概率，不能保证领取时 ready。

应对：保留两个 pending slots 和当前 readiness polling；用硬件 trace 验证，不把静态推导当成性能结论。

### 8.2 可能破坏流水

过强的 level 分组会形成阶段屏障。尤其不能使用单纯 `top_level` 升序。

应对：第一候选使用 bottom-level criticality；若硬件上形成 barrier 或未通过性能门槛，
本任务停止并撤回生产实现，不继续尝试第二种排序，也不恢复 ReadyQ。

### 8.3 旧 forward-progress 证明失效

任意 shuffle typed stream 可能让有限 pending slots 形成跨类型占位死锁。

应对：只允许满足 dependency-monotonic priority 的排序；增加交叉依赖和随机 DAG 模型测试。

### 8.4 Host 临时内存和构图时间

Case1 有 65,536 个 executable task，不能为每个 task 构建独立 successor vector。

实际实现利用 `producer_id < consumer_id` 约束，逆序扫描 consumer 后直接松弛 fanin producer，
不保存 successor adjacency。临时结构为 bottom level、core type、覆盖 bitmap 和 typed stream，
空间为 `O(V)`；planner 使用 `simpler_run.bind.ticket_stream_plan` 单独打点。

### 8.5 与现有未提交修改冲突

开始实现时目标 Runtime 路径没有其他用户修改；用户已有的未跟踪
`tests/st/a5/host_build_graph/paged_attention_unroll/` 全程未修改、未纳入测试范围。
硬件基线从明确 SHA `884ec07f` 创建独立 detached worktree，避免把当前实现混入基线。

## 9. 明确否决的方案

### 9.1 恢复动态 ReadyQ

否决原因：重新引入所有 worker 对共享队头的 GM load/CAS、ReadyQ push/pop、CompletionQ 和 resolver 热点。历史 Case1 已显示数量级回退。

### 9.2 只按正向 dependency level 排序

否决原因：PA 所有 QK 都是 root，会先领取全部 QK，形成过大 runahead 和近似阶段屏障。

### 9.3 在生产 Runtime 硬编码 PA task pattern

否决原因：Runtime 不能依赖 func ID、batch 或 block 的用例语义；会破坏通用 HBG 调度边界。

### 9.4 直接交换 PA 的 batch/block 循环

否决原因：每个 batch 的 `oi/li/mi`、`out_view` 和 `PTO2_SCOPE` 生命周期跨越全部 block。直接交换循环需要保存并管理每个 batch 的中间 tensor/scope 状态，改动大且只服务一个用例。

### 9.5 任意随机或 round-robin shuffle

否决原因：如果排序不满足 producer-before-consumer priority，两个 private pending slots 可能被跨类型远期 consumer 占满，破坏现有 forward-progress 依据。

## 10. 如果第一候选无收益

第一候选若没有通过性能门槛，立即停止，不在本任务中尝试第二种排序算法；撤回生产实现，
仅保留 A/B 数据和调查结论。不得转入 ReadyQ、bounded lookahead、seed-only 重排或 PA unroll。

本次实测第一候选通过性能门槛，因此没有触发该停止条件。

## 11. 建议提交拆分

1. `Add: build dependency-aware HBG ticket streams on host`
   - planner、sidecar 接入，生产路径只有 critical-path order。
2. `Test: cover critical-path ticket ordering and progress`
   - 非单调 stream、交叉类型 DAG、synthetic PA seed 分布。
3. `Docs: document critical-path order and A5 validation`
   - forward-progress 证明、同卡数据、DFX 与已知基线硬件失败。

不增加 A/B 配置，也不把 PA unroll 或其他 scheduler 修改混入本改动。

## 12. 新会话交接清单

新会话开始时按顺序完成：

1. 阅读根目录 `CLAUDE.md` 和 `.claude/rules/`。
2. 阅读本文以及：

   ```text
   doc_aic_sched/a5_hbg_ticket_cursor_pending_scheduler_design.md
   doc_aic_sched/PAGED_ATTENTION_CASE1_PERFORMANCE_REPORT.md
   ```

3. 确认目标 baseline 和现有 dirty worktree 的处置方式。
4. 先完成阶段 0 的离线 stream/seed 证明。
5. 再实现 Host-only planner 和单元测试。
6. simulation 通过后，按 `onboard-arch-precheck` 和 benchmark 流程运行 A5。
7. 根据第 7 节门槛决定保留或撤回唯一的生产排序；不在 Runtime 内切换模式。

## 13. 完成定义

只有同时满足以下条件才算完成：

- Host planner 和所有正确性测试合入；
- 设计文档中的 forward-progress 证明已更新；
- Case1 同卡 profiling-off 数据达到验收门槛；
- level-1 trace 证明 seed 分布改变且 PV/UP PendingWait 下降；
- 代表性非 PA DAG 无稳定性能回退；
- 默认排序的选择有数据支持；
- 未恢复 ReadyQ/CompletionQ，也未在 Runtime 中引入 PA 特化。

截至本次实现，除“所有通用 DAG onboard ST 通过”外其余项目已完成。该剩余项在
`884ec07f` 基线上同样失败，需作为既有硬件正确性问题单独处理；本方案没有修改这些
scene 的 kernel 或 golden。
