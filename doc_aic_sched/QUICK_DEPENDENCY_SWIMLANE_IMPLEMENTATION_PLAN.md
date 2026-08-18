# A5 HBG 依赖泳道快速获取实施方案

```text
状态：待实施（临时调试方案）
适用分支：feat/hbg-aicore-dependency-integration
适用运行时：A5 host_build_graph + AICore ticket scheduler
目标用例：TestPagedAttentionHostBuildGraphA5::Case1
日期：2026-08-12
```

## 1. 结论

为了在当前版本快速获得带依赖关系的泳道图，本方案不补齐正式的 A5 platform
dep-gen runner，而是在 Host orchestration 构图完成后，直接在同一线程调用
`dep_gen_host_graph_emit()` 输出 `deps.json`，随后使用现有
`swimlane_converter` 将其与 Level-1 `chip_swimlane_records.json` 离线合并。

推荐通过临时环境变量启用：

```text
SIMPLER_HBG_QUICK_DEP_GEN=1
```

运行测试时只传 `--enable-chip-swimlane 1`，不要传 `--enable-dep-gen`。这样可以：

- 复用当前已经实现的 Host DAG capture；
- 避开 A5 runner 中返回 `-1` 的 device replay stub；
- 避开无数据的 dep-gen device ring 和 collector；
- 保证 capture 和 emit 访问同一个 `thread_local HostGraphState`；
- 在一次运行中同时得到 timing 和静态 DAG；
- 或只复用新生成的 `deps.json`，与已有的干净 profiling 数据离线合并。

该方案用于当前问题定位，完成分析后应删除，不作为最终产品接口。

## 2. 背景与现状

当前分支已经具备以下能力：

1. `run_host_orchestration()` 开始时调用
   `dep_gen_host_graph_begin_capture()`；
2. `submit_task_common()` 调用 `begin_task/end_task`，并记录 explicit、creator、
   tensormap 三类依赖边；
3. `dep_gen_host_graph.cpp` 可以输出包含 `tasks`、`tensors`、`edges` 的
   `deps.json`；
4. `swimlane_converter` 可以读取单独的 `deps.json`，解析 task 到 kernel 的映射，
   显示 fanin/fanout，并生成依赖 flow event；
5. Level-1 profiling 已能输出 ticket scheduler 的 `TicketClaim`、
   `PendingWait`、`Kernel`、`CompletionPublish` 等阶段。

当前缺口在 platform runner：

- `set_dep_gen_enabled()` 没有启用 Host graph capture；
- A5 runner 不区分 host-direct capture 与 device replay；
- 启用 `--enable-dep-gen` 后仍会创建并启动 device collector；
- drain 时总是调用 HBG 未实现的 replay stub，返回 `-1`，不生成 `deps.json`；
- Host graph 状态是 `thread_local`，直接在另一个 drain 线程 emit 还可能返回
  “no capture was adopted on this thread”。

因此，快速方案应在 Host orchestration 所在线程就地写文件，而不是临时拼接
runner 生命周期。

## 3. 目标与非目标

### 3.1 目标

- 为 A5 HBG 当前 ticket scheduler 生成静态任务依赖图；
- 支持 Case1 规模的图；
- 输出现有工具可直接消费的 `deps.json`；
- 在 Perfetto 中显示 executable task 之间的依赖箭头及 fanin/fanout；
- 不改变 AICore scheduler 热路径和 profiling sidecar ABI；
- 不启用当前无效的 device dep-gen collector；
- 改动小、门控明确、易于删除。

### 3.2 非目标

- 不实现正式的 A5 onboard/sim runner 接线；
- 不解决跨线程 capture 的正式所有权模型；
- 不恢复 main 上面向 AICPU scheduler 的 Level 2–4 语义；
- 不在 AICore 热路径记录每次 fanin poll；
- 不把依赖箭头直接锚定到 `PendingWait`；
- 不完成 critical producer、true dependency wait、pickup lag 的自动归因；
- 不保证 inline alloc、dummy、predicate 等无 Kernel 节点都具有可视化锚点；
- 不把该环境变量作为公开或长期支持的 DFX 接口。

## 4. 快速路径设计

```text
CallConfig(--enable-chip-swimlane 1)
                 +
SIMPLER_HBG_QUICK_DEP_GEN=1
                 │
                 ▼
configure_runtime_diagnostics_impl
  ├─ enable AICore Level-1 scheduler trace
  └─ enable dep_gen_host_graph capture
                 │
                 ▼
run_host_orchestration（Host 构图线程）
  ├─ begin_capture
  ├─ submit_task / record edges
  ├─ rt_orchestration_done
  └─ dep_gen_host_graph_emit(<output>/deps.json)
                 │
                 ├──────────────┐
                 ▼              ▼
chip_swimlane_records.json    deps.json
                 └──────┬───────┘
                        ▼
              swimlane_converter
                        ▼
          merged_swimlane_with_deps.json
```

关键点是 `dep_gen_host_graph_emit()` 紧跟在 `rt_orchestration_done()` 后执行。
此时任务图已经完整，且调用仍位于 capture 所在的线程。

## 5. 代码改动计划

### 5.1 增加临时环境变量门控

修改：

```text
src/a5/runtime/host_build_graph/host/runtime_maker.cpp
```

在 `configure_runtime_diagnostics_impl()` 中读取环境变量并启用 Host graph capture：

```cpp
extern "C" void configure_runtime_diagnostics_impl(Runtime *runtime, const CallConfig *config) {
    if (runtime == nullptr || config == nullptr) return;

    const char *quick_dep_gen = std::getenv("SIMPLER_HBG_QUICK_DEP_GEN");
    dep_gen_host_graph_set_enabled(
        config->enable_dep_gen != 0 ||
        (quick_dep_gen != nullptr && std::strcmp(quick_dep_gen, "0") != 0)
    );

    runtime->set_aicore_scheduler_trace(
        config->enable_chip_swimlane == static_cast<int32_t>(ChipSwimlaneLevel::AICORE_TIMING),
        config->output_prefix
    );
}
```

说明：

- 当前文件已经包含 `<cstdlib>` 和 `<cstring>`，不需要增加 include；
- 保留 `config->enable_dep_gen` 是为了不改变函数本身的含义；
- 本方案实际运行时不传 `--enable-dep-gen`，只使用环境变量，避免 platform runner
  进入 device collector/replay 分支；
- 环境变量值为 `0` 时按关闭处理；其他非空值按开启处理。

### 5.2 在 Host orchestration 线程直接输出

在 `run_host_orchestration()` 的以下代码之后：

```cpp
rt_scope_end(rt);
rt_orchestration_done(rt);
```

插入：

```cpp
if (dep_gen_host_graph_enabled()) {
    const std::string &output_prefix = runtime->aicore_scheduler_trace_output_prefix();
    if (output_prefix.empty()) {
        LOG_ERROR("quick dep_gen requires an output prefix");
        return -1;
    }

    const auto deps_path = std::filesystem::path(output_prefix) / "deps.json";
    if (dep_gen_host_graph_emit(deps_path.c_str()) != 0) {
        LOG_ERROR("quick dep_gen emit failed");
        return -1;
    }
}
```

输出失败必须中止本次 bind/run，避免测试成功但依赖文件静默缺失。

### 5.3 本阶段不修改的文件

临时方案不应修改：

```text
src/a5/platform/onboard/host/device_runner.cpp
src/a5/platform/onboard/host/device_runner.h
src/a5/platform/sim/host/device_runner.cpp
src/a5/platform/sim/host/device_runner.h
src/a5/runtime/host_build_graph/runtime/aicore_execution_sidecar_v1.h
simpler_setup/tools/swimlane_converter.py
```

这能把临时改动约束在一个运行时源文件内。

## 6. 构建与运行计划

### 6.1 环境前提

- 在能够正常 `import simpler` 的项目虚拟环境或已安装环境中执行；
- 当前工作树的 `python/` 子树若仍处于删除状态，不应直接依赖源码目录导入；
- onboard 执行前确认实际芯片为 A5，并通过项目既有设备锁定流程选择 device；
- 修改 runtime C++ 后必须重新构建，测试不会自动重编译 runtime。

### 6.2 增量重建

```bash
pip install --no-build-isolation -e .
```

预期产物更新到：

```text
build/lib/a5/onboard/host_build_graph/
```

### 6.3 先用小用例验证

建议先验证 `SmallCase1`，减少调试周期：

```bash
SIMPLER_HBG_QUICK_DEP_GEN=1 \
python tests/st/a5/host_build_graph/paged_attention/test_paged_attention.py \
  -p a5 \
  -d <device_id> \
  --case SmallCase1 \
  --rounds 1 \
  --skip-golden \
  --enable-chip-swimlane 1
```

### 6.4 运行 Case1

```bash
SIMPLER_HBG_QUICK_DEP_GEN=1 \
python tests/st/a5/host_build_graph/paged_attention/test_paged_attention.py \
  -p a5 \
  -d <device_id> \
  --case Case1 \
  --manual include \
  --rounds 1 \
  --skip-golden \
  --enable-chip-swimlane 1
```

重要约束：不要追加 `--enable-dep-gen`。该参数会使 A5 platform runner 启动当前
无效的 device collector/replay 路径。

## 7. 产物检查与离线合并

### 7.1 检查新运行产物

输出目录应至少包含：

```text
chip_swimlane_records.json
deps.json
name_map_TestPagedAttentionHostBuildGraphA5_Case1.json
```

基础检查：

```bash
test -s outputs/<new_run>/deps.json
jq '.tasks | length' outputs/<new_run>/deps.json
jq '.edges | length' outputs/<new_run>/deps.json
jq -e '(.tasks | type == "array") and (.edges | type == "array")' \
  outputs/<new_run>/deps.json
```

同时检查运行日志包含：

```text
dep_gen host graph: wrote deps.json
```

且不应包含：

```text
dep_gen replay failed
no capture was adopted on this thread
```

### 7.2 合并当前 Case1 profiling

复用已有 timing：

```text
outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/
```

执行：

```bash
python -m simpler_setup.tools.swimlane_converter \
  outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/chip_swimlane_records.json \
  --deps-json outputs/<new_run>/deps.json \
  --func-names outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/name_map_TestPagedAttentionHostBuildGraphA5_Case1.json \
  -o outputs/TestPagedAttentionHostBuildGraphA5_Case1_20260812_084219/merged_swimlane_with_deps.json \
  -v
```

使用显式 `--deps-json`，不要依赖“最新目录”自动选择，避免将不同 topology 的图与
profiling 错配。

### 7.3 Perfetto 验证

打开：

```text
merged_swimlane_with_deps.json
```

至少确认：

- Kernel task 名称能够由 `deps.json::tasks[].kernel_ids` 正确解析；
- task 详情中出现 fanin/fanout hint；
- executable producer 与 consumer 之间出现 dependency flow；
- `TicketClaim`、`PendingWait`、`Kernel`、`CompletionPublish` 轨道仍然存在；
- 转换日志的 dependency flow 数量大于 0；
- 没有大规模 “references non-existent successor” 警告。

## 8. 验收标准

### 8.1 必须满足

- [ ] 未设置 `SIMPLER_HBG_QUICK_DEP_GEN` 时行为与当前版本一致；
- [ ] 设置环境变量后，SmallCase1 能生成非空 `deps.json`；
- [ ] Case1 能生成非空 `deps.json`；
- [ ] 运行过程没有调用失败的 replay 路径；
- [ ] `deps.json` 能被 `swimlane_converter` 正常解析；
- [ ] 合并产物能在 Perfetto 中打开；
- [ ] 合并产物包含依赖 flow；
- [ ] 原始 Level-1 task 数和 Kernel/CompletionPublish 数量不因合并改变；
- [ ] 不修改 AICore scheduler、sidecar ABI 或 hot path。

### 8.2 建议统计

记录以下数据，供后续正式实现对照：

```text
deps.json task 数
deps.json edge 数
explicit / creator / tensormap 各类 edge 数
转换后 dependency flow 数
缺失 producer endpoint 数
缺失 consumer endpoint 数
未知 kernel_id 数
deps.json 文件大小
Host graph capture + JSON 输出耗时
```

## 9. 已知限制

### 9.1 Inline task 可视化不完整

当前 `begin_task()` 位于 executable `submit_task_common()` 路径，alloc、dummy 或其他
inline-completed 节点可能不进入 task table。若依赖边端点是这类 task，转换器可能跳过
对应 flow。

Case1 中执行任务与总 runtime task 数存在差异，因此需要关注 converter 的
non-existent endpoint 警告。临时分析时可以先聚焦 executable task 之间的依赖。

### 9.2 依赖箭头不会连接到 PendingWait

当前 converter 将静态依赖箭头锚定到 Worker View 的 task/Kernel slice，而不是
`PendingWait`。因此该方案可以回答“consumer 有哪些前驱”，但不能直接回答“本段
PendingWait 主要被哪个前驱阻塞”。

进一步分析时可以人工结合：

```text
critical producer = 最晚结束 CompletionPublish 的前驱
dependency ready  = max(所有前驱 CompletionPublish.end)
true dep wait     = max(0, dependency ready - PendingWait.start)
pickup lag        = max(0, PendingWait.end - dependency ready)
```

该离线归因不属于本临时实施范围。

### 9.3 Graph 必须与 timing 同拓扑

复用 `deps.json` 时，以下条件必须一致：

- Case 参数；
- orchestration 逻辑；
- task ID 分配顺序；
- kernel 配置；
- manual scope、展开因子及动态控制输入。

若 topology 改变，必须重新生成 `deps.json`。

### 9.4 Profiling 开销

Host graph capture 和 JSON 序列化会增加 Host bind 阶段开销。它发生在设备执行前，
通常不会改变 Device wall，但不能把该次运行用于严格的 Host 总时延比较。

建议用本次运行只获取 DAG，再将 `deps.json` 合并到未启用 quick dep-gen 的 profiling
数据中。

## 10. 风险与缓解

| 风险 | 表现 | 缓解方式 |
| --- | --- | --- |
| 忘记去掉 `--enable-dep-gen` | runner 启动空 collector，最后 replay `-1` | 命令模板只使用环境变量 |
| output prefix 为空 | emit 无目标路径 | 与 `--enable-chip-swimlane 1` 配套，并严格返回错误 |
| capture/emit 跨线程 | emit 返回 `-3` | 在 `run_host_orchestration()` 内立即 emit |
| Case1 JSON 较大 | Host bind 和写盘时间增加 | 先跑 SmallCase1；DAG 只采一次并离线复用 |
| task ID 与旧 timing 不一致 | 箭头错误或大量缺失端点 | 使用同一提交、同一 Case 参数并检查 task 数 |
| inline 节点无锚点 | 部分边不显示 | 记录缺失端点数量，临时分析只使用 executable 子图 |
| 临时代码遗留 | 非正式接口被依赖 | 环境变量统一以 `QUICK` 命名，分析结束后按回退清单删除 |

## 11. 回退方案

完成临时分析后：

1. 删除 `configure_runtime_diagnostics_impl()` 中的
   `SIMPLER_HBG_QUICK_DEP_GEN` 读取和 `dep_gen_host_graph_set_enabled()` 调用；
2. 删除 `rt_orchestration_done()` 后的直接 emit 代码；
3. 重新运行：

   ```bash
   pip install --no-build-isolation -e .
   ```

4. 不删除已经生成的 `deps.json` 和 `merged_swimlane_with_deps.json`，它们是可复用的
   调试产物；
5. 确认未设置环境变量的普通 Case1 行为和 Level-1 trace 与改动前一致。

## 12. 后续正式方案边界

正式实现应单独推进，不在本临时补丁上继续叠加：

1. 将 A2A3 host-direct dep-gen runner 分支移植到 A5 onboard/sim；
2. 在 bind 前正式调用 `dep_gen_host_graph_set_enabled()`；
3. host graph active 时跳过 device DFX flag、ring、collector 和 replay；
4. 对齐 main 的同线程 capture/emit 生命周期，移除废弃 snapshot API；
5. 补齐 alloc/dummy/inline task 的 DAG 节点和可视化锚点；
6. 使用 `deps.json + CompletionPublish + PendingWait` 做 critical producer 和
   pickup lag 离线归因；
7. 增加 A5 unit、sim、onboard、converter E2E 和 DFX CI smoke。

临时方案验收后，应保留本文件作为分析记录，但代码实现应在正式方案落地前回退。
