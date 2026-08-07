# HBG-AICore Scheduler M0～M3 原子提交计划

## 1. 摘要

- 本计划结合 `host-build-graph-aicore-scheduler-plan.zh.md` 和
  `host-build-graph-aicore-shared-contention-analysis.zh.md`。
- 共规划 36 个提交：设计校正 1 个、准备性重构 3 个、M0 8 个、M1 12 个、
  M2 5 个、M3 7 个。
- M0 至 M7 固定使用基础 Vyukov bounded MPMC。M3 不实现 sharded MPMC、
  ready bitmap 或 bitmask。
- M4 使用同一基础 MPMC 完成 A5 真机正确性闭环。sharded MPMC、两级
  bitmap 和 per-shard non-empty bitmask 的实现、比较和选型全部属于 M8。
- 每个提交的目标改动量不超过 300～400 行，不混入无关格式化、重命名或清理。
- 新运行时仅通过显式选择 `host_build_graph_aicore` 启用，不新增环境变量、
  编译宏或 feature flag。

## 2. 已锁定的接口与行为

### 2.1 CALLABLE 编译规格

仅 `runtime="host_build_graph_aicore"` 要求 CALLABLE 顶层包含：

```python
"aicore_adapter": {
    "source": "aicore_adapter.cpp",
    "extra_include_dirs": [],
}
```

- 缺少 `aicore_adapter` 或 `source` 时在编译前报 schema 错误。
- adapter 源码必须定义 `aicore_before_task` 和 `aicore_after_task`，缺少任一
  符号时链接失败。
- hook 接收 `AicoreGraphLaunchDesc` 、`AicoreTaskRecord` 和 `WorkerExecutionSlot`，
  返回 `0` 表示成功，非零进入 first-error。
- 调用顺序固定为 `before_task -> kernel -> after_task -> release completion`。
- DUMMY 和 predicate-false task 不调用 kernel 或 adapter，仅发布完成。
- Worker、Orchestrator、submit 和现有 `ChipCallable` Python 使用接口不变。

### 2.2 内部 ABI

- `AicoreCallableBundleHeader` 保存 magic、ABI 版本、adapter ABI、Host orchestration
  SO 和 linked AICore image 的 offset/size 以及 image identity。
- 仅新运行时将 bundle 放入 `ChipCallable.binary`；现有 HBG/TRB 格式不变。
- graph image 内部只使用 task id、整数、GM 地址和基于 image base 的 offset。
- `AicoreTaskControl` 按 128B 对齐隔离，wire 类型需通过 trivial-copy、
  standard-layout 和布局静态断言。
- 单 lane 仅接受 AIC/AIV 和 `block_num == 1`；MIX、wide、`sync_start` 和 async
  在启动前拒绝。
- AIC 与 AIV 各有一个基础 MPMC ReadyQ；DUMMY ReadyQ 可由两类 core 领取。
- 成功 MPMC pop 是唯一 task ownership 线性化点，不增加 claim CAS、
  `WAITING -> READY` CAS 或 idle bitmap。
- 任务不 retry、不 requeue；first-error 后停止新领取并执行有界 drain。

## 3. 通用绿色门禁

代码提交除各自定向测试外，执行适用的公共门禁：

```bash
source .venv/bin/activate
pip install --no-build-isolation -e .
pytest tests/ut
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build --output-on-failure -LE requires_hardware
python simpler_setup/build_runtimes.py --platforms a5sim
```

具有 CCEC/A5 工具链的 CI 额外执行：

```bash
python simpler_setup/build_runtimes.py --platforms a5
```

里程碑场景测试使用 CI 的 600 秒 session timeout：

```bash
pytest examples tests/st --platform a5sim \
  --runtime host_build_graph_aicore --device 0-15 \
  -p no:xdist --pto-session-timeout 600 --require-pto-isa -v
```

M0～M3 不运行 A5 真机测试；A5 二进制必须可编译，真机正确性证据属于 M4。

## 4. 提交序列

### 4.1 设计校正与准备性重构

#### 提交 1：`docs: 将 ReadySet 优化统一归入 M8`

- **详细说明**：修正两份源设计文档的里程碑冲突，保留已增加的 M8 内容。
- **主要变更**：修正 ReadySet、风险表、设计决策、WP4、M3/M4/M8、模型测试、
  DFX 和验收清单。
- **验证**：运行 Markdown pre-commit，用 `rg` 确认 M3/M4 不再要求候选实现，
  M8 仍完整包含候选协议和采用条件。

#### 提交 2：`refactor: 提取 Host orchestration 加载与调用`

- **详细说明**：从 A5 HBG `runtime_maker.cpp` 提取 SO 校验、`dlopen`、符号解析、
  调用和释放，不改变当前 HBG 行为。
- **主要变更**：新增 `src/a5/runtime/host_graph_common/host_orchestration.{h,cpp}`，
  HBG runtime maker 改用公共组件，添加成功、缺符号、异常返回和重复释放 UT。
- **验证**：公共门禁、相关 C++ UT 和现有 A5sim HBG vector/prepared-callable 场景通过。

#### 提交 3：`refactor: 提取不可变 Host 图快照`

- **详细说明**：将 HBG 构图完成后的 task、fanin、payload、predicate 和预完成信息
  暴露为只读 `FrozenHostGraph`，并提取 relocation/边界检查。
- **主要变更**：新增 `host_graph_common/frozen_graph.{h,cpp}`，调整 HBG freeze 路径，
  添加空图、稳定 task id、fanin 顺序、predicate、预完成和非法 offset UT。
- **验证**：现有 HBG arena 语义不变，归一化图和 A5sim 执行结果不变。

#### 提交 4：`refactor: 从运行时配置选择 orchestration 工具链`

- **详细说明**：删除 `KernelCompiler._orchestration_toolchain()` 对 runtime name 的硬编码，
  由 `build_config.py` 的 `orchestration.execution` 选择 `host` 或 `aicpu`。
- **主要变更**：HBG 声明 `host`，TRB 声明 `aicpu`，更新 compiler/config loader 及
  A2/A3、A5、sim/onboard UT。
- **验证**：现有 runtime 的工具链选择与改动前一致，callable 编译和 a5sim build 全绿。

### 4.2 M0：空图生命周期与运行时骨架

#### 提交 5：`feat: 定义 HBG-AICore graph image ABI`

- 建立 launch descriptor、task record、128B task control、run control、worker slot、状态和 sentinel。
- 增加 size/alignment/offset/trivial-copy/standard-layout 断言和序列化 UT。
- 新接口尚无调用者，不影响现有 runtime。

#### 提交 6：`feat: 添加 sidecar 布局与容量规划器`

- 以 checked arithmetic 计算 records、controls、fanin、payload、slots、ReadyQ 和 DFX 区域。
- UT 覆盖空图、边界容量、2 的幂队列、DUMMY 最坏容量和溢出。

#### 提交 7：`feat: 搭建 A5 HBG-AICore 运行时目标`

- 新增仅 A5/A5sim 可发现的 runtime 和 host/AICPU/AICore 最小入口。
- 尚未支持的图返回确定性 milestone-not-ready 错误。
- runtime discovery UT 确认 A2/A3 不暴露该 runtime，a5sim/a5 build smoke 通过。

#### 提交 8：`feat: 将空图与 DUMMY 图编码为 device sidecar`

- 从 `FrozenHostGraph` 生成 M0 graph image，支持空图、单 DUMMY、relocation 和 generation。
- UT 覆盖确定性字节布局、容量不足、非法 offset 和跨 run 复用。

#### 提交 9：`feat: 添加 AICore 最小 atomic ABI`

- 在 raw integer 上封装 acquire/release/relaxed load/store、CAS、exchange 和 fetch-add。
- 增加 Host model 和 a5sim/a5 compile probe，不实现 bitmap/bitmask 原语。

#### 提交 10：`feat: 实现 resident AICore 启动与退出状态机`

- AICore 校验 descriptor，发布 ready，等待 start，处理空图/DUMMY 并发布 exit/error。
- Model UT 覆盖 ready-before-start、非法 ABI、重复 stop 和 first-error first-winner。

#### 提交 11：`feat: 接通 AICPU 控制面与 M0 回收流程`

- AICPU 只负责拓扑/start、resident launch、exit/error 监控、有界 drain 和回收。
- 删除临时拒绝桩，UT 覆盖 launch failure、timeout、重复 cleanup 和复用无泄漏。

#### 提交 12：`test: 增加 M0 生命周期验收场景`

- A5sim 覆盖空图、DUMMY、重复运行、generation、非法 descriptor、容量不足和有限退出。
- 对可比用例记录 HBG/HBG-AICore 的归一化终态和错误类别。

### 4.3 M1：单 task 纵向链路与 per-callable image

#### 提交 13：`feat: 要求显式 AICore adapter 编译规格`

- 解析并校验 `aicore_adapter`，将源码、include 和 ABI 纳入 artifact cache key。
- UT 覆盖缺字段、路径不存在、相对路径和 cache miss。

#### 提交 14：`refactor: 拆分 AICore 对象编译与最终链接`

- 将 `compile_incore()` 拆成 compile-object 和 link-objects，原 API 输出保持不变。
- UT 比较输出、链接参数、entry、relocation 消除和临时文件清理。

#### 提交 15：`feat: 生成 per-callable linked AICore image`

- 将 incore entry 重命名为 `aicore_callable_kernel_<func_id>`，生成 switch dispatcher，
  与 scheduler/adapter 一次链接。
- UT 覆盖重复 func id、core type 不匹配、缺 hook、未解析 relocation 和稳定 entry。

#### 提交 16：`feat: 定义 callable bundle 格式与校验器`

- 定义 orchestration SO 和 linked image 的版本化 pack/parser。
- UT 覆盖截断、slice 重叠、magic/version、identity 篡改和往返解析。

#### 提交 17：`refactor: 在 CallableArtifacts 中携带可选 AICore image`

- 增加 image data/size/identity 和 bundle orchestration slice，无 bundle 时保持现有行为。
- 扩展 immutable upload/layout UT，验证 optional image 和生命周期。

#### 提交 18：`feat: 缓存并选择 sim per-callable image`

- a5sim 按 identity 缓存 SO/dlopen handle，bind/run 前选择，finalize 统一释放。
- UT 覆盖 hit/miss、绑定切换、失败回滚和 finalize。

#### 提交 19：`feat: 缓存并选择 onboard per-callable image`

- A5 onboard 按 identity 注册并复用 device binary handle，保留至 Worker finalize。
- CANN mock/compile UT 覆盖复用、失败回滚、绑定切换和释放策略。

#### 提交 20：`feat: 为新运行时启用 callable bundle`

- 将 linked image 和 orchestration SO 写入 bundle，并让新 runtime 注册路径解析。
- identity 覆盖 platform/runtime/toolchain/source/include/macro/PTO-ISA/adapter ABI。

#### 提交 21：`feat: 物化单 lane WorkerExecutionSlot`

- 物化 tensor/scalar 参数，校验 task kind、core type、kernel id 和边界。
- 启动前拒绝 MIX、wide、async、`block_num != 1` 和无兼容 core。

#### 提交 22：`feat: 添加基础 Vyukov task-id MPMC`

- 实现固定容量 sequence-counter MPMC，仅存 task id，push full 视为内部错误。
- UT 覆盖 empty/full、FIFO、wrap、sequence 复用和非法 task id。

#### 提交 23：`feat: 执行单个 AIC 或 AIV root task`

- 接通 classify、MPMC push/pop、参数物化、adapter、kernel、completion 和错误返回。
- UT 覆盖调用顺序、一次执行、hook 错误、未知 kernel、队列隔离和 first-error。

#### 提交 24：`test: 增加 M1 单 task 强差分验收`

- A5sim 覆盖 AIC/AIV root、predicate-false、DUMMY、adapter 错误、非法 dispatch 和重复运行。
- 对 HBG/HBG-AICore 比较 graph signature、输出、terminal status 和 completion count。

### 4.4 M2：单核依赖推进与 wake relay

#### 提交 25：`feat: 实现单核首次依赖分类`

- 按稳定 fanin 顺序查找第一个未完成 producer，无未完成项时发布 ReadyQ。
- Model UT 覆盖 chain、diamond、fanout/fanin、多 root、长 fanin、预完成和隐藏分配。

#### 提交 26：`feat: 实现 task-id wake-list 注册协议`

- consumer 先发布 `next_waiter`，再 CAS producer head；观察到 CLOSED 后立即重分类。
- 模型测试穷举 register/close 关键交错、CAS retry 和长链。

#### 提交 27：`feat: 关闭 wake-list 并 relay 等待任务`

- `after_task` 后 release-store completion，exchange head 为 CLOSED，逐个重分类 waiter。
- Model UT 覆盖完成/注册交错、多级 chain、diamond、fanout/fanin 和 predicate producer。

#### 提交 28：`feat: 添加 M2 错误收敛与依赖 DFX`

- 为 queue full、重复 ready、非法 task、adapter/dispatch failure 和异常 wake 链提供 first-error。
- 增加本地分类、注册、relay、push/pop 和 CAS retry 计数，UT 验证有限退出。

#### 提交 29：`test: 增加 M2 图族与交错验收`

- A5sim 覆盖基础图族、DUMMY、predicate、预完成和长 fanin。
- 固定 seed 验证 graph signature、依赖边顺序、输出、错误终态和 HBG 强差分。

### 4.5 M3：基础 MPMC 多核单 lane 调度

#### 提交 30：`test: 验证基础 MPMC 并发 wrap 与 ABA`

- 多 producer/multi consumer 压力覆盖并发 push/pop、容量 wrap、sequence 复用和长序列。
- 每个 task id 必须恰好有一个 pop winner。

#### 提交 31：`feat: 分片首次分类并加入全核启动 barrier`

- 按连续 task-id 区间分配唯一 initial owner，全核分类后通过 release/acquire barrier。
- Model UT 覆盖空区间、慢 classifier、无遗漏/重叠和 barrier 前零执行。

#### 提交 32：`feat: 启用多核兼容 ReadyQ Pull`

- AIC 扫描 AIC/DUMMY，AIV 扫描 AIV/DUMMY，各核以私有旋转起点进行 round-robin。
- Model UT 覆盖并发 roots、wake burst、队列隔离、DUMMY 竞争、慢核和 exactly-once。

#### 提交 33：`feat: 批量提交每核 completed count`

- task 完成只增加 core-local delta；完整空 ReadyQ 扫描后才以 relaxed fetch-add 提交。
- Model UT 覆盖 busy/idle 切换、flush 与新 ready 交错、无早退或漏计。

#### 提交 34：`feat: 实现多核 first-error 有界 drain`

- first-error 后停止新 pop，已领取 task 在安全点收敛，各核发布 phase/current-task。
- Model UT 覆盖并发错误、执行中错误、queue full、慢核、timeout 和重复 stop。

#### 提交 35：`test: 增加 M3 随机 DAG 与队列压力验收`

- 固定 seed 随机 DAG 覆盖 root/wake burst、queue wrap、临界容量、长 chain、宽 fanout/fanin。
- 检查每 task 单次 ready/pop/execute/complete，并对 HBG 做强差分。

#### 提交 36：`test: 完成 M3 公平性与可观测性验收`

- 覆盖慢核、忙核、AIC/AIV 负载不均、DUMMY 竞争、busy/idle 多轮、错误 drain
  和长压力。
- 记录 MPMC CAS retry/sequence wait、wake CAS、completed-count flush、每核任务分布和
  HBG/HBG-AICore 固定 workload 基线，不设统一性能硬阈值。
- 本提交不实现或测试 bitmap、bitmask 或 sharded MPMC 候选协议。

## 5. 里程碑验收矩阵

| 里程碑 | 必须通过的能力 | 明确不包含 |
| -------- | ---------------- | ---------- |
| M0 | 空图/DUMMY、ready/start/exit/error/reclaim、容量与 relocation | callable kernel、依赖、多核 |
| M1 | 单 AIC/AIV root、per-callable image、adapter、参数物化、基础 MPMC | 多 task 依赖、多核 |
| M2 | 单核 initial classify、wake register/close/relay、图族强差分 | 并行分类、多核执行 |
| M3 | 并行分类、全核 barrier、基础 MPMC 多核 Pull、批量完成计数、有限 drain | sharded MPMC、bitmap、bitmask、M4 A5 真机结论 |

## 6. 提交纪律与假设

- 保留用户已加入主计划文档的 M8 内容，设计校正只在其基础上修正归属。
- 提交标题按用户要求使用 Conventional Commits，优先于仓库默认 `Add:/Fix:` 风格。
- 每个生产提交同时携带最小自证 UT，跨场景验收放入独立 `test:` 提交。
- 新 runtime 仅位于 `src/a5/runtime/`，不会被 A2/A3 自动发现。
- M0～M3 不声称 A5 真机正确；M4 在基础 MPMC 上完成真机验证，M8 才允许引入并
  选择 bitmask/bitmap/sharded 优化。
