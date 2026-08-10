# HBG-AICore Scheduler 第一阶段（M0～M3）原子提交计划

## 1. 总体描述与阶段出口

新增 Runtime 命名为 `host_build_graph_aicore`，位于
`simpler/src/a5/runtime/host_build_graph_aicore/`，与现有
`simpler/src/a5/runtime/host_build_graph/` 平级。第一阶段从当前 A5 HBG 复制完整实现代码作为
冻结基准，包括 Host 构图、orchestration、graph/SM/arena layout、AICPU/AICore 和构建配置；
复制提交只允许调整 Runtime 名称、构建目标和必要的 include/namespace，不改变调度语义。

复制完成后，新 Runtime 在自己的目录中独立演进，逐步把首次分类、依赖解析、ReadyQ 领取和
完成传播迁移到 AICore。原 HBG 保持不变并作为正确性与性能 oracle；新 Runtime 不在构建或
运行时包含、链接 HBG 私有源码。HBG 后续变化也不会自动同步，任何同步都必须经过显式差分
评审和独立提交。

第一阶段按以下顺序交付 16 个提交：

1. M0 复制并冻结 HBG 基线，先建立 128B 隔离 sidecar 的功能门禁，再建立独立 Runtime 的
   resident 空图生命周期；
2. M1 跑通单 core、单 root 的真实 AIC/AIV task；
3. M2 在单 core 上跑通 success-only DAG；
4. M3 扩展到多 core，形成 A5 真机可行性与性能报告后暂停评审。

第一阶段只回答“Host 已知完整 DAG 能否由 AICore scheduler 正确、高效执行”。M3 完成后不得
直接进入 M4，必须先根据 ABI v0 的实测结果决定是否继续以及 v1 的具体形态。

当前分支已经实现 M0～M3 的首轮功能闭环：除 M2 的纯 AIC/纯 AIV 图外，同一图中的普通
单 lane AIC/AIV task、分片 initial classify、全兼容核 Pull、CompletionQ 完成转发和跨类型
tensor 可见性协议均已落地并通过 A5sim。正确性矩阵覆盖 1/2/63/64/65/256/1024 task，压力
矩阵覆盖 4096-task multi-root；HBG 以相同 orchestration 作为输出 oracle。A5 真机可见性、长
压力与 HBG/HBG-AICore 性能签署仍须在架构预检恢复后补齐，不能用 A5sim 时延替代。

## 2. 已锁定的实现契约

### 2.1 Runtime、复制基线与公共接口

- 新增显式 A5/A5sim Runtime：`host_build_graph_aicore`；现有 `host_build_graph` 保持不变。
- 基线复制提交记录来源 commit、源目录 tree hash 和复制后差分清单；该提交本身必须能构建并与
  HBG 运行相同的基础用例。
- 基线之后所有 Runtime-specific 修改只发生在 `host_build_graph_aicore/` 中，不反向修改 HBG。
- Worker、Orchestrator、submit、`ChipCallable` 和业务 kernel 编译格式保持不变。
- 不新增 CALLABLE bundle、callable-scoped scheduler image、`aicore_adapter`、环境变量、编译宏或
  pytest 平台参数。
- resident scheduler 使用 Runtime 自身的静态 AICore image；业务 kernel 继续通过现有
  `func_id_to_addr_`、`CoreCallable::resolved_addr()` 和 `PTO2DispatchPayload` 动态调用。

### 2.2 scheduler 与 worker 的动态角色

M3 将依赖 resolver 与 task executor 解耦。所有存在对应 task 类型的 AIC/AIV core 都竞争兼容
ReadyQ 并执行 kernel；只有选中的 resolver core 负责 initial classify 和 CompletionQ 的
complete-and-wake。resolver 仍可同时 Pull/执行，不是永久保留的专用核。M0～M2 不增加公开
scheduler 数量配置：

- M0 只验证空图 resident attach/exit，使用 Runtime 内部的确定性生命周期策略；
- M1、M2 内部固定单个执行 core，先闭合单核 root 和 DAG 正确性；
- 不按“前 N 个 core”永久划分 scheduler/worker，也不在依赖算法形成前升级 `CallConfig`、mailbox
  或 remote protocol；
- M3 只在 `host_build_graph_aicore` 自己的 `PTO2OrchestrationConfig` 中增加
  `aic_dependency_scheduler_limit` 和 `aiv_dependency_scheduler_limit`：`-1` 禁用该类型 resolver，
  `0` 自动选择，正整数为上限；默认仅启用 AIV resolver（AIC=`-1`、AIV=`0`），两者都不限制
  executor 宽度；
- Host loader 从本 Runtime 的 orchestration SO 读取并校验配置；不修改公共 `CallConfig`、binding、
  parent/child mailbox、remote L3 protocol、公共 device runner 或其他 Runtime。非空图若两种 resolver
  都禁用则在启动前拒绝。

### 2.3 Graph ABI v0 与 128B 隔离执行 sidecar

v0 只把复制自 HBG 的 Host graph、task descriptor、task payload 和 fanin local id 当作只读输入。
HBG Host 已完成 graph image 的指针重定位和整体 H2D，image 中的指针在设备侧是有效 GM 地址，
不是 Host 裸指针。AICore 不得把 HBG 的以下对象直接作为多核可变调度状态：

- 一字节连续排列的 ring `completion_flags`；
- 64B `PTO2TaskSlotState` 中的 completion mirror 和 pointer wake links；
- 紧凑排列的 `PTO2ReadyQueueSlot {sequence, slot_state, task_id_snapshot}`。

这些对象当前由 AICPU 使用 `std::atomic` 和 AICPU cache/coherency 语义访问。多 AICore 若对同一
cache line 中的不同字节或普通字段执行 store + DCCI，可能覆盖相邻 core 已发布的更新；CCEC
raw atomic 也不能由 `std::atomic<uint8_t>` 的 Host 表面自动推出。因此 128B 隔离是 M0 的功能
前置条件，不得推迟到 v1/M4。

M0 在 HBG 只读图旁建立 graph-sized execution sidecar：

- `alignas(128) AicoreTaskControlV0`：每 task 独占 128B，保存 word-sized completion、
  `wake_list_head`、`next_waiter` 和最小状态；所有跨核 RMW 字段只用已验证的 AICore GM 原子；
  需要 DCCI 发布的 `next_waiter` 位于独立 cache line，不与 completion/wake-head 原子同行；
- `AicoreReadyQueueV0`：只保存 task id 的有界 Vyukov MPMC；cursor、sequence 和发布字段使用
  AICore 可证明安全的 word-sized GM 原子，不复用 HBG 的 pointer slot 字节布局；
- graph-sized `CompletionQ`：executor 只发布已执行 task id，resolver 成功 pop 后才发布 completion、
  关闭 wake list 并重分类 waiter；slot 的 sequence/task-id 全程使用 raw-GM word 操作；
- `AicoreRunControlV0`：启动/分类 barrier、active AIC/AIV 数、expected/completed、退出状态和聚合
  DFX；
- `AicoreWorkerContextV0`：每核类型/rank、run control、只读 graph、execution sidecar 和私有
  `PTO2DispatchPayload` 地址；
- 每核私有 `local_completed_delta` 与统计计数。

sidecar 由 Host 按实际 task 数规划、做溢出/容量校验并随 run 发布；AICPU supervisor 只完成设备
地址绑定和启动发布。Host/device 类型必须是 C/POD、只含整数/offset/task id，不含指向 sidecar
内部的 Host 裸指针，并补齐 trivially-copyable、standard-layout、`sizeof`、`alignof` 和 `offsetof`
断言。v1/M4 负责正式版本化 immutable record、checked layout 和生命周期，不再首次引入 128B
control 或 task-id ReadyQ。

### 2.4 依赖和完成协议

v0 迁移当前 HBG 已有算法，但把可变状态落在隔离 sidecar 中：

1. initial classify 完整扫描任务的全部 fanin；全部满足则路由到对应 ReadyQ，否则注册到首个
   未满足 producer 的 wake list；
2. waiter 以 task id 通过 CAS 压入 producer control 的 intrusive wake list；producer 已经 CLOSED 时，waiter
   重新扫描全部 fanin并转挂到下一个未满足 producer，或进入 ReadyQ；
3. kernel 返回后，executor 先以 Runtime 通用 whole-data-cache clean + barrier 发布普通输出，再把
   task id 推入 CompletionQ；resolver pop 后发布 128B control 中的 word-sized completion，原子
   关闭并摘取 wake list，对每个 waiter 重新分类；consumer 物化/执行前先 invalidate data cache；
4. 不增加 remaining-fanin counter、逐边 atomic decrement、`WAITING -> READY` CAS、claim CAS、
   retry 或 requeue；
5. task-id ReadyQ 成功 pop 是 task ownership 的唯一线性化点；AICore 不写 HBG completion flag、
   task-state mirror、pointer wake link 或 ReadyQ slot。

每个 resolver 完成 wake relay 后增加自己的 `local_completed_delta`；CompletionQ 暂时为空时把
非零 delta 一次性 `fetch_add` 到全局 completed。core 仅在全局
`completed == expected_task_count` 后退出，不读取其他 resolver 尚未 flush 的尾数。

### 2.5 第一阶段明确不实现

M0～M3 只覆盖 success-only、单 lane、`block_num == 1`：

- 不实现 predicate、DUMMY、hidden allocation、manual scope、wide、MIX、`sync_start` 或 async；
- 不实现 queue-full 恢复、kernel error、first-error、drain、watchdog 或 Runtime timeout；
- 不实现 depth-two、ready bitmap、sharded queue 或自适应仲裁；
- 不承诺 v0 ABI 稳定，也不提前增加 v1 的正式 identity/lifecycle 字段；128B control 和 task-id
  ReadyQ 是正确性基线，不属于可延期字段。

所有图必须在 Host 启动前证明 task window、arena、sidecar 和最大 ready frontier 不超过已分配
容量；不支持的 task shape 必须在设备启动前拒绝，不能依赖 queue-full 后恢复或外层 timeout。
共享设备的外层 timeout 必须保留用于资源回收，但不能算作 Runtime 错误契约验证。

## 3. 通用提交与验证规则

- 基线复制是唯一允许的大型机械提交；不得在其中混入调度行为变化、清理或格式化。
- 其余提交只完成一个可审查目标，不以 300～400 行的形式标准强行拆开同一并发协议。
- 新增 wire/ABI view 时，同一提交补齐 Host/device 编译测试和布局断言。
- 新增并发状态转换时，先提交 CPU model test，再提交设备实现；模型与设备使用相同状态名。
- 新增 Runtime 公共路由或 `CallConfig` wire 时，运行所有相关旧 Runtime 回归。
- 第一阶段新增 SceneTestCase 标记 `manual: true`，用现有 `--manual only` 显式运行。
- 每提交运行定向测试；每个里程碑末尾运行相关 A5sim 全量测试、HBG 差分和公共门禁。

基础 Host 门禁：

```bash
source .venv/bin/activate
pytest tests/ut
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build -j
ctest --test-dir tests/ut/cpp/build --output-on-failure
pre-commit run --all-files
```

A5 真机测试在占用设备前必须执行架构预检，随后通过项目已有 `task-submit` 流程运行；禁止直接
占用共享设备。

## 4. 原子提交序列

### 4.1 M0：复制基线与空图 resident 生命周期（提交 1～6）

#### 提交 1：`docs: define HBG-AICore sibling runtime and frozen-copy baseline`

- 同步修订 scheduler plan、commit plan 和 shared-contention analysis。
- 固化目录关系、复制/独立维护边界、HBG oracle、第一阶段能力边界和 M3 人工闸门。
- 验证：文档链接、术语和 M0～M3 交叉引用一致。

#### 提交 2：`build: route orchestration toolchain by runtime name`

- 由共享 `KernelCompiler` 按 Runtime 名称选择 orchestration toolchain，不增加 build-config capability：
  `host_build_graph` 和 `host_build_graph_aicore` 固定使用 Host GXX；
  `tensormap_and_ringbuffer` 在仿真使用 Host GXX、板端使用 AArch64 GXX。
- 保持各 Runtime 的编译 flags、cache key 和产物不变；未知 Runtime 名称明确失败。
- 验证：RuntimeBuilder/KernelCompiler UT 覆盖上述名称映射和未知名称失败。

#### 提交 3：`refactor: import frozen A5 HBG runtime as HBG-AICore baseline`

- 将 `src/a5/runtime/host_build_graph/` 的实现机械复制到平级
  `src/a5/runtime/host_build_graph_aicore/`。
- 只调整 Runtime 名称、构建目标和必要 include/namespace；保存来源 commit、tree hash 和允许的
  差分清单。
- 验证：新 Runtime 可被动态发现和构建；graph layout golden、基础 A5sim 输出与 HBG 一致。

#### 提交 4：`feat: add 128B-isolated AICore execution sidecar`

- 为 HBG 只读 descriptor/payload/fanin 定义 AICore ABI view；新增 graph-sized 128B task control 和
  task-id-only MPMC，不把 HBG mutable state 暴露给 AICore 写。
- 从 FDWIC 收窄复制 word-sized raw GM load/store/CAS/exchange/fetch-add、DCCI 和 barrier；明确
  completion、wake head、next waiter 和 ReadyQ slot 的逐字段发布协议。
- 验证：POD/布局断言、容量溢出、A5sim 编译；A5 手工 probe 必须先通过同 control 邻域压力、
  completion/wake 并发、queue slot 发布和普通数据可见性。任何邻位 clobber 都阻塞 M0。

这是冻结复制后的第一个功能提交。后续 resident lifecycle、root 执行和 DAG
调度提交都以本提交及其 A5 probe 通过为前置条件。

#### 提交 5：`feat: run resident scheduler on an empty graph`

- 用 FDWIC resident attach/start/finish 骨架替换复制来的逐任务 AICPU dispatch loop。
- AICPU supervisor 建立 run/worker control，启动 scheduler，等待空图收敛并发布退出。
- 空图不进入 Pull、不创建伪 task；inactive core 只参与受控 attach/exit。
- 验证：A5sim 多轮启动/退出，无 hang、重复完成或遗留状态。

#### 提交 6：`test: accept M0 copy parity and empty lifecycle`

- 增加 manual A5sim/A5 空图和 primitive SceneTestCase。
- 重复 submit/finalize，验证 barrier、active/inactive core、退出计数和资源回收。
- M0 出口：独立复制 Runtime 与 HBG 基线一致，并能完成 resident 空图生命周期。

### 4.2 M1：单 core、单 root、真实 kernel（提交 7～9）

#### 提交 7：`test: lock task-id queue and graph-view parity`

- 复用现有 `test_ready_queue.cpp` 的空/满、环回和 MPMC stress 轨迹，但验证 v0 task-id queue
  自身的 cursor、sequence、task id 和 AICore raw-atomic 发布协议。
- HBG ReadyQ 只作为 Vyukov 算法参考，不要求与其 pointer slot 字节布局一致；额外验证相邻 queue
  slot 并发写不会 clobber。

#### 提交 8：`feat: execute one root through existing callable addresses`

- 从复制后的只读 graph image 读取唯一零 fanin task，并把 task id 推入 sidecar AIC/AIV ReadyQ。
- scheduler pop 后在本核 payload 中物化参数，通过现有 `func_id_to_addr_` 和
  `CoreCallable::resolved_addr()` 调用真实 kernel。
- kernel 前沿用 HBG payload/input DCCI，kernel 后完成可见性顺序再发布 completion。

#### 提交 9：`test: accept single AIC and AIV roots`

- A5sim/A5 分别使用仓内已验证的 AIC/AIV kernel。
- 覆盖重复 submit、输出、task owner 和 exactly-once。
- M1 出口：单 AIC 或 AIV scheduler core 可执行真实 root，无 bundle、adapter 或直链接业务镜像。

### 4.3 M2：单 core success-only DAG（提交 10～12）

#### 提交 10：`test: model HBG first-unmet wake protocol`

- CPU 模型逐项对应 HBG `classify_fanin_state`、`register_wake` 和 completion wake drain。
- 覆盖 register-before-close、close-before-register、多 waiter、多 fanin 转挂和单 fanin 快路径。
- 验证每个 task 最多 ready 一次，CLOSED 后 waiter 不丢失且所有 fanin 满足后才进入 ReadyQ。
- 当前增强用例还直接调用生产 `classify/complete-and-wake` 接口，覆盖 16 waiter 并发 drain 和
  40 轮 close/register 竞态；CPU 模型另覆盖 32 路 fanin，不以独立模型替代生产协议验证。

#### 提交 11：`feat: migrate HBG classify and wake relay to AICore`

- 单 core 首次扫描同时完成 root/已满足 task 路由和非 root waiter 注册。
- 完成路径依次执行普通数据发布、128B control 的 word-sized completion、wake-list close/drain 和
  waiter 重分类。
- 直接使用 128B control 中的 task-id wake links 和 task-id ReadyQ，不增加 remaining-fanin。

#### 提交 12：`test: accept single-core DAG families`

- manual A5sim/A5 覆盖 chain、repeated diamond、fanout、32-way layered fanin、multi-root 和固定
  seed 随机 DAG。
- 将输出、task 执行次数和依赖顺序与原 HBG/CPU reference 对比。
- 除显式状态检查图外，使用真实 AIV kernel 和 TensorMap 推导依赖覆盖 chain、diamond、
  multi-root，并以相同 orchestration 在 HBG 上运行作为输出 oracle。
- M2 出口：单 core DAG 无漏 task、重复 task 或依赖可见性错误。

### 4.4 M3：多 core 正确性与真机可行性（提交 13～16）

#### 提交 13：`feat: shard initial classify behind a global barrier`

- 选中的 AIC/AIV resolver 按稳定连续 task-id 区间扫描，各 task 只分类一次；全部 resolver 完成
  ready/wake 初始化后才进入 RUN phase。
- 在 HBG-AICore 本地 orchestration config 中增加独立 AIC/AIV resolver limit 及 Host validation；
  `-1/0/正数` 分别表示禁用、自动和上限。resolver 不是专用核，仍参与兼容 ReadyQ Pull；未选中的
  核只 Pull/执行。
- 验证 task 数小于、等于、大于 core 数，不整除分片以及 AIC/AIV 数量不同。

#### 提交 14：`feat: run multi-core pull and completion accounting`

- AIC/AIV core 分别竞争兼容 ReadyQ，成功 pop 是唯一 ownership 线性化点。
- executor 在 Runtime 发布 kernel 输出后推送共享 CompletionQ；resolver 优先 pop CompletionQ，完成
  completion/wake relay 后批量 flush 本地 completed delta。
- 每 core 只写自己的 worker context/payload 和 per-core DFX；所有兼容 executor core 都参与 Pull，
  resolver limit 不缩窄执行宽度。
- 全局 `completed == expected` 后退出；不增加 claim CAS、requeue、idle bitmap 或 per-core tail scan。

#### 提交 15：`test: stress multi-core scheduling races`

- 覆盖 queue 多次环回、并发 wake register/close、ready-after-empty-scan、最后一次 flush 和不同
  task/core 数组合。
- 使用同质和交替 AIC/AIV 标准 DAG family 与固定随机图，对单类型/双类型、`1/auto` resolver cap
  验证 exactly-once、
  计数守恒和无饥饿；fairness 只记录分布，不设置硬门槛。

#### 提交 16：`perf: report M3 feasibility and pause`

- 在 A5 真机执行第 5 节矩阵，逐 case 保存完整 `[STRACE]` 原始日志和 JSON 汇总。
- 报告 HBG/HBG-AICore、环境身份、startup/classify、queue、wake、kernel、完成计数和 per-core 分布。
- 使用 `simpler_setup.tools.hbg_aicore_m3_compare` 报告 p50/p95/p99；不删除离群点或设置自动通过线，
  但任何超过 +2% 的 p50 变化必须标记 review。真机报告提交后暂停，由人工决定是否进入 M4。

## 5. M3 A5 真机采样与报告

### 5.1 正确性矩阵

- task 数：`1`、`2`、`63`、`64`、`65`、`256`、`1024`；
- DAG：chain、repeated diamond、fanout、32-way layered fanin、multi-root、随机 DAG；
- random fanin `0～4`；resolver 覆盖默认 AIV-only、AIC-only、双类型 auto 和显式 cap；
- AIC/AIV 同质图和交替 mixed 图分别比较 HBG 输出、执行次数和依赖顺序。

### 5.2 拓扑 microbench

- task 数：`64`、`1024`、`4096`；
- DAG：chain、32-way layered fanin、multi-root；
- resolver 配置：单类型、双类型 `1` cap 和 auto；executor 始终使用全部兼容核；
- 每点预热 10 次、记录 30 次，报告 p50/p95，不报告样本量不足的 p99。

### 5.3 真实 kernel benchmark

- AIV 使用仓内已验证的 vector-add；AIC 使用已验证的 paged-attention QK；
- 每个 workload 使用 256 个独立 root，调度并发上限为 `1/2/4/auto`；
- 每点预热 10 次、记录 100 次，报告 p50/p95/p99；
- repetition 交替执行 `HBG -> HBG-AICore` 与 `HBG-AICore -> HBG`，降低温度和时间漂移。

### 5.4 原始字段和汇总

JSONL 至少包含 simpler commit、dirty 状态、HBG 复制来源、芯片/驱动/CANN/PTO-ISA/编译器版本、
编译 flags、graph family、task 数、core type、AIC/AIV resolver limit、实际 resolver/executor 数、seed、
所有原始时延与失败标记。

聚合统计至少包含 graph build、H2D、scheduler startup、initial classify、ReadyQ push/pop/miss、
wake/reclassify、kernel cycles、completed、总 device latency、吞吐和 per-core task 分布。v0 不要求
bundle/adapter identity 或 graph-sized per-task ready/start/end 时间戳。

### 5.5 当前验证与签署状态（2026-08-10）

- Runtime-local config、sidecar、MPMC 环回、wake register/close race 和 CompletionQ relay 的定向
  C++ 测试通过；公共 `CallConfig`、mailbox 和 remote protocol 保持原 ABI。相关 Python 回归和性能
  汇总测试通过。
- `tests/st/a5/host_build_graph_aicore` 在 A5sim 上以 `--manual include` 执行：
  HBG-AICore 8/8 通过，同一场景的 HBG oracle 5/5 通过。
- `tools/benchmark_rounds.sh` 已支持同 case 的 `host_build_graph`/`host_build_graph_aicore` 采样并保留
  原始日志；`simpler_setup.tools.hbg_aicore_m3_compare` 保留全部样本，输出 p50/p95/p99，并将 p50
  超过 +2% 标为人工 review。A5sim 的 4096-task multi-root 采集烟测得到 Host p50 +0.23%、Device
  p50 -0.38%，只证明采集链路可用，不作为性能签署。
- 当前主机的 `npu-smi` 初始化 DCMI 失败（`ret is -8005`），无法识别芯片型号，架构预检因此拒绝
  启动 `--platform a5`。恢复设备环境后仍必须完成 5.1～5.4 的同设备交替采样、长压力和原始日志
  归档；在此之前 M3 功能闭环可评审，但 A5 真机可行性与性能出口保持未签署。

## 6. 里程碑验收

| 里程碑 | 必须通过的功能 | 阶段出口 |
| ------ | -------------- | -------- |
| M0 | 冻结复制、独立构建、ABI/atomic seam、resident 空图生命周期 | 新 Runtime 可独立启动和收敛 |
| M1 | 单 AIC/AIV root、动态 kernel 调用 | 单 task exactly-once |
| M2 | 单 core 标准 DAG family | HBG first-unmet/wake 协议正确 |
| M3 | 多 core、mixed DAG、CompletionQ、完成计数和性能矩阵 | A5sim 已通过；A5 报告后暂停评审 |

所有里程碑必须证明：未选择 `host_build_graph_aicore` 时，现有 HBG/TRB 行为、格式和测试结果
不变。M3 评审通过前不得提前实现 v1、adapter、predicate、DUMMY、first-error、wide/MIX 或
depth-two 调度。
