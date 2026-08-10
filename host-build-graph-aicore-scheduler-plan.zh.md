# Host 提前构图与 AICore 解依赖调度实现计划书

| 项目 | 内容 |
| ---- | ---- |
| 状态 | M0～M2 已实现并通过 A5sim；M3、A5 真机与性能签署待完成 |
| 文档类型 | 架构设计、当前实现状态与后续实施计划 |
| 首发平台 | A5sim、A5 |
| 方案代号 | `host_build_graph_aicore`，简称 HBG-AICore |
| 基线 | `host_build_graph`（HBG） |
| 参考 | `../simpler-dist/src/a5/runtime/fully_distributed_within_core/`（FDWIC）实现、设计和验证 |
| 日期 | 2026-08-10 |

## 1. 结论、目标和边界

### 1.1 架构结论

当前分支已新增显式 Runtime 变体 `host_build_graph_aicore`，并完成 M0～M2：单 root 与单核
同质 DAG 的首次分类、依赖等待、完成传播和任务领取已在 A5sim 验证；验证同时覆盖生产
close/register 并发协议、CPU 模型的长 fanin，以及真实 TensorMap AIV 数据流的 chain、diamond
和 multi-root HBG 差分。A5 真机可见性、性能收益和 M3 多核扩展尚未证明，完成后仍须由人工
架构评审决定是否进入完整实现；现有 `host_build_graph` 的默认行为不变。

新 Runtime 位于 `simpler/src/a5/runtime/host_build_graph_aicore/`，与现有
`simpler/src/a5/runtime/host_build_graph/` 平级。第一阶段从当前 A5 HBG 复制完整实现代码作为
冻结基准，包括 Host 构图、orchestration、graph/SM/arena layout、AICPU/AICore 和构建配置。
基线复制提交只允许调整 Runtime 名称、构建目标和必要的 include/namespace，不改变调度语义，
并记录来源 commit、源目录 tree hash 和复制后差分清单。

复制完成后，新 Runtime 在自己的目录中独立演进，逐步把首次分类、依赖等待、完成传播和
单 lane 任务领取迁移到 AICore。原 HBG 保持不变并作为正确性和性能 oracle；新 Runtime 不在
构建或运行时包含、链接 HBG 私有源码。HBG 后续变化不会自动同步，任何同步都必须经过显式
差分评审和独立提交。第一阶段允许使用实验性 Graph ABI v0，目标是尽快形成可信性能结论，
不提前承担完整异常、diagnostics、并发 prepare 和全部 task shape 的工程成本；但 AICore 多核
写安全所需的 graph-sized 128B task control 和 task-id ReadyQ 必须从 M0 建立，不能延期。

人工评审通过后，第二阶段用正式 Graph ABI v1 重建 immutable record、版本化 control、容量和
生命周期契约，并逐步补齐单 lane、wide、MIX、`sync_start`、deferred completion、diagnostics
以及 HBG 的优化能力。AICPU 最终只保留设备启动、AICore 镜像装载、硬件资源管理、终止和错误
回收职责，不再参与逐任务解依赖决策。

`simpler-dist` 的 FDWIC 用于参考 AICore 常驻调度、跨核原子状态、任务领取、可见性协议、
MIX/多核协作和硬件验证方式。允许把其中适用的 atomic、AICore attach/lifecycle、AICPU
control-plane 和独立 runtime 接入代码复制到新目录后裁剪并独立维护；新 Runtime 不形成对
`simpler-dist` 的构建或运行时依赖。本方案不迁入 FDWIC 的设备侧 Orchestrator replay、
claim cursor、per-core TensorMap、固定容量 task ring 或业务工作负载。

### 1.2 目标

1. 用最小修改范围在 A5 上证明 AICore 解依赖可运行、可扩展，并形成可复现的 HBG 性能对比。
2. 第一阶段每个里程碑均可独立验证；M3 输出性能报告后暂停，由用户人工决定继续、先优化或终止。
3. 第二阶段 Host 在设备启动前构造完整 DAG，并完成容量校验、地址重定位和不可变图数据发布。
4. AICore 在图发布后并行完成首次依赖分类，并在运行期完成依赖释放和任务调度。
5. 保持现有 Orchestrator、任务提交、Worker、`ChipCallable` 和 kernel 编译格式不变；M0～M2
   不增加 scheduler 配置，M3 多核动态策略形成后才按需增加可选调度并发上限，新 Runtime 始终
   显式选择。
6. 最终对齐项目启动时冻结的 A5 HBG 能力和优化矩阵，并建立 A5sim/A5 强差分证据。
7. 先完成 AIC/AIV 单 lane，再增加 wide、MIX、`sync_start`，最后完成组合能力、异常、DFX、
   生命周期和长期稳定性验收。

### 1.3 非目标

- 本计划不支持 A2/A3。
- 不把新 Runtime 设为默认 Runtime，也不自动替换 HBG。
- 不实现 HBG 基线本身不具备的任务重试或 AICPU 调度回退。
- 不允许执行期继续增长整图或动态扩容图存储。
- 不要求与 HBG 产生完全相同的合法执行顺序，只要求依赖关系、结果和错误语义等价。
- 不把 `simpler-dist` 的设备 Orchestrator replay 迁入本方案。
- SIMT 分类、CompletionQ 和替代 ReadyQ 是最终验收后的候选优化，不阻塞主计划交付。

### 1.4 关键术语

| 术语 | 含义 |
| ---- | ---- |
| Host graph image | Host Orchestrator 构造完成、可重定位并整体复制到设备的完整图数据 |
| Graph ABI v0 | 第一阶段实验 ABI；HBG descriptor/payload/fanin 只读，加 graph-sized 128B task control、task-id ReadyQ 和 run/worker control |
| Graph ABI v1 | 第二阶段正式 ABI；immutable record、版本化 128B control、checked layout 和生命周期 |
| Run/worker control | v0 由 AICPU 创建的最小启动 barrier、完成计数、退出、每核 payload 地址和聚合 DFX |
| Immutable task record | 构图完成后只读的任务描述、fanin、参数元数据和资源需求 |
| Task control | 每任务独立的可变调度状态，固定 128B、缓存行隔离 |
| Initial classify | 所有任务执行前，由 AICore 并行判断 root、已满足或等待状态 |
| Wake relay | 生产者完成后关闭并摘取 wake list，再重新分类其消费者的过程 |
| Frozen HBG baseline | M0 记录的 HBG commit、能力、优化、测试和性能基线；后续 HBG 新能力不自动扩大本计划 |

### 1.5 分阶段交付和人工闸门

| 阶段 | 目的 | 主要范围 | 退出方式 |
| ---- | ---- | -------- | -------- |
| 第一阶段：M0～M3 | 最小安全基线验证 AICore 解依赖和性能 | Graph ABI v0、128B 隔离 control、depth-one、成功路径、AIC/AIV 多核单 lane | M3 输出报告后暂停，由用户人工评审 |
| 第二阶段 A：M4～M6 | 单 lane 完备 | Graph ABI v1、真实数据可见性、异常/DFX、deferred completion、depth-two | 单 lane 独立签收 |
| 第二阶段 B：M7～M8 | 完整 task shape | wide、MIX、`sync_start` 及其资源协议 | 各 task shape 独立签收 |
| 第二阶段 C：M9 | 完整功能完备 | 冻结 HBG 能力/优化矩阵、组合测试、长期稳定性 | 完整 Runtime 准入 |
| 后续 O1～O3 | 可选性能优化 | SIMT、CompletionQ、ReadyQ 候选 | 各自评审，不阻塞 M9 |

第一阶段代码可以合入，但 Runtime 默认关闭。Host/model/build 测试可进入常规 CI；第一阶段
明确只支持成功路径，A5sim/A5 设备用例只能手工 opt-in，不进入默认回归。共享设备外层 timeout
仍必须用于资源回收，但不属于 Runtime 错误语义。第二阶段只能在 M3 报告经用户人工评审并
明确启动后实施。

## 2. 总体架构方案

### 2.1 端到端流程

第一阶段只替换解依赖和单 lane 领取位置，数据路径保持最短：

```text
Host Worker
  -> 选择 host_build_graph_aicore（实验、depth-one）
  -> 使用现有 ChipCallable 注册和动态 kernel 地址
  -> 在冻结复制的 HBG Host 路径同步运行 Orchestrator 并生成完整图
  -> 沿用 HBG 只读 graph/SM/arena descriptor、payload 和 fanin
  -> Host 创建 graph-sized 128B task control 与 task-id ReadyQ sidecar
  -> AICPU 绑定 run/worker control 并启动 Runtime 静态 AICore image
  -> AICore 完成 classify、wake relay、Pull 和单 lane kernel 执行
  -> 记录性能数据；只运行已知成功用例
```

第二阶段 Graph ABI v1 的目标流程为：

```text
Host Worker
  -> 选择 host_build_graph_aicore
  -> 编译并装载 Host Orchestrator SO
  -> 在新 Runtime 自己的 Host 路径同步执行 Orchestrator，构造完整图
  -> 冻结图；校验任务数、fanin、队列、sidecar 和设备内存上限
  -> 生成 AICore execution sidecar 和地址重定位表
  -> 使用 Runtime 静态 scheduler image 和现有动态 child-kernel 地址
  -> H2D 发布 graph image + sidecar
  -> 启动 AICPU 控制核和 AICore 常驻 scheduler

AICore scheduler
  -> 并行 initial classify
  -> 全核 barrier，确认所有任务已入 ready queue 或 wake list
  -> 每个空闲 core 按自身类型轮询共享 ready queue
  -> 原子领取兼容任务；领取者直接成为执行者
  -> 在本 core 私有 execution slot 中物化任务参数
  -> 按 Runtime 可见性协议观察输入
  -> 通过动态函数地址执行 kernel
  -> 按 Runtime 可见性协议发布输出
  -> 发布完成状态，摘取 wake list 并重新分类消费者
  -> 释放资源，继续调度

AICPU control plane
  -> 监控终止、超时和 first-error
  -> 必要时进入 fail-stop drain
  -> 收集最终状态并返回 Host
```

两个 ABI 都必须满足两个全局顺序：图和控制区全部发布后才能启动首次分类；所有参与分类的
AICore 完成 barrier 后才能执行普通任务。第二个顺序避免“分类尚未注册 waiter、生产者已经
完成”的启动竞态，也让首次分类可以按 task-id 区间并行分片。

### 2.2 图表示与 execution sidecar

新 Runtime 始终拥有自己的源码和构建产物，不读取或链接原 HBG 的私有实现。Graph ABI v0
来自 M0 对 HBG Runtime 源码的冻结复制，但只保留其 Host 构图、TensorMap、task descriptor、
task payload、fanin 和 SM/arena 作为只读 graph input。HBG Host 已完成 graph image 的指针
重定位和整体 H2D，因此 image 中的指针在设备侧是有效 GM 地址，不是 Host 裸指针。

HBG 的 `completion_flags` 是连续排列的 `std::atomic<uint8_t>`，其 64B `PTO2TaskSlotState` 和紧凑
ReadyQ slot 也混合了原子字段、普通字段与 device pointer。这些布局对当前 AICPU scheduler
成立，但不能直接推出多 AICore 的 sub-word atomic、DCCI 邻位安全或 CCEC ABI 安全。v0 因此从
M0 建立最小、graph-sized execution sidecar：

- `alignas(128) AicoreTaskControlV0`：word-sized completion、task-id wake head/next 和最小状态；
  每 task 独占 128B，DCCI 发布字段与 concurrently-mutated atomic 字段分处不同 cache line；
- `AicoreReadyQueueV0`：只保存 task id 的有界 Vyukov MPMC，全部共享发布字段使用已验证的
  word-sized AICore GM 原子；
- `AicoreRunControlV0`：启动/分类 barrier、active AIC/AIV 数、expected/completed、退出状态和
  聚合 DFX；
- `AicoreWorkerContextV0`：每核类型/rank、run control、只读 graph、sidecar 和私有
  `PTO2DispatchPayload` 地址。

sidecar 由 Host 按实际 task 数和 ready frontier 规划并校验，AICPU 只绑定设备地址和发布启动。
sidecar ABI 使用 C/POD、offset/task id 和显式布局断言，不含指向自身区域的 Host pointer。v0
不承诺稳定缓存兼容面，但 128B 隔离与 task-id queue 是正确性基线，不是 v1 优化。

Graph ABI v1 在 M4 整体替换 v0：Host 继续保持相同公开 Orchestrator/submit 语义，但重新生成
HBG-AICore 自己的 immutable graph image，并正式版本化从 v0 已存在的 execution sidecar。
graph/control identity 同时升版，所有 v0 graph cache 强制失效并重建；生产 Runtime 不同时兼容
v0/v1。一致性由 normalized graph 强差分约束。

Graph ABI v1 固定使用以下对象：

| 对象 | 可变性 | 主要字段和职责 |
| ---- | ------ | -------------- |
| `AicoreGraphLaunchDesc` | 启动后只读 | 图基址、任务数、各区域容量、ready-set 地址、拓扑、镜像身份、DFX 地址 |
| `AicoreTaskRecord` | 只读 | task id、kernel id、fanin、资源 shape、block 数、predicate、参数元数据 |
| `AicoreTaskControl` | 可变 | 延续 v0 的 128B 隔离；completion、wake head、next waiter，以及 M7/M8 使用的 task epoch、参与者计数和错误码 |
| `AicoreRunControl` | 可变 | `expected_task_count`、批量更新的全局 `completed_count`、first-error 和退出状态 |
| `TaskReadySet` | 可变 | v1 的 M4 至 M9 使用只保存 task id 的 Vyukov bounded MPMC；替代 ReadyQ 属于 O3 候选优化 |
| `WorkerExecutionSlot` | 每核私有可变 | v1 中由领取者物化 tensor/scalar 参数、`LocalContext` 和 `GlobalContext`；单 lane 路径无跨核发布 |
| `AicoreSchedulerDfx` | 可变 | 分类、领取、空轮询、等待、完成、重分类、冲突和错误计数 |

`AicoreTaskControl` 从 v0 起固定为 128B 并做静态布局断言。高频原子控制字段不能与 kernel
普通输出数据或其他 task control 共用缓存行；需要 DCCI 发布的普通字段也不能与并发原子字段
同行，防止 clean/invalidate 覆盖相邻更新。不可变记录和可变控制分离，既降低误写风险，也便于
HBG 与新 Runtime 对同一图做差分。

v1 布局在 Host 构图冻结后按实际图统计生成：M4 建立 descriptor、record、control、run-control、
fanin/wake、per-core execution slot 和 ReadyQ；后续里程碑只增加已经启用能力所需的 DFX、
participant 和 deferred-completion 区域。现有 HBG 的固定 SM/arena 写法只作为 checked-layout
参考，不能继续充当 v1 的生产布局。

### 2.3 首次依赖分类

1. 每个 scheduler core 根据 task id 连续区间领取一段任务，避免分类阶段共享游标热点。
2. 对任务的 fanin 做完整扫描。
3. 全部生产者已完成或任务没有 fanin：按资源 shape 路由到 AIC、AIV、MIX 或 DUMMY
   ready queue。
4. 发现第一个未完成生产者：把当前任务注册到该生产者的 wake list，并停止本次扫描。
5. 若注册时发现生产者的 wake list 已关闭，立即重新扫描，不得把任务遗留在关闭链表上。
6. 分类结束后进入全核 barrier；barrier 前禁止领取普通任务执行。

隐藏分配等 Host 阶段已经完成的任务，应在 graph image 中预置完成状态，因此其消费者
会在首次分类时自然越过这些 fanin。

### 2.4 运行期 wake relay

采用复制后 HBG 的“扫描 fanin、等待第一个未满足生产者”算法，但把可变状态落在 128B
execution sidecar，再把同一协议迁移到 AICore：

1. 生产者执行完成后，先按 Runtime 的普通数据可见性协议发布 kernel 输出。
2. 以 AICore 已验证的 word-sized publish store 写入 task control completion；不写 HBG 的一字节
   completion flag 或 task-state mirror。
3. 原子交换 task-id `wake_list_head` 为关闭哨兵，禁止新 waiter 注册。
4. 遍历摘取的 task-id waiter 链；每个 waiter 重新扫描全部 fanin。
5. waiter 若已满足，进入对应 ready queue；否则注册到新的第一个未完成生产者。
6. waiter 注册与生产者关闭并发时，以 CAS 结果和关闭哨兵决定重试，不允许丢唤醒。

此算法不维护完整 fanout 邻接表，内存开销与任务数和 fanin 成正比；一个消费者在每次
被唤醒后最多只挂在一个生产者上。代价是可能重复扫描 fanin，需用 DFX 统计
`fanin_scan_count` 和 `wake_reclassify_count` 验证真实成本。

### 2.5 Ready queue 和仲裁

- v0 使用 AICore-safe task-id 有界 Vyukov MPMC。HBG queue 只提供算法和 CPU 轨迹 oracle，不复用
  其 pointer slot 字节布局；cursor、sequence 和 task-id 发布均使用 word-sized GM 原子。
- v0 所有图必须在启动前保证 task window、arena、sidecar 和最大 ready frontier 不超过已分配
  容量；第一阶段不定义 queue-full 运行期恢复语义，因此超限必须在 Host 阶段拒绝。
- v1 正式版本化 task-id MPMC，并在构图结束时按最坏 ready frontier 计算容量；DUMMY/predicate
  容量也从对应能力启用的里程碑开始计算。
- 队列选择采用 executable-first：先过滤当前 core/cluster 能执行的 shape，再在可执行队列间
  round-robin，避免固定优先级长期饿死某类任务。
- 单 lane ReadyQ 按 AIC/AIV 类型共享；任务入队时不绑定具体 core，空闲 core 只轮询自身可执行的
  queue。MPMC 成功 pop 后，领取者直接执行，不再进行二次 core 选择或 task-local claim。
- v1 中 MPMC `push` 返回 full 属于内部容量计算错误，设置 first-error，不能无限自旋。

M1 至 M3 使用 v0 task-id Vyukov MPMC；M4 至 M9 使用版本化 v1 task-id Vyukov MPMC。M9 完成后，
O3 再基于固定正确性和性能
基线对比 sharded MPMC、per-shard 非空 bitmask 和两级 ready bitmap，并按正确性、公平性、
A5 可见性与性能证据决定是否替换基线。具体协议和采用条件见
[AICore 共享竞争与 ReadyQ 替代方案分析](host-build-graph-aicore-shared-contention-analysis.zh.md)。

### 2.6 AIC/AIV 单 lane 调度

基础单 lane 调度采用对称 Pull 模型。每个 AIC/AIV core 既是 scheduler，也是 worker：core
空闲时进入调度循环，从与自身类型兼容的共享 ReadyQ 主动领取 task id；成功领取的 core
直接在本 core 上执行该任务。任务在入队时不绑定具体 core，不存在“scheduler core 领取后再
选择目标 lane”的二次派发。

M0～M2 不暴露固定 scheduler 数量。M0 仅验证空图 resident 生命周期，M1/M2 使用 Runtime 内部
单核策略。M3 引入多核 classify/Pull 时，Runtime 根据图规模、设备拓扑和空闲 core 动态决定
实际调度并发度。若正确性和性能矩阵需要可控上限，M3 才增加可选
`CallConfig.aicore_scheduler_limit`：`0` 表示自动策略，正整数只限制同时进行调度工作的 core
数量，不指定永久 scheduler 核，也不限制能够执行 kernel 的 worker 总数。

M1 至 M9 中，ReadySet 的唯一消费操作就是 task 所有权转移：MPMC 成功 pop 的 core
取得对应 task 所有权。O3 候选实现也必须保留“候选结构的唯一消费操作即所有权转移”
的约束。单 lane 不维护独立 claim 字段或二次
claim CAS。exactly-once 依赖“每 task 只发布一次”和“每 ReadySet 元素只被领取一次”两个
不变量。单 lane 不 requeue/retry，也不会因为“目标 lane 不可用”重新入队。core 只有在未执行
任务时才进入调度循环，因此空闲状态由控制流隐式保证，不维护共享 AIC/AIV idle bitmap。

依赖等待位置由 wake list 和 ready set 表达，不额外维护 `WAITING -> READY` 共享 CAS。initial
classify 和 wake relay 必须保证 waiter 所有权只转移、不复制，task 一生只发布到 ReadySet
一次。跨 run 复用时，ReadySet 必须在所有 core 退出后整体重建或按 generation 隔离，不能让
旧 queue slot/ready bit 进入新一轮。详细分析见共享竞争分析文档。

v0 每个 core 继续使用 HBG `PTO2DispatchPayload` 布局，从只读 task payload 读取参数，并通过
现有 `func_id_to_addr_` 和 `CoreCallable::resolved_addr()` 动态调用业务 kernel。v1 才从
`AicoreTaskRecord` 物化到独立 `WorkerExecutionSlot`。两种 slot 均由领取者写入并由同一 core
同步消费，不需要跨核 epoch、doorbell 或 ack；kernel 返回后，该 core 完成状态发布和 wake
relay，再回到调度循环。

从 M5 开始，predicate 在依赖满足后读取。若任务被判定为 DUMMY 或 predicate 为 false，则 scheduler
执行语义上的空完成，不启动业务 kernel，但仍按普通任务发布完成并唤醒消费者。

每个 core 私有维护 `local_completed_delta`。完成状态发布和 wake relay 全部结束后，只增加
本地 delta；当本核完整扫描所有兼容 ReadyQ、均无任务可领取时，才以一次 relaxed
`fetch_add` 把非零 delta 批量提交到 `AicoreRunControl.completed_count`，成功后清零。提交后若
又出现 ready task，则进入下一段 busy period 并在下次空闲时提交。这里不维护共享 idle bitmap，
也不先判断所有 core 是否全局空闲。

### 2.7 AICore 可见性契约

A5 上普通 GM 数据不存在可假定的跨核自动一致性。调度控制原子和 kernel payload 的
发布必须分成两套协议：

- task control 和 queue sequence 使用经过硬件验证的 AICore 原子操作和
  acquire/release 顺序。
- graph record、producer output 和其他跨核普通数据使用 DCCI 加 barrier 的显式发布/观察。
- 单 lane payload/execution slot 是领取 core 的私有数据，不参与核间一致性协议。
- 控制原子字段与被 DCCI 操作的普通数据至少缓存行隔离；不同 task control 之间按 128B 隔离。

v0 沿用 HBG 的通用顺序：读取 graph/payload 前执行必要 DCCI，kernel 返回后完成普通数据发布，
再写 sidecar completion 和关闭 wake list。128B control 邻位安全、word-sized 原子和 task-id queue
发布必须在 M0 A5 probe 中成立；第一阶段只复用已在 HBG 中验证的 kernel 和参数形状，
并通过跨 AIC/AIV 依赖 seam 验证该顺序。是否需要业务感知、按 tensor 范围操作的 adapter，留到
M3 数据评审和 v1/M5 设计决定；v0 不增加 callable hook 或静默扩展 kernel ABI。

### 2.8 Runtime 镜像与 Worker 生命周期

HBG-AICore 使用和其他 Runtime 相同的静态 AICore scheduler image。现有 callable 注册流程已经
上传整个 `ChipCallable`、计算每个 child kernel 的设备地址，并按 content hash 管理 device buffer
生命周期；scheduler 通过 `func_id_to_addr_` 取得 `CoreCallable`，再调用其 `resolved_addr()`。
因此 v0 不增加 bundle、callable-scoped image、额外 `dlopen`/`rtRegisterAllKernel` cache，也不
使用 FDWIC 的 `build_aicore_with_extra_sources`。

FDWIC 需要直链接镜像是因为它在 AICore replay orchestration；HBG-AICore 的图已经由 Host 完整
构造，不具备相同前提。M0 空图直接启动 Runtime 静态 image，M1 起用现有动态 child-kernel 地址。

v1 若引入独立 graph/cache 生命周期，每个 `PreparedExecution` 必须固定持有不可变 callable
artifact 引用、graph/sidecar、pipeline slot、arena bank、lease generation 和 run epoch；不能用
一个可变的“当前 graph”作为运行事实。Runtime image 仍保持 runtime-scoped，除非 M3 评审得到
必须直链接的硬件证据。

### 2.9 AICPU 控制面

AICPU 的最终职责为：

- 启动 Runtime 静态 AICore scheduler image。
- 建立物理 core id、core type、cluster 与 worker control 的映射。
- 发布 graph launch descriptor，等待 scheduler ready。
- 监控超时、first-error、deferred completion、正常完成和退出确认。
- 执行 AICore 无法完成的寄存器/MMIO 控制和最终资源回收。

AICPU 不扫描 fanin、不维护普通 ready queue，也不逐任务选择执行 core。这样可以明确证明
“解依赖在 AICore”，同时保留 AICore 无权访问的硬件控制边界。

实现可复制 FDWIC `runtime/dist_engine/aicpu/control_plane.h` 的一次性初始化、拓扑、GM state
发布和 worker ready/finish 骨架，并替换为本方案的 graph/run control。公共平台已有的寄存器、
MMIO、launch 和 teardown 入口继续直接复用。复制基线中的 graph、ReadyQ 和 payload 布局保留；
首次分类、ReadyQ 消费、payload 物化和完成传播迁移到 AICore，逐任务 AICPU dispatch 被删除。

第一阶段只实现正常 ready/exit 成功路径，不承诺本节中的超时、首错或异常退出；这些职责从
M6 开始成为 Runtime 验收契约。

### 2.10 MIX、wide 和 `sync_start` 扩展

这些能力在 M7、M8 逐项实现，并从对应里程碑开始引入 idle bitmap、participant slot 发布和
跨核 task epoch；这些对象不属于单 lane Pull 路径：

| 能力 | 调度方案 |
| ---- | -------- |
| MIX | 每 cluster 由 AIC leader 领取 MIX task；即使 AIC kernel 不活跃，也保留协调 leader；一次性预留所需 AIC/AIV lane，再发布同一 task epoch |
| Wide | leader 从 idle bitmap 预留多个 participant core，为每个 block 分配 ticket；最后一个 participant 完成后发布任务完成 |
| `sync_start` | 使用 task-local epoch barrier；所有资源预留且 execution slot 发布后，participants 到达 barrier，同一 epoch 放行 |

资源预留必须按全局固定顺序进行，并且采用“全部成功或全部回滚”，禁止持有部分资源等待
其余资源，否则 MIX 和 wide 组合会形成环路死锁。leader 只负责协调，不代表它一定执行
业务 AIC kernel。

### 2.11 Deferred completion 与 HBG 优化能力

最终 Runtime 必须支持冻结 HBG 基线中的 deferred/async completion。kernel 返回不再必然等于
task 完成：没有 deferred condition 的 task 由执行 core 直接完成；注册了 condition 的 task
进入 `DEFERRED`，在条件满足前不得发布 completion 或唤醒消费者。AICPU 只负责轮询 AICore
无权访问的硬件/SDMA 条件，并把 `(task_id, run_epoch, condition_result)` 发布到有界 deferred
completion queue；AICore scheduler 校验 epoch 后完成 task control、wake relay 和后继发布。
AICPU 不在该路径执行 fanin 扫描或普通 task 仲裁。

M6 先完成单 lane deferred completion；M7/M8 增加 wide/MIX 参与者汇聚后的 deferred 组合；
M9 按冻结矩阵完成全部组合和 drain。`allow_early_resolve`、early-dispatch 等 HBG 优化也在 M9
形成能力对齐：必须有独立开关、关闭时语义不变、开启时不早于依赖满足执行，并用同一固定
workload 证明具有预期的预物化/等待隐藏效果。实现不要求复制 HBG 的 AICPU pending-slot 协议。

### 2.12 错误和退出

以下契约从 M6 开始生效；M0～M3 明确只支持人工运行的成功路径：

- 采用 first-failure fail-stop。第一个错误通过 CAS 固化错误码、task id、core id 和 phase。
- first-error 后停止领取新任务；已领取任务进入受控 drain，不能继续释放新的业务后继。
- 不重试，不切换到 AICPU 执行，不掩盖首错。
- AICPU 等待所有 scheduler/participant 确认退出，再把错误返回 Host。
- 超时诊断必须输出队列深度、未完成任务、所等待生产者、各 core 当前状态/任务、在途 task
  和最后事件；M7/M8 还需输出资源 idle bitmap、participant 和 task epoch。
- AICPU 以批量提交的全局 `completed_count == expected_task_count` 且无 first-error 判断正常完成。
  exactly-once completion 成立时，完成数到达总数已经隐含队列为空且不存在未完成的在途 task，
  因此单 lane 路径不维护共享 idle bitmap 或 in-flight counter；每 core 当前状态只用于 DFX。

## 3. 可行性、风险与设计选择

### 3.1 可行性依据

| 能力 | 当前基础 | 参考程度 | 新 Runtime 结论 |
| ---- | -------- | -------- | --------------- |
| Host 提前构图 | HBG 已在 Host 同步运行 Orchestrator 并整体 H2D | 高 | v0 复制到独立目录；v1 重建正式图 ABI |
| 依赖生成 | HBG 的 TensorMap 和显式依赖已经产出 fanin | 高 | v0 复制以快速验证；v1 用 normalized graph 保证等价 |
| 解依赖算法 | HBG 已有 completion flag、首次分类、wake list 和重新分类 | 高 | 复用算法，不复用 AICPU mutable layout；从 v0 起写入 128B sidecar |
| 有界队列 | HBG 已有 Vyukov MPMC 实现和 A5 MPMC UT | 高 | 复用算法/测试轨迹；v0 起使用 AICore-safe task-id queue |
| 参数物化 | HBG 已有 `PTO2DispatchPayload` | 中 | v0 直接适配复制布局；v1 使用 per-core execution slot |
| AICore 常驻调度 | FDWIC 的 `aicore/onboard_entry.h`、`core_main.h` 已实现 GM attach 和 resident 生命周期 | 高 | 可复制适用骨架，不带入设备 Orchestrator replay |
| AICore raw atomic | FDWIC 的 `common/atomic.h`、`runtime_state.h` 已实现 CCEC GM 原子和 first-error 发布 | 高 | 复制并收窄为本方案所需 load/store/CAS/exchange/fetch-add/barrier |
| AICPU 控制面 | FDWIC 的 `aicpu/control_plane.h` 已实现初始化、拓扑、共享状态发布和错误读取 | 高 | 复制 supervisor 骨架，删除 FDWIC 配置和 SPMD 状态 |
| 动态 kernel 调用 | HBG 已有 `func_id_to_addr_`、`CoreCallable::resolved_addr()` 和 AICore 间接调用 | 高 | v0 直接复用，不引入 FDWIC 直链接构建 |
| 仿真和真机验证 | 两仓都有 A5sim/A5 测试基础 | 高 | 建立新 Runtime 自己的分层验证 |

可行性的前提不是“把 AICPU C++ 原样编译到 AICore”，而是复制 HBG 行为和数据基线，针对
AICore 内存模型、编译器和资源拓扑重写热路径。第一阶段只回答“能否正确运行、竞争成本和
性能是否值得继续”；Graph ABI v1 和完备性不应成为拿到这个答案的前置成本。

FDWIC 的固定 `DistGlobal`、claim/replay 和每核 TensorMap 不适合 Host-prebuilt DAG：其状态按
编译期容量冻结，当前实现尾部 offset 接近 1GB。FDWIC 对 v0 的价值是异构入口、raw GM atomic、
缓存行隔离和控制面协议已经经过实现验证，而不是提供 graph ABI、直链接镜像或整套 Runtime。
按冻结图计算容量的独立 sidecar planner 属于 v1，不能成为取得 M3 数据的前置成本。

### 3.2 主要风险和控制措施

| 风险 | 影响 | 控制措施 | 阶段闸门 |
| ---- | ---- | -------- | -------- |
| 第一阶段无性能收益 | 完整实现投入没有价值 | M0 冻结 HBG 基线，M3 同机测量微图和真实单 lane workload | M3 人工评审后才可启动第二阶段 |
| v0 与 v1 双重开发 | v0 临时代码渗入生产或迁移成本失控 | v0 已采用 128B control/task-id queue；M4 主要正式化 immutable record、identity、checked layout 并整体升版 | M4 删除生产 v0 路径 |
| 第一阶段异常卡死 | 手工 A5sim/A5 运行永久等待或污染设备 | 仅运行已知成功用例、设备测试手工 opt-in；接受人工回收风险 | 不进入默认 CI，M6 前不宣称异常安全 |
| A5 跨核 cache 不一致 | 读到旧参数或旧输出，产生静默错算 | v0 复用 HBG 通用 DCCI/barrier 并做 MB-8 seam；v1 根据证据决定是否需要 adapter | M5 seam 未通过不得签收完整业务 |
| 原子和 DCCI 同行 clobber | 丢完成、重复执行或死锁 | M0 即使用 128B control、word-sized atomic、DCCI/atomic 分行和邻居压力 | M0 原子探针未通过不得继续 |
| HBG byte completion/紧凑 slot 被误复用 | sub-word 写或 cache-line clean 覆盖相邻 core 更新 | HBG mutable layout 仅作 oracle；AICore 只写隔离 sidecar 和 task-id queue | M0 静态检查与 A5 clobber probe |
| AICore 编译器不支持现有 C++ 原语 | HBG 的 `std::atomic`/指针状态不能进入 CCEC 热路径 | 复制并收窄 FDWIC raw GM atomic，增加独立编译探针 | WP1 前完成工具链探针 |
| 动态函数入口不稳定 | kernel 无法调用或调用错误版本 | 复用 HBG callable 地址上传/解析路径并做 AIC/AIV 真机 smoke | M1 A5 smoke |
| 多 scheduler 原子热点 | 调度开销抵消收益 | 分类分片、现有 Vyukov 队列、可执行队列 RR、冲突 DFX | M3 报告，不设自动数值门槛 |
| ReadyQ 基线成为性能热点 | MPMC cursor 串行抵消多 scheduler 收益 | v0/v1 各自保持 Vyukov MPMC 基线并采集竞争 DFX；O3 单独评估替代结构 | 不阻塞 M9 正确性交付 |
| graph/queue 最坏容量过大 | HBM 占用不可控或启动失败 | Host 精确预估、配置 cap、启动前 fail | 容量边界 UT/ST |
| MIX/wide 部分资源持有 | 环路死锁 | 全局预留顺序、all-or-rollback、task epoch barrier | 模型检查 + 故障注入 |
| 复制基线长期漂移 | HBG 修复未自动进入新 Runtime | 来源 commit/tree hash、显式差分评审、独立同步提交 | 每次同步单独评审 |
| sim 与真机内存模型差异 | sim 通过、真机失败 | 第一阶段即在 A5 测量；A5sim 只作功能预检，A5 seam 和压力为最终依据 | 真机证据单独签收 |
| 错误路径不能退出 | 测试或生产永久挂起 | first-error、fail-stop drain、watchdog 状态快照 | 每类错误注入必须有限时退出 |
| 单个里程碑范围过大 | 正确性闭环延迟 | 单 root、单核 DAG、多核单 lane、wide、MIX/sync 逐级交付 | 每个里程碑独立验收 |
| HBG 基线持续变化 | 验收范围成为移动目标 | M0 冻结 commit、能力、优化和测试矩阵；新增能力另行评审 | M9 只对冻结矩阵签收 |
| successor prepare 串扰 | prepared run 覆盖 active graph/slot | per-run artifact pin、不同 slot/bank/generation；diagnostics depth-one | M6 active+prepared 多 callable ST |

### 3.3 已冻结的设计选择

下列问题已经完成决策，是实现基线；任何变更应重新进行架构评审。

| 序号 | 设计选择问题 | 已冻结决策 |
| ---- | ------------ | ---------- |
| 1 | 修改 HBG 还是新增 Runtime？ | 新增显式 `host_build_graph_aicore`，保留 HBG |
| 2 | 首发平台覆盖哪些架构？ | A5sim、A5；本计划不覆盖 A2/A3 |
| 3 | Orchestrator 在哪里、如何运行？ | Host 同步运行到结束，设备启动前完成全图 |
| 4 | 图 ABI 如何演进？ | 第一阶段使用复制 HBG 布局的实验 v0；第二阶段整体替换为正式 v1，不长期双栈兼容 |
| 5 | 解依赖使用 refcount 还是扫描 fanin？ | 扫描 fanin，等待首个未完成 producer，wake relay 后重分类 |
| 6 | 首次分类在哪里完成？ | AICore 多核并行分类，完成后全局 barrier |
| 7 | ReadyQ 使用什么实现？ | M1～M3 使用 AICore-safe v0 task-id Vyukov MPMC；M4～M9 正式版本化为 v1；HBG pointer queue 仅作算法 oracle；替代结构移到 O3 |
| 8 | task shape 如何递进？ | M1～M6 完成单 lane；M7 wide；M8 MIX/`sync_start`；M9 完整组合验收 |
| 9 | wide task 如何组织？ | leader + participant cores + block tickets |
| 10 | MIX 的协调者是谁？ | cluster AIC leader，必要时为纯协调角色 |
| 11 | `sync_start` 如何实现？ | task-local epoch barrier |
| 12 | 每任务控制块如何布局？ | v0 起固定 128B sidecar，原子控制、DCCI 普通字段和相邻 task 隔离；v1 正式版本化并扩展字段 |
| 13 | kernel 数据可见性由谁负责？ | v0 复用 HBG 通用 DCCI/barrier；v1 是否需要业务感知 adapter 由 M3/M5 seam 证据决定 |
| 14 | AICore 产物如何组织？ | 使用 Runtime 静态 scheduler image，业务 kernel 继续走现有动态地址；无 callable-scoped image |
| 15 | 是否改变公开 submit/Orchestrator API？ | submit/Orchestrator/ChipCallable 不变；M0～M2 不改 `CallConfig`，M3 仅在需要实验上限时增加可选 `aicore_scheduler_limit` |
| 16 | 完整图超过容量如何处理？ | 设备启动前失败，不运行期扩容 |
| 17 | 任务失败后是否重试或回退？ | first-failure fail-stop，不重试、不回退 AICPU |
| 18 | 多 ready queue 如何仲裁？ | executable-first，再 round-robin |
| 19 | 单 lane 任务如何选择执行 core？ | 空闲 core 从兼容共享 ReadyQ 主动 Pull；MPMC pop winner 直接执行，不二次派发 |
| 20 | 如何使用 `simpler-dist`？ | 允许复制适用的 runtime primitive/control 骨架到独立目录；不复制 SPMD 图模型和业务工作负载，不形成跨仓依赖 |
| 21 | 验证顺序是什么？ | 每个里程碑先 Host/model/A5sim，再做所需 A5；M3 必须用 A5 形成性能报告 |
| 22 | 正确性 oracle 是什么？ | M0 冻结版本 HBG 的 graph、输出、错误、能力和优化矩阵 |
| 23 | 第一阶段性能如何决策？ | 不设自动数值门槛；M3 输出可信报告并暂停，由用户人工评审是否继续 |
| 24 | 新 Runtime 是否默认启用？ | 不默认，显式选择 |
| 25 | v0 是否增加 callable bundle？ | 不增加；沿用现有 Host SO、ChipCallable 和 Runtime 静态 image 生命周期 |
| 26 | successor prepare 如何处理？ | v0 depth-one；M6 起 nondiagnostic depth-two，PreparedExecution 固定持有全部 per-run artifact；diagnostics depth-one |
| 27 | 是否支持 deferred/async completion？ | 最终与冻结 HBG 对齐；M6 单 lane，M7/M8 增加 shape 组合，M9 完整验收 |
| 28 | `allow_early_resolve` 如何处理？ | 最终对齐冻结 HBG 的优化能力；M9 交付独立开关、语义保护和性能证据 |
| 29 | 第一阶段异常策略是什么？ | 只支持成功路径且无 Runtime timeout；设备测试手工 opt-in，外层 timeout 仅负责资源回收 |
| 30 | 第二阶段如何启动？ | M3 后必须暂停，只有用户人工评审并明确启动后才能继续 |

仍需通过数据校准、但不改变架构的参数包括：v1 默认 graph memory cap、event ring 容量、
AICore 调度并发自动策略/可选上限和队列回退次数。第一阶段 workload、规模、采样轮数和报告字段在 M0
冻结，M3 之后不得选择性删改不利样本。

## 4. 基于 pypto/runtime 的关键修改点和实现路径

### 4.1 关键修改点列表

| 区域 | 当前参考点 | 计划修改 |
| ---- | ---------- | -------- |
| Runtime 发现 | `simpler_setup/runtime_builder.py` | 已按 `build_config.py` 动态发现；新增平级目录即可，无额外 registry |
| Orchestrator 编译 | `simpler_setup/kernel_compiler.py` | 按 Runtime 名称选择：两个 HBG Runtime 使用 Host GXX；TRB 仿真使用 Host GXX、板端使用 AArch64 GXX；未知名称失败 |
| 每次运行配置 | `CallConfig`、binding、mailbox、remote protocol、SceneTest | M0～M2 不变；M3 若实验需要，增加可选 `aicore_scheduler_limit`，0=自动、正数=调度并发上限；其他 Runtime 忽略 |
| Runtime 基线 | 新增 `src/a5/runtime/host_build_graph_aicore/` | 机械复制完整 A5 HBG 实现，记录来源 commit/tree hash 后独立维护 |
| Runtime 构建清单 | 新 Runtime 的 `build_config.py` | 保持复制基线的 Host/orchestration，后续替换 AICPU supervisor 和 AICore scheduler |
| Host 构图 | 复制后的 `host/runtime_maker.cpp` | v0 保持 HBG 构图、重定位和 H2D 行为；v1 才重建 checked-layout builder |
| 依赖生成 | HBG 的 `runtime/pto_dep_compute.h`、`orchestrator_core/pto_orchestrator.cpp` | v0 复制 fanin 生成；v1 重新编码并用 normalized graph 校验 |
| 图 ABI | HBG 的 `runtime/pto_runtime2_types.h`、`runtime/pto_shared_memory.h` | v0 只读 descriptor/payload/fanin，加 128B control/task-id queue sidecar；M4 用独立 immutable record 和正式 ABI v1 整体替换 |
| HBG 调度语义 | HBG 的 `runtime/scheduler/` | v0 复制 classify/wake/MPMC 算法并迁到 AICore-safe sidecar；不复用 AICPU mutable layout |
| AICore 执行 | `aicore/aicore_executor.cpp`、`runtime/pto2_dispatch_payload.h` | 新建常驻 scheduler；每 core 自主 Pull，v0 继续使用 HBG payload 和动态 kernel 地址 |
| AICPU 执行 | `aicpu/aicpu_executor.cpp` | 新 Runtime 只保留 launch/control/drain；删除逐任务解依赖职责 |
| FDWIC atomic | `../simpler-dist/src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/{atomic,runtime_state}.h` | 复制到新目录并收窄；保留 CCEC raw GM intrinsic 和 first-error 顺序，不保留 FDWIC 全局状态 |
| FDWIC resident/control | `../simpler-dist/src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/{onboard_entry,core_main}.h`、`aicpu/control_plane.h` | 复制适用的 attach/start/finish/supervisor 骨架，替换为 graph launch/run control |
| Worker 内部 | `src/common/worker/chip_worker.*`、`pto_runtime_c_api.h` | v0 保持现有接口；M3 若增加调度并发上限再透传对应 CallConfig；v1 让 PreparedExecution 固定持有 graph/slot/bank/generation |
| 错误状态 | `common/pto_runtime_status.h`、公共 error-name 映射 | M6 增加 sidecar 容量、queue、epoch、deferred 和 drain 错误 |
| DFX | 现有 L2 swimlane、PMU、args dump 基础 | M3 只交付性能计数；M6 补齐诊断，M7/M8 增加资源事件，M9 完整对齐 HBG |
| Tests | `tests/ut`、`tests/st/a5` | 增加模型、差分、A5sim、A5 seam、压力、故障注入和生命周期测试 |

Runtime-specific 代码不在 HBG 与 HBG-AICore 之间抽取共享层。M0 将 HBG 实现完整复制到
`host_build_graph_aicore/`；复制后新 Runtime 不读取、包含或链接原 HBG 私有源码，两个目录由
各自测试保护并独立演进。仓库既有的通用 Worker、平台和公开 ABI 组件不属于这项隔离限制。

### 4.2 工作包和依赖顺序

| 工作包 | 内容 | 前置 | 交付物和退出条件 |
| ------ | ---- | ---- | ---------------- |
| WP0 基线与实验接入 | 冻结并机械复制 HBG；建立平级 v0 Runtime 和只读 graph seam | 无 | 来源证明、复制差分、空图生命周期和构建 UT |
| WP1 AICore 原语与安全 sidecar | 从 `simpler-dist` 复制并收窄 resident、word-sized atomic、DCCI、barrier；建立 128B task control 和 task-id queue | WP0 | A5sim/A5 clobber probe；不带入 replay/SPMD 图模型 |
| WP2 v0 最小调度 | 使用 HBG 只读 graph + AICore-safe execution sidecar；交付 root、单核 DAG、多核动态单 lane，并在 M3 按需增加调度并发上限 | WP0、WP1 | M1～M3 正确性证据和性能报告 |
| WP3 v1 图 ABI | 重建 immutable record、正式版本化 128B control、checked layout 和 graph identity | M3 人工批准 | v0 graph cache 失效；v1 Host/model UT 和 graph 差分 |
| WP4 单 lane 完备 | 真实 tensor 可见性、错误/drain、DFX、deferred completion、early resolve 和 depth-two | WP3 | M4～M6 单 lane 独立签收 |
| WP5 高级资源调度 | wide、MIX、`sync_start`、participant epoch 和 all-or-rollback | WP4 | M7/M8 task-shape 独立签收 |
| WP6 全能力准入 | 冻结 HBG 矩阵组合、diagnostics、长期稳定性和最终性能画像 | WP5 | M9 完整准入报告 |
| WP7 可选优化 | SIMT、普通 CompletionQ、替代 ReadyQ | M9 | O1～O3 独立开关、对比和回退 |

### 4.3 里程碑

里程碑按“快速验证 → 单 lane 完备 → task shape 补充 → 全能力完备”递进。M0～M3 优先缩小
修改面并尽早取得 A5 数据；M4 之后才承担正式 ABI 和工程完备性。

本节是 M0～M9 的架构里程碑来源。第一阶段的原子提交边界、接口落点和逐提交验证以
[M0～M3 原子提交计划](host-build-graph-aicore-m0-m3-commit-plan.zh.md) 为实施依据；若两者冲突，
以本文的阶段边界和冻结决策为准。

共同规则：

1. 每个里程碑只增加一组可观察能力，并有 Host/model/A5sim/A5 中适合该阶段的独立证据。
2. 新 Runtime 始终显式选择；HBG 构建产物、默认行为和回归结果不变；M0～M2 不改变公共
   `CallConfig` 或 wire，M3 仅在需要可控实验上限时同步增加 `aicore_scheduler_limit` 和 wire 升版。
3. M0～M3 仅支持成功执行路径，设备用例手工 opt-in；Host 必须在设备启动前拒绝超容量和未支持
   task shape，但不承诺已启动后的 kernel 错误或卡死能由 Runtime 自身退出。
4. M4 起补齐正式 ABI 的全部输入/identity 校验；M6 起设备错误和 timeout 必须有限时退出。
5. M3 完成后必须暂停，只有用户人工批准才能开始 M4；性能报告不设置自动通过线。
6. M4 通过 ABI 升版整体替换 v0，不允许 v0/v1 生产双栈；M1～M3 的算法和性能 workload 保留。
7. 每个里程碑保存固定 workload、环境、identity、关键 counters 和分阶段时延，并与 HBG、
   前序里程碑比较。

#### M0：冻结基线与 v0 空图生命周期

新增功能：

- 记录项目启动时的 HBG commit、源目录 tree hash、A5 能力/优化矩阵、固定测试集、环境和性能
  采样协议。
- 机械复制完整 A5 HBG Runtime 到平级目录；复制提交只包含名称/构建接入，并用差分清单证明
  没有调度语义变化。
- 新 Runtime 可被显式选择、构建和装载，固定发布 depth-one 和实验 Graph ABI v0；HBG graph
  仅供只读，v0 从首版即包含 graph-sized 128B task control 和 task-id ReadyQ；继续使用 Runtime
  静态 AICore image 和现有 ChipCallable 格式。
- 128B sidecar 是冻结复制后的第一个功能改动；其 A5 邻位无 clobber probe 通过后，才允许接入
  resident lifecycle 或任何真实 task 执行路径。
- 从 `simpler-dist` 复制并裁剪 AICore resident attach、word-sized raw atomic、DCCI、barrier 和 AICPU
  control-plane 骨架，不复制 replay、SPMD TensorMap 或业务 workload。

独立验证：

- Host/build UT 覆盖 Runtime 动态发现、只读 graph view、128B/POD 布局、容量溢出和 v0 空图控制。
- A5sim/A5 手工运行空图 ready/正常 exit；验证 atomic、DCCI、barrier、相邻 task/queue slot
  clobber 和动态函数地址探针。该 probe 是 M0 功能闸门。
- 保存 HBG 与 v0 的构建、H2D、AICore 启动和空图退出基线；不运行异常或 timeout 用例。

#### M1：单 core、单 root 任务执行

新增功能：

- 在 v0 上分别仅启用一个 AIC 或一个 AIV，执行一个无依赖 root task。
- v0 root classifier 明确校验“单 task、零 fanin”后把 task id 发布到 sidecar ReadyQ；不由 Host 直接
  指定执行 core，避免出现 M1 使用 ReadyQ、M2 才定义分类来源的断层。
- 打通复制后 HBG payload 参数读取、MPMC pop、现有动态 child-kernel 调用和正常完成计数的
  最短纵向链路。

独立验证和性能对比：

- Host/model 测试覆盖 root 分类、v0 task-id queue、只读 HBG graph view、参数映射和 callable 地址解析。
- A5sim/A5 手工分别运行 no-op/短 kernel 的单 AIC、单 AIV 成功用例；不运行 predicate、
  kernel 错误或非法输入。
- 比较 M0、HBG 和 M1 的 ready-to-start、payload materialization、kernel、completion 和总时延。

#### M2：单 core DAG 解依赖

新增功能：

- 在单 core 上交付 initial classify、fanin 扫描、wake list 和 wake relay。
- 支持成功路径的 chain、diamond、fanout/fanin、多个 root 和固定 seed 随机 DAG；predicate、
  DUMMY、hidden allocation、manual scope 和 deferred completion 留到 v1。
- 交付 classify、fanin scan、wake register/close/reclassify、queue push/pop 和 completion 的
  性能 counters；不引入 first-error 状态机。

独立验证和性能对比：

- CPU 模型覆盖成功路径 waiter 注册/关闭竞态、重复唤醒和长 fanin，不注入 queue 满或首错。
- 生产 `classify/complete-and-wake` 接口直接覆盖多 waiter 并发 drain 和 close/register 竞态，
  避免只由独立模型证明状态机。
- A5sim/A5 手工对基础图族逐边验证 `producer complete-before-consumer start`，并与 HBG 比较
  graph signature 和成功输出。
- 对 chain、diamond 和 multi-root 增加真实 AIV kernel/TensorMap 推导依赖，以相同 orchestration
  分别运行 HBG-AICore 和 HBG，比较业务输出；显式状态检查图继续覆盖 fanout、宽 fanin 和随机图。
- 固定单核 DAG 比较 M1、HBG 和本里程碑的分类、重复 fanin 扫描、完成传播及总时延。

#### M3：多 core 单 lane 调度

新增功能：

- AIC/AIV 空闲 core 在 scheduler/worker 角色间动态切换，并行 initial classify 后在全核 barrier
  后开始执行。
- Runtime 根据图规模、设备拓扑和空闲 core 选择实际调度并发度；若正确性/性能实验需要显式
  上限，本里程碑增加可选 `CallConfig.aicore_scheduler_limit` 及 binding、mailbox、remote、
  SceneTest、布局和非法值验证，`0` 为自动策略，正整数不固定专用 scheduler 核。
- 使用 v0 task-id 有界 MPMC，交付 ReadySet exactly-once 领取、兼容队列 round-robin 和领取者本核执行。
- 复用并扩展现有 Vyukov bounded MPMC 的算法/CPU 轨迹，在 v0 raw-atomic 布局上建立 A5sim/A5
  正确性和压力证据，不在本阶段引入
  sharded MPMC、ready bitmap 或 bitmask 优化。
- 支持随机 DAG、queue wrap、多 root 并发和成功路径公平进展；不实现 first-error/drain。

独立验证和性能对比：

- 并发模型测试覆盖 MPMC wrap/ABA、task 单次发布/领取、无丢唤醒、无重复执行，以及 busy/idle
  反复切换和 flush 后立即出现新任务时批量完成计数无漏计、重复或提前退出。
- A5sim/A5 设备用例继续手工 opt-in，只运行预先验证的成功 workload。
- 正确性覆盖 1/2/63/64/65/256/1024 task 和标准/随机 DAG；拓扑 microbench 固定测量
  64/1024/4096 task 的 chain、32-way fanin 和 multi-root；调度并发上限覆盖 1/2/4/自动，并记录
  每轮实际并发度。
- 真实 kernel benchmark 使用 256 个独立 root，分别覆盖仓内已验证的 AIC 和 AIV workload。
- 同机比较 HBG 与 M3 的 graph publish、initial classify、ready-to-first-start、producer complete
  到 consumer start、queue/atomic/fanin/wake、task throughput、device 总时延和 p50/p95/p99。
- 输出完整环境、identity、原始样本和瓶颈解释后暂停。没有自动性能阈值，由用户人工决定
  继续 M4、先优化 v0 算法或终止方案。

#### M4：Graph ABI v1 与正式构建契约

前置：用户已人工批准 M3 报告并明确启动第二阶段。

新增功能：

- 用版本化 `AicoreGraphLaunchDesc`、immutable `AicoreTaskRecord`、正式 `AicoreTaskControl`、
  `AicoreRunControl`、graph-sized ReadyQ/sidecar 和 checked relocation 整体替换 v0；保持 M0 已
  验证的 128B 隔离和 task-id ownership 契约。
- graph ABI、control ABI 和 graph identity 同时升版；强制失效并重建 v0 graph cache，删除生产
  v0 执行分支，不兼容双 ABI。
- 建立 normalized graph exporter，冻结 task/kernel/shape/fanin/predicate/参数分类的差分格式。
- v1 仍发布 depth-one，先验证正式数据模型，不在本里程碑引入 successor overlap。

独立验证：

- Host UT 覆盖空图、布局/对齐、整数溢出、区域重叠、重定位、graph cap 和 identity mismatch。
- 对 M1～M3 全部成功图比较 v0、v1 与 HBG graph signature 和输出；性能变化必须分解到新布局、
  H2D、分类和参数物化。
- 验证旧 graph cache 被确定性拒绝并重建，Worker 不会把 v0 graph 与 v1 control 混用。

#### M5：单 lane 语义与数据可见性完备

新增功能：

- 完成 TensorMap 自动依赖、显式依赖、manual scope、predicate、DUMMY、hidden allocation、
  预完成 producer、真实 tensor/scalar 参数和 HBG 单 lane graph 语义。
- 完成 AIC→AIC、AIV→AIV、AIC→AIV、AIV→AIC 的 producer-output/consumer-input DCCI
  发布观察协议，并在真实 tensor seam 上复验 M0 的 128B control/普通数据缓存行隔离。
- 根据上述 seam 的正确性和性能证据决定是否只保留 Runtime 通用协议，或引入业务感知的
  tensor-range adapter；若引入，必须在本里程碑单独定义 ABI、identity 和缺失时的构建错误。
- 保持 depth-one，只验证正常完成路径；完整异常、diagnostics 和 deferred completion 留到 M6。

独立验证和性能对比：

- Host/model 覆盖 normalized graph、predicate/DUMMY、hidden task、manual scope；只有实际采用
  adapter 时才增加其 identity 测试。
- MB-2、MB-5、MB-8 等价 seam 在 A5 长压力下无旧读、丢写或邻居 clobber。
- 冻结 HBG 基线中的单 lane 成功用例全部做 graph、依赖边、输出和 task 次数强差分。
- 比较 M3、M4、M5 与 HBG，量化 v1 布局、可见性协议和真实参数物化成本。

#### M6：单 lane 生命周期、异常和并发 prepare 完备

新增功能：

- 完成 first-error、停止领取、有限时 drain、watchdog、容量/queue/kernel/deferred 错误
  映射，以及从 timeout 定位 task、producer、queue、core 和 phase 的 bounded DFX。
- 支持单 lane deferred/async completion：AICPU 只发布带 run epoch 的硬件 completion 事件，
  AICore 校验后完成 task 和 wake relay。
- 补齐冻结 HBG 基线的 dump args、PMU、swimlane、dep-gen、scope stats 和单 lane
  `allow_early_resolve`/early-dispatch 能力。
- 完成多 callable/graph cache 生命周期；`PreparedExecution` 固定持有 artifact、graph/sidecar、
  slot、bank、generation 和 run epoch。
- nondiagnostic run 发布并验证 depth-two successor prepare；任何 diagnostics run 及其相邻 run
  保持 depth-one，successor 在 predecessor finalize 前不得 launch/accept。

独立验证和性能对比：

- 模型和故障注入覆盖首错幂等、queue full、stale epoch、deferred condition、timeout 和 drain。
- A5sim/A5 交替运行 callable/graph A 与 B，覆盖 A active + B prepared、predecessor 失败、stale
  activation、shutdown 和 unlaunched token exactly-once finalize。
- 验证 diagnostics 下没有 prepared successor；关闭 diagnostics 后 Host prepare 与 predecessor
  device execution 有重叠但设备 launch 仍严格 FIFO。
- 冻结 HBG 单 lane 能力、错误、DFX 和优化矩阵全部差分通过后，单 lane 才可独立签收。

#### M7：Wide 同类型多核任务

新增功能：

- leader 按固定顺序一次性预留多个同类型 participant core，为每个 block 分配唯一 ticket。
- participant 按 task epoch 读取发布后的 execution slot；最后一个 participant 汇聚普通或
  deferred completion 后，唯一发布 task 完成。
- 资源不足、部分预留、participant 迟到、执行失败和 deferred condition 均遵循
  all-or-rollback、generation 隔离和有限时 drain。

独立验证和性能对比：

- 模型测试覆盖唯一 leader、ticket exactly-once、部分预留回滚、迟到 epoch、最后完成竞争和
  wide deferred completion。
- A5sim/A5 覆盖不同 block 数、资源紧张、慢 participant、故障注入和 successor prepare；
  M6 单 lane 完整套件保持通过。
- 对同一可表达 workload 比较单 lane 拆分、HBG wide 和 M7 的预留、同步、kernel、完成及总时延。

#### M8：MIX 和 `sync_start`

新增功能：

- cluster AIC leader 按固定全局顺序预留 AIC/AIV participant，并统一发布 task epoch。
- 支持 leader 仅协调、MIX 多资源执行、task-local `sync_start` barrier、普通/deferred 完成汇聚。
- 异构资源不足、部分预留、leader 退出、participant 迟到和 deferred failure 均能全部回滚或
  有限时失败，不允许持有部分资源等待另一类资源。

独立验证和性能对比：

- 模型测试覆盖异构预留顺序、barrier generation、无部分持有等待、唯一完成和组合 drain。
- MB-4、MB-7 等价用例在 A5sim/A5 形成证据；M6 单 lane 与 M7 wide 套件继续通过。
- 比较冻结 HBG、无 `sync_start` 和启用 `sync_start` 的预留成本、参与者到达偏差、kernel、
  completion 和总时延。

#### M9：冻结 HBG 全能力和优化准入

新增保证：

- 对 M0 冻结的 HBG 能力矩阵逐项完成 graph、输出、错误、diagnostics、生命周期和优化能力
  验收，覆盖 single lane、wide、MIX、`sync_start` 与 deferred completion 的合法组合。
- 完成 `allow_early_resolve`/early-dispatch 的独立开关、语义保护、性能证据和关闭回退；不要求
  复刻 HBG 内部调度协议，但开启后必须体现对应优化效果。
- counters、bounded event ring 和泳道能从 timeout 定位 task、producer、queue、core、participant、
  deferred condition 和 phase；全量容量、故障、depth-two、多 callable 和长期稳定性通过。

独立验证和性能对比：

- 运行冻结版本完整矩阵，确认 HBG 默认行为、公开 API 和既有业务结果无变化；M0～M8 全套回归通过。
- 输出 HBG 与 HBG-AICore 的构图、H2D、分类、ready-to-start、执行、完成传播、deferred、drain
  和总时延对比，并解释全部显著退化。
- 只有 M9 完成后，HBG-AICore 才可声明对冻结 A5 HBG 完整能力和优化对齐；仍保持显式启用。

#### O1～O3：M9 后可选优化

- **O1 SIMT**：并行 graph/sidecar 初始化、task 区间分类和 fanin 扫描。
- **O2 普通 CompletionQ**：把普通 task 的 wake relay 与执行 core 解耦；不替代 M6 已要求的
  deferred-completion 事件队列。
- **O3 ReadyQ**：对比 sharded MPMC、per-shard 非空 bitmask、两级 ready bitmap、local-first
  和批量 reservation；具体候选见
  [AICore 共享竞争与 ReadyQ 替代方案分析](host-build-graph-aicore-shared-contention-analysis.zh.md)。

三项优化必须独立开关、独立验证并可回退到 M9 基线。任一优化失败或无收益都不影响 M9
功能验收，也不能改变 task 所有权、依赖、完成、错误或 FIFO run 生命周期语义。

## 5. 功能验证方案

### 5.1 验证原则

1. M0～M3 以最小实现尽早获得正确性和性能事实；M4～M9 再以正确性和完备性优先。
2. M0 冻结版本 HBG 是强 oracle：比较 normalized graph、输出、错误、能力、diagnostics、
   生命周期和优化效果；后续 HBG 新功能不自动纳入。
3. 合法调度顺序可以不同，因此不逐项比较全序 trace；验证每条依赖边的
   `producer complete-before-consumer start`。
4. `simpler-dist` 的适用实现可以复制并裁剪，但其用例仍需按本 Runtime 的图模型和并发契约
   重建，不能把 FDWIC 历史 PASS 当作本 Runtime 的证据。
5. M0～M3 不执行失败用例，设备测试仅手工 opt-in；M6 起每个失败用例必须保留 first-error
   和 bounded event ring，禁止只报告 timeout。
6. M3 的性能报告必须包含全部预先冻结样本并暂停等待人工评审，不用单一加速比自动判定。
7. 每个里程碑新增测试后继续运行全部适用的前序套件，新增能力不能替代已有回归证据。
8. 每个里程碑使用固定 workload 分别与 HBG 和前序里程碑比较；保存环境、Runtime/graph identity、
   DFX counters、分阶段时延和总时延，保证结果可重复。

### 5.2 Host/CPU 单元和模型测试

覆盖以下对象：

- HBG 复制来源/tree hash、v0 只读 graph ABI view、128B task control、task-id ReadyQ 和最小
  run/worker control，以及 v1 sidecar 字节布局、对齐、重定位和容量计算。
- HBG MPMC queue 提供空、满、wrap、ABA 和并发 push/pop 的算法轨迹；v0/v1 task-id queue 分别
  验证 raw-atomic 布局、task-id 有效性和相邻 slot 无 clobber。
- O3 候选 ReadySet 的 set/acquire、summary 清位竞态、fallback scan、热点 word、公平性和
  非 64 对齐任务数；这些用例不属于 M0～M9 基线。
- wake list 的注册/关闭竞态、关闭后重扫、重复唤醒和长 fanin。
- wake list 除模型测试外，必须直接调用生产分类/完成接口做并发 close/register 和多 waiter drain；
  当前 v0 生产接口用例覆盖 16 waiter 和 40 轮竞态，CPU 模型另覆盖 32 路 fanin。
- initial classify 的区间分片、预完成 producer、隐藏分配和 barrier。
- task 单次 ready 发布、ReadySet exactly-once 领取和 first-error 幂等性。
- M7/M8 的 idle bitmap、MIX/wide all-or-rollback、participant 和 task epoch barrier 模型。
- callable/graph identity、v0→v1 graph-cache 失效、多 callable 切换、PreparedExecution pin 和 Worker 销毁。
- graph/queue cap、非法 shape、deferred completion、depth-two 和 diagnostics depth-one；只有 M5
  决定引入 adapter 时才增加 adapter 缺失测试。

并发状态机建议用可重复 seed 的随机调度器做模型测试：每次随机选择 classify、register、
complete、pop、ready-set acquire 或 release 动作，并在每步检查“任务最多执行一次、完成只在 fanin
满足后发生、每个 pending 任务必在队列或某个开放 wake list 中”。

### 5.3 HBG 强差分

建立同一 Orchestrator 输入的双运行 harness：一次运行 HBG，一次运行 HBG-AICore。

| 差分项 | 比较规则 |
| ------ | -------- |
| Graph signature | task 数、kernel identity、resource shape、fanin 边、predicate、参数类别完全一致 |
| 执行约束 | 每条 fanin 边满足完成先于启动；任务执行次数一致 |
| 输出 | tensor/scalar 输出按业务容差比较；整数和状态字段精确比较 |
| 错误 | 同一非法输入返回同类错误，且新 Runtime 不超时 |
| DUMMY/predicate | 空完成集合和后继可达性一致 |
| Deferred completion | condition 类型、完成时点、后继释放、错误和 drain 结果等价 |
| Run 生命周期 | nondiagnostic depth-two、diagnostics depth-one、FIFO launch/accept/finalize 等价 |
| Diagnostics | dump/PMU/swimlane/dep-gen/scope-stats 的可用能力和关键字段等价 |
| 优化能力 | `allow_early_resolve`/early-dispatch 开关语义、关闭回退和预期优化效果成立 |

基础图族包括：空图、单 root、长 chain、diamond、宽 fanout、宽 fanin、多 root、多层随机
DAG、重复显式依赖、TensorMap overlap、manual dependency/scope、predicate true/false、
hidden allocation、多 callable 连续运行、active+prepared successor、deferred completion、
diagnostics、early-resolve 开关和容量边界。M0 冻结能力矩阵时必须把每项映射到具体测试；
M9 不接受仅凭“基础图通过”推断组合能力通过。

### 5.4 `simpler-dist` atomic minibench 等价验证矩阵

这里的“相同用例”指验证相同内部契约，不要求使用相同实现或业务源码。

| MB ID | `simpler-dist` 契约 | 本方案等价用例 | 阶段 |
| ----- | ------------------- | -------------- | ---- |
| MB-1 | sharded claim 恰一 winner、无跳号 | 多 scheduler 竞争 ReadySet；M3/M9 检查 MPMC slot 恰一 winner，O3 对候选实现复验 | M3、M9；O3 候选 |
| MB-2 | 64B completion flag，邻居无 clobber | v0 从 M0 验证相邻 128B task control 的 completion/wake/DCCI 压力；v1 复验正式 control | M0 A5 功能闸门，M3 压力，M5 复验 |
| MB-4 | `block.won` 多 lane 协作 | MIX leader/participants 竞争、唯一 leader、参数发布和最后 participant 完成 | M8，A5 必测 |
| MB-5 | shared map 定序、seq 和回收 | 多核读取 v0 HBG 只读 graph + sidecar control 或 v1 immutable graph/control，验证发布顺序和无旧 generation | M3 模型，M5/M6 A5 |
| MB-6 | 确定性 GM heap、容量和反压 | v0 sidecar/queue 与 v1 graph/sidecar/queue 临界容量、启动前拒绝和长序列无越界 | M0/M4 Host，M6 运行期 |
| MB-7 | `core_progress[]` 与 run-ahead | 慢核/忙核下检查 executable-first RR、公平进展和资源预留 | M3 单 lane，M7/M8 多资源 |
| MB-8 | `Coherent<T>`/DCCI seam | producer 写输出、Runtime 发布完成、consumer invalidate 后读取；对照缺失可见性操作的负例 | M3 smoke，M5 A5 必测 |
| MB-9 | private map 每核确定性 | 同一输入和 seed 重复构图/运行，graph signature、task 映射和结果稳定 | M3、M9 |

MB-2、MB-4、MB-6、MB-7、MB-8 还应保留 `simpler-dist` 测试风格中的 Normal、Heavy、
边界/慢核或多轮变体。测试报告必须区分“数值通过”和“完整调度契约通过”，并记录
graph signature、Runtime identity、image build identity、平台和迭代次数。

### 5.5 A5sim 验证

- M0～M3 仅手工 opt-in：依次运行空图、单 root、单核 DAG 和多核单 lane 成功路径；不注入错误。
- M4/M5 运行 v1 布局、normalized graph、predicate/DUMMY/hidden/manual scope 和真实单 lane 差分。
- M6 起纳入受保护回归，并注入 queue full、非法/重复 task id、kernel/deferred 错误、
  timeout、stale successor 和 diagnostics/depth-two 组合。
- M7 增加 wide participant 迟到和资源不足；M8 增加 MIX、`sync_start`、异构资源不足和 deferred。
- M9 运行冻结 HBG 全能力/优化矩阵；O3 的 bitmap/summary 错误只属于后续候选验证。

A5sim 用于功能和状态机覆盖，不把 sim cache 行为或 sim 时延作为 A5 可见性/性能结论。

### 5.6 A5 真机验证

按以下顺序执行：

1. M0 primitive probe：word-sized raw atomic、128B control 邻位无 clobber、task-id queue slot 发布、
   DCCI、barrier、动态 kernel 地址和 resident ready/exit。
2. M1～M3 手工成功路径：单 root、单核 DAG、多核单 lane，并完成固定性能矩阵；外层 timeout
   负责资源回收，但不作为 Runtime 错误语义证据。
3. M5 MB-2/MB-8 seam：固定数据模式、多轮、跨 AIC/AIV 组合、同行负例和邻居压力。
4. M6 单 lane 完整矩阵：predicate、随机 DAG、queue wrap、deferred、diagnostics、depth-two、
   kernel 错误、watchdog 和 successor 失败；只有采用 adapter 时才增加其错误用例。
5. M7/M8 wide、MIX、`sync_start`：不同 cluster、资源紧张、慢 participant、deferred 和组合 shape。
6. M9 真实业务 callable、HBG 优化开关、全量故障和长期稳定性。

真机每次证据需保存芯片/驱动/CANN/PTO-ISA/Runtime commit、Runtime binary/graph identity、编译宏、随机
seed、重复次数、原始时延、首错和 DFX 摘要。M3 报告必须保存未筛选样本；M5 后短 smoke
通过再逐级提高到长压力，避免把偶发一致性问题误判为通过。

### 5.7 DFX 不变量和性能画像

固定 counters 至少包括：

- task 总数、initial-ready、initial-waiting、completed、dummy-completed。
- fanin 扫描次数/边数、wake 注册/关闭/重分类次数、关闭竞态重试。
- 各 ready queue push/pop/full/empty、最大深度和回退次数。
- MPMC cursor/sequence atomic 和 CAS retry；O3 候选实现增加 bitmap L0/L1 atomic、
  bit-clear retry、fallback scan 和 hot-word 最大重试。
- ReadySet acquire success/retry、重复 ready 检测、各 core Pull/执行任务数；M7/M8 增加 idle
  bitmap 预留/回滚、participant、epoch 和 barrier。
- payload/可见性操作、scheduler、kernel 和完成传播周期；只有采用 adapter 时才增加 hook 指标。
- deferred register/event/complete、first-error、drain task/core 数、watchdog 和 successor prepare 状态。

可选 bounded event ring 记录 `CLASSIFY`、`WAIT_ON`、`READY`、`CLAIM`、`START`、
`COMPLETE`、`WAKE`、`RESERVE`、`ROLLBACK`、`ERROR`，记录满后只置 overflow 标志，
不能阻塞 scheduler。

M3 正确性覆盖 1/2/63/64/65/256/1024 task 和标准/随机 DAG。拓扑 microbench 覆盖
64/1024/4096 task 的 chain、32-way fanin、multi-root，每点预热 10 次、记录 30 次并报告
p50/p95。真实 kernel benchmark 使用 256 个独立 root，覆盖至少一个真实 AIC 和一个真实 AIV
workload，每点预热 10 次、记录 100 次并报告 p50/p95/p99；scheduler 数均为 1/2/4/全部可用
core。同机交替运行 HBG 与 HBG-AICore，并报告各 core 负载、queue/atomic/fanin/wake 和 cache
成本。M3 不设置自动门槛，报告完成即暂停等待人工评审；M4～M9 继续使用同一组 workload
观察正式化成本和功能扩展趋势。

### 5.8 最终验收清单

- [ ] 新 Runtime 只能显式启用，HBG 默认行为和回归测试无变化。
- [ ] HBG-AICore 与 HBG 位于 A5 runtime 下的平级目录；M0 机械复制后不再包含或链接原 HBG
  私有实现，两份代码独立维护。
- [ ] M0 已冻结 HBG commit、源目录 tree hash、允许差分、能力/优化矩阵、测试、性能 workload
  和环境。
- [ ] M0～M3 设备测试仅手工 opt-in；M3 完整报告已人工评审并明确批准启动第二阶段。
- [ ] v0 从 M0 起使用 128B task control 和 task-id ReadyQ；HBG mutable completion/slot/queue 布局
  未被 AICore 写入。
- [ ] v0 已由 v1 整体替换，旧 graph cache 确定性失效，生产路径不兼容双 ABI。
- [ ] M0 至 M9 均可独立构建、验证和回退，且保留固定 workload 的 HBG/前序性能基线。
- [ ] Host Orchestrator 构造完整图后才启动设备 scheduler。
- [ ] AICore 完成首次分类、依赖等待、wake relay、任务领取和完成传播。
- [ ] v0/v1 task-id MPMC 均通过 CPU 模型、A5sim 和 A5 验证，无丢任务、重复领取、邻位 clobber 或饥饿；
  O3 替代 ReadyQ 不阻塞 M9。
- [ ] Orchestrator/submit/Worker/ChipCallable 接口未改变；若 M3 引入
  `CallConfig.aicore_scheduler_limit`，其 binding、mailbox 和 remote protocol 已同步升版并通过
  兼容性测试；否则公共 CallConfig/wire 保持不变。
- [ ] 冻结 HBG normalized graph、输出、错误、task shape、deferred、diagnostics、生命周期和优化矩阵通过。
- [ ] MB-1/MB-2、MB-4 至 MB-9 等价契约均有明确用例和证据；未到对应里程碑的能力均明确拒绝。
- [ ] MB-2/MB-8 A5 真机 seam 和长压力通过。
- [ ] 单 lane、wide、MIX、`sync_start` 的普通和合法 deferred completion 组合均通过。
- [ ] 非法 shape、容量溢出、stale epoch 和不合法 deferred condition 启动前或有限时失败；若
  M5 采用 adapter，其缺失/版本错误同样确定性失败。
- [ ] first-error 后有限时 drain，无任务重复执行、丢唤醒或永久等待。
- [ ] nondiagnostic depth-two successor prepare 保留，diagnostics depth-one；active/prepared 多 callable 不串扰且 launch FIFO。
- [ ] 多 callable/graph cache 选择正确，PreparedExecution 固定资源、生命周期和内存占用可观测。
- [ ] DFX 可从 timeout 定位到 task、producer、queue、core、participant、deferred condition 和 phase。
- [ ] `allow_early_resolve`/early-dispatch 开关语义、关闭回退和优化效果达到冻结 HBG 能力要求。
- [ ] 形成 HBG 与 HBG-AICore 的完整性能画像和已知限制清单。
- [ ] O1/O2/O3 未完成不影响 M9 验收；任何已启用候选均可独立关闭并回退到 M9 基线。
