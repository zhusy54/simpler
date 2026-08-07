# Host 提前构图与 AICore 解依赖调度实现计划书

| 项目 | 内容 |
| ---- | ---- |
| 状态 | 已批准，待实现 |
| 文档类型 | 架构设计与实施计划，不代表功能已经落地 |
| 首发平台 | A5sim、A5 |
| 方案代号 | `host_build_graph_aicore`，简称 HBG-AICore |
| 基线 | `host_build_graph`（HBG） |
| 参考 | `simpler-dist` 的 fully-distributed-within-core（FDWIC）调度和验证思路 |
| 日期 | 2026-08-06 |

## 1. 结论、目标和边界

### 1.1 架构结论

本方案可行，建议新增显式 Runtime 变体 `host_build_graph_aicore`，不直接改变现有
`host_build_graph` 的默认行为。

该方案保留 HBG 已经成熟的“Host 同步执行 Orchestrator、一次性构造完整计算图”能力，将图构造后的首次就绪分类、依赖等待、完成传播、资源选择和任务领取迁移到 AICore。
AICPU 保留设备启动、AICore 镜像装载、硬件资源管理、终止和错误回收职责，但不再参与逐任务解依赖决策。

`simpler-dist` 的 FDWIC 用于参考 AICore 常驻调度、跨核原子状态、任务领取、可见性协议、
MIX/多核协作和硬件验证方式。本方案不照搬设备侧 Orchestrator replay，也不复制其测试源码或业务工作负载，只复用相同的并发和一致性契约。

### 1.2 目标

1. Host 在设备启动前构造完整 DAG，并完成容量校验、地址重定位和不可变图数据发布。
2. AICore 在图发布后并行完成首次依赖分类，并在运行期完成依赖释放和任务调度。
3. 保持现有 Orchestrator、任务提交和 Worker 对外接口不变。
4. 按里程碑逐步支持 A5sim 和 A5，形成可与 HBG 强差分的正确性基线。
5. 先交付 AIC/AIV 单 lane 任务，再在同一数据模型上增加 wide、MIX 和
   `sync_start`；每项能力均有独立验收节点。
6. 建立可验证原子正确性、跨核可见性、无死锁、公平性和调度开销的 DFX 能力。

### 1.3 非目标

- 本计划不支持 A2/A3。
- 不把新 Runtime 设为默认 Runtime，也不自动替换 HBG。
- 本计划不支持异步或延迟完成条件。
- 本计划不实现任务重试或 AICPU 调度回退。
- 不允许执行期继续增长整图或动态扩容图存储。
- 不要求与 HBG 产生完全相同的合法执行顺序，只要求依赖关系、结果和错误语义等价。
- 不把 `simpler-dist` 的设备 Orchestrator replay 迁入本方案。

### 1.4 关键术语

| 术语 | 含义 |
| ---- | ---- |
| Host graph image | Host Orchestrator 构造完成、可重定位并整体复制到设备的完整图数据 |
| Execution sidecar | 为 AICore 调度增加的任务控制、队列、资源位图、执行槽和 DFX 数据 |
| Immutable task record | 构图完成后只读的任务描述、fanin、参数元数据和资源需求 |
| Task control | 每任务独立的可变调度状态，固定 128B、缓存行隔离 |
| Initial classify | 所有任务执行前，由 AICore 并行判断 root、已满足或等待状态 |
| Wake relay | 生产者完成后关闭并摘取 wake list，再重新分类其消费者的过程 |
| Linked adapter | 每个 callable 必须提供的任务执行前后数据可见性适配层 |

## 2. 总体架构方案

### 2.1 端到端流程

```text
Host Worker
  -> 选择 host_build_graph_aicore
  -> 编译并装载 Host Orchestrator SO
  -> 同步执行 Orchestrator，构造完整 HBG 图
  -> 冻结图；校验任务数、fanin、队列、sidecar 和设备内存上限
  -> 生成 AICore execution sidecar 和地址重定位表
  -> 选择当前顶层 callable 对应的直链接 AICore 镜像
  -> H2D 发布 graph image + sidecar
  -> 启动 AICPU 控制核和 AICore 常驻 scheduler

AICore scheduler
  -> 并行 initial classify
  -> 全核 barrier，确认所有任务已入 ready queue 或 wake list
  -> 每个空闲 core 按自身类型轮询共享 ready queue
  -> 原子领取兼容任务；领取者直接成为执行者
  -> 在本 core 私有 execution slot 中物化任务参数
  -> 调用 linked adapter before hook
  -> 执行直链接 kernel
  -> 调用 linked adapter after hook
  -> 发布完成状态，摘取 wake list 并重新分类消费者
  -> 释放资源，继续调度

AICPU control plane
  -> 监控终止、超时和 first-error
  -> 必要时进入 fail-stop drain
  -> 收集最终状态并返回 Host
```

必须满足两个全局顺序：图和 sidecar 全部发布后才能启动首次分类；所有 AICore 完成首次分类后才能执行任何普通任务。第二个顺序避免“分类尚未注册 waiter、生产者已经完成”的启动竞态，也让首次分类可以按任务区间并行分片。

### 2.2 图表示与 execution sidecar

继续复用 HBG 的完整图、TensorMap 推导结果、任务参数和 fanin 表示，不另建第二套
Orchestrator 语义。新增 sidecar 只承载 AICore 调度需要的设备 ABI。

建议的数据对象如下：

| 对象 | 可变性 | 主要字段和职责 |
| ---- | ------ | -------------- |
| `AicoreGraphLaunchDesc` | 启动后只读 | 图基址、任务数、各区域容量、ready-set 地址、拓扑、镜像身份、DFX 地址 |
| `AicoreTaskRecord` | 只读 | task id、kernel id、fanin、资源 shape、block 数、predicate、参数元数据 |
| `AicoreTaskControl` | 可变 | completion、wake head、next waiter，以及 M5/M6 使用的 task epoch、参与者计数和错误码 |
| `AicoreRunControl` | 可变 | `expected_task_count`、批量更新的全局 `completed_count`、first-error 和退出状态 |
| `TaskReadySet` | 可变 | M0 至 M7 固定使用只保存 task id 的 Vyukov bounded MPMC；sharded MPMC 和 ready bitmap 属于 M8 候选优化 |
| `WorkerExecutionSlot` | 每核私有可变 | 领取者本地物化的 tensor/scalar 参数、`LocalContext` 和 `GlobalContext`；单 lane 路径无跨核发布 |
| `AicoreSchedulerDfx` | 可变 | 分类、领取、空轮询、等待、完成、重分类、冲突和错误计数 |

`AicoreTaskControl` 固定为 128B 并做静态布局断言。高频原子控制字段不能与 kernel
普通输出数据共用缓存行，防止 DCCI clean/invalidate 覆盖相邻原子更新。不可变记录和
可变控制分离，既降低误写风险，也便于 HBG 与新 Runtime 对同一图做差分。

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

沿用 HBG 已验证的“扫描 fanin、等待第一个未满足生产者”语义，但控制对象改为 AICore
可访问的 task-id 链表：

1. 生产者执行完成后，先通过 adapter after hook 发布 kernel 输出。
2. 以 release 语义写入任务完成状态。
3. 原子交换 `wake_list_head` 为关闭哨兵，禁止新 waiter 注册。
4. 遍历摘取的 waiter task id；每个 waiter 重新扫描全部 fanin。
5. waiter 若已满足，进入对应 ready queue；否则注册到新的第一个未完成生产者。
6. waiter 注册与生产者关闭并发时，以 CAS 结果和关闭哨兵决定重试，不允许丢唤醒。

此算法不维护完整 fanout 邻接表，内存开销与任务数和 fanin 成正比；一个消费者在每次
被唤醒后最多只挂在一个生产者上。代价是可能重复扫描 fanin，需用 DFX 统计
`fanin_scan_count` 和 `wake_reclassify_count` 验证真实成本。

### 2.5 Ready queue 和仲裁

- 基线使用有界 MPMC 队列，slot 采用 sequence counter 防 ABA。
- MPMC payload 是稳定 task id；弹出后通过 graph base 定位 record/control。
- MPMC 在构图结束时按“该类型可能同时进入队列的任务数”计算最坏容量并向上取 2 的幂。
- MPMC 的 DUMMY 容量必须覆盖所有可能受 predicate 控制的任务，不能假定多数 predicate 为真。
- 任一容量或总 sidecar 内存超过配置上限，Host 在启动设备前失败。
- 队列选择采用 executable-first：先过滤当前 core/cluster 能执行的 shape，再在可执行队列间
  round-robin，避免固定优先级长期饿死某类任务。
- 单 lane ReadyQ 按 AIC/AIV 类型共享；任务入队时不绑定具体 core，空闲 core 只轮询自身可执行的
  queue。MPMC 成功 pop 后，领取者直接执行，不再进行二次 core 选择或 task-local claim。
- MPMC `push` 返回 full 属于内部容量计算错误，设置 first-error，不能无限自旋。

M0 至 M7 的交付实现固定为 Vyukov bounded MPMC。M8 再基于固定正确性和性能基线，
对比 sharded MPMC、per-shard 非空 bitmask 和两级 ready bitmap，并按正确性、公平性、
A5 可见性与性能证据决定是否替换基线。具体协议和采用条件见
[AICore 共享竞争与 ReadyQ 替代方案分析](host-build-graph-aicore-shared-contention-analysis.zh.md)。

### 2.6 AIC/AIV 单 lane 调度

基础单 lane 调度采用对称 Pull 模型。每个 AIC/AIV core 既是 scheduler，也是 worker：core
空闲时进入调度循环，从与自身类型兼容的共享 ReadyQ 主动领取 task id；成功领取的 core
直接在本 core 上执行该任务。任务在入队时不绑定具体 core，不存在“scheduler core 领取后再
选择目标 lane”的二次派发。

M0 至 M7 中，ReadySet 的唯一消费操作就是 task 所有权转移：MPMC 成功 pop 的 core
取得对应 task 所有权。M8 候选实现也必须保留“候选结构的唯一消费操作即所有权转移”
的约束。单 lane 不维护独立 claim 字段或二次
claim CAS。exactly-once 依赖“每 task 只发布一次”和“每 ReadySet 元素只被领取一次”两个
不变量。单 lane 不 requeue/retry，也不会因为“目标 lane 不可用”重新入队。core 只有在未执行
任务时才进入调度循环，因此空闲状态由控制流隐式保证，不维护共享 AIC/AIV idle bitmap。

依赖等待位置由 wake list 和 ready set 表达，不额外维护 `WAITING -> READY` 共享 CAS。initial
classify 和 wake relay 必须保证 waiter 所有权只转移、不复制，task 一生只发布到 ReadySet
一次。跨 run 复用时，ReadySet 必须在所有 core 退出后整体重建或按 generation 隔离，不能让
旧 queue slot/ready bit 进入新一轮。详细分析见共享竞争分析文档。

每个 core 使用本地 `WorkerExecutionSlot`，从 `AicoreTaskRecord` 物化与现有
`PTO2DispatchPayload` 等价的参数：tensor 指针、scalar、`LocalContext` 和
`GlobalContext`。slot 由领取者写入并由同一 core 同步消费，不需要跨核 epoch、doorbell 或
ack；kernel 返回后，该 core 完成状态发布和 wake relay，再回到调度循环。

predicate 在依赖满足后读取。若任务被判定为 DUMMY 或 predicate 为 false，则 scheduler
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
- 单 lane `WorkerExecutionSlot` 是领取 core 的私有数据，不参与核间一致性协议。
- 控制原子字段与被 DCCI 操作的普通数据至少缓存行隔离。

每个顶层 callable 的直链接 AICore 镜像必须链接一对 adapter hook：

- `before_task`：在 kernel 开始前使参数和生产者输出对当前 lane 可见。
- `after_task`：在完成状态发布前，使当前 kernel 输出对后继 lane 可见。

adapter 负责理解该 callable 的 kernel ABI 和 tensor 访问范围。Runtime 不猜测业务数据
范围。缺少 hook 必须在链接或构建校验阶段失败，不提供静默空实现。MB-8 等硬件 seam
验证通过之前，不得宣称跨 AICore 数据依赖正确。

### 2.8 直链接镜像与 Worker 生命周期

借鉴 `simpler-dist` 的 `build_aicore_with_extra_sources`，为每个顶层 callable 构建包含
scheduler、adapter、其 Orchestrator 所引用 kernel 入口的直链接 AICore 镜像。缓存键至少
包含：平台、Runtime、callable build identity、全部额外源码哈希、编译宏、PTO-ISA 版本
和 adapter ABI 版本。

Worker 内部维护 `callable identity -> AICore image/handle` 映射；每次运行选择当前 callable
对应的镜像，不能使用 scene-test 的 Worker 全局 override 覆盖所有 callable。CANN 没有
可靠公开的 AICore handle unregister 接口，因此 handle 随 Worker 生命周期缓存并在 Worker
销毁时整体释放。缓存必须有数量和内存 DFX，防止长寿命 Worker 注册无限 callable。

### 2.9 AICPU 控制面

AICPU 继续承担以下职责：

- 装载并启动正确的 callable AICore 镜像。
- 建立物理 core id、core type、cluster 与 sidecar lane 的映射。
- 发布 graph launch descriptor，等待 scheduler ready。
- 监控超时、first-error、正常完成和退出确认。
- 执行 AICore 无法完成的寄存器/MMIO 控制和最终资源回收。

AICPU 不再扫描 fanin、不维护 ready queue，也不逐任务选择执行 core。这样可以明确证明
“解依赖在 AICore”，同时保留 AICore 无权访问的硬件控制边界。

### 2.10 MIX、wide 和 `sync_start` 扩展

这些能力在 M5、M6 逐项实现，并从对应里程碑开始引入 idle bitmap、participant slot 发布和
跨核 task epoch；这些对象不属于单 lane Pull 路径：

| 能力 | 调度方案 |
| ---- | -------- |
| MIX | 每 cluster 由 AIC leader 领取 MIX task；即使 AIC kernel 不活跃，也保留协调 leader；一次性预留所需 AIC/AIV lane，再发布同一 task epoch |
| Wide | leader 从 idle bitmap 预留多个 participant core，为每个 block 分配 ticket；最后一个 participant 完成后发布任务完成 |
| `sync_start` | 使用 task-local epoch barrier；所有资源预留且 execution slot 发布后，participants 到达 barrier，同一 epoch 放行 |

资源预留必须按全局固定顺序进行，并且采用“全部成功或全部回滚”，禁止持有部分资源等待
其余资源，否则 MIX 和 wide 组合会形成环路死锁。leader 只负责协调，不代表它一定执行
业务 AIC kernel。

### 2.11 错误和退出

- 采用 first-failure fail-stop。第一个错误通过 CAS 固化错误码、task id、core id 和 phase。
- first-error 后停止领取新任务；已领取任务进入受控 drain，不能继续释放新的业务后继。
- 不重试，不切换到 AICPU 执行，不掩盖首错。
- AICPU 等待所有 scheduler/participant 确认退出，再把错误返回 Host。
- 超时诊断必须输出队列深度、未完成任务、所等待生产者、各 core 当前状态/任务、在途 task
  和最后事件；M5/M6 还需输出资源 idle bitmap。
- AICPU 以批量提交的全局 `completed_count == expected_task_count` 且无 first-error 判断正常完成。
  exactly-once completion 成立时，完成数到达总数已经隐含队列为空且不存在未完成的在途 task，
  因此单 lane 路径不维护共享 idle bitmap 或 in-flight counter；每 core 当前状态只用于 DFX。

## 3. 可行性、风险与设计选择

### 3.1 可行性依据

| 能力 | 当前基础 | 可复用程度 | 结论 |
| ---- | -------- | ---------- | ---- |
| Host 提前构图 | HBG 已在 Host 同步运行 Orchestrator 并整体 H2D | 高 | 无需引入新图语义 |
| 依赖生成 | `compute_task_fanin`、TensorMap 和显式依赖已经产出 fanin | 高 | 复用构图结果 |
| 解依赖算法 | HBG 已有 completion flag、首次分类、wake list 和重新分类 | 高 | 保留语义，替换执行位置和 ABI |
| 有界队列 | HBG 已有 Vyukov MPMC 实现 | 中 | 算法可参考，payload 和内存语义需重写 |
| 参数物化 | HBG 已有 `PTO2DispatchPayload` | 中 | 需改成 AICore per-lane execution slot |
| AICore 常驻调度 | FDWIC 已证明该方向可编译、可运行和可观测 | 中 | 不复用设备 Orchestrator，只参考并发机制 |
| 直链接构建 | FDWIC 已有 extra-source AICore 构建与缓存 | 中高 | 需升级为 per-callable Worker 映射 |
| 仿真和真机验证 | 两仓都有 A5sim/A5 测试基础 | 高 | 可建立分层验证 |

可行性的前提不是“把 AICPU C++ 原样编译到 AICore”，而是冻结 HBG 的语义契约后，针对 AICore 内存模型、编译器和资源拓扑实现新的控制 ABI。

### 3.2 主要风险和控制措施

| 风险 | 影响 | 控制措施 | 阶段闸门 |
| ---- | ---- | -------- | -------- |
| A5 跨核 cache 不一致 | 读到旧参数或旧输出，产生静默错算 | mandatory adapter、缓存行隔离、MB-8 seam、长时间压力 | seam 未通过不得进入业务 ST |
| 原子和 DCCI 同行 clobber | 丢完成、重复执行或死锁 | 128B control、普通数据分区、MB-2 邻居压力 | 原子探针和布局断言通过 |
| AICore 编译器不支持现有 C++ 原语 | 无法直接复用 MPMC/wake 代码 | 建立最小 atomic ABI 封装和独立编译探针 | WP1 前完成工具链探针 |
| 动态函数入口或链接不稳定 | kernel 无法调用或调用错误版本 | per-callable 直链接、强 identity、链接期 hook 校验 | image identity UT + A5 smoke |
| 多 scheduler 原子热点 | 调度开销抵消收益 | 分类分片、task-id 队列、可执行队列 RR、冲突 DFX | 性能 profile，不设统一硬门槛 |
| ReadyQ 基线成为性能热点 | MPMC cursor 串行抵消多 scheduler 收益 | M0 至 M7 保持 MPMC 正确性基线并采集竞争 DFX；M8 模型检查并对比候选优化 | M8 冻结是否替换基线 |
| graph/queue 最坏容量过大 | HBM 占用不可控或启动失败 | Host 精确预估、配置 cap、启动前 fail | 容量边界 UT/ST |
| MIX/wide 部分资源持有 | 环路死锁 | 全局预留顺序、all-or-rollback、task epoch barrier | 模型检查 + 故障注入 |
| 多 callable 镜像泄漏 | Worker 长时间运行内存增长 | Worker 生命周期缓存、数量/字节统计、上限 | 生命周期 ST |
| sim 与真机内存模型差异 | sim 通过、真机失败 | A5sim 只作第一阶段；A5 seam 和压力为最终依据 | 真机证据单独签收 |
| 错误路径不能退出 | 测试或生产永久挂起 | first-error、fail-stop drain、watchdog 状态快照 | 每类错误注入必须有限时退出 |
| 单个里程碑范围过大 | 正确性闭环延迟 | 单 root、单核 DAG、多核单 lane、wide、MIX/sync 逐级交付 | 每个里程碑独立验收 |
| 与 HBG 语义漂移 | 同图结果或错误不同 | normalized graph + output + error 强差分 | 差分套件为合入门槛 |

### 3.3 已批准的设计选择

下列问题已经完成决策，是实现基线；任何变更应重新进行架构评审。

| 序号 | 设计选择问题 | 已批准决策 |
| ---- | ------------ | ---------- |
| 1 | 修改 HBG 还是新增 Runtime？ | 新增显式 `host_build_graph_aicore`，保留 HBG |
| 2 | 首发平台覆盖哪些架构？ | A5sim、A5；本计划不覆盖 A2/A3 |
| 3 | Orchestrator 在哪里、如何运行？ | Host 同步运行到结束，设备启动前完成全图 |
| 4 | 是否建立第二套图格式？ | 复用 HBG 图，增加 AICore execution sidecar |
| 5 | 解依赖使用 refcount 还是扫描 fanin？ | 扫描 fanin，等待首个未完成 producer，wake relay 后重分类 |
| 6 | 首次分类在哪里完成？ | AICore 多核并行分类，完成后全局 barrier |
| 7 | ReadyQ 使用什么实现？ | M0 至 M7 固定使用只保存 task id 的 Vyukov bounded MPMC；M8 对比 sharded MPMC、per-shard 非空 bitmask 和两级 ready bitmap，再决定是否替换 |
| 8 | task shape 如何递进？ | M1 至 M4 完成 AIC/AIV 单 lane；M5 交付 wide；M6 交付 MIX 和 `sync_start` |
| 9 | wide task 如何组织？ | leader + participant cores + block tickets |
| 10 | MIX 的协调者是谁？ | cluster AIC leader，必要时为纯协调角色 |
| 11 | `sync_start` 如何实现？ | task-local epoch barrier |
| 12 | 每任务控制块如何布局？ | 固定 128B，原子控制与普通数据隔离 |
| 13 | kernel 数据可见性由谁负责？ | mandatory linked adapter before/after hooks；缺失则构建失败 |
| 14 | AICore 产物如何组织？ | 每个顶层 callable 的直链接镜像，按完整 build identity 缓存 |
| 15 | 是否改变公开 submit/Orchestrator API？ | 不改变，仅增加内部 Runtime 选择和产物映射 |
| 16 | 完整图超过容量如何处理？ | 设备启动前失败，不运行期扩容 |
| 17 | 任务失败后是否重试或回退？ | first-failure fail-stop，不重试、不回退 AICPU |
| 18 | 多 ready queue 如何仲裁？ | executable-first，再 round-robin |
| 19 | 单 lane 任务如何选择执行 core？ | 空闲 core 从兼容共享 ReadyQ 主动 Pull；M0 至 M7 中 MPMC pop winner 直接执行，不维护额外 task 所有权字段、idle bitmap 或跨核派发 |
| 20 | 如何使用 `simpler-dist` 测试？ | 复用验证思想和契约，不复制源码或业务工作负载 |
| 21 | 验证顺序是什么？ | A5sim 第一阶段，A5 真机第二阶段 |
| 22 | 正确性 oracle 是什么？ | HBG 的图、输出和错误强差分 |
| 23 | 里程碑是否设置硬性能门槛？ | 不设置统一加速比；每个里程碑必须与 HBG 和前序里程碑形成性能对比，显著退化必须可解释 |
| 24 | 新 Runtime 是否默认启用？ | 不默认，显式选择 |
| 25 | DFX 如何随能力递进？ | counters 与对应功能在同一里程碑交付；M7 完成 bounded event ring 和泳道准入 |
| 26 | 是否支持 deferred/async completion？ | 本计划不支持；检测到即启动前拒绝 |
| 27 | `allow_early_resolve` 如何处理？ | 本计划不做优化，按普通依赖语义执行并记录诊断 |

仍需在实现阶段通过数据校准、但不改变架构的参数包括：默认 graph memory cap、event ring
容量、AICore scheduler 数量、队列回退重试次数和性能采样轮数。

## 4. 基于 pypto/runtime 的关键修改点和实现路径

### 4.1 关键修改点列表

| 区域 | 当前参考点 | 计划修改 |
| ---- | ---------- | -------- |
| Runtime 注册 | `simpler_setup/runtime_builder.py` | 注册 `host_build_graph_aicore`；纳入平台、变体、ABI 和 callable identity |
| Orchestrator 编译 | `simpler_setup/kernel_compiler.py` | 新 Runtime 继续用 Host G++ 编译 Orchestrator SO；复用 HBG 分支判断 |
| per-callable 镜像 | `simpler_setup/runtime_builder.py` | 泛化 extra-source 构建；加入 adapter、kernel 集合、宏和 PTO-ISA 指纹 |
| 场景测试缓存 | `simpler_setup/scene_test.py` | 从 Worker-wide override 改为 callable-scoped artifact 选择和缓存校验 |
| Runtime 构建清单 | 新增 `src/a5/runtime/host_build_graph_aicore/build_config.py` | 分离 host、AICPU control、AICore scheduler/adapter 构建目标 |
| Host 构图 | `src/a5/runtime/host_build_graph/host/runtime_maker.cpp` | 抽取可复用的构图、冻结、重定位和 H2D builder；HBG 行为保持不变 |
| 依赖生成 | `runtime/pto_dep_compute.h`、`orchestrator_core/pto_orchestrator.cpp` | 复用 fanin 生成；增加 sidecar finalize 和静态能力检查 |
| 图 ABI | `runtime/pto_runtime2_types.h`、`runtime/pto_shared_memory.h` | 保持 HBG ABI；新 Runtime 定义只读 record 和 128B control sidecar |
| HBG 调度语义 | `runtime/scheduler/pto_scheduler.h`、`scheduler_completion.cpp`、`scheduler_dispatch.cpp` | 抽取可共享的状态机规范/Host 模型；不把 AICPU 指针队列直接用于 AICore |
| AICore 执行 | `aicore/aicore_executor.cpp`、`runtime/pto2_dispatch_payload.h` | 新建常驻 AICore scheduler；每 core 自主 Pull，并用本地 `WorkerExecutionSlot` 取代 AICPU 逐任务 payload 发布 |
| AICPU 执行 | `aicpu/aicpu_executor.cpp` | 新 Runtime 只保留 launch/control/drain；删除逐任务解依赖职责 |
| Worker 内部 | `src/common/worker/chip_worker.*`、`pto_runtime_c_api.h` | 内部选择 callable image/handle 和 launch descriptor；公开 API 不变 |
| 错误状态 | `common/pto_runtime_status.h`、公共 error-name 映射 | 增加 sidecar 容量、adapter 缺失、queue full、重复 ready、epoch 和 drain 错误 |
| DFX | 现有 L2 swimlane、PMU、args dump 基础 | 增加 scheduler counters、event ring、依赖等待和 per-core Pull 事件；M5/M6 增加资源预留事件 |
| Tests | `tests/ut`、`tests/st/a5` | 增加模型、差分、A5sim、A5 seam、压力、故障注入和生命周期测试 |

共享代码抽取遵循“先写等价测试，再移动”的原则。不能让 HBG 依赖 AICore 专用类型，也不能让新 Runtime 读取 HBG 队列中的设备指针。

### 4.2 工作包和依赖顺序

| 工作包 | 内容 | 前置 | 交付物和退出条件 |
| ------ | ---- | ---- | ---------------- |
| WP0 契约冻结 | 固化 HBG 图、依赖、predicate、错误、参数 ABI 和差分输出格式 | 无 | graph signature 工具；HBG 基线用例全通过 |
| WP1 AICore 原语探针 | 验证 atomic load/CAS/exchange/fetch-or/fetch-and、DCCI、barrier、函数链接和 128B 布局 | WP0 | A5sim/A5 probe；不支持的原语有替代封装 |
| WP2 Host sidecar builder | 冻结图、容量计算、record/control/queue/slot 布局、重定位 | WP0 | Host UT 覆盖零任务、上限、溢出、地址重定位 |
| WP3 per-callable 构建 | 直链接 scheduler+adapter+kernels；强缓存 identity；Worker 映射 | WP1 | 多 callable 重复运行和缓存生命周期 UT |
| WP4 AICore scheduler | initial classify、barrier、MPMC 基线、唯一 Pull、单 lane、wake relay、完成 | WP1、WP2 | CPU 模型、A5sim 随机 DAG 和长压力通过；A5 完成 MPMC 正确性闭环 |
| WP5 端到端控制面 | AICPU launch/drain、Worker 集成、状态返回、公开 API 兼容 | WP3、WP4 | HBG 差分 ST；故障能有限时退出 |
| WP6 可见性闭环 | mandatory adapter、MB-2/MB-8 seam、真实 tensor 链 | WP3、WP5 | A5 真机压力无旧读/丢写/邻居 clobber |
| WP7 高级资源调度 | MIX leader、wide tickets、all-or-rollback、`sync_start` epoch | WP5、WP6 | M5/M6 分别完成 wide 和 MIX/sync 独立验收 |
| WP8 DFX 与性能 | counters、event ring、泳道、调度开销和公平性报告 | WP0 后贯穿各工作包 | 每个里程碑可独立定位停滞并形成 HBG/前序基线对比 |

### 4.3 里程碑

里程碑按可运行功能递进，而不是按代码模块拆分。每个里程碑只增加一组可独立观察的能力，
并满足以下共同退出条件：

1. 新增状态机和数据结构有 Host/CPU 单元测试或可重复模型测试。
2. 新增功能有 A5sim 或 A5 端到端用例，并与 HBG 比较 graph signature、输出和错误语义。
3. 新 Runtime 仅能显式选择；HBG 和已有 Runtime 的构建产物、公开 API 与回归结果不变。
4. 保存本里程碑的固定 workload、环境、DFX counters 和分阶段时延，分别与 HBG、前序里程碑
   比较。未设置统一加速比，但显著退化必须能定位和解释。
5. 当前不支持的下一阶段能力在设备启动前返回明确错误，不能静默降级或走未验证路径。
6. 每个里程碑都能作为独立集成点提交和回退；下一阶段可以预留 ABI 字段，但不能提前改变
   已交付能力的运行语义。

#### M0：独立 Runtime 生命周期

新增功能：

- 新 Runtime 可被显式选择、构建和装载。
- Host Orchestrator 沿用 HBG 路径，能冻结并发布空 graph image 和最小 sidecar。
- AICPU 能启动 AICore 常驻入口，完成 ready、正常退出、first-error 返回和资源回收。

独立验证和性能基线：

- Host UT 覆盖零任务布局、重定位、容量溢出和非法 launch descriptor。
- A5sim 空图和单 dummy 图反复启动/退出，无泄漏、超时或遗留状态。
- HBG 全量既有测试不变；记录 Runtime 构建、H2D、启动和退出时延，形成后续统一基线。

#### M1：单 core、单 root 任务执行

新增功能：

- 仅启用一个 AIC 或一个 AIV，执行无依赖 root task。
- 打通 per-callable 直链接镜像、`WorkerExecutionSlot`、adapter before/after hook 和 kernel 调用。
- 完成 ReadySet 领取、参数物化、业务执行、完成发布和错误返回的最短纵向链路。

独立验证和性能对比：

- UT 覆盖 core-local slot 布局、参数物化、image identity 和 adapter 缺失错误。
- A5sim 分别运行单 AIC、单 AIV、predicate false 和 kernel/adapter 错误用例，与 HBG 强差分。
- 比较 M0、HBG 和本里程碑的 ready-to-start、kernel 前后 adapter、完成及总时延。

#### M2：单 core DAG 解依赖

新增功能：

- 在单 core 上交付 initial classify、fanin 扫描、wake list 和 wake relay。
- 支持 chain、diamond、fanout/fanin、多个 root、DUMMY、predicate 和预完成 producer。
- 交付 queue/wake/first-error 的最小 counters，确保依赖停滞可定位。

独立验证和性能对比：

- CPU 模型覆盖 waiter 注册/关闭竞态、重复唤醒、长 fanin、queue 满和 first-error 幂等性。
- A5sim 对基础图族逐边验证 `producer complete-before-consumer start`，并与 HBG 比较图、输出和错误。
- 固定单核 DAG 比较 M1、HBG 和本里程碑的分类、重复 fanin 扫描、完成传播及总时延。

#### M3：多 core 单 lane 调度

新增功能：

- AIC/AIV 多 scheduler 并行 initial classify，并在全核 barrier 后开始执行。
- 交付有界 MPMC、ReadySet exactly-once 领取、兼容队列 round-robin 和领取者本核执行。
- 交付基础 Vyukov bounded MPMC 的 CPU/A5sim 正确性和压力证据，不在本阶段引入
  sharded MPMC、ready bitmap 或 bitmask 优化。
- 支持随机 DAG、queue wrap、多 root 并发、公平进展和 first-error 后有限时 drain。

独立验证和性能对比：

- 并发模型测试覆盖 MPMC wrap/ABA、task 单次发布/领取、无丢唤醒、无重复执行，以及 busy/idle
  反复切换和 flush 后立即出现新任务时批量完成计数无漏计、重复或提前退出。
- A5sim 运行多 seed、多 scheduler 数、慢核和长压力；所有单核用例继续通过。
- 固定 DAG 比较 M2、HBG 和本里程碑的并行分类、queue contention、各 core 负载及总时延。

#### M4：A5 单 lane 真机正确性

新增保证：

- 完成 AICore atomic、DCCI、128B control 隔离和 adapter 数据可见性闭环。
- producer output 到 consumer input 在 AIC→AIC、AIV→AIV、AIC→AIV 和 AIV→AIC 下均无旧读。
- 完成多 callable image 选择、缓存 identity 和 Worker 生命周期验证。

独立验证和性能对比：

- MB-2、MB-5、MB-8 等价 seam 在 A5 重复压力下无旧读、丢写或邻居 clobber。
- M0 至 M3 的适用用例全部在 A5 复验；真实单 lane callable 与 HBG 强差分。
- 在 root burst、wake burst 和长压力 workload 上验证基础 MPMC 的无丢任务/饥饿、A5 可见性、
  atomic retry、ready-to-start、尾延迟和总时延，并保存供 M8 对比的固定基线。
- 比较 A5sim、A5、M3 和 HBG，量化 atomic contention、adapter/DCCI、调度与总时延。

完成 M4 后，AIC/AIV 单 lane 能力才形成可独立签收的正确性闭环。

#### M5：Wide 同类型多核任务

新增功能：

- leader 一次性预留多个同类型 participant core，为每个 block 分配 ticket。
- participant 按 task epoch 读取 execution slot，最后一个 participant 发布任务完成。
- 资源不足、participant 迟到和执行失败均遵循 all-or-rollback 与有限时 drain。

独立验证和性能对比：

- 模型测试覆盖唯一 leader、ticket 恰好一次、部分预留回滚、迟到 epoch 和最后完成者竞争。
- A5sim/A5 分别运行不同 block 数、资源紧张、慢 participant 和故障注入；M4 单 lane 套件不变。
- 对同一可表达 workload 比较单 lane 拆分执行、HBG wide 和本里程碑的预留、同步、kernel 与总时延。

#### M6：MIX 和 `sync_start`

新增功能：

- cluster AIC leader 按固定全局顺序预留 AIC/AIV participant，并统一发布 task epoch。
- 支持 leader 仅协调、MIX 多资源执行和 task-local `sync_start` epoch barrier。
- 组合资源不足、leader 退出和 participant 迟到均能全部回滚或有限时失败。

独立验证和性能对比：

- 模型测试覆盖异构资源预留顺序、无部分持有等待、barrier generation 和完成汇聚。
- MB-4、MB-7 等价用例在 A5sim/A5 形成证据；M4 单 lane 与 M5 wide 套件继续通过。
- 对 MIX workload 比较 HBG、无 `sync_start` 和启用 `sync_start` 的预留、到达偏差及总时延。

#### M7：完整功能准入

新增保证：

- 全量差分、压力、错误、容量、生命周期和长期稳定性测试通过。
- counters、bounded event ring 和泳道能够从 timeout 定位到 task、producer、queue、core 和 phase。
- 所有已交付 task shape 均具备 A5sim/A5 证据和明确的已知限制。

独立验证和性能对比：

- 运行固定版本的完整回归矩阵，确认 HBG 默认行为、公开 API 和既有业务结果无变化。
- 输出 HBG 与 HBG-AICore 的构图、H2D、分类、ready-to-start、执行、完成传播、drain 和总时延对比。
- 汇总 M0 至 M7 的趋势，解释所有显著退化并形成 M8 优化基线；性能采集本身不能改变调度语义。

#### M8：SIMT、CompletionQ 和 MPMC 优化

新增功能：

- 使用 SIMT 并行处理 graph image/sidecar 启动初始化、task 区间分类和 fanin 扫描，减少大图
  启动与 initial classify 串行周期；task 只分类一次、waiter 只挂一条链和全核 barrier 语义不变。
- 增加可独立开关的 CompletionQ：执行 core 发布输出和 completion 后入队稳定 task id，
  scheduler core 批量消费并执行 wake-list close、wake relay 和后继重新分类，使执行与完成传播解耦。
- 在 Vyukov bounded MPMC 正确性基线上评估 sharded MPMC、per-shard 非空 bitmask/两级 summary、
  旋转 local-first 扫描、批量 enqueue reservation 和 cursor/slot cache-line 隔离。成功 pop 仍是
  task 所有权转移的唯一线性化点，不增加 task-local claim CAS。具体候选和采用门槛见
  [AICore 共享竞争与 ReadyQ 替代方案分析](host-build-graph-aicore-shared-contention-analysis.zh.md)。

独立验证和性能对比：

- SIMT 覆盖线程间归并、非整齐 task/fanin 数、长 fanin、wake head 热点和 barrier 可见性，
  对比 M7 的启动、分类、注册竞争和 ready-to-start 时延。
- CompletionQ 覆盖 task 单次发布/消费、completion-before-enqueue、queue 满、突发 fanout 和尾部
  drain，对比 M7 直接 relay 的原子次数、完成传播延迟和总时延。
- MPMC 候选覆盖 wrap/ABA、并发 push/pop、稀疏尾部、热点 shard、无饥饿和 A5 可见性，
  对比 M7 的 atomic 次数、CAS retry、ready-to-start、尾延迟和总时延。
- 三项优化均可独立启停和回退；组合启用时 M0 至 M7 的完整回归、强差分和 A5 长压力继续通过。

## 5. 功能验证方案

### 5.1 验证原则

1. 正确性优先于性能，真机内存一致性证据优先于 sim 结果。
2. HBG 是强 oracle：比较 normalized graph、最终输出、错误类型和依赖不变量。
3. 合法调度顺序可以不同，因此不逐项比较全序 trace；验证每条依赖边的
   `producer complete-before-consumer start`。
4. `simpler-dist` 用例按并发契约重建，不能把其历史 PASS 当作本 Runtime 的证据。
5. 每个失败用例必须保留 first-error 和 bounded event ring，禁止只报告 timeout。
6. 每个里程碑新增测试后继续运行全部前序里程碑套件，新增能力不能替代已有回归证据。
7. 每个里程碑使用固定 workload 分别与 HBG 和前序里程碑比较；保存环境、镜像 identity、
   DFX counters、分阶段时延和总时延，保证结果可重复。

### 5.2 Host/CPU 单元和模型测试

覆盖以下对象：

- sidecar 字节布局、对齐、重定位和图容量计算。
- MPMC queue 的空、满、wrap、ABA、并发 push/pop 和 task-id 有效性。
- M8 候选 ReadySet 的 set/acquire、summary 清位竞态、fallback scan、热点 word、
  公平性和非 64 对齐任务数；这些用例不属于 M0 至 M7 基线。
- wake list 的注册/关闭竞态、关闭后重扫、重复唤醒和长 fanin。
- initial classify 的区间分片、预完成 producer、隐藏分配和 barrier。
- task 单次 ready 发布、ReadySet exactly-once 领取和 first-error 幂等性。
- M5/M6 的 idle bitmap、MIX/wide all-or-rollback 和 task epoch barrier 状态机模型。
- callable build identity、缓存命中/失效、多 callable 切换和 Worker 销毁。
- graph cap、queue cap、缺 adapter、非法 shape、async completion 的启动前错误。

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

基础图族包括：空图、单 root、长 chain、diamond、宽 fanout、宽 fanin、多 root、多层随机
DAG、重复显式依赖、TensorMap overlap、manual dependency/scope、predicate true/false、
hidden allocation、多 callable 连续运行和容量边界。

### 5.4 `simpler-dist` atomic minibench 等价验证矩阵

这里的“相同用例”指验证相同内部契约，不要求使用相同实现或业务源码。

| MB ID | `simpler-dist` 契约 | 本方案等价用例 | 阶段 |
| ----- | ------------------- | -------------- | ---- |
| MB-1 | sharded claim 恰一 winner、无跳号 | 多 scheduler 竞争 ReadySet；M3 检查 MPMC slot 恰一 winner 和 task-id 连续性，M8 对候选实现做同等检查 | M3 基线，M8 候选 |
| MB-2 | 64B completion flag，邻居无 clobber | 相邻 task control 高频完成/重分类，检查完成位和错误位不丢失 | M3 模型，M4 A5 |
| MB-4 | `block.won` 多 lane 协作 | MIX leader/participants 竞争、唯一 leader、参数发布和最后 participant 完成 | M6，A5 必测 |
| MB-5 | shared map 定序、seq 和回收 | 多核读取同一 immutable graph record/control，验证发布 epoch、task-id 映射和无旧 generation | M3 模型，M4 A5 |
| MB-6 | 确定性 GM heap、容量和反压 | graph/sidecar/queue 临界容量、满队列、启动前拒绝和长序列无越界 | M0 Host，M3 运行期 |
| MB-7 | `core_progress[]` 与 run-ahead | 注入慢核/忙核，检查 executable-first RR、公平进展、无无限超前或饥饿 | M3 单 lane，M5/M6 多资源 |
| MB-8 | `Coherent<T>`/DCCI seam | producer 写 payload、发布完成、consumer invalidate 后读取；对照无 adapter 负例 | M4，A5 必测 |
| MB-9 | private map 每核确定性 | 同一输入和 seed 重复构图/运行，graph signature、task 映射和结果稳定 | M3 |

MB-2、MB-4、MB-6、MB-7、MB-8 还应保留 `simpler-dist` 测试风格中的 Normal、Heavy、
边界/慢核或多轮变体。测试报告必须区分“数值通过”和“完整调度契约通过”，并记录
graph signature、Runtime identity、image build identity、平台和迭代次数。

### 5.5 A5sim 验证

- 先运行构建、加载、initial classify 和单 lane 端到端 smoke。
- 运行 HBG 双 Runtime 差分图族。
- 对随机 DAG 做多 seed、多 scheduler 数和多 queue wrap 压力。
- 注入 queue full、非法/重复 task id、adapter 返回错误、kernel 错误和超时；bitmap
  summary 错误注入属于 M8 候选实现验证。
- M5 增加 wide participant 迟到和资源不足；M6 增加 MIX、`sync_start` 和异构资源不足。

A5sim 的退出标准是功能和状态机覆盖完整，不把 sim 的 cache 行为作为 A5 可见性结论。

### 5.6 A5 真机验证

按以下顺序执行：

1. Atomic/DCCI 原语 probe：单控制块、相邻控制块、同缓存行负例和分离缓存行正例。
2. MB-2/MB-8 seam：固定数据模式、多轮、跨 AIC/AIV 组合和邻居压力。
3. 单 lane 图族：chain、fanin/fanout、predicate、随机 DAG、长时间 queue wrap。
4. 故障和 drain：kernel 错误、adapter 错误、watchdog、部分 core 迟到。
5. M5/M6 wide、MIX 和 `sync_start`：不同 cluster、资源紧张、慢 participant、组合 shape。
6. 真实业务 callable：使用现有 pypto/runtime 业务用例，分别运行 HBG 和 HBG-AICore。

真机每次证据需保存芯片/驱动/CANN/PTO-ISA/Runtime commit、镜像 identity、编译宏、随机
seed、重复次数、首错和 DFX 摘要。短 smoke 通过后再逐级提高到长压力，避免把偶发一致性
问题误判为通过。

### 5.7 DFX 不变量和性能画像

固定 counters 至少包括：

- task 总数、initial-ready、initial-waiting、completed、dummy-completed。
- fanin 扫描次数/边数、wake 注册/关闭/重分类次数、关闭竞态重试。
- 各 ready queue push/pop/full/empty、最大深度和回退次数。
- MPMC cursor/sequence atomic 和 CAS retry；M8 候选实现增加 bitmap L0/L1 atomic、
  bit-clear retry、fallback scan 和 hot-word 最大重试。
- ReadySet acquire success/retry、重复 ready 检测、各 core Pull/执行任务数；M5/M6 增加 idle
  bitmap 预留/回滚。
- adapter before/after 次数和周期、scheduler 周期、kernel 周期、完成传播周期。
- first-error、drain task/core 数、watchdog 状态。

可选 bounded event ring 记录 `CLASSIFY`、`WAIT_ON`、`READY`、`CLAIM`、`START`、
`COMPLETE`、`WAKE`、`RESERVE`、`ROLLBACK`、`ERROR`，记录满后只置 overflow 标志，
不能阻塞 scheduler。

性能报告对比 HBG 和 HBG-AICore 的 Host 构图、H2D、首次分类、ready-to-start、kernel、
完成传播、尾部 drain 和总耗时，同时报告各 core 负载分布、queue contention 和 cache
维护成本。每个里程碑的签收条件都是数据完整、结果可解释，不以单一加速比作为合入门槛。

### 5.8 最终验收清单

- [ ] 新 Runtime 只能显式启用，HBG 默认行为和回归测试无变化。
- [ ] M0 至 M8 均可独立构建、验证和回退，且保留固定 workload 的 HBG/前序性能基线。
- [ ] Host Orchestrator 构造完整图后才启动设备 scheduler。
- [ ] AICore 完成首次分类、依赖等待、wake relay、任务领取和完成传播。
- [ ] M0 至 M7 的 MPMC 交付实现通过 CPU 模型、A5sim 和 A5 验证，且无丢任务、
  重复领取或饥饿；M8 另行完成候选 ReadySet 的对比和采用决策。
- [ ] 公开 Orchestrator/submit/Worker API 未改变。
- [ ] HBG normalized graph、输出和错误强差分通过。
- [ ] MB-1/MB-2、MB-4 至 MB-9 等价契约均有明确用例和证据；未到对应里程碑的能力均明确拒绝。
- [ ] MB-2/MB-8 A5 真机 seam 和长压力通过。
- [ ] 不支持的 async、非法 shape 和容量溢出均在启动前失败。
- [ ] first-error 后有限时 drain，无任务重复执行、丢唤醒或永久等待。
- [ ] 多 callable image 缓存选择正确，生命周期和内存占用可观测。
- [ ] DFX 可从 timeout 定位到具体 task、producer、queue、core 和 phase。
- [ ] 形成 HBG 与 HBG-AICore 的完整性能画像和已知限制清单。
