# HBG-AICore 共享竞争与 ReadyQ 替代方案分析

| 项目 | 内容 |
| ---- | ---- |
| 状态 | 设计分析，供 M3/M4 实现和 A5 验证使用 |
| 调度模型 | 共享 ReadyQ、自主 Pull、领取者本核执行 |
| 范围 | 单 lane 正常运行热路径 |
| 不包含 | 启动/退出、first-error/drain、M5/M6 wide/MIX/`sync_start` |
| 关联计划 | [Host 提前构图与 AICore 解依赖调度实现计划书](host-build-graph-aicore-scheduler-plan.zh.md) |

## 1. 结论

从原始设计中的 task/wake 共享操作开始消减后，`WAITING -> READY` 和独立 claim 不进入最终
运行期 ABI；`next_waiter` 发布并入 wake-list 注册协议。最终保留 completion、wake-list 注册、
wake-list 关闭和 `completed_count` 四项。ReadyQ 自身另作一组，因为其内部包含 cursor、slot
sequence 或 bitmap word 等多项原子操作。

| # | 共享操作 | 正确性作用 | 推荐结论 |
| ---: | -------- | ---------- | ---------- |
| 1 | `WAITING -> READY` | 可选的重复发布诊断 | 正确 Pull 协议没有竞争，建议从正确性热路径删除 |
| 2 | ReadySet 所有权转移 | 保证 task exactly-once | MPMC 成功 pop 或 bitmap 原子清 bit 即完成领取，不设独立 claim 字段 |
| 3 | completion 发布 | 让其他 core 观察 kernel/DUMMY 完成 | 唯一 writer，使用 release store，不做 CAS |
| 4 | wake-list 注册协议 | 发布 `next_waiter`，并把 waiter 挂到 producer | `next_waiter` 不做 RMW；仅对 head 做 CAS push |
| 5 | `wake_list_head` 关闭 | completion 与 waiter 注册并发时不丢唤醒 | 保留 atomic exchange |
| 6 | `completed_count` | 判断整图完成 | 每核本地累计；完整空闲扫描后批量 `fetch_add` |

真正预期发生多 writer 竞争的是 ReadyQ、wake-list head，以及低频批量更新的 `completed_count`。
`WAITING -> READY` 和独立 task claim 都不属于最终运行期共享操作；task 所有权由 ReadySet 的
领取操作直接转移。completion 以及注册协议中的 `next_waiter` 发布主要解决跨核可见性，
不是多 writer 热点。

## 2. ReadyQ 实现选择

### 2.1 本方案对 ReadyQ 的特殊约束

Host 在启动前已经生成完整图，因此：

- task id 稳定，task 数量和资源类型已知；
- 每个 task 最多从 `WAITING` 进入一次 `READY`；
- 每个 ready task 只被领取一次；
- 单 lane Pull 不会因为目标 core 不可用而重新入队；
- 调度不要求 FIFO，只要求无丢失、exactly-once 和无饥饿。

这些条件比通用 MPMC queue 更强，允许用 ready set 代替循环队列。

### 2.2 Vyukov bounded MPMC 基线

MPMC 为每种资源类型维护一个有界队列：

- producer CAS `enqueue_pos`，再以 release store 发布 slot `sequence`；
- consumer CAS `dequeue_pos`，以 acquire load 读取 `sequence`；
- slot generation 防止环回后的 ABA；
- task id 是普通 payload，不保存设备指针。

优点是算法成熟、支持任意入队顺序和 slot 复用。主要成本是每次 push/pop 都访问全局 cursor
和 per-slot sequence；大量 core 同时 Pull 时，`dequeue_pos` 是集中热点。batch push/pop 可以
摊薄 cursor CAS，但会让一个 core 一次取得多个 task，不适合直接用于“领取后同步执行”的
单-task Pull，除非额外维护本地待执行缓存。

### 2.3 其他免锁实现比较

| 方案 | 原子热点 | 优点 | 主要问题 |
| ---- | -------- | ---- | -------- |
| 单个 Vyukov MPMC | enqueue/dequeue cursor、slot sequence | 成熟、FIFO、支持复用 | cursor 集中竞争，slot 元数据大 |
| Sharded MPMC | 每 shard cursor | 分散竞争，可 local-first | 需要分片路由、跨 shard 扫描和 work stealing |
| 单层 ready bitmap | bitmap word | 1 bit/task，无 cursor/slot ABA | 空闲时可能扫描整个图，尾部代价高 |
| 两级 ready bitmap | L0 task word、L1 summary word | 空间小，快速定位非空 word | summary 清位存在竞态，需要修复协议 |

理论上，两级 ready bitmap 最匹配本方案的一次 ready/一次领取语义。它不维护 FIFO，也不
为永不重新入队的 task 支付 sequence/generation 成本。

### 2.4 两级 ready bitmap 提案

每个资源类型维护：

```text
L0 ready_words[ceil(task_count / 64)]
  bit i = task i 已 READY 且尚未被领取

L1 summary_words[ceil(L0_word_count / 64)]
  bit j = L0 ready_words[j] 可能非零
```

Host 可建立 task id 到 bitmap 位置的稳定映射。各 core 保存私有旋转扫描游标，避免总从最低
task id 开始造成饥饿。必要时对 task id 做稳定条带化映射，避免大量相邻 task 集中竞争同一
L0 word。

Ready 发布顺序：

1. `L0[word].fetch_or(task_bit, release)`；旧值已含该 bit 表示重复发布错误。
2. `L1[summary_word].fetch_or(word_bit, release)`。

Pull 顺序：

1. 从私有游标开始 acquire-load L1，选择一个非空 L0 word。
2. acquire-load L0，选择一个 task bit。
3. CAS 或 `fetch_and` 清除该 bit；观察到旧值含该 bit 的 core 是唯一领取者。
4. winner 取得 task 所有权并执行本 core task；DFX 记录 `START` event。
5. L0 变空时清除 L1 bit，然后重新 acquire-load L0；若非空则重新置 L1 bit。

第 5 步用于覆盖 producer 置 L0 与 consumer 清 L1 的交错。producer 总是按 L0 后 L1 的顺序
发布，因此重新检查能够修复 summary。实现还应保留低频 L0 fallback scan；L1 只能作为
加速 hint，不能成为丢失 task 的唯一故障点。

同一 word 上不同 task bit 的 `fetch_or/fetch_and` 仍会串行，因此 bitmap 不是“无竞争”，而是
把全局 cursor 热点分散到多个 word。收益取决于 ready task 的分布、core 数和 A5 原子代价。

### 2.5 理论推荐和采用条件

理论推荐使用两级 ready bitmap，原因是它直接表达固定 task 集合的 ready/unacquired 状态，
内存约为 1 bit/task，且原子清除 bit 本身就是 exactly-once 领取。Vyukov MPMC 保留为正确性和
性能基线，在 A5 证据形成前不从主方案删除。

bitmap 只有同时满足以下条件才替换 MPMC：

- CPU 模型和 A5 压力均无丢 bit、重复领取或永久 summary 漏报；
- root burst、wake burst、稀疏尾部和热点 word 下均能公平完成；
- atomic 次数、CAS retry、ready-to-start 和总时延整体优于 MPMC，且尾延迟无不可解释退化。

若 bitmap 的 summary 维护或热点 word 抵消收益，则保留 Vyukov MPMC；若只有 cursor 热点，
优先比较 sharded MPMC，再考虑增加 bitmap 复杂度。

## 3. Task 状态是否需要直接 CAS

技术上可以，但 Pull 正确性不需要独立 task state CAS。依赖子系统通过 fanin、wake list 和
ReadySet 判断 task 所在位置；任务所有权由 ReadySet 自身的唯一领取操作转移：

```text
逻辑位置：wake list -> ReadyQ/bitmap -> executing -> complete
MPMC 所有权：queue slot -> 成功 pop 的 core
Bitmap 所有权：ready bit -> 成功清 bit 的 core
```

不维护独立 claim 字段。MPMC 的 slot sequence/cursor 保证一个 queue item 只被 pop 一次，
“每 task 只 push 一次”的调度不变量保证 queue item 与 task 一一对应；bitmap 的原子清 bit
同时保证唯一领取。completion 仍使用独立 release store。

### 3.1 `WAITING -> READY`

initial classify 对 task 静态分片；运行期 waiter 每次只挂在一个 producer 上，因此正确路径
只有一个 core 负责发布 ready。fanin 扫描和 consumer 执行都不需要读取一个独立的 `READY`
状态，因此这里不存在必须仲裁的共享所有权。

建议不做 `WAITING -> READY` CAS：MPMC 依赖单-owner reclassify 保证只 push 一次；bitmap 的
`fetch_or` 返回旧 bit，可直接检测重复发布。READY 只作为 DFX event 记录，不成为调度正确性
状态。这样既减少一次逐 task RMW，也避免维护一套与 wake list/ReadySet 重复的状态。

### 3.2 ReadySet 领取即所有权转移

- MPMC：成功 pop 的 core 直接取得 task 所有权。MPMC 保证 queue slot 唯一消费，调度协议
  保证 task id 只进入一个 slot，因此不再执行 task-local CAS。
- Bitmap：清除 ready bit 的 CAS/`fetch_and` 选出唯一 winner，也不再执行第二次 CAS。

MPMC 方案必须把“task 只 push 一次”提升为正式不变量：initial classify 单 owner；waiter 同时
只挂一个 producer；wake-list close 将 waiter 所有权转移给唯一 relayer；relayer 只能重新挂链
或 push，不能复制；单 lane 执行不 requeue/retry。模型测试和 event ring负责验证该不变量。

## 4. 候选共享操作的必要性和消减方式

### 4.1 `WAITING -> READY`

- **在做什么**：这只是一个逻辑描述——consumer 已经从 wake-list 等待阶段进入 ReadyQ/bitmap。
  真正承载位置的是“挂在哪个 wake list”以及“是否已发布到 ready set”，不是独立 state 字段。
- **谁会使用**：如果保留，reclassify core 写 `READY`，DFX 读取它做展示；Pull core 不需要
  读取它，成功取得 ReadySet 元素后即可执行。依赖扫描不靠它判断 fanin，wake relay 不靠它
  找 waiter，ReadyQ 也不靠它保存 task，因此它没有不可替代的 correctness consumer。
- **竞争在哪里**：正确协议中没有竞争。initial classify 为 task 指定唯一 shard；运行期 waiter
  同时只挂在一个 producer 上，摘取后由一个 relayer 独占，直到重新挂链或发布 ready。只有协议
  已经出错、同一 waiter 被重复持有时，才会有两个 core 同时写 `READY`。
- **为什么不需要 CAS**：用一个额外 CAS 检测重复 ready 可以更早报错，但不是防止重复执行的
  唯一位置。最终方案选择用结构性单-owner 不变量避免 MPMC 重复 queue item；bitmap 还可以
  通过 `fetch_or` 的旧 bit 检测重复发布，并通过原子清 bit 保证唯一领取。
- **如何消减**：从运行期正确性状态中删除 `WAITING -> READY` CAS。保持单-owner classify/
  reclassify 不变量；MPMC 用模型测试和 per-core event ring 检查重复发布，bitmap 在 set bit 时
  直接报告重复发布；READY 不再修改共享 task state。
- **为什么有效**：它直接删除每 task 一次 RMW。exactly-once 由“只发布一次 + ReadySet 唯一
  领取”共同保证，不再维护 wake list、ReadySet 和 task state 三套重复事实。

### 4.2 ReadySet 所有权转移

- **在做什么**：空闲 core 从共享 ReadySet 取得 task id 后，直接成为该 task 的唯一执行者，
  随后物化本地 execution slot并调用 kernel。这里仍有“领取”这个逻辑动作，但不再有独立
  所有权字段或第二次原子操作。
- **为什么需要唯一所有权**：kernel、completion 和 wake relay 都不能执行两次。exactly-once
  必须有明确线性化点，但线性化点可以复用 ReadySet 已经存在的唯一消费操作。
- **竞争在哪里**：MPMC 下多个 core 竞争 `dequeue_pos` 和 slot sequence，成功 pop 的 core 是
  winner；同一 queue item 不会被第二个 core取得。Bitmap 下多个 core 竞争同一 L0 word，只有
  观察到旧值含目标 bit并成功清除的 core 是 winner。不存在额外 task-local 竞争地址。
- **为什么不需要独立 CAS**：MPMC 的唯一 pop 结合“每 task 只 push 一次”，已经推出 task 只被
  一个 core 取得；bitmap 清 bit 直接完成相同证明。再增加 claim CAS 只是重复验证同一所有权，
  每 task 多一次 RMW 和 cache-line 转移。
- **如何消减**：删除 claim 字段及其 CAS。MPMC 把成功 pop 定义为 claim 线性化点；bitmap 把
  成功清 bit 定义为线性化点。禁止单 lane requeue/retry，并用模型测试穷举 waiter 所有权转移，
  用 event ring 离线检查每个 task 恰好出现一次 READY、START 和 COMPLETE。
- **为什么有效**：所有权检查被合并进 ReadySet 本来就必须执行的原子操作，减少一次逐 task
  RMW。代价是重复 push 不再由生产运行时 claim CAS 兜底，因此单-owner 发布不变量必须作为
  合入门槛，而不是仅靠约定。

### 4.3 Completion 发布

- **在做什么**：业务 kernel 或 DUMMY 路径结束后，执行 core 先发布输出数据，再把 producer
  标记为完成。其他 core 在 initial classify 或 wake reclassify 中读取该标志，决定 consumer
  是否仍需等待。
- **为什么需要**：完成标志不仅表示“kernel 已返回”，还必须表示“producer output 已对其他
  core 可见”。若先写 completion、后 clean 输出，consumer 可能观察到完成并立即读取旧 tensor，
  产生不报错的错误结果。
- **竞争在哪里**：ReadySet exactly-once 领取保证只有一个 completion writer，因此没有 writer-writer
  竞争；竞争表现为一个 core 发布时，多个 classifier/relayer core 并发 acquire-load 同一个
  completion word。它是发布/观察同步点，而不是 CAS winner 竞争。
- **为什么同步有必要**：A5 普通 GM 不提供可假定的跨核自动一致性。需要 `after_task` 对输出
  做 DCCI clean 加 barrier，再以 release 语义发布 completion；reader acquire 观察完成后，
  `before_task` 负责 invalidate 对应输入。只有这条 happens-before 链能把数据和控制状态关联。
- **如何消减**：completion 由唯一 writer 用 release store，不使用 exchange/CAS；fanin 扫描
  遇到第一个未完成 producer 即停止，减少 completion load；每 task 独立 control cache line，
  避免相邻 task 的 DCCI 或原子访问互相覆盖。
- **为什么有效**：release store 保留所需的单向发布顺序，但避免 RMW 带来的独占 cache-line
  争抢。减少 fanin load 和 cache-line 隔离进一步降低 reader 流量，而不改变完成语义。

### 4.4 Wake-list 注册协议（合并原 `next_waiter` 发布）

- **在做什么**：consumer 扫描 fanin 时选择一个未完成 producer，先把自己的 `next_waiter`
  指向当前旧 head，再用 CAS 把自己安装为新 head。producer 完成后只需摘取这条 intrusive
  list，沿 `next_waiter` 重新分类 consumer，不需要轮询全图或保存完整 fanout 表。
- **为什么需要**：一个 `WAITING` task 必须处于可被推进的位置：要么已经在 ReadyQ，要么挂在
  某个尚未完成 producer 上。没有 wake registration，consumer 在 producer 完成后无人重新
  检查，会永久停留在 `WAITING`。
- **竞争在哪里**：宽 fanout 或许多 consumer 共享同一早期 producer 时，不同 classifier core
  会同时 CAS 同一个 `wake_list_head`。CAS winner 修改 head，其余 core 的 expected 失效并重试，
  这是该协议唯一的多 writer 竞争地址。每个 waiter 的 `next_waiter` 在注册期间只有当前 owner
  写，不存在 writer-writer 竞争；它与完成 core 的读取之间只有跨核可见性要求。
- **为什么 CAS 和发布顺序有必要**：普通 head store 会发生 lost update：两个 core 都读取旧
  head，分别写入自己的 node，后写者覆盖先写者，其中一条 waiter 链永久丢失。CAS push 把
  “观察旧 head、把 next 指向旧 head、安装新 head”线性化。同时必须保证 `next_waiter` 先于
  head 对完成 core 可见，否则 head CAS 虽然正确，producer 仍可能只能遍历到链首。
- **完整操作顺序**：读取旧 head；写 `waiter.next_waiter = old_head`；用 release atomic store
  或明确的 DCCI clean + barrier 发布 next；CAS `wake_list_head`。CAS 失败后使用返回的新 head
  重写 `next_waiter` 并重试。`next_waiter` 保存稳定 task id，不保存设备指针，也不单独做 CAS。
- **如何消减**：第一层采用稳定的旋转 fanin 起点，例如由 consumer task id 派生扫描起点，
  使不同 consumer 不总选择同一个“第一个 fanin”。第二层可让一个 classifier shard 先把同一
  producer 的多个 waiter 串成本地链，再用一次 CAS 批量挂接。只有 DFX 证明单 head 仍为热点
  时，才增加 per-producer 分片 heads；producer 完成时遍历所有 shard。A5 seam 测试需要比较
  `next_waiter` 的 atomic store/load 与独立 cache line DCCI 方案，但两者都不增加竞争性 RMW。
- **为什么有效**：旋转起点把注册流量分散到多个尚未完成 producer；批量挂接把 N 次 head CAS
  降为每 shard 一次；分片 head 则把同一原子地址拆成多个独立 cache line。三者依次增加复杂度，
  应按 CAS retry 和等待周期逐级启用，避免无测量地扩大 control 内存。把 `next_waiter` 合入
  注册协议还能避免把单 writer 字段误实现为第二个 CAS；它本身不会减少 head CAS 次数。

### 4.5 Wake-list 关闭

- **在做什么**：producer 发布 completion 后，把 `wake_list_head` 原子替换为 CLOSED sentinel，
  并取得替换前的完整 waiter 链。之后任何注册者看到 CLOSED 都不能再挂链，只能重新扫描 fanin。
- **为什么需要**：completion 与注册可能交错。若 producer 先读取 head、waiter 随后挂入，而
  producer 再也不检查 list，新 waiter 会挂在一个已经完成且不会再次触发的 producer 上，形成
  永久丢唤醒。
- **竞争在哪里**：只有 producer completion core 执行一次 close，但它与所有正在 CAS push 的
  classifier core 竞争同一个 head。关键不是多个 closer，而是 close exchange 和 register CAS
  必须形成全序：要么 waiter 在线性化点之前进入摘取链，要么在之后观察 CLOSED 并重分类。
- **为什么 exchange 有必要**：分开的“写 CLOSED”和“读取旧 head”之间存在窗口，无法同时
  保证禁止新注册并保留旧链。atomic exchange 在一个操作中完成封口和所有权转移。
- **如何消减**：initial classify barrier 已保证普通 kernel 在全图初始注册完成前不会执行，
  因而启动阶段没有 close/register 竞争；运行期每 producer 只 close 一次。head 可使用带 CLOSED
  bit 的 tagged atomic，把关闭状态和指针放在同一字中，避免额外 completion 检查。若 Host
  构建静态 fanout，producer 可直接遍历不可变 consumer 列表，动态注册和关闭都能删除。
- **为什么有效**：barrier 消除了最大规模的启动竞态；单次 exchange 限制每 producer 的写入
  次数；tagged head 减少控制字段访问。静态 fanout 能彻底消除 head 竞争，但会增加 O(edge)
  内存和完成时 fanout 扫描成本，因此不是默认方案。

### 4.6 `completed_count`

- **在做什么**：每个 core 私有维护 `local_completed_delta`。kernel 或 DUMMY task 完成完整的
  completion 发布和 wake relay 后，只增加本地 delta；当本核完整扫描所有兼容 ReadyQ 均无任务
  可领取时，才把 delta 批量 `fetch_add` 到全局 `completed_count`。AICPU 控制面在
  `completed_count == expected_task_count` 时判断正常完成。
- **为什么需要**：ReadyQ 为空不能表示完成——所有剩余 task 可能仍挂在 wake list，或某个 core
  正在执行最后一个 task。逐 task 扫描所有 completion flag 开销为 O(task 数)，全局 count 提供
  O(1) 的完成判定。
- **竞争在哪里**：多个 core 可能同时从 busy 转为空闲，并发 `fetch_add` 同一个全局 counter。
  竞争没有完全消失，但从“每 task 一次”降为“每个 core 每段 busy period 一次”；持续执行期间
  只修改本地 delta，不触碰共享 cache line。
- **为什么原子有必要**：普通 read-modify-write 会丢增量：两个 core 同时读到 N，各自写 N+1，
  最终少计一次并导致永不退出，因此批量提交仍需 atomic `fetch_add`。该原子只聚合计数，
  不负责发布 kernel 数据，可以使用 relaxed 语义；数据可见性由 completion 协议单独承担。
- **如何消减**：`local_completed_delta` 放在寄存器或每核私有存储中；delta 为零时不执行 RMW。
  core 完整空闲扫描后提交非零 delta，并且只在 `fetch_add` 完成后清零。提交后若又出现 ready
  task，正常执行并累计下一批；不尝试先判断“所有 core 全局空闲”，避免引入另一套 idle 同步。
  wide/MIX task 只由唯一逻辑完成者累计一次，错误路径仍走 first-error/drain。
- **为什么有效**：全局 counter 和 O(1) 完成判断保持不变，但原子次数从 O(task 数) 降到
  O(busy period 数)。最后一批任务完成后，执行 core 必然回到调度循环、完成一次空闲扫描并
  提交剩余 delta，因此不会漏计；新任务在提交后出现也只会进入下一批，不会提前完成。

## 5. DFX 和验证

每 core 记录局部 counters，结束时汇总，避免 DFX 自身制造共享热点：

- MPMC enqueue/dequeue CAS attempt、fail、sequence wait cycle；
- bitmap L0/L1 atomic、bit-clear retry、fallback scan、hot-word max retry；
- MPMC pop winner、bitmap duplicate-set/clear-bit winner；
- wake register CAS retry、close-race reclassify；
- completion publish、completed-batch flush count/size/atomic cycle。

CPU 模型必须覆盖：

- READY 重复发布、重复 queue entry 和 ReadySet 唯一领取；
- bitmap set/acquire、L1 清位与 producer set 的全部交错；
- wake 注册协议中 `next_waiter` 发布、head CAS 与 exchange close 的全部交错；
- 乱序完成、busy/idle 反复切换以及 flush 后立即出现新任务时，批量计数无漏计、重复或提前退出。

A5 对比 workload 至少包括 root burst、单 producer 多 waiter、高 fanin、随机 DAG、稀疏尾部、
热点 bitmap word 和非 64 对齐任务数。报告 MPMC、sharded MPMC 和 bitmap 的 atomic 次数、
CAS retry、ready-to-start、调度周期、总时延和各 core 任务分布。
