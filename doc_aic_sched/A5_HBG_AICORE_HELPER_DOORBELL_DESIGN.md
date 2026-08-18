# A5 HBG 无 AICPU Helper-Doorbell 调度方案

## 1. 结论与硬件前提

本文描述一个假设性 HBG 调度方案：AICPU 只负责初始化、启动和退出，运行时的
claim、依赖推进、payload 构造、派发和完成处理全部由 AICore 执行。

主方案依赖一种当前硬件尚未验证存在的跨核 doorbell/event 寄存器。已验证的
`DATA_MAIN_BASE` 和 AIC_CTRL MMIO 窗口不满足要求：AICore 不能写自己的或其他核的
`DATA_MAIN_BASE`，AICore LSU 访问该 MMIO 窗口会挂核。因此本文不能直接用现有 DMB/COND
实现，相关硬件结论见 `docs/hardware/mmio-performance.md`。

主方案要求硬件同时满足以下契约：

1. AIC 和 AIV 都能向指定物理 AICore 写 event，目标核能读取或消费自己的 event。
2. dispatch 和 completion 两个方向都可用，不需要 AICPU 中继。
3. event 是可靠锁存或有队列/credit；不能因连续写入或多生产者写入静默丢事件。
4. event token 至少能携带 `generation + slot_id + phase`，并有明确的原子写入语义。
5. GM publish 后再写 event 具有 release 顺序；目标核消费 event 后能 acquire 对应 GM payload。
6. event 不抢占正在运行的 kernel。Executor 只在 kernel 边界消费 event。
7. 每个目标核有独立通道，或硬件能保证不同目标、dispatch 和 completion 之间不会错误合并。

建议的逻辑 token 为：

```text
{generation, target_slot, phase=STAGED|READY|COMPLETE, task_token}
```

如果 token 宽度不足，GM 保存完整 task token，寄存器只保存 generation、slot 和 phase；但这会让
目标核多一次 GM slot acquire。可靠、足宽 token 是获得 TMR 级切换开销的必要条件。

## 2. 总体调度方案

### 2.1 角色

- AICPU：初始化 graph、ticket stream、物理核映射和 event 通道；启动 persistent executor；运行时不参与。
- AIC/AIV Executor：执行已经 READY 的计算任务。
- 空闲 AIV helper：批量 claim、依赖 route、目标核预留和 payload 构造。
- AIV resolver：消费 completion、推进 fanin、把 STAGED 任务释放为 READY。

helper 和 resolver 是空闲 AIV 的动态角色，不永久绑定固定核。系统保留一个可迁移的 helper lease，
避免所有 AIV 同时进入计算后控制面失去前进能力。

### 2.2 两级 claim

方案继承 TMR 的资源约束，但把承担者从 AICPU 换成 AIV helper：

1. 逻辑 claim：helper 批量领取 ticket，解析 shape，注册 fanin；此时不绑定目标核。
2. 物理 claim：任务已经 READY，或满足 TMR 风格的 early-eligibility 后，才预留目标核 pending slot。

策略约束：

- 普通 READY 永远优先于 early-dispatch。
- early-eligibility 表示所有直接 producer 已完成物理派发，而不是依赖已经完成。
- 每个目标核最多一个物理 early-pending slot；逻辑 BLOCKED task 可以继续保持未绑定。
- MIX、SPMD 和 sync-start 必须以 cohort 为单位原子预留，不能逐核留下半完成 placement。

### 2.3 派发和完成

helper 完成物理 claim 后构造完整 `PTO2DispatchPayload`，包括 callable、args 和 local context：

```text
helper:
  reserve target slot
  -> materialize full payload
  -> publish payload to GM
  -> memory barrier
  -> write target STAGED event

resolver on final fanin:
  publish READY state
  -> memory barrier
  -> write target READY event

target Executor at kernel boundary:
  consume event
  -> prefer normal READY
  -> validate generation/slot
  -> acquire prepared payload
  -> execute kernel
  -> write resolver COMPLETE event

resolver:
  consume COMPLETE
  -> publish task DONE in GM
  -> release fanout
  -> recycle target slot
```

目标核看到 `STAGED` 但尚未 `READY` 时不能进入 gated spin。它应继续执行其他 READY task；AIV
目标也可以继续做 helper/resolver 工作。这样避免 TMR early-dispatch 中 blocked task 占住执行核。

如果目标核第一次观察该 slot 时已经是 READY，可以直接 acquire payload 并执行，不要求先观察 STAGED。
generation 用于丢弃迟到或合并后的旧 event，GM 状态仅用于异常恢复。

## 3. 目标 Executor 相比 TMR 的热路径

### 3.1 可靠双向 doorbell 主方案

TMR 目标核路径为：

```text
read DMB -> observe payload -> COND ACK -> kernel -> COND FIN
```

新方案目标核路径为：

```text
read READY event -> observe payload -> kernel -> write helper COMPLETE event
```

在可靠、足宽、带顺序保证的 doorbell 假设下，相比 TMR：

| 项目 | 新增量 |
| ---- | ------ |
| 目标核 GM 访存 | 0；payload acquire 与 TMR 等价 |
| 目标核 GM 原子 | 0 |
| READY queue/slot CAS | 0；event 直接携带目标 slot |
| completion GM publish | 0；由 COMPLETE event 代替 TMR 的 COND FIN |
| 本地指令 | phase/generation 校验和 ready-first 分支 |

需要从目标热路径删除当前 completion-inbox 的以下操作：

```text
GM store task.state = DONE
GM atomicExch(inbox.head, task_id)
GM store task.completion_next
cache-line writeback + dsb
```

这些操作当前位于
`src/a5/runtime/host_build_graph/runtime/aicore_ticket_scheduler_v1.h::aicore_enqueue_completion_v1`。
尤其是 MPSC `atomicExch(inbox.head)` 会让同时完成的 Executor 串行化，不适合 TMR 级 ready-to-ready
目标。

### 3.2 剩余跨核竞争

主方案仍有以下系统级竞争，但不增加目标 Executor 的 GM 原子：

- helper 写 payload、Executor 读 payload产生 AIV 到 AIC 的 cache/NoC ownership handoff。
- helper 的 ticket、fanin 和 target reservation 原子会间接占用 GM、NoC 和 atomic 单元。
- 多个 helper 写同一目标 event 时可能在寄存器 fabric 上竞争。

避免方式是每个物理 pending slot 只有一个 helper transaction owner，payload 和 slot 独占 cache line，
helper 使用有界批次。payload handoff 在 TMR 中也存在，但生产者是 AICPU；新方案把流量移入 AICore
共享的 L2/GM/NoC 域，因此需要硬件实测其对 kernel 的干扰。

## 4. GM 退化方案与 MTE 次选优化

### 4.1 纯 GM SPSC 退化方案

如果不存在可靠双向 doorbell，不能退回当前共享 MPSC completion inbox，否则目标核重新承担
`atomicExch` 和 cache-line 竞争。建议为每个 Executor 分配两条独占 SPSC 通道：

```text
helper   -> per-Executor dispatch ring   -> Executor
Executor -> per-Executor completion ring -> resolver
```

每条 ring 使用单调 sequence/credit；producer 和 consumer index 分离到不同 cache line。由于每条通道
只有一个 producer 和一个 consumer，正常路径不需要 CAS、`fetch_add` 或 `atomicExch`。

目标 Executor 的退化热路径为：

```text
poll dispatch sequence
-> acquire payload
-> kernel
-> store completion token to private ring slot
-> publish slot/sequence
```

相比 TMR，目标核新增一次无竞争 completion GM 发布，包括 entry/sequence store、cache publish 和顺序
屏障；跨核竞争退化为 Executor 与 resolver 之间的单生产者单消费者 cache-line handoff。其延迟会高于
寄存器 FIN，但不会出现共享 inbox 的多生产者原子串行化。

dispatch ring 的 READY sequence 应与 payload 分离：Executor 轮询小的 sequence cache line，观察 READY
后才读取完整 payload，避免反复 invalidate 512-byte payload。GM 轮询和通知延迟需要在 A5 实测；现有
AICore 到 AICPU 的 GM+dcci 数据约为 1 us，不能直接当作 AICore 到 AICore 的结论。

### 4.2 AICore Executor 上的 MTE pipeline

MTE 次选方案建立在 SPSC 上，不能修复 MPSC inbox。MTE 不适合小 flag 轮询，也不能隐藏
`atomicExch`/CAS 的串行化；它只用于 READY 已确认后的 bulk payload 读取和 completion record 写回。

在 kernel N 和 N+1 之间，如果对应 core type 同时提供可并发的 GM load/store MTE 通道，可在不同
queue/event 上并行发起：

```text
MTE store: local completion[N] -> GM completion ring
MTE load : GM payload[N+1]     -> local staging buffer
scalar   : validate token, select callable, update local indices
```

然后分别等待必要的 MTE event：

```text
completion store done -> publish completion sequence/notification
payload load done     -> launch kernel N+1
```

理想情况下，边界成本从：

```text
T_completion_store + T_payload_load
```

变为：

```text
max(T_completion_store, T_payload_load) + T_sync
```

该优化需要额外验证以下 ABI 和硬件条件：

1. persistent executor 能保留独占的本地双缓冲、MTE queue 和 event ID。AIV 可考虑 UB；AIC 的可访问
   staging memory、数据搬运方向和 scalar 读取能力需要按实际 ISA 单独确认，不能直接套用 AIV 路径。
2. 被调用 kernel 不会覆盖 executor 尚未完成的本地 staging/MTE 状态；否则必须在调用前等待，无法跨
   kernel 隐藏。
3. MTE 写 GM 完成后，sequence publish 和 helper 观察之间具有正确的 release/acquire 顺序。
4. `PTO2DispatchPayload::args` 的 kernel ABI 当前是 GM 指针；不能直接把 UB staging 地址传给 kernel。
   可用 MTE 预取控制字段帮助解析，但最终 args 存放和调用 ABI 需要单独设计或保持 GM payload。
5. completion 通知不能早于 MTE 写回完成；若硬件不能把 MTE-complete 自动链到 event，Executor 仍需等待
   MTE event 后再发布 sequence/doorbell。
6. completion token 很小，单次 MTE setup 可能比 scalar store 更贵；应按 cache line 批量写回或实测确认
   单条记录也有收益。

因此 MTE 能重叠两个 GM 数据移动，却不能保证完全消除 ready-to-ready 延迟。READY 来得太晚、只有一个
可执行任务、kernel 使用全部 MTE/UB 资源，或 completion 处于依赖关键路径时，重叠收益都会下降。

## 5. 推荐顺序与验收指标

实现优先级：

1. 可靠双向 targetable doorbell：目标核无新增 GM/原子，最接近 TMR。
2. 每 Executor SPSC dispatch/completion ring：无 GM 原子，但保留一次 completion publish。
3. 在 SPSC 上增加 MTE 双缓冲：把 completion store 与下一 payload load 并行化。
4. 不采用共享 MPSC completion inbox 作为低延迟目标路径；它只适合作为兼容或异常恢复通道。

验收必须同时测量：

- profiling 关闭时，同一 Executor 的 ready-to-ready P50/P95/P99；
- 1、N 个 Executor 同时完成时的 completion 尾延迟；
- helper 开启前后，繁忙 AIC kernel 的执行时长变化，用于量化 GM/NoC 干扰；
- READY、STAGED、completion burst 和 generation wrap/ABA 压力下是否丢事件；
- AIV 全忙时 helper lease 的控制面前进性；
- MTE 开启前后的边界成本是否从两段求和下降为接近两段最大值。

在没有满足第 1 节硬件契约的实测证据前，本方案只能作为架构设计；当前 A5 已验证的寄存器接口不能直接
实现该协议。
