# Scheduler 处理流程

> Scheduler 的线程模型、主循环、完成处理、任务调度、AICore 交互协议与水位推进机制。

## 1. 线程模型

### 1.1 四线程分配

```mermaid
flowchart TD
    subgraph AICPU["AICPU (4 线程)"]
        direction TB
        T0["Thread 0: Scheduler<br/>管理 8 AIC + 16 AIV"]
        T1["Thread 1: Scheduler<br/>管理 8 AIC + 16 AIV"]
        T2["Thread 2: Scheduler<br/>管理 8 AIC + 16 AIV"]
        T3["Thread 3: Orchestrator<br/>构建任务图"]
    end

    subgraph AICore["AICore 计算核心 (24 blocks)"]
        AIC["AIC ×24 (CUBE)"]
        AIV["AIV ×48 (VECTOR)"]
    end

    T0 -->|"寄存器派发"| AICore
    T1 -->|"寄存器派发"| AICore
    T2 -->|"寄存器派发"| AICore
    T3 -->|"共享内存"| T0
    T3 -->|"共享内存"| T1
    T3 -->|"共享内存"| T2
```

每个 block = 1 AIC + 2 AIV。24 blocks 的 AIC/AIV 核心均分给 3 个 Scheduler 线程。

### 1.2 启动同步

```mermaid
sequenceDiagram
    participant T3 as Thread 3 (Orchestrator)
    participant T0 as Threads 0-2 (Schedulers)

    T3->>T3: 创建 SM handle + Runtime
    T3->>T0: 设置 runtime_init_ready_
    T0->>T0: 等待 runtime_init_ready_
    T0->>T0: 竞争 pto2_init_done_<br/>(胜者 memset 初始化)
    T0->>T3: 设置 pto2_init_complete_
    T3->>T3: 等待 pto2_init_complete_
    T3->>T3: 配置 orchestrator-scheduler 指针
    par
        T3->>T3: 调用编排函数
    and
        T0->>T0: 进入调度主循环
    end
```

---

## 2. Scheduler 主循环

```mermaid
flowchart TD
    START["进入主循环"] --> P1

    P1["Phase 1: 完成处理<br/>轮询每个核心的 COND 寄存器<br/>处理已完成的 subtask"]
    P1 --> P2

    P2["Phase 2: 任务调度<br/>对每个空闲核心<br/>从 ReadyQueue pop → 派发"]
    P2 --> ADV

    ADV["尝试推进水位<br/>(try-lock per-ring)"]
    ADV --> TERM

    TERM{"终止条件?<br/>all_completed &&<br/>orchestrator_done"}
    TERM -->|"否"| P1
    TERM -->|"是"| EXIT["退出主循环<br/>发送 EXIT_SIGNAL 到 AICore"]
```

---

## 3. 完成处理流程 (Phase 1)

### 3.1 两阶段完成

```mermaid
sequenceDiagram
    participant AICore as AICore 工作核心
    participant Reg as COND 寄存器
    participant Sched as Scheduler 线程
    participant SlotState as TaskSlotState
    participant Consumer as 消费者 Task

    AICore->>Reg: 写入 FIN (bit31=1 | dispatch_seq)
    Sched->>Reg: 轮询 COND 检测 TASK_FIN_STATE

    Note over Sched: Stage 1: on_subtask_complete
    Sched->>SlotState: fetch_or(subtask_done_mask, done_bit)

    alt subtask_done_mask == active_mask
        Note over Sched: Stage 2: on_mixed_task_complete
        Sched->>SlotState: 获取 fanout_lock
        Sched->>SlotState: task_state = COMPLETED
        Sched->>SlotState: 读取 fanout_head
        Sched->>SlotState: 释放 fanout_lock

        loop 遍历 fanout 链表
            Sched->>Consumer: fanin_refcount++
            alt fanin_refcount == fanin_count
                Sched->>Sched: 推入 ReadyQueue<br/>(或 local buffer)
            end
        end

        Note over Sched: Cold path: on_task_release
        Sched->>Sched: 释放 producer 引用<br/>(fanout_refcount++)
        Sched->>Sched: check_and_handle_consumed<br/>若 CONSUMED → 触发水位推进
    else 还有未完成的 subtask
        Note over Sched: 等待其他 subtask
    end
```

### 3.2 完成处理关键步骤

```mermaid
flowchart TD
    FIN["检测 COND = FIN"] --> SUB["on_subtask_complete<br/>设置 done bit"]
    SUB --> ALL{"所有 subtask<br/>都完成了?"}
    ALL -->|"否"| DONE["继续轮询"]
    ALL -->|"是"| LOCK["获取 fanout_lock"]
    LOCK --> MARK["task_state = COMPLETED<br/>读取 fanout_head"]
    MARK --> UNLOCK["释放 fanout_lock"]
    UNLOCK --> FANOUT["遍历 fanout 链表"]
    FANOUT --> NOTIFY["对每个消费者:<br/>fanin_refcount++"]
    NOTIFY --> READY{"消费者<br/>依赖全部满足?"}
    READY -->|"是"| PUSH["推入 ReadyQueue"]
    READY -->|"否"| CONT["继续遍历"]
    PUSH --> CONT
    CONT --> RELEASE["on_task_release:<br/>释放 producer 引用"]
    RELEASE --> SELF{"自身<br/>CONSUMED?"}
    SELF -->|"是"| ADVANCE["尝试推进水位"]
    SELF -->|"否"| DONE
```

---

## 4. 任务调度流程 (Phase 2)

### 4.1 调度流程

```mermaid
flowchart TD
    IDLE["检测空闲核心"] --> SHAPE["确定资源形状<br/>(AIC/AIV 组合)"]
    SHAPE --> LOCAL{"本地缓冲区<br/>有匹配任务?"}
    LOCAL -->|"是"| POP_L["从 local buffer pop"]
    LOCAL -->|"否"| POP_G["从全局 ReadyQueue pop"]
    POP_L --> BUILD
    POP_G --> GOT{"获取到任务?"}
    GOT -->|"否"| SKIP["跳过该核心"]
    GOT -->|"是"| BUILD

    BUILD["构建 DispatchPayload<br/>mixed_task_id, kernel_id,<br/>function_bin_addr, args[]"]
    BUILD --> WRITE["写 payload 指针<br/>到 Handshake.task"]
    WRITE --> REG["写 dispatch_seq<br/>到 DATA_MAIN_BASE 寄存器"]
    REG --> NEXT["处理下一个空闲核心"]
```

### 4.2 Ready Queue 设计

5 种 **ResourceShape** 队列，基于 Lock-free Bounded MPMC (Vyukov) 设计：

| Shape | active_mask 含义 | 说明 |
|-------|-----------------|------|
| `AIC_ONLY` | 仅 AIC | 矩阵运算 (matmul) |
| `AIV_X1` | 仅 AIV0 或 AIV1 | 单 AIV 向量运算 |
| `AIV_X2` | AIV0 + AIV1 | 双 AIV 向量运算 |
| `AIC_AIV_X1` | AIC + 1个 AIV | AIC + 单 AIV 混合 |
| `AIC_AIV_X2` | AIC + AIV0 + AIV1 | 全 cluster 占用 |

**Vyukov MPMC Queue 特性**：
- `enqueue_pos` 和 `dequeue_pos` 在不同 cache line（避免 false sharing）
- 每个 slot 有 sequence counter（防止 ABA 问题）
- 空队列 pop 只需一次 atomic load（快速路径）

### 4.3 Local-first 优化

每个 Scheduler 线程维护 2 个**线程本地缓冲区**（按 CoreType 分类：AIC=0, AIV=1）：

```mermaid
flowchart LR
    COMPLETE["任务完成<br/>消费者变 READY"] --> ROUTE{"active_mask<br/>包含 AIC?"}
    ROUTE -->|"是"| BUF0["local_bufs[0] (AIC)"]
    ROUTE -->|"否"| BUF1["local_bufs[1] (AIV)"]

    BUF0 --> DISPATCH["本线程优先调度<br/>(低延迟)"]
    BUF1 --> DISPATCH

    BUF0 -->|"溢出"| GLOBAL["全局 ReadyQueue"]
    BUF1 -->|"溢出"| GLOBAL
```

优势：当生产者和消费者在同一线程管理的核心上时，避免全局队列的 CAS 竞争。

---

## 5. AICore 交互协议

### 5.1 寄存器交互

```mermaid
sequenceDiagram
    participant Sched as Scheduler (AICPU)
    participant REG as 硬件寄存器
    participant Core as AICore 工作核心

    Note over Sched, Core: === 派发阶段 ===
    Sched->>REG: 写 DATA_MAIN_BASE = dispatch_seq
    Core->>REG: 轮询 DATA_MAIN_BASE<br/>!= IDLE (0x7FFFFFFD)
    Core->>Core: 读取 Handshake.task<br/>获取 DispatchPayload
    Core->>REG: 写 COND = ACK<br/>(bit31=0 | dispatch_seq)

    Note over Sched, Core: === 执行阶段 ===
    Core->>Core: 通过 func_id_to_addr<br/>查找并执行内核函数

    Note over Sched, Core: === 完成阶段 ===
    Core->>REG: 写 COND = FIN<br/>(bit31=1 | dispatch_seq)
    Sched->>REG: 轮询 COND<br/>检测 TASK_FIN_STATE
```

**寄存器定义** (32-bit, per-core)：

| 寄存器 | 方向 | 编码 |
|--------|------|------|
| `DATA_MAIN_BASE` | AICPU→AICore | dispatch_seq (单调递增)；`0xFFFFFFFE` = EXIT |
| `COND` | AICore→AICPU | bit31=0: ACK, bit31=1: FIN；低31位 = dispatch_seq |

### 5.2 Multi-Ring dispatch_seq

32 位寄存器无法容纳 64 位 `mixed_task_id`。如果直接截断，不同 ring 的 `local_id=0` 会产生**碰撞**。

**解决方案**：每个核心维护单调递增的 `s_dispatch_seq[core_id]`，替代 `mixed_task_id` 写入寄存器：

```
Ring 0, local_id=0 → dispatch_seq = 1  (唯一)
Ring 1, local_id=0 → dispatch_seq = 2  (唯一)
Ring 0, local_id=1 → dispatch_seq = 3  (唯一)
```

AICore 通过 `last_reg_val` 对比检测新任务到来，保证不会误判。

### 5.3 DispatchPayload 结构

| 字段 | 说明 |
|------|------|
| `mixed_task_id` | 64-bit 混合任务标识（用于完成聚合） |
| `subslot` | 子任务槽位 (AIC / AIV0 / AIV1) |
| `kernel_id` | 内核函数 ID |
| `core_type` | AIC 或 AIV |
| `function_bin_addr` | GM 中的内核二进制地址 |
| `num_args` | 参数数量 |
| `args[]` | Tensor 地址和 scalar 值 |

---

## 6. 水位推进 (Watermark Advancement)

### 6.1 推进流程

```mermaid
flowchart TD
    TRIGGER["任务变为 CONSUMED"] --> TRY{"try-lock<br/>ring advance_lock"}
    TRY -->|"获取失败"| SKIP["跳过<br/>(其他线程正在推进)"]
    TRY -->|"获取成功"| SCAN

    SCAN["从 last_task_alive 开始扫描"]
    SCAN --> CHECK{"task_state[slot]<br/>== CONSUMED?"}
    CHECK -->|"是"| ADV["last_task_alive++"]
    ADV --> HEAP["更新 heap_tail<br/>= packed_buffer_end"]
    HEAP --> CHECK
    CHECK -->|"否"| SYNC["sync_to_sm()<br/>写入共享内存"]
    SYNC --> UNLOCK["释放 advance_lock"]
```

### 6.2 Per-Ring 独立推进

```mermaid
flowchart LR
    subgraph Ring0["Ring 0"]
        LA0["last_task_alive = 15"]
        HT0["heap_tail = 0x1000"]
    end
    subgraph Ring1["Ring 1"]
        LA1["last_task_alive = 42"]
        HT1["heap_tail = 0x800"]
    end
    subgraph Ring2["Ring 2"]
        LA2["last_task_alive = 3"]
        HT2["heap_tail = 0x200"]
    end

    Ring0 -.->|"独立推进"| Ring0
    Ring1 -.->|"独立推进"| Ring1
    Ring2 -.->|"独立推进"| Ring2
```

每个 ring 有独立的 `advance_lock` (try-lock)：
- **多线程安全**：多个 Scheduler 线程可能同时尝试推进同一 ring，try-lock 保证只有一个线程执行扫描
- **无阻塞**：获取锁失败的线程直接跳过，不会被阻塞——获胜的线程会扫描到当前已 CONSUMED 的所有任务

### 6.3 水位推进对流控的影响

```mermaid
flowchart LR
    subgraph Scheduler["Scheduler 侧"]
        CONSUMED["任务 CONSUMED"] --> ADV_LA["推进 last_task_alive"]
        ADV_LA --> ADV_HT["推进 heap_tail"]
    end

    subgraph SharedMem["共享内存"]
        SM_LA["last_task_alive"]
        SM_HT["heap_tail"]
    end

    subgraph Orchestrator["Orchestrator 侧"]
        TR_BP["TaskRing 背压解除<br/>可分配新槽位"]
        HR_BP["HeapRing 背压解除<br/>可分配新缓冲区"]
        TM_BP["TensorMap 清理触发<br/>回收过期条目"]
    end

    ADV_LA -->|"写入 SM"| SM_LA
    ADV_HT -->|"写入 SM"| SM_HT
    SM_LA -->|"读取"| TR_BP
    SM_LA -->|"读取"| TM_BP
    SM_HT -->|"读取"| HR_BP
```

**循环闭合**：Scheduler 推进水位 → Orchestrator 解除背压 → 提交更多任务 → Scheduler 执行更多任务 → 推进水位。这是系统**稳态流水线**运行的核心反馈环。

---

> 系统总体架构和数据结构详见 [01-数据流与核心设计.md](01-数据流与核心设计.md)，Orchestration 处理流程详见 [02-Orchestration处理流程.md](02-Orchestration处理流程.md)。
