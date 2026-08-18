# A5 HBG Ticket Cursor + Private Pending 调度器设计

```text
状态：v1 已实现（直接替换旧 queue/resolver 路径）；静态 dependency-priority stream 已接入
目标：A5 host_build_graph_aicore v0
日期：2026-08-12
```

## 1. 结论

本文建议在当前 A5 `host_build_graph_aicore` v0 调度器中，用以下组合同时替代 AIC/AIV ReadyQ、CompletionQ 和 wake list：

- Host 按剩余关键路径长度构建 dependency-priority AIC/AIV task-ID 流；
- AIC/AIV 各使用一个 ticket cursor，通过 `atomicAdd` 唯一分配任务；
- 每个 AICore 维护少量 owner-only private pending slots；
- 任务 owner 本地轮询 pending task 的 completion flags 判断依赖；
- kernel 完成后直接发布 per-task completion flag；
- cursor 耗尽且本核 pending 为空后，本核一次性发布 drained 状态并进入低频等待。

推荐的初始参数为：

```text
pending slot 数 K = 2
每种 core type 的 active worker 数 = min(可用核数, 该类型可执行任务数)
首轮任务按 type_rank 静态分配
后续任务通过 atomicAdd ticket cursor 动态分配
```

该设计的目标不是消除所有原子操作，而是把当前共享队列的协调成本从“随空闲轮询时间增长”改为“主要随任务数和依赖边数增长”。

## 2. 背景与当前问题

当前调度器使用：

- AIC ReadyQ：多个 resolver 生产、多个 AIC 消费；
- AIV ReadyQ：多个 resolver 生产、多个 AIV 消费；
- CompletionQ：所有 executor 生产、多个 resolver 消费；
- per-task wake list：任务完成后重新分类 waiter。

当前空队列 `pop()` 虽然通常不会执行 dequeue CAS，但仍会加载共享 `dequeue_pos` 和队头 `sequence`。A5 上这些 GM load 通过 `atomicAdd(0)` 或 `atomicMax(identity)` 实现，因此大量空闲核会持续访问相同的原子地址。当前 `SPIN_WAIT_HINT()` 还是空操作，空转成本会随等待时间持续增长。

当前队列还需要分别发布 slot 的 `task_id` 和 `sequence`。两者是不同 GM word，必须额外处理 sequence 已可见而 payload 尚未可见的情况。

本设计要同时消除：

```text
ReadyQ push / pop
CompletionQ push / pop
slot payload publication
wake-list registration / close / reclassification
专职 resolver 角色
```

## 3. 设计范围

### 3.1 支持范围

本设计只覆盖当前 A5 HBG AICore scheduler v0 已经接受的任务形态：

- graph 在 AICore 执行前已经由 Host 完整构建；
- fanin task ID 严格小于 consumer task ID；
- 每个 executable task 只有一个 active subtask；
- `logical_block_num == 1`；
- task 不包含 predicate、sync-start 或 executable early-resolve；
- task core type 为单独的 AIC 或 AIV；
- inline-completed task 可以存在，但不分配给 AICore。

当前实现已经在 `aicore_classify_task_v0()` 中检查 `producer < task_id`，因此 task ID 是合法拓扑序。

### 3.2 非目标

本设计暂不支持：

- 一个逻辑任务同时占用 AIC/AIV；
- 多 block 或多 subtask 联合完成；
- owner 故障后的任务恢复；
- task claim 后的跨核 work stealing；
- graph 执行期间动态追加任务；
- 不满足 producer-before-consumer priority 的任意 task-ID 流。

未来支持 mixed/multi-block task 时，需要增加 group owner、lane deposit、remaining-subtask counter 和联合 completion publication，不能直接复用本文的单 owner 状态机。

## 4. 核心架构

```text
Host orchestration 完成
          |
          v
校验 graph，计算 bottom level，并生成两个 priority-ordered task-ID 流
  aic_task_ids[]       aiv_task_ids[]
          |                   |
          v                   v
  AIC ticket cursor    AIV ticket cursor
          |                   |
          +---------+---------+
                    |
                    v
           每核 private pending[K]
                    |
              检查 fanin flags
                    |
             +------+------+
             |             |
          未完成          全部完成
             |             |
        本地退避/轮询     execute_task
                           |
                    flush kernel output
                           |
                  publish completion flag
                           |
                      释放 pending slot
```

任务所有权在任务 ready 之前确定。任务 readiness 不再作为跨核事件发布，而由 owner 根据 monotonic completion flags 本地判断。

## 5. 关键不变量

实现必须维持以下不变量。

### 5.1 唯一所有权

每个 executable task 恰好来自一个 type-specific task-ID 流，每个有效 stream index 只由一个 worker 获得。

```text
task_id 在 aic_task_ids[] 或 aiv_task_ids[] 中恰好出现一次
atomicAdd(cursor, 1) 返回的有效 index 全局唯一
```

### 5.2 拓扑 priority

对任意 fanin edge：

```text
producer_task_id < consumer_task_id
bottom_level(producer) > bottom_level(consumer)
```

AIC/AIV task-ID 流分别按 `(bottom_level desc, task_id asc)` 排序。跨类型依赖仍通过全局
completion cells 连接；task ID 仍是 graph 的合法拓扑序，但 typed stream 本身可以是非单调 task ID。

### 5.3 单调完成

每个 completion flag 每轮只允许：

```text
NOT_DONE(0) -> DONE(1)
```

禁止回退和重复执行。inline-completed task 在 RUN 前初始化为 DONE。

### 5.4 owner-only pending

private pending slot 只由所属物理 AICore 读写，不允许其他 AICore、AICPU 或 Host 在 RUN 期间修改。DFX 如需读取 pending 状态，应写独立的快照结构，不能把 pending slot 变为共享控制面。

### 5.5 数据先于完成可见

completion flag 只能在 kernel 输出已经 publish 后设置为 DONE。consumer 观察到 DONE 后，在执行 dependent kernel 前必须完成必要的 cache invalidate。

## 6. 无死锁依据

该方案的进展性依赖 typed stream 对所有依赖边保持严格 priority 单调。

假设：

- 每种存在 executable task 的 core type 至少有一个 active worker；
- active worker 最终会运行；
- worker 不会只阻塞在一个 pending slot，而是轮询所有 pending slots；
- 有空 slot 且 cursor 未耗尽时，worker 最终会领取任务。

考虑全局 priority 最高的未完成 executable task `T`：

- `T` 的所有 executable fanin priority 都严格高于 `T`；
- 如果这些 fanin 仍未完成，它们就是比 `T` 更高 priority 的未完成任务，与 `T` 的选择矛盾；
- 因此 `T` 的 fanin 均已完成或是 inline-completed；
- 因此 `T` 一旦被 owner 领取就是 ready；
- type-specific cursor 按相同 priority 分配，不能永久跳过 `T` 去领取更低 priority task；
- 如果 `T` 尚未被领取，则它前面的同类型 pending 只可能具有更高或相同 priority；相同 priority
  task 之间不存在依赖边，更高 priority task 又递归满足上述推进条件。

所以，只要 worker 公平运行，系统中始终存在能够完成的 claimed task，不能形成仅由未完成 fanin 构成的闭环。

这个证明不适用于任意 shuffle，也不适用于 worker 只等待 slot 0、完全不检查其他 pending slot 的实现。

## 7. Host 侧设计

### 7.1 构建 typed task-ID 流

Host 先逐 task 调用 `aicore_classify_task_v0()`，确定 executable 和 inline-completed task，然后由
Host-only planner 构建：

```cpp
std::vector<uint32_t> aic_task_ids;
std::vector<uint32_t> aiv_task_ids;
std::vector<uint32_t> bottom_level;
```

规则如下：

- 利用 `producer_task_id < consumer_task_id`，按 consumer task ID 逆序遍历 fanin edge；
- 对每条边更新 `bottom_level[producer] = max(bottom_level[producer], bottom_level[consumer] + 1)`；
- executable task 按 core type 分流，再按 `(bottom_level desc, task_id asc)` 排序；
- inline-completed task 只设置 completion flag，不进入 typed stream；
- 任何 unsupported task 直接拒绝该 scheduler；

Host 必须额外校验：

```text
aic_task_ids.size() == aic_task_count
aiv_task_ids.size() == aiv_task_count
两个 stream 无重复、无交集
stream task core type 与 descriptor 一致
每条 edge 的 producer priority 严格高于 consumer
两个 typed stream 都符合完整的 priority 排序规则
executable_count + inline_completed_count == total_task_count
```

planner 的时间复杂度为 `O(V + E + V log V)`，临时内存为 `O(V)`；不构建 successor adjacency，
也不增加 device steady-state 数据结构。

### 7.2 Sidecar 初始化

Host 初始化：

- typed task-ID arrays；
- 所有 completion cells 为 NOT_DONE；
- inline-completed task 的 completion cell 为 DONE；
- cursor metadata；
- run control 和 worker context；
- DFX counters。

任务列表和 graph descriptor/payload 在进入 RUN 后保持只读。

## 8. AICPU 控制面设计

### 8.1 active worker 选择

每种 core type 的 active worker 数：

```cpp
active_aic = min(available_aic, aic_task_count);
active_aiv = min(available_aiv, aiv_task_count);
```

这样可以避免只有少量任务时启动所有物理核参与竞争。

active worker 必须使用连续的 type rank `[0, active_type_workers)`；如果物理核选择不是连续前缀，AICPU 必须额外生成 dense `active_type_rank`，首轮 seed 不能直接使用带空洞的物理 `type_rank`。

如果未来需要进一步限制功耗或调度开销，可以增加显式配置上限：

```cpp
active_aic = min(active_aic, configured_aic_limit);
active_aiv = min(active_aiv, configured_aiv_limit);
```

如果 graph 只有 inline-completed task，`active_worker_count == 0`，AICPU 应直接完成本轮控制流程，不进入 ticket RUN phase。

### 8.2 首轮 rank-seeded 分配

为了避免 RUN 刚开始时最快的核连续领取多个最前面的任务，前 `active_type_workers` 个任务不通过 atomic cursor 竞争，而由 `type_rank` 唯一确定：

```cpp
initial_index = worker.type_rank;
initial_task_id = type_task_ids[initial_index];
```

AICPU 在发布 RUN 前把 cursor 初始化为：

```cpp
aic_stream.next_index = active_aic;
aiv_stream.next_index = active_aiv;
```

这保证：

- 每个 active worker 初始至少拥有一个任务；
- task 数不超过 core 数时，ownership 分配不需要 atomicAdd；
- 后续任务仍可由完成较快的 worker 动态领取。

### 8.3 结束条件

每个 active worker 在满足以下条件后只发布一次 drained：

```text
本类型 cursor 已确认耗尽
本核所有 pending slots 为空
本核没有正在执行的 kernel
```

AICPU 等待：

```cpp
drained_worker_count == active_worker_count
```

该条件比每轮高频读取全局 `completed_count` 更适合本设计。每核在 drained 时批量累加本地 executed count，用于最终校验：

```text
sum(worker.executed_count) + inline_completed_count == total_task_count
```

AICPU 确认所有 active worker drained 后发布 EXIT。已经 drained 的 AICore 使用带退避的低频方式等待 EXIT，不再轮询 task cursor 或 completion flags。

## 9. 建议的数据结构

以下结构是设计草案，字段和 ABI 版本号可在实现时调整。

### 9.1 Completion cell

```cpp
enum class AicoreTaskCompletionV1 : int64_t {
    NOT_DONE = 0,
    DONE = 1,
};

struct alignas(128) AicoreTaskCompletionCellV1 {
    volatile int64_t completion;
    uint8_t atomic_line_padding[120];
};
```

v1 首版继续使用每任务独立 cache line，原因是：

- 避免多个 producer completion flags 共享 cache line；
- 避免对某个 flag 的高 fan-out 轮询干扰相邻任务；
- 保持与当前 per-task 128-byte control cell 接近的内存规模；
- 后续可以基于数据再评估压缩为更紧凑的 flag array。

禁止对 completion cell 所在 cache line 使用可能写回旧值的普通 cached DCCI publication。

### 9.2 Type-specific ticket stream

```cpp
struct alignas(128) AicoreTaskStreamV1 {
    uint64_t task_ids_offset;
    uint64_t task_count;
    uint64_t initial_ticket_count;
    uint8_t metadata_padding[104];

    volatile uint64_t next_index;
    uint8_t cursor_padding[120];
};
```

AIC 和 AIV 各有一个独立 stream。metadata 和 hot cursor 分离 cache line。

task ID 可以使用 `uint32_t`，因为当前 graph task count 和 fanin ID 已受 `int32_t` 范围限制。

### 9.3 Core-local pending slot

```cpp
struct AicorePendingSlotV1 {
    int64_t task_id;
    int32_t fanin_count;
    int32_t next_fanin_index;
    int32_t kernel_id;
    int32_t subtask_slot;
};
```

推荐把 `AicorePendingSlotV1 pending[K]` 放在 persistent AICore executor 的 core-local 栈/局部存储中，而不是 sidecar GM。K=2 时结构很小，可以避免 pending 管理本身产生 GM 流量。

如果编译器产生不可接受的 spill，再退化为 sidecar 中的 per-worker owner-only 区域；即便放在 GM，也不能被其他核写入。

### 9.4 Worker debug snapshot

pending 放在 core-local 存储后，AICPU 无法在超时时直接读取它。建议为每个 worker 保留一个低频更新的 GM debug snapshot：

```cpp
struct AicoreWorkerDebugV1 {
    int64_t pending_task_id[K];
    int32_t pending_next_fanin[K];
    int32_t pending_waiting_producer[K];
    uint64_t cursor_exhausted;
    uint64_t lifecycle_state;
};
```

snapshot 只在以下状态转换时由 owner 更新并 publish：

- task 被放入 pending；
- 当前等待 producer 发生变化；
- task 开始执行；
- pending slot 被释放；
- worker drained 或进入 fatal 状态。

禁止在每次 completion poll 中更新 snapshot。这样既保留 hang 诊断能力，也不会把 owner-only pending 重新变成高频共享状态。

### 9.5 Run control

建议保留或增加：

```cpp
struct AicoreRunControlV1 {
    uint64_t attached_count;
    uint64_t active_worker_count;
    uint64_t drained_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t executed_task_count;
    uint64_t startup_phase;
    uint64_t exit_requested;
    uint64_t scheduler_error;
    uint64_t error_task_id;
};
```

`executed_task_count` 不应每完成一个 task 就在共享 cache line 上 atomicAdd。每核本地累计，在 drained 时一次性合并。

## 10. Worker 算法

### 10.1 初始化

每个 active worker：

1. 等待所有 worker attach；
2. 观察 AICPU 发布的 active topology 和 RUN phase；
3. 根据 core type 选择 AIC/AIV stream；
4. 初始化两个空 pending slots 和本地统计计数器；
5. 用 dense active type rank 取得首轮静态 task并放入 slot 0。

inactive worker 不访问 task cursor，直接进入带退避的 EXIT 等待。

### 10.2 任务领取

后续 ticket 领取：

```cpp
uint64_t index = atomicAdd(stream.next_index, 1);
if (index < stream.task_count) {
    task_id = stream.task_ids[index];
} else {
    local_cursor_exhausted = true;
}
```

多个 worker 在尾部可能各自取得一个越界 index。越界只表示本核确认 stream 已耗尽，不是错误。实现和 DFX 不应要求最终 cursor 精确等于 `task_count`。

每个 worker 确认耗尽后不得再次 atomicAdd cursor。

### 10.3 Pending 填充策略

推荐采用“执行优先、阻塞时才扩充 pending”的策略：

1. 先扫描已有 pending slots；
2. 如果有 ready task，立即执行一个，不继续 claim；
3. 如果没有 ready task且存在空 slot，claim 一个新任务；
4. 如果 pending 已满且都未 ready，进入一次 backoff；
5. 下一轮从上次 slot 的后继位置开始扫描。

该策略让 K=2 的第二个 slot 主要用于绕过依赖阻塞，而不是让快核提前囤积大量 ready task。

### 10.4 Readiness 检查

每个 slot 只轮询第一个尚未确认完成的 fanin：

```cpp
while (slot.next_fanin_index < slot.fanin_count) {
    int32_t producer = fanins[slot.next_fanin_index];
    if (!completion_ready_acquire(producer)) {
        return NOT_READY;
    }
    ++slot.next_fanin_index;
}
return READY;
```

completion flag 在本轮执行中单调，因此已经确认完成的 fanin 前缀不需要再次读取。这样：

- 每条已完成 edge 的成功检查通常只发生一次；
- 一个 blocked slot 每轮最多轮询一个未完成 producer；
- 不会在每个 poll 中重新扫描全部 fanin。

worker 必须 round-robin 检查所有 occupied pending slots，不能因为 slot 0 未 ready 就跳过 slot 1。

### 10.5 执行与完成发布

任务 ready 后：

1. 重新观察 descriptor/callable 等执行元数据；
2. invalidate 执行所需的数据 cache；
3. materialize per-worker dispatch payload；
4. 执行 kernel；
5. publish/flush kernel 产生的数据；
6. 以 release 语义 atomic publish `completion[task_id] = DONE`；
7. 记录本地 DFX；
8. 释放 pending slot。

伪代码：

```cpp
observe_task_metadata(task_id);
materialize_payload(task_id, dispatch_payload);
execute_task(dispatch_payload);
publish_kernel_outputs();
atomicExch(completion[task_id], DONE);
release_pending_slot(slot);
```

### 10.6 主循环伪代码

```cpp
seed_initial_task_by_type_rank();

while (!fatal_error) {
    int ready_slot = find_ready_pending_round_robin();
    if (ready_slot >= 0) {
        execute_and_complete(pending[ready_slot]);
        local_executed_count++;
        reset_backoff();
        continue;
    }

    int free_slot = find_free_pending_slot();
    if (free_slot >= 0 && !cursor_exhausted) {
        ClaimResult claim = claim_next_ticket();
        if (claim.valid) {
            initialize_pending(pending[free_slot], claim.task_id);
            reset_backoff();
            continue;
        }
        cursor_exhausted = true;
    }

    if (cursor_exhausted && pending_empty()) {
        flush_local_dfx_and_publish_drained();
        break;
    }

    local_backoff();
}

wait_for_exit_with_backoff();
```

## 11. Backoff 策略

不能继续使用空实现的 `SPIN_WAIT_HINT()` 作为唯一等待措施。

推荐使用纯 core-local 指数退避：

```text
首次无进展：短 nop/计算循环
连续无进展：逐步增加等待迭代数
观察到 completion、成功 claim 或执行 task：立即重置退避
达到上限后保持固定最大退避
```

要求：

- 退避过程不得更新共享 GM counter；
- pending slot 较多时，每轮先完整 round-robin 扫描再退避；
- EXIT/fatal flag 可以低频检查，避免每个内层 poll 都访问全局控制字；
- DFX 记录本地 `idle_iterations` 和 `backoff_cycles`，drained 时一次性发布。

具体 nop 指令或等待实现需要在 A5 sim/onboard 上验证，不能假设普通空循环一定不会被编译器删除。

## 12. AIC/AIV 跨类型依赖

两个 type cursor 只负责 ownership 分配，不隔离 completion namespace。

例如：

```text
AIC task 3 ----> AIV task 5 ----> AIC task 8
```

- task 3 由 AIC stream 分配；
- task 5 由 AIV stream 分配并轮询 completion[3]；
- task 8 由 AIC stream 分配并轮询 completion[5]。

只要两种存在任务的 core type 都有 active worker，就不需要跨核邮箱或 CompletionQ。

## 13. Cache 与内存序协议

A5 GM cache 非一致，C++ 风格的 acquire/release 名称不能替代实际硬件协议。实现必须复用已经在 A5 上验证过的 raw-GM atomic 和 DCCI 组合。

### 13.1 Host/AICPU 发布只读 graph 与 stream

```text
Host 写 graph/typed task IDs/sidecar
    -> copy_to_device / cache flush
AICPU 写 active topology/cursor 初值
    -> cache flush
AICore 观察 RUN
    -> invalidate/observe metadata
```

### 13.2 Producer 完成发布

```text
kernel 写输出
    -> publish/flush data cache
    -> dsb/store barrier
    -> atomicExch(completion, DONE)
```

completion flag 不得早于 kernel output flush 可见。

### 13.3 Consumer 观察完成

```text
atomic load completion == DONE
    -> 执行前 invalidate 所需数据
    -> materialize/execute dependent task
```

owner 可以提前读取 immutable fanin ID 和 descriptor，但不能在依赖完成前缓存并直接使用 producer 的可变输出数据。

## 14. 错误处理

### 14.1 首错误发布

沿用 first-error-wins：

```cpp
if (atomicCAS(error_claimed, 0, 1) == 0) {
    publish_error_metadata();
    atomicExch(scheduler_error, error_code);
}
```

错误元数据必须先于非零 `scheduler_error` 可见。

### 14.2 Claim 后异常

ticket cursor 一旦推进，task ownership 不可回滚。如果 owner 在 claim 后永久退出，该 task 会丢失。本设计与当前 persistent AICore 执行模型一样，不提供单核故障恢复。

普通 task validation 失败应触发全局 fatal exit，而不是尝试把 ticket 放回 cursor。

### 14.3 超时诊断

超时 dump 至少包括：

- AIC/AIV cursor 当前值与 task count；
- 每核 cursor_exhausted、pending occupancy；
- 每个 pending task ID 和 `next_fanin_index`；
- 当前等待的 producer task ID 及 completion value；
- drained worker count；
- first error metadata。

## 15. DFX 设计

所有高频统计优先保存在 core-local 变量中，禁止在每次 poll 中 atomicAdd GM counter。

### 15.1 每核计数

建议记录：

```text
seeded_task_count
ticket_claim_count
ticket_exhaustion_count
executed_task_count
pending_peak
pending_scan_count
fanin_flag_load_count
fanin_wait_count
idle_iteration_count
backoff_cycles
claim_cycles
dependency_wait_cycles
payload_cycles
kernel_cycles
completion_publish_cycles
```

这些计数在 worker drained 时一次性写入自己的 context。

### 15.2 全局一致性校验

执行结束后校验：

```text
所有 typed stream 中的 task 恰好执行一次
sum(executed_task_count) == aic_task_count + aiv_task_count
sum(executed_task_count) + inline_completed_count == total_task_count
drained_worker_count == active_worker_count
每个 executable task completion == DONE
cursor >= task_count（不要求等于）
```

### 15.3 Swimlane phase

建议保留或增加以下阶段：

```text
SeedClaim
TicketClaim
PendingWait
Payload
Kernel
CompletionPublish
Drain
```

这样可以直接区分 cursor 获取、依赖等待和 kernel 执行成本。

### 15.4 原子操作成本模型

定义：

```text
T = executable task 数
C = active worker 数
E = fanin edge 数
W = blocked pending 对尚未完成 flag 的重复检查次数
```

采用 rank seed 后，ownership 路径预计为：

```text
首轮 seed ownership atomic：0
有效 ticket atomicAdd：max(0, T - C)
尾部 exhaustion atomicAdd：最多约 C
completion atomicExch：T
fanin flag load：约 E + W
worker drained atomicAdd：C
```

其中 `W` 是本设计最重要的不确定项，受 DAG 形态、kernel 时长、K 和 backoff 影响，必须通过 DFX 实测。与当前 queue 方案不同，ownership 和 exhaustion 原子次数不随 kernel 空闲等待时间无限增长。

## 16. 方案取舍

### 16.1 相比当前 ReadyQ/CompletionQ

收益：

- 删除三个共享 MPMC queue；
- 删除 slot `task_id/sequence` 双字段 publication；
- 删除 wake-list CAS 和 completion reclassification；
- ownership 原子操作次数主要与任务数相关；
- 空闲 worker 不再持续 pop 同一个队头；
- completion traffic 分散到 per-task cache line；
- sidecar 控制结构更简单。

代价：

- task 在 ready 前绑定 owner；
- owner 需要轮询 fanin flag；
- 高 fan-out 时多个 consumer 可能轮询同一个 producer flag；
- private pending window 大小影响负载均衡和依赖隐藏；
- 没有 work stealing，claim 后的负载偏斜无法修正。

### 16.2 相比 per-task READY scan/CAS

ticket + pending 的优势：

- 不需要所有 executor 扫描全部 READY state；
- task ownership 通常只需要一次 `atomicAdd`；
- 不存在多个核对同一个 READY task 执行 claim CAS 的突发竞争；
- cursor 耗尽后，worker 可以永久停止 claim。

READY scan/CAS 的优势是 late binding，面对 kernel 时长差异很大的宽 DAG 时负载均衡可能更好。

### 16.3 相比 simpler-dist atomicMax claim

simpler-dist 的 claim race 要求所有候选核重放同一个任务并执行 `atomicMax(cursor, task_id)`。对每个任务可能产生接近候选核数的 ownership 原子操作。

本文利用 HBG 已经静态构建好的 typed task-ID arrays，使用唯一 ticket：

```text
atomicMax replay：约 eligible_cores * tasks
ticket cursor：约 remaining_tasks + exhaustion_attempts
```

因此不建议直接照搬全核 atomicMax claim。

## 17. 性能风险与缓解

### 17.1 高 fan-out completion 热点

一个 producer 被很多 pending task 依赖时，多个 owner 会读取同一个 completion cell。

缓解措施：

- 每个 blocked slot 每轮只检查一个 producer；
- 指数退避；
- active worker 数不超过 task 数；
- round-robin pending，避免紧盯一个 producer；
- 通过 DFX 统计每个 producer 的 wait/load 热度后再决定是否引入分层通知。

### 17.2 快核囤积任务

缓解措施：

- rank-seeded 首轮分配；
- 初始 K=2；
- 有 ready pending 时优先执行，不继续 claim；
- 只有所有现有 pending 都 blocked 时才占用第二个 slot；
- 不使用大块 ticket chunk。

### 17.3 Cursor 热点

每种 core type 仍有一个共享 `next_index`，但每个 task 通常只访问一次，且不在空闲循环中重复访问。

只有在大量极短 task 下 cursor 才可能成为显著热点。若实测需要优化，可评估小 chunk ticket，但 chunk 会增加任务囤积，不应作为 v1 默认方案。

### 17.4 Pending 本地存储 spill

需要检查 CCEC 生成代码和 core-local 栈占用。如果 K=2 仍发生明显 spill：

- 减少 slot 缓存字段；
- kernel_id/subtask_slot 在执行时重新读取；
- 或将 pending 放到 per-worker owner-only GM 区域。

## 18. 实施步骤

### 阶段 1：Host 数据与模型验证

- 构建 AIC/AIV typed task-ID arrays；
- 增加 host-only ticket/pending reference model；
- 用随机拓扑 DAG 验证 exactly-once 和无死锁；
- 验证 inline-completed、纯 AIC、纯 AIV、AIC/AIV 交叉依赖。

### 阶段 2：Sidecar v1

- 增加 task streams、completion cells 和新的 run control；
- 直接替换 sidecar v0，不增加 feature/runtime gate；
- 加入 layout/offset/static-assert 单测。

### 阶段 3：AICore executor

- 实现 rank seed；
- 实现 ticket claim；
- 实现 K=2 core-local pending；
- 实现 fanin prefix polling；
- 实现直接 completion publication；
- 实现 drained 和 backoff。

### 阶段 4：删除旧路径

v1 直接替换方案删除：

- AIC/AIV ReadyQ；
- CompletionQ；
- wake list fields；
- resolver/classification phase；
- queue-specific DFX 和 validation。

## 19. 测试矩阵

### 19.1 Host/UT

- 0、1、2、`cores-1`、`cores`、`cores+1` 个任务；
- cursor index 唯一性和越界 exhaustion；
- type-rank seed 不重复；
- AIC/AIV typed list 完整性；
- K=1/2/4 pending reference model；
- randomized DAG，所有 edge 满足 `producer < consumer`；
- 非法逆向 edge 必须拒绝；
- 每个 task exactly once；
- inline-completed task 不进入 stream；
- error first-writer-wins。

### 19.2 Sim/ST

- 单任务 AIC；
- 单任务 AIV；
- AIC-only chain；
- AIV-only chain；
- AIC/AIV 交替 chain；
- diamond DAG；
- 多 independent roots；
- 高 fan-in；
- 高 fan-out；
- 长短 kernel 混合；
- task 数远小于核数；
- task 数远大于核数。

### 19.3 Onboard 性能

对比现有 queue scheduler，至少采集：

```text
Total / scheduler / kernel cycles
GM atomic 次数及热点地址
ticket claim cycles
fanin flag load/wait 次数
idle/backoff cycles
每核执行任务数分布
pending occupancy peak
task ready-to-start latency
尾核完成时间
```

K=1、2、4 必须使用相同 graph 和相同 active core 策略对比。最终 K 值由实测决定，本文只推荐 K=2 作为首个实现。

## 20. 验收标准

功能验收：

- 所有当前 v0 支持的合法 graph 结果一致；
- 所有 executable task exactly once；
- AIC/AIV 交叉依赖正确；
- inline-completed task 正确传播；
- sim/onboard 无 hang；
- fatal error 能停止所有 worker 并给出 pending/cursor 诊断。

架构验收：

- 执行路径不存在 ReadyQ/CompletionQ push/pop；
- 执行路径不存在 wake-list registration/reclassification；
- 空闲 worker 不重复访问 task cursor；
- 高频 DFX counter 不在轮询循环中 atomicAdd；
- completion publish 遵守数据 flush -> flag publish 顺序。

性能验收：

- 稀疏 task 场景显著降低空闲 GM atomic traffic；
- 不以不可接受的 fanin polling 增长换取 queue 删除；
- 宽 DAG 下每核任务分布和尾延迟没有明显恶化；
- K=2 相对 K=1/4 给出可解释的选择依据。

## 21. 最终推荐配置

首版实现采用：

```text
所有权：type-specific ticket cursor
首轮：type_rank 静态 seed
后续：atomicAdd(next_index, 1)
pending：每核 K=2，core-local
调度策略：ready task 优先；仅全部 blocked 时扩充 pending
依赖：每 slot 轮询第一个未完成 fanin，完成前缀只检查一次
完成：per-task padded completion flag
结束：cursor exhausted + local pending empty -> publish drained once
等待：core-local exponential backoff
统计：本地累计，drained 时批量发布
```

这套配置最符合当前 HBG 的静态拓扑图、单 owner task 和任务较少的目标场景，同时避免 simpler-dist 全核 replay/atomicMax 的额外竞争。

## 22. 当前实现参考点

- 当前 graph 对 `producer < task_id` 的校验：[`aicore_graph_view_v0.h`](../src/a5/runtime/host_build_graph/runtime/aicore_graph_view_v0.h)
- Host 已有的逐 task 分类和 core-type 统计：[`runtime_maker.cpp`](../src/a5/runtime/host_build_graph/host/runtime_maker.cpp)
- 当前 single-slot、single-block v0 约束：[`runtime_maker.cpp`](../src/a5/runtime/host_build_graph/host/runtime_maker.cpp)
- v1 sidecar：[`aicore_execution_sidecar_v1.h`](../src/a5/runtime/host_build_graph/runtime/aicore_execution_sidecar_v1.h)
- ticket/pending 原语：[`aicore_ticket_scheduler_v1.h`](../src/a5/runtime/host_build_graph/runtime/aicore_ticket_scheduler_v1.h)
- AICore poll/execute/complete 主循环：[`aicore_executor.cpp`](../src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp)
- 当前 A5 raw-GM atomic 包装：[`aicore_gm_atomic.h`](../src/a5/runtime/host_build_graph/runtime/aicore_gm_atomic.h)
