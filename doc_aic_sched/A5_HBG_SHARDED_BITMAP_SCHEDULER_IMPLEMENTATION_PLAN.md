# A5 HBG 静态分片 Bitmap 调度器实现方案

```text
目标分支：feat/hbg-mpmc-scheduler-optimization
目标基点：000903a4
范围：A5 host_build_graph AICore scheduler
状态：方案已确定，待实现
```

## 1. 目标与结论

用三个静态分片 event workset 直接替换当前三个共享 MPMC 队列：

- AIC Ready workset；
- AIV Ready workset；
- Completion workset。

每个 workset 为每个 task 保留一个事件 bit。task 只有在真正满足全部
fanin 后才发布 Ready bit，因此执行所有权仍是 late binding，不会像 ticket
cursor 一样让某个 worker 提前持有未 Ready task。

每个 workset 固定包含 64 个 shard。task 到 shard 的存储映射是静态的；
shard 的 home worker 根据本轮 active topology 分配；空闲 worker 可以从其他
shard steal。静态映射只决定事件存放位置，不预分配 task 的执行核。

该方案删除：

- 全局 `enqueue_pos` / `dequeue_pos` 热点；
- 有界队列容量和 overflow；
- queue slot 的 `sequence` / payload 双字段发布；
- FIFO 队头阻塞。

首版直接替换旧路径，不增加环境变量、宏或 runtime 行为开关。

支持范围保持当前 A5 HBG AICore v0 约束：

- graph 在 AICore 执行前已经完整构建；
- `producer_task_id < consumer_task_id`；
- 每个 executable task 只有一个 active subtask 和一个 logical block；
- 不支持 executable predicate、sync-start 或 early-resolve；
- task core type 为单独的 AIC 或 AIV；
- inline-completed task 可以存在，但不发布 Ready/Completion 事件。

## 2. 数据结构与静态映射

### 2.1 Task 到事件 bit 的映射

每个 workset 固定使用 64 个 shard：

```cpp
shard_index = task_id & 63;
local_index = task_id >> 6;
leaf_index = local_index >> 6;
bit_index = local_index & 63;
```

逆映射：

```cpp
task_id = ((leaf_index * 64 + bit_index) * 64) + shard_index;
```

claim 后必须检查恢复出的 `task_id < graph.task_count`。

该映射把每组连续的 64 个 task 分散到 64 个 shard，不依赖 active worker
数量，也不需要额外的 task-to-slot 映射表。

以 65,536 个 task 为例，每个 workset 为：

```text
64 shards
每 shard 1,024 个 task bit
每 shard 16 个 uint64_t leaf word
每 shard 1 个 uint64_t summary，低 16 bit 有效
leaf 总计 8 KiB；summary 逻辑数据总计 512 B
```

当前 131,072 task window 对应每 shard 32 个 leaf word，仍只需一个 summary。
单层 summary 的明确上限为：

```text
64 shards * 64 summary bits * 64 task bits = 262,144 tasks
```

超过该上限时先拒绝 layout；不要在没有设计和测试的情况下静默扩展。

### 2.2 Workset 布局

Sidecar 继续使用 POD + offset，不能在上传镜像内保存 host 指针。逻辑结构：

```cpp
inline constexpr uint32_t AICORE_EVENT_SHARD_COUNT_V1 = 64;

struct alignas(128) AicoreEventShardV1 {
    uint64_t leaf_words_offset;
    uint32_t leaf_word_count;
    uint32_t reserved0;
    uint8_t metadata_padding[112];

    volatile uint64_t summary;
    uint8_t summary_padding[120];
};

struct AicoreEventWorksetLayoutV1 {
    uint64_t shards_offset;
    uint64_t leaf_words_offset;
    uint64_t leaf_words_per_shard;
};
```

summary 独占 cache line；每个 shard 的 leaf block 按 cache line 对齐。首版不要
为每个 leaf word 单独填充 128 byte，应先测量 leaf 竞争再决定。

Sidecar 保存三个独立 workset：

```text
aic_ready_workset
aiv_ready_workset
completion_workset
```

AIC/AIV Ready workset 可以直接使用 task ID，下错 core type 的 bit 永远不置位。
65,536 task 时三个 workset 的 leaf 总空间为 24 KiB，换取无需 dense-index 表。

### 2.3 Task 状态机

把二值 completion word 改为明确的 scheduler state：

```cpp
enum class AicoreTaskEventStateV1 : int64_t {
    BLOCKED = 0,
    READY = 1,
    RUNNING = 2,
    COMPLETED = 3,
    RESOLVING = 4,
    RESOLVED = 5,
};
```

状态转换：

```text
BLOCKED -> READY -> RUNNING -> COMPLETED -> RESOLVING -> RESOLVED
```

inline-completed task 初始化为 `RESOLVED`。producer 处于 `COMPLETED`、
`RESOLVING` 或 `RESOLVED` 时，均表示 kernel 输出已经可被依赖方使用。

task state 和 `wake_list_head` 放在原子 cache line；`next_waiter` 等
single-writer 元数据保留在单独的 publication line。

### 2.4 Home shard 与 stealing

当某类 active worker 数为 `W` 时，rank 为 `r` 的 worker 优先管理：

```text
shard r, r + W, r + 2W, ... < 64
```

例如 28 个 AIC worker 各管理 2～3 个 AIC Ready shard；56 个 AIV worker
各管理 1～2 个 AIV Ready shard。Completion shard 分配给 resolver-active
worker。

Ready stealing 只允许匹配 core type 的 worker；首版 Completion stealing 只允许
resolver-capable worker。home 是优先扫描责任，不是执行所有权。

## 3. 原子协议与执行流程

### 3.1 原子 bit 操作

leaf 和 summary 的置位、清位都必须是原子 RMW，因为多个 producer 可能同时
修改同一 word 的不同 bit，consumer 也可能并发清 bit。

在 `aicore_gm_atomic.h` 中基于已有 64-bit `atomicCAS` 增加：

```cpp
aicore_gm_fetch_or_v0(word, mask)
aicore_gm_fetch_and_v0(word, mask)
```

helper 返回旧值；CAS 失败后必须使用硬件返回的 observed value 继续重试。
host/sim 路径使用对应的 `__atomic` builtin。禁止用 load + 普通 store 更新 bit。

### 3.2 Ready 发布

resolver 确认 task 所有 fanin 已完成后：

```text
CAS task.state: BLOCKED -> READY
atomic OR Ready leaf 的 task bit
atomic OR Ready summary 的 leaf bit
```

只有 state CAS 成功者发布 bit。若已观察到 `READY` 或更晚状态，按 stale
notification 处理；其他非法跳转记录 scheduler error。

### 3.3 Ready claim 与执行

worker 保存 core-local 的 home shard、victim shard、summary bit 和 task bit
轮转游标。claim 一项 Ready 工作：

```text
读取一个 shard summary
从轮转起点选择一个 set summary bit
读取对应 leaf
从轮转起点选择一个 set task bit
CAS leaf: old -> old & ~task_bit
恢复并校验 task ID
CAS task.state: READY -> RUNNING
```

state CAS 成功的 worker 才获得执行权。失败表示 stale bitmap entry，丢弃后
继续扫描。

每次 shard visit 只 claim 一个 task。禁止 `exchange(leaf, 0)` 后把整批任务
留在 core-local：这会让单核囤积最多 64 个 Ready task，重新产生尾延迟。

执行路径：

```text
读取 descriptor/callable
materialize 本 worker 的 dispatch payload
执行 kernel
publish/flush kernel 输出
```

### 3.4 Completion 发布与解析

kernel 返回后：

```text
publish/flush kernel 输出
CAS task.state: RUNNING -> COMPLETED
atomic OR Completion leaf 的 task bit
atomic OR Completion summary 的 leaf bit
```

resolver claim Completion bit 后：

```text
CAS task.state: COMPLETED -> RESOLVING
atomic exchange 关闭 producer wake list
遍历 detached waiters
重新 classify 每个 waiter
为刚满足依赖的 waiter 发布 Ready bit
设置 producer state = RESOLVED
```

现有 wake-list closed-sentinel retry 协议必须保留。classifier 与 completion
并发时，要么在 close 前成功注册，要么观察到 closed 并基于完成状态重新分类。

### 3.5 Summary cleanup 的无丢事件协议

summary 只是提示；leaf 和 task state 才是事实源。leaf CAS 清掉最后一个 bit
后必须执行：

```cpp
atomic_and(summary, ~leaf_mask);
if (atomic_load(leaf) != 0) {
    atomic_or(summary, leaf_mask);
}
```

必须先清 summary，再重新读取 leaf：

- publisher 在 clear 前写入时，recheck 会发现并重新置 summary；
- publisher 在 clear 后写入时，会自行置 summary；
- 并发导致重复置 summary 是允许的；
- leaf 非空但 summary 永久为 0 是不允许的。

summary 指向空 leaf 时也执行相同的 clear + recheck 修复。

### 3.6 Worker 主循环与公平性

主循环：

```text
按固定迭代间隔检查 fatal/exit
按 fairness turn 选择 Ready 或 Completion
  Ready：先轮转 home shards，再检查一个轮转 victim shard
  Completion：先轮转 home shards，再检查一个轮转 victim shard
成功后推进对应 shard/bit 游标
均失败时推进 victim 游标并继续 spin
```

resolver-capable worker 使用固定配额：

```text
最多连续执行 4 个 Ready task，随后必须尝试一次 Completion
最多连续解析 4 个 Completion，随后必须尝试一次 Ready
```

没有 Ready 时立即处理 Completion；非 resolver worker 只扫描匹配 core type
的 Ready workset。

dispatch 路径不得 sleep、`sched_yield` 或使用定时 backoff。空闲 worker 只轮询
自己的少量 home summary 和一个轮转 victim summary，不读取全局队列游标。

AICore 进入 `execute_task()` 后无法并发轮询。进展性假设 kernel 最终返回；若
要求 Completion 延迟完全不受 kernel 时长影响，需要 dedicated resolver 或
producer inline resolve，不属于首版范围。

### 3.7 结束条件

不能用 bitmap 暂时为空判断 graph 完成。维护：

```text
expected_executable_task_count
resolved_task_count
```

resolver 本地累计 resolved 数量，在以下时机 flush 到全局计数：

- 达到固定 batch，例如 32；
- Completion 扫描未找到事件时。

这样既减少全局 counter 竞争，也保证最后一个不足 batch 的尾数能够发布。
AICPU 在下式成立后请求 EXIT：

```text
resolved_task_count == expected_executable_task_count
```

最终校验三个 workset 全空，并且所有 executable task 均为 `RESOLVED`。

## 4. 正确性与无饥饿条件

实现必须维持：

1. 每个 executable task 每轮至多发布一次 Ready 和一次 Completion 事件。
2. `BLOCKED -> READY` 成功后才能发布 Ready bit。
3. 恰好一个 worker 成功执行 `READY -> RUNNING`。
4. kernel 输出全局可见后才能执行 `RUNNING -> COMPLETED`。
5. 恰好一个 resolver 成功执行 `COMPLETED -> RESOLVING`。
6. summary cleanup 不能隐藏并发发布的 leaf bit。
7. 每个 shard 在整轮运行期间都有 home worker。
8. home shard、victim shard、summary bit 和 task bit 都采用轮转起点，不能每次
   从 0 开始。
9. 每次 shard visit 最多 claim 一个 task。
10. Ready 与 Completion 之间存在有限的互相服务配额。

在 graph 有限、worker 最终被调度、kernel 最终返回的前提下，上述规则保证：

- 每个事件都有不依赖 stealing 的 home polling 路径；
- home worker 有界轮转其全部 shard；
- 非空 leaf 要么已被 summary 指示，要么在 cleanup race 后被重新置位；
- 轮转 bit 选择避免低编号事件持续压住高编号事件；
- 轮转 victim 保证空闲 thief 最终访问每个 shard；
- fairness budget 防止 Ready 和 Completion 互相饿死。

worker claim 后永久故障不做恢复；沿用当前 persistent AICore 模型，通过现有
timeout/error diagnostics 终止整轮。

## 5. 实施与验证清单

### 5.1 代码修改范围

- 新增 `runtime/aicore_event_workset_v1.h`，包含映射、publish、claim、cleanup
  及 host-testable helper。
- 用新的 v1 sidecar layout 替换
  `runtime/aicore_execution_sidecar_v0.h` 中的三组 queue 字段，加入 task state
  和三个 workset。
- 在 `runtime/aicore_gm_atomic.h` 增加 CAS-based 64-bit OR/AND。
- 修改 `runtime/aicore_dependency_scheduler_v0.h`，以 Ready bit 替换 ReadyQ
  push，并解析 claim 到的 Completion 事件。
- 重写 `aicore/aicore_executor.cpp` 的 Ready/Completion polling、home、steal
  和 fairness loop。
- 修改 `host/runtime_maker.cpp`，规划、初始化、上传并校验新 sidecar。
- 修改 AICPU lifecycle：分配 home shard，以 resolved task count 结束，不再
  校验 queue push/pop。
- queue DFX 替换为 per-worker bitmap 统计；只在退出或现有冷诊断点批量发布，
  禁止在 hot path 打日志。
- 所有生产和测试引用删除后，删除 `runtime/aicore_ready_queue_v0.h`。
- 同步更新 HBG `RUNTIME_LOGIC.md` 和 profiling 文档。

不要修改 a2a3 runtime；该 scheduler 路径为 A5 专用。

### 5.2 必需单测

- task ID 0、63、64、4,095、4,096、65,535、131,071 的正反映射。
- layout 大小计算，以及超过 262,144 task 时拒绝。
- 两个 producer 并发设置同一 leaf 的不同 bit，两个 bit 均保留。
- 多 consumer 对所有已发布 bit exact-once claim。
- publisher 与最后一个 bit 的 summary cleanup 并发时不丢事件。
- stale summary 能被清理且不会产生虚假 task。
- Ready/Completion state CAS 拒绝重复事件。
- 持续发布低编号 bit 时，轮转策略仍能取得高编号 bit。
- home worker 忙时，thief 能取得该 shard 的事件。
- Ready/Completion 持续有工作时，fairness budget 两边都能推进。
- 65,536 task 随机并发 publish/claim，最终 exact-once reconciliation。
- chain、diamond、高 fanin、高 fanout、纯 AIC、纯 AIV 和跨类型 DAG。

先为当前 queue 行为建立失败复现，再替换实现。可以扩展
`tests/ut/cpp/a5/test_aicore_sidecar_v0.cpp`，也可以新增聚焦的
`test_aicore_event_workset_v1.cpp` 并注册到 C++ test build。

### 5.3 Sim 与板载验收

先运行目标 A5 sim tests，再按仓库要求完成 architecture precheck，并通过
`task-submit` 运行 Paged Attention Case1。

65,536-task Case1 至少记录：

```text
成功次数和 exact task count
Device wall mean / sd / p50 / p95
Ready/Completion publish 和 claim 数量
各 workset leaf CAS 尝试和失败数
summary load、false positive、clear、re-arm 数量
home claim 和 steal claim 数量
Ready publish-to-claim 最大延迟
Completion publish-to-resolve 最大延迟
每 worker task/completion 分布
resolved count 和最终非空 bitmap word 数量
```

直接基线使用 `000903a4` queue scheduler。历史数据为：

```text
Device wall: 522.411 +/- 18.045 ms
Ready queue CAS failure: 94.578%
Completion queue CAS failure: 85.566%
```

验收要求：exact-once 正确、重复运行无 stall、无 queue capacity failure，并且
显著降低该竞争主导的 Device wall。

首轮性能分析必须区分：

- leaf publish 竞争；
- leaf claim 竞争；
- summary polling 成本；
- stealing 频率；
- dependency/wake resolution 成本；
- 所有 eligible worker 均在执行 kernel、暂时无人轮询的时间。

如果热点仍集中在某个 shard summary，可在该 shard 内按 producer 增加 banks，
由 home worker 轮转扫描；不要重新引入全局 summary 或全局 cursor。
