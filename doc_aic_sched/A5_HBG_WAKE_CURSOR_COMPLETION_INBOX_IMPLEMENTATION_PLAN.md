# A5 Host-Build-Graph Wake-list + Cursor Completion Inbox 实现方案

本文是下一会话的实现交接文档。实现基线固定为
`b01dddda8db9329ff18613363f7becc3f9b5f5d4` 的 wake-list + cursor 方案；目标是在不改变
Task 归属、不引入 ReadyQ 的前提下，把完成后的 wake-list 解依赖从 Executor 路径移到空闲
AICore，并通过低竞争的 Completion Inbox 传递完成事件。

结论：建议采用“每个 worker 一个 intrusive completion inbox + 空闲 worker 批量摘取”的方案。
Executor 完成一个 Task 时只发布输出、标记 `COMPLETE`、向某个 inbox 做一次无重试的
`atomicExch`，随后释放自己的 pending slot。空闲 Scheduler AICore 在进入现有 backoff 前，
优先处理本地 inbox，必要时轮转偷取另一个 inbox 的整批事件；它复用现有 wake-list 路由协议，
关闭已完成 producer 的 wake list，并将可以运行的 waiter 更新为 `READY`。

## 1. 基线、目标与边界

### 1.1 固定实现基线

- 基线对象：`b01dddda8db9329ff18613363f7becc3f9b5f5d4`。
- 基线父提交：`884ec07f985ef35a9b5047562748e2c0e3c8a5fd`。
- 基线 PTO ISA pin：`83d01313d9bfc247c4b7c8bcf969d1019f0d106f`。
- 下一会话开始时先执行
  `git cat-file -e b01dddda8db9329ff18613363f7becc3f9b5f5d4^{commit}`，确认对象仍可访问。
- 当前工作树可能包含用户未提交修改。不得覆盖、restore 或 reset 这些修改；实现与基准应使用独立
  worktree，或从干净分支开始。

若需要把 stash commit 中的实现变成正常开发分支，推荐从父提交建立分支，再恢复该对象涉及的八个
文件，而不是把 stash merge 对象直接作为历史基础：

```bash
git switch -c feat/hbg-completion-inbox 884ec07f985ef35a9b5047562748e2c0e3c8a5fd
git restore --source=b01dddda8db9329ff18613363f7becc3f9b5f5d4 -- \
  src/a5/runtime/host_build_graph/aicore/aicore_executor.cpp \
  src/a5/runtime/host_build_graph/docs/RUNTIME_LOGIC.md \
  src/a5/runtime/host_build_graph/docs/profiling_levels.md \
  src/a5/runtime/host_build_graph/host/runtime_maker.cpp \
  src/a5/runtime/host_build_graph/runtime/aicore_execution_sidecar_v1.h \
  src/a5/runtime/host_build_graph/runtime/aicore_ticket_model_v1.h \
  src/a5/runtime/host_build_graph/runtime/aicore_ticket_scheduler_v1.h \
  tests/ut/cpp/a5/test_aicore_ticket_scheduler_v1.cpp
```

只有在目标工作树干净且确认用户修改不在其中时才执行上述命令。

### 1.2 保留的 wake-list + cursor 机制

以下语义保持不变：

- AIC、AIV 各有顺序 Task 流和共享 cursor；worker 通过 cursor claim 新 Task。
- 每个 worker 保留两个 owner-private pending slot；已 claim 的 Task 不迁移执行所有权。
- blocked consumer 向第一个未完成 producer 的 wake list 注册。
- consumer 被唤醒后仍由其 owner 从 pending slot 取出并执行。
- `aicore_route_task_v1` 的“扫描依赖、注册到下一未完成 producer、全部完成后置
  `READY`”协议继续作为唯一解依赖逻辑。
- 不增加全局 ReadyQ，也不让 resolver 直接执行其他 worker 已拥有的 Task。

### 1.3 本次改变

基线中 Executor 在 kernel 结束后同步执行 `aicore_complete_and_wake_v1`：写完成状态、关闭
wake list、遍历 waiter，并递归或迭代完成重新挂接/置 `READY`。本次将其拆成：

1. Executor completion publish：只发布结果、写 `COMPLETE`、投递 completion event。
2. Scheduler resolve：由有空闲窗口的 AICore 关闭 wake list，处理 waiter，并发布 `READY`。

“空闲”不是新增的永久 resolver 核，而是 worker 当前没有可立即执行的已拥有 Task、准备进入
`local_backoff()` 的时间窗口。原有 claim 和 pending 检查仍排在 resolver 工作之前，以免后台解依赖
反过来拖慢本核的直接执行路径。

### 1.4 非目标和硬约束

- 不做 Task 所有权迁移，不做 work stealing 执行；只允许 completion resolve 的批量偷取。
- 不增加行为控制环境变量或编译宏。参数调优通过独立构建或临时实验 patch 完成。
- 不依赖 AICore 之间的 peer MMIO/COND 唤醒；AICore 只能通过 GM 中的 inbox 状态发现工作。
- 因此无法做到字面意义上的“零轮询”。目标是每次 idle 周期只检查常数个 inbox，批量处理，空时
  继续现有指数 backoff，避免扫描全部 Task 或全部 worker。
- 不在热路径输出日志；仅累加 worker-local DFX counter，结束时汇总。
- Completion Inbox 的一次共享竞争仅指 enqueue 本身。wake-list 注册 CAS 和关闭 wake list 的
  exchange 仍是正确性协议所必需的 per-Task 原子操作，但不会形成全局队列热点。

## 2. Completion Inbox 数据结构与并发协议

### 2.1 为什么不用全局 MPMC ring

全局 MPMC CompletionQ 会让所有 Executor 和 resolver 竞争同一 head/tail；满队列处理还会带来
CAS retry、容量判断和轮询。历史实现中 CompletionQ/ReadyQ 已观察到很高 CAS 失败率，不适合作为
65,536 Task 场景的中心热点。

这里使用分片 intrusive inbox。每个 Task 自身就是一个单次 completion node，不需要额外分配，
也不存在队列容量不足。顺序采用 LIFO；wake-list resolve 之间没有必须保持的完成顺序，所以无需为
FIFO 付出 tail CAS 或双原子操作。

### 2.2 建议布局

在 execution sidecar 中按 active worker 数量分配 inbox，并让每个 head 独占一个缓存行：

```cpp
constexpr int64_t kCompletionEmpty = -1;
constexpr int64_t kCompletionLinkUnpublished = -2;

struct alignas(128) CompletionInboxV1 {
    volatile int64_t head;  // task id, or kCompletionEmpty
    // pad to one GM cache line
};
```

在每个 Task control 中增加：

```cpp
volatile int64_t completion_next;  // initialized to kCompletionLinkUnpublished
```

约束：

- `completion_next` 只由完成该 Task 的 Executor 写一次。
- 一个 Task 只允许 enqueue 一次；Task 的 cursor 所有权保证单执行者，单测和 DFX 再验证 exact-once。
- `completion_next` 尽量不要与高频修改的 `state`、`wake_list_head` 共用热点缓存行；以实际
  sidecar 对齐和大小静态断言为准。
- inbox 数量等于 active Scheduler worker 数，而不是物理核上限。
- sidecar 初始化所有 inbox head 为 empty，所有 Task link 为 unpublished。

### 2.3 Executor enqueue：一次共享 RMW，无 CAS retry

完成 Task `T` 的顺序必须是：

1. kernel 返回。
2. 完成输出数据的 publish/flush，并执行现有协议要求的 GM memory barrier。
3. 由唯一 Executor 以 release 语义发布 `T.state = COMPLETE`。
4. 用稳定散列选择 inbox，例如 `T.task_id % active_worker_count`。若已知当前 worker id，也可优先
   投递到当前 worker inbox，但必须用基准确认是否造成局部堆积。
5. 执行一次 `old = atomicExch(inbox.head, T.task_id)`；这是 completion enqueue 唯一的共享
   竞争操作，没有 retry loop。
6. raw-GM store `T.completion_next = old`，并用 release/barrier 保证 resolver 最终可见。
7. 清除 owner 的 pending slot，更新 worker-local completion counter。

步骤 5 先于步骤 6 是 intrusive exchange stack 的必要代价：只有 exchange 返回后才能知道旧 head。
resolver 可能在这两个步骤之间摘到 `T`，因此 link 必须有 unpublished 哨兵，不能把初始值误解为
队尾。

`COMPLETE` 必须先于 enqueue 对 resolver 可见；output 又必须先于 `COMPLETE` 对后续 kernel 可见。
具体 barrier 使用应沿用项目 A5 raw-GM/atomic 封装，不自行用普通 C++ volatile 猜测设备内存序。

### 2.4 Resolver 批量摘取

worker 进入 idle service 点后执行：

1. acquire-load 自己的 inbox head；仅在非 empty 时才做
   `batch = atomicExch(head, kCompletionEmpty)`。
2. 若本地为空，按照 worker-local rotating victim cursor 检查一个其他 inbox；非空才 exchange。
3. exchange 一次摘下整个链。两个 resolver 竞争同一 inbox 时，只有一个得到该批；另一个得到 empty
   或随后到达的新批，因此不需要 per-node claim CAS。
4. 遍历链时，若当前 node 的 `completion_next` 仍为 unpublished，只等待这个极短的 publish window；
   使用受限的 load/backoff，并记录次数和最大等待。调试构建或 host 结束校验必须能把永久 unpublished
   报为协议错误，不能静默死循环。
5. 对每个 Task 验证状态为 `COMPLETE`，再执行 wake resolve。

resolver 可设置常数 batch budget，防止一个长链长时间阻止本 worker 执行自己的 ready Task。budget
耗尽时不能丢弃剩余 detached list；应保存在 worker-local resolver cursor，下一个 idle service 点继续。
实现第一版可先完整处理一批，以降低协议复杂度，再通过 DFX 决定是否需要 budget。

### 2.5 Wake resolve 与 late registration

对已完成 producer `P`：

1. `waiter = atomicExch(P.wake_list_head, CLOSED)`，关闭注册入口并摘下当前 waiter 链。
2. 对每个 waiter 调用保留下来的 `aicore_route_task_v1`。
3. route 从当前依赖位置继续扫描：producer 状态为 `COMPLETE` 即已满足；遇到下一未完成 producer
   则 CAS 注册到其 wake list；全部依赖满足则 release-store waiter 状态为 `READY`。

late registration 仍由现有协议保证：

- waiter CAS 先成功、resolver 后 exchange：waiter 包含在摘下的链中。
- resolver 先写 CLOSED、waiter 后 CAS：CAS 观察到 CLOSED 而失败，重新扫描时观察到 `COMPLETE`，
  跳过该 producer。

Task 的最终完成态定义为：`state == COMPLETE && wake_list_head == CLOSED`。前者表示计算和输出已
发布，后者表示 completion 已被 resolver 消费。无需增加 `RUNNING` 或 `RESOLVED` 状态。

### 2.6 Worker 主循环中的服务点

一次正常调度迭代建议按以下优先级：

1. 扫描本 worker pending slot；发现 `READY` 立即执行。
2. pending 有空位时，通过 AIC/AIV cursor claim 新 Task，并 route/执行。
3. 若本轮没有有效 claim 或可执行 pending，处理 worker-local detached completion batch。
4. 没有 detached batch 时检查本 inbox；仍为空时轮转检查一个 victim inbox。
5. resolve 后立刻回到 pending 扫描，使刚发布的 `READY` 尽快被 owner 看到。
6. 本轮没有任何进展才调用现有 `local_backoff()`。

这意味着同一 AICore 连续执行 Task A、B 时，最快路径是：A kernel 完成 → 发布输出 → A 标记
`COMPLETE` → A enqueue → 清 A pending → 发现 B 已 `READY` → 直接运行 B。若 B 等待 A，且本核
没有其他可 claim 工作，则本核或其他空闲核摘取 A completion → 关闭 A wake list → B 置
`READY` → B owner 重扫 pending → materialize B payload → 执行 B。

为避免持续有 claim 工作时 completion 长时间无人处理，可在每完成固定数量 kernel 后设置一次
低优先级 fairness service point。固定间隔候选值 1、2、4、8 只用于独立构建的实验矩阵；不得未经
批准新增运行时环境变量。首版建议从 4 开始，再依据 completion lag 调整。

### 2.7 退出协议

cursor exhausted 和 pending empty 只表示该 worker `EXECUTOR_DRAINED`，不能立即退出，因为它仍可能
需要作为 resolver 服务其他 worker。

分两阶段退出：

- `EXECUTOR_DRAINED`：所有 typed cursor 已耗尽，本 worker pending 为空。
- `RESOLVER_DRAINED`：所有 worker 均 executor-drained，所有 inbox head empty，本 worker 没有
  detached batch，且全局 emitted completion 数等于 resolved completion 数。

避免每个 Task 对全局 resolved counter 做原子加。worker 维护本地 emitted/resolved 数，在冷路径
阶段性或最终汇总；退出协调者在确认所有 executor-drained 后进行稳定的双重空检查。host 结束时还要
校验全部 Task `COMPLETE`、全部 wake head `CLOSED`、全部 inbox 为空，以及 emitted/resolved 等于
实际执行 Task 数。若设备侧退出判定需要计数，优先每批而非每 Task 更新共享计数。

## 3. 实现步骤与文件落点

### 3.1 数据模型和初始化

主要修改：

- `runtime/aicore_execution_sidecar_v1.h`：增加分片 inbox、worker-local resolver cursor/counter，
  必要时增加退出阶段字段。
- `runtime/aicore_ticket_model_v1.h`：Task state 语义调整为 `BLOCKED/READY/COMPLETE`，增加
  `completion_next` 和哨兵定义。
- `host/runtime_maker.cpp`：按 active worker 数初始化 inbox、Task link、统计字段。

补充 `static_assert` 检查结构对齐、字段宽度、sidecar 容量和 host/device ABI 一致。先检查现有第二
缓存行 padding 是否足够，再决定字段位置，不能仅靠表面空洞假设不会产生 false sharing。

### 3.2 拆分完成函数

在 `runtime/aicore_ticket_scheduler_v1.h` 中把原同步函数拆成三层：

- `aicore_publish_completion_v1(task, inbox)`：Executor-only，写 `COMPLETE` 并 enqueue。
- `aicore_try_detach_completion_batch_v1(worker, own_or_victim)`：idle service，摘取整批。
- `aicore_resolve_completion_v1(task)`：关闭 wake list，逐 waiter 调用现有 route。

`aicore_route_task_v1` 应只把 `COMPLETE` producer 视为已满足，并正确处理 CLOSED；不得把
`COMPLETE` 等价为“wake 已经 resolve”。这两个概念分别由 state 和 wake head 表达。

### 3.3 接入 Executor 主循环

在 `aicore/aicore_executor.cpp`：

- 用 publish/enqueue 替换 kernel 后的 inline complete-and-wake。
- enqueue 成功发布后再释放 pending slot。
- 在现有 `local_backoff()` 之前增加 completion service 点。
- 保持 pending scan 和 cursor claim 的优先级高于后台 resolve。
- 增加 rotating victim cursor，单次 idle 只探测一个远端 inbox。
- 将 executor drain 和 resolver drain 分开；所有错误分支仍必须可终止并被 host 诊断。

禁止在 completion/wake 热路径添加 `printf`。需要调试时使用结构化 ring/stat 字段，并确保 profiling
关闭时不会显著增加每 Task GM 写。

### 3.4 DFX 和文档同步

在 profiling 定义中增加或重命名以下阶段：

- `CompletionEnqueue`
- `CompletionBatchClaim`
- `WakeResolve` / `ReadyPublish`
- `ResolverIdle` / backoff

worker-local counter 至少包括：enqueue 数、own/steal batch 数、batch size、resolved Task 数、
wake waiter 数、route 到下一 producer 数、发布 READY 数、unpublished-link 等待次数/最大迭代、
completion enqueue-to-resolve lag，以及 READY-to-kernel-start lag。仅在 drain 或已有 DFX flush 点发布。

同步更新：

- `src/a5/runtime/host_build_graph/docs/RUNTIME_LOGIC.md`
- `src/a5/runtime/host_build_graph/docs/profiling_levels.md`

文档需要明确两种完成时刻、LIFO 不影响正确性的原因、late registration 线性化点、退出条件和“没有
peer wake，所以 idle 仍需有界 GM 检查/backoff”的限制。

### 3.5 分阶段提交建议

1. 数据结构、初始化和 host/device ABI 单测。
2. completion enqueue/detach 模型与并发单测。
3. Executor 接入、退出协议和 DFX。
4. 文档、仿真、onboard Case1 和性能报告。

每阶段保持可编译。不要在同一提交混入无关格式化或用户工作树中的既有修改。

## 4. 正确性与测试验证

### 4.1 C++ 模型单测

扩展 `tests/ut/cpp/a5/test_aicore_ticket_scheduler_v1.cpp`，覆盖：

- 单 producer、单 waiter：completion 被消费一次，waiter 最终 READY。
- 多 Executor 并发 push 同一 inbox：无丢失、无重复，链中每个 Task 恰好一次。
- resolver 在 exchange 与 `completion_next` publish 之间摘取：识别 unpublished，最终继续。
- 注册 CAS 先于 close，以及 close 先于注册 CAS 两种 late-registration 时序。
- AIC → AIV → AIC 跨类型依赖链。
- 本地 inbox 消费与轮转 victim 批量偷取。
- LIFO completion 顺序下结果与依赖拓扑一致。
- 高 fan-out producer，一次 close 唤醒或重新挂接全部 waiter。
- worker 已 executor-drained 但仍持续 resolve，直到全局 resolver-drained。
- 随机 DAG 压测：每个 Task exact-once、无永久 BLOCKED、全部 wake head CLOSED。
- intrusive node 无容量溢出；重复 enqueue 和永久 unpublished 能被诊断。

本地 venv 和 C++ UT：

```bash
source .venv/bin/activate
pip install --no-build-isolation -e .
cmake -S tests/ut/cpp -B build/tests_ut_cpp
cmake --build build/tests_ut_cpp -j
ctest --test-dir build/tests_ut_cpp --output-on-failure
```

若仓库当前 C++ UT 已有标准 build 目录或 preset，下一会话应以 `CLAUDE.md`、testing skill 和 CI 的
最新命令为准，不并行创建冲突的 build tree。

### 4.2 A5 simulation

Case1 仅支持真实 A5，因此 simulation 用较小 case 验证功能和调度不变量：

```bash
source .venv/bin/activate
pytest tests/st/a5/host_build_graph/paged_attention_unroll \
  --platform a5sim --case SmallCaseMixedGroups
```

随后按可用时间运行完整 A5 host-build-graph simulation 集合。仿真只断言状态、输出、exact-once 和
无死锁，不对 wall time 设门槛。

### 4.3 Onboard 前的明确例外

本任务的用户明确要求：测试验证时跳过 A5 芯片预检。因此下一会话运行 Case1 时：

- **不要调用** `.claude/skills/onboard-arch-precheck/check.sh a5`。
- 不执行替代性的自动 silicon detection；把用户指派的独占设备视为 A5。
- 仍先检查 `command -v task-submit` 和 `task-submit --list`，并只通过 `task-submit` 锁定设备。
- 该例外只适用于本方案的这次验证，不应修改仓库默认 precheck 流程或 skill。
- 性能报告必须显式写明“按用户要求跳过 A5 precheck”，避免把该次运行误当成已自动验证芯片型号。

这是一项有意识的验证范围豁免：如果设备实际不是 A5，Case1 结果无效；出现架构相关编译/运行错误
时应停止并报告，不能通过修改 platform 参数绕过。

### 4.4 Case1 正确性运行

Case1 配置为 batch 256、16 heads、1 KV head、head dim 128、block 128、context 8192、
max model length 32768、BF16。先做一次保留 golden 的正确性运行：

```bash
RUN_DIR="$(mktemp -d -p tmp completion-inbox-case1.XXXXXX)"
mkdir -p "$RUN_DIR/ascend"
ASCEND_PROCESS_LOG_PATH="$RUN_DIR/ascend" \
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
  --run "source .venv/bin/activate && python \
tests/st/a5/host_build_graph/paged_attention_unroll/test_paged_attention_unroll.py \
-p a5 -d \$TASK_DEVICE --case Case1 --rounds 1"
```

记录 task-submit job id、实际 `TASK_DEVICE`、完整 git SHA、PTO ISA pin、编译命令、测试返回码和日志
目录。失败时保留临时目录，不自动删除。

### 4.5 必须满足的正确性门槛

- golden 比对通过，无设备异常、超时或 host hang。
- 执行 Task 数与图中 Task 数一致；Case1 当前预期为 65,536，若图生成逻辑变化则以本次 dump 为准。
- 每个 Task 只执行、enqueue、resolve 一次。
- 结束时无 BLOCKED/READY 残留，全部 Task 为 COMPLETE，全部 wake list CLOSED。
- 所有 inbox empty，无 worker-local detached list，无 unpublished link。
- emitted completion 数等于 resolved completion 数等于执行 Task 数。
- 跨 AIC/AIV 依赖和 PendingWait 都能收敛。

## 5. Case1 性能实验与报告交付

### 5.1 严格 A/B 基准

不要把既有文档中的约 35 ms 数值直接当作统计基线，因为设备、构建和环境可能不同。使用同一独占
设备、同一 PTO ISA pin、同一 ring heap/task window、同一编译选项，在一次 task-submit 锁内比较：

- A：完整基线对象 `b01dddda8db9329ff18613363f7becc3f9b5f5d4`。
- B：Completion Inbox 实现的完整提交 SHA。

为避免污染当前工作树，使用两个 detached worktree 和各自 venv/build 产物。例如：

```bash
WORKTREE_PARENT="$(mktemp -d -p tmp hbg-completion-ab.XXXXXX)"
BASELINE_WT="$WORKTREE_PARENT/baseline"
IMPL_WT="$WORKTREE_PARENT/implementation"
IMPL_SHA="$(git rev-parse HEAD)"
git worktree add --detach "$BASELINE_WT" b01dddda8db9329ff18613363f7becc3f9b5f5d4
git worktree add --detach "$IMPL_WT" "$IMPL_SHA"
```

不要在脚本中用 `rm -rf` 清理这些路径；验证结束后人工确认内容，再使用 `git worktree remove` 处理。

每个版本先 warmup 1 次，再至少采集 10 次、推荐 20 次 profiling-off 样本。正式采样使用
`--skip-golden`，但它不能替代前述 golden correctness run。A/B 按 ABBA 或 BAAB 交错，避免温度和频率
漂移系统性偏向某一版本。完整命令应放入报告附录，并继续遵守“跳过 A5 precheck、只通过
task-submit 独占设备”的要求。

### 5.2 调优矩阵

只在正确性通过后比较：

- fairness service interval：1、2、4、8。
- batch 不限额与小的固定 batch budget（仅当 DFX 显示长批阻塞本核工作）。
- inbox 映射：task-id hash 与 current-worker-local。

每项用独立 build/commit 标识，不增加生产环境变量。先用 3～5 次样本筛选，再对胜出配置做完整
20 次 A/B。优化目标首先是 Device wall，其次才是 queue lag 或 Scheduler 单项时间。

### 5.3 Profiling 和统计

profiling-off 正式样本完成后，每个版本再各运行一次
`--enable-chip-swimlane 1`，用于解释差异；profiling runner/export 开销不计入 Device wall。

报告至少给出：

- Device wall、Runner、full simpler_run 的 mean、标准差、min、p50、p95、max、95% CI。
- B 相对 A 的绝对差和百分比，正式样本成功率。
- Scheduler/Orchestrator/尾部开销分解，以及 PendingWait 时间或次数。
- completion enqueue-to-resolve lag、READY-to-kernel-start lag。
- own/steal batch 数、batch size 分布、每 inbox backlog、unpublished-link wait。
- wake close、waiter route、READY publish、emitted/resolved 数及任何协议错误。
- 编译器、PTO ISA pin、完整 SHA、设备 id、轮数、Case 参数、日志和原始结果路径。

如果性能回退，先区分：completion 投递成本、空闲核探测成本、batch 长尾、READY 发布变晚、退出尾部，
再决定修改；不能只根据总时间猜测。

### 5.4 报告文件

下一会话最终新增：

`doc_aic_sched/A5_HBG_COMPLETION_INBOX_CASE1_PERFORMANCE_REPORT.md`

并在已有 `doc_aic_sched/PAGED_ATTENTION_CASE1_PERFORMANCE_REPORT.md` 增加一条简短索引，避免继续把
历史总览膨胀成难维护的超长文档。新报告结构建议为：

1. 结论与是否接受该方案。
2. A/B 提交、环境和 A5 precheck 豁免声明。
3. 正确性结果。
4. profiling-off 统计表和原始样本。
5. profiling-on DFX 分解。
6. 风险、回退条件和下一步。

接受标准建议：全部正确性门槛通过；B 的 Device wall 均值和 p95 不劣于 A，或虽有轻微总时延回退
但能用稳定数据证明关键长尾显著改善并由维护者明确接受。若 B 回退超过噪声范围、出现 completion
积压/unpublished 异常、或需要高频全局轮询才能收敛，则保留基线 inline wake，实现不合入。
