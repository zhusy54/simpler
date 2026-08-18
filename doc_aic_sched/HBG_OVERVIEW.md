# HBG（host_build_graph）运行时方案介绍

> 本文基于 `main` 分支代码总结，介绍 simpler 的两大 L2 芯片级运行时之一 ——
> **HBG（host_build_graph）** 的整体设计、执行模型、关键数据结构、与
> TRB（`tensormap_and_ringbuffer`）的差异，以及图执行（Graph Execution）、两帧流水线、
> 依赖生成等扩展机制。源码位于 `src/{a2a3,a5}/runtime/host_build_graph/`，
> 架构原理对应 `docs/chip-level-arch.md` 与 `src/{arch}/runtime/host_build_graph/docs/`。

## 1. 概述：HBG 是什么

simpler 在 L2（单芯片）层提供**两个运行时**，二者共享同一套调度器（Scheduler）、
环形缓冲（Ring）与共享内存布局，**唯一区别在于「编排器（Orchestrator）在何时运行」**：

| 运行时 | 缩写 | 编排器运行位置 | 编排器与调度器的关系 |
| ------ | ---- | -------------- | -------------------- |
| `host_build_graph` | **HBG** | **Host CPU 上一次性运行到完成** | 设备上**没有**编排线程，全部 AICPU 线程都做调度 |
| `tensormap_and_ringbuffer` | TRB | **AICPU 线程 N-1 在设备上运行** | 编排线程与调度线程（0..N-2）**在设备上并发** |

HBG 的名字即来自其核心约束：**Host 端把整张任务图先构建完成，再整体下发到设备**。
设备的 AICPU 不再负责构建图，只负责调度与派发。这是 HBG 的「定义性约束」——
在任何一个设备任务完成之前，Host 已经把完整的图构建完毕并下发。

这一选择带来的连锁后果构成了 HBG 的全部特征：

- 设备没有编排线程，所有 AICPU 线程都是调度线程，参与对自己负责的 AICore 核分区的分类与派发；
- 依赖图（DAG）在 Host 进程内被完整、有序地观察到，因此可以**就地、无损地记录**真实的依赖路径（见 §10 dep_gen）；
- Host 可以在编排期间读写外部张量的宿主视图（`get_tensor_data` / `set_tensor_data`），因为此时设备调度尚未启动；
- 同一次 run 内，编排与调度不重叠；跨 run 的流水线化（prepared successor）则可以在 run N 执行时准备 run N+1。

## 2. 三程序模型与编译产物

HBG 遵循 simpler 的「三程序」模型（Host `.so` + AICPU `.so` + AICore `.o`），独立编译、
运行时链接。`build_config.py` 定义了四个编译目标及其源文件归属：

| 目标 | 编译产物 | 包含的源目录（相对 `runtime/host_build_graph/`） |
| ---- | -------- | ------------------------------------------------ |
| `host` | 设备编排与管理的 host 共享库（DeviceRunner / Runtime / 编排核） | `host/`、`runtime/orchestrator_core/`、`runtime/shared/`、`orchestration/` |
| `aicpu` | AICPU 设备调度程序 | `aicpu/`、`runtime/`、`orchestration/` |
| `aicore` | AICore 计算核（每个 `func_id` 一个 `.cpp` → `.o`） | `aicore/`、`orchestration/` |
| `orchestration` | 用户编排函数 `.so`（`dlopen` 加载） | `orchestration/` |

要点：

- **`orchestration/` 源目录被同时编入 host、aicpu、orchestration 三个目标**。它提供
  编排函数与运行时都需要的方法（如 `ChipTensor` 构造校验、张量方法），保证三端类型一致。
- **编排 `.so` 对运行时 `.cpp` 零链接依赖**：编排源码只包含
  `pto_orchestration_api.h`，所有运行时调用都通过内嵌在 `PTO2Runtime` 中的
  **函数指针表 `PTO2RuntimeOps`** 完成（`submit_task` / `scope_begin/end` /
  `orchestration_done` / `alloc_tensors` / `graph_begin/end` 等）。这使得同一个编排 `.so`
  既能在 Host（HBG）被调用，也能在设备（TRB）被调用。

## 3. 运行生命周期

一次 HBG run 的端到端流程如下（见 `docs/RUNTIME_LOGIC.md` §1–2 与 `docs/task-flow.md`）：

```text
Host register  : 物化编排字节 → 临时 .so → dlopen(RTLD_LOCAL) → 解析
                 aicpu_orchestration_config / aicpu_orchestration_entry / framework_bind_runtime
        ↓
Host bind/run  : 校验并 staging 外部张量到 run 私有的 host accessor
        ↓
Host orchestrate: 同步调用编排入口函数，编排函数通过 rt_submit_* 构建整张 DAG
        ↓
Host finalize graph : 确定 task 总数与图镜像，把 per-slot 任务/负载指针重写为设备地址
        ↓
Host H2D       : 把共享内存镜像 + 运行时 arena 拷贝到设备
        ↓
Device boot    : 最高编号的 AICPU 线程 attach 已填充的 arena（不 reset），发布 boot 屏障
        ↓
Device schedule: 所有 AICPU 线程分类/派发各自核分区，完成任务后关核
        ↓
Device teardown: 最后到达的线程销毁已 attach 的运行时，发布清理资格；唯一一个返回线程认领并 reset
        ↓
Host collect   : 收集输出，销毁/重置 per-run 状态
```

关键不变量：

- **boot 线程 attach arena 时不做 reset**，arena 由 Host 预先填充。`INIT_READY` 之前，
  每个 chip 子进程在启动 mailbox 中公布其运行时契约的 `pipeline_depth`。
- **清理资格只在销毁之后发布**，防止 `deinit()` 与运行时 arena 或本次 run 的 host accessor 竞争。
- 子 AIC/AIV callable 被单独上传，其解析出的设备函数地址填入 per-run 派发表。

## 4. Host 端构图细节

对每一次 run，Host 依次执行（`docs/RUNTIME_LOGIC.md` §2.2）：

1. 校验并把外部张量 staging 到 run 私有的 host accessor；
2. 为运行时/共享内存子区预留一个 backing arena；
3. 把运行时绑定到编排 DSO；
4. **同步调用编排入口**（此时设备调度尚未启动）；
5. 确定 task 总数与图镜像；
6. 把 per-slot 的任务/负载指针重写为设备地址；
7. 把共享内存镜像与 arena 拷贝到设备。

编排期间的任何 fatal 会停止该序列，并通过 `orch_error_code` 上抛。

编排函数通过函数指针表调用以下原语（`pto_orchestration_api.h`）：

- `rt_submit_task(MixedKernels, CoreTaskArgs)` —— 通用任务提交（AIC / AIV / MIX / SPMD）；
- `rt_submit_aic_task` / `rt_submit_aiv_task` —— 单核类型快捷提交；
- `rt_submit_dummy_task` —— 仅参与依赖、不跑核的同步屏障；
- `rt_submit_graph(...)` —— 提交一个图执行任务（见 §9）；
- `rt_scope_begin/end`、`alloc_tensors`、`get_tensor_data` / `set_tensor_data`；
- `rt_available_cluster_count` / `rt_available_aiv_count` —— 本 run 的 MIX cluster 数与独立 AIV 核数（**禁止硬编码 24/36**）。

> **重要**：`get_tensor_data` / `set_tensor_data` 操作的是外部张量的宿主视图，只在编排构建图期间有效。
> 已提交设备 producer 产出的张量在构图期间不可读，运行时创建的图堆输出没有宿主视图，
> 二者都会作为非法参数上报。

## 5. 预构建图镜像（Prebuilt Graph Image）

整张图被序列化为一组 POD、位置无关的结构，三类 per-slot 结构构成共享镜像（`docs/RUNTIME_LOGIC.md` §3）：

| 结构 | 职责 |
| ---- | ---- |
| `PTO2TaskDescriptor` | 完整 task ID、kernel ID、packed-buffer 地址 |
| `PTO2TaskPayload` | 张量、标量、谓词、local-ID fanin、派发元数据 |
| `PTO2TaskSlotState` | active mask、属性、block/subtask 计数、完成状态、task/payload 绑定 |

Host↔设备边界是 **POD + 位置无关**的：fanin 用**整数 producer ID**而非指针表示；
唯一的 per-slot 指针在下发前被重绑定到设备地址。这满足「一次 `memcpy` 到设备、零指针修正」的契约。

### 5.1 有界 H2D 上传

共享内存镜像按 ring 容量（task window）开空间，但一次 run 只写 `[0, total_tasks)`，
设备以 scheduler-only 方式启动、不会读到 `total_tasks` 之后的 SM slot。因此每 run 的 SM H2D
是**按工作量有界**的，而非按容量——这正是让 `bind` 开销与工作量成正比的契约。

- **共享内存**：header 在 Host 清零；`descriptors / payloads / slot_states / completion_flags`
  每个 task 在 submit 时逐个写入，H2D 上界为 `[0, total_tasks)`。每个 slot 在 `orch::prepare_task`
  认领时按需初始化（**没有全窗口重置**）。
- **运行时 arena**：
  - **编排块**（`fanin_seen_epoch` / `scope_tasks` / TensorMap，约 8.5 MB）**完全不下发**——
    它是 Host 专用的依赖计算 scratch，AICPU 调度器对它零引用。
  - **调度块及之后**（ready-queue slot 池、runtime header、completion mailbox）**整块下发**。
    ready queue **不**按 `total_tasks` 收窄：图执行会回放一个缓存的 GRAPH 任务，
    设备 Scheduler 把它展开成多个设备端节点，这些节点会把 ready queue 推过 host task 数，
    因此队列 slot 必须全部携带有效的 Vyukov 序列。

`bind_callable_to_runtime_impl` 在上传前 `always_assert` 校验 `orch_start <= orch_end`，
使未来 `runtime_reserve_layout` 的任何把调度块挪到编排块之前的重排都会直接 fault，而非下发错位镜像。

## 6. 全图容量模型（Whole-Graph Capacity）

HBG 使用**一个任务环、一个图堆、一个 TensorMap 池**，它们是容量受限的存储，而非流式流控缓冲
（`docs/RUNTIME_LOGIC.md` §4）：

- run 进行中 `last_task_alive` 不前进；
- `heap_tail` 不在 run 中退役任务输出缓冲；
- task slot 与堆字节**不在 run 中途回收**；
- TensorMap 表项在 host 构图期间不回收。

`completed_watermark` 记录已完成设备任务的连续前缀，仅用于完成/消费者元数据，不回收任务环或堆。
没有 post-run 扫描让图空间可复用；运行时销毁会释放整个 arena，下一次 run 从全新初始化的镜像开始。

### 6.1 分配失败

图必须能装进配置的 task window、heap、fanin capacity 和 TensorMap pool。当分配无法推进时，
一个墙钟 backstop 会 latch 一个 fatal，而不是空等一个尚未启动的调度器：

```text
FATAL: Task Allocator Deadlock - Heap Exhausted!
  Task ring:  current=..., last_alive=..., active=.../...
  Heap ring:  top=..., tail=..., size=..., available=...
  Requested:  ... bytes
```

这是 host 编排侧日志；allocator 记录对应的运行时错误并 unwind，不直接终止进程。

## 7. 提交与依赖（Submission and Dependencies）

### 7.1 混合任务与逻辑 block

active mask 选择 AIC、AIV0、AIV1 三条 lane。`block_num` 是逻辑 SPMD 宽度，sync-start 未请求时可超过
物理设备宽度，调度器按波次派发。slot 分配/发布前要求：

```text
block_num >= 1
block_num * popcount(active_mask) <= INT16_MAX
```

该乘积存入 16 位 `total_required_subtasks`。sync-start 另加共驻留限制：AIV 任务用可用 AIV 数，
AIC/MIX 任务用可用 cluster 数。

### 7.2 TensorMap 与 fanin

TensorMap 把张量区域映射到 producer task ID。对每个任务：

1. INPUT/INOUT 区域查找重叠的 producer；
2. 显式与发现的 producer 去重进 `fanin_local_ids[]`；
3. OUTPUT/INOUT 区域把新任务注册为 producer；
4. 每个 producer 记录其最高消费者 local ID，用于完成元数据。

**没有 fanout 邻接表或依赖池**——per-slot 完成标志就是设备上的就绪真相。

## 8. 设备端调度模型（Boot Classification & Wake Lists）

**submit 不把任务推进 ready queue**。图到达设备后，boot 分类（`docs/RUNTIME_LOGIC.md` §6）对每个已提交
任务**恰好扫描一次**：

- fanin 全部完成的任务 → 路由到它的形状队列；
- 否则 → 注册到第一个未满足 producer 的侵入式 **wake list**；
- producer 完成 → 重新分类它唤醒的所有 waiter。

完成标志单调，因此这种**消费者拉取（consumer-pull）方案不会漏掉 producer 状态翻转，也不需要周期性依赖轮询**。

可派发形状为 `AIC`、`AIV`、`MIX`；仅依赖的 `DUMMY` 任务用专用队列，无需 AICore 派发即可完成。

> **早 producer 传播（early staging）在 HBG 中当前被关闭**。共享调度器保留了该代码以与 TRB 保持对等，
> 但 HBG 的 boot 分类器与 wake list 是实际生效的就绪路径。

派发与完成（§7）：

- AIC/AIV 派发从 `next_block_idx` 认领逻辑 block 索引区间，宽任务未完则重新入队；
- MIX 派发选择「已用 lane 共享一个合法 placement」的 cluster offset，tracker 用 128-bit bitset
  （因扁平 offset 为 `cluster * 3`，在支持的设备上超过 bit 63）；
- sync-start cohort 尽量本地 staging，更宽的 ownership span 用 generation-tagged 全局 drain；
- 每条 lane 完成递增 `completed_subtasks`，当该计数等于 `block_num * popcount(active_mask)` 时任务完成；
- 完成设置任务标志、推进连续 `completed_watermark`、并重新分类其 wake-list 消费者。

drain 的 `pending_task` 在完整尝试期间保持有效：所有参与线程在协调者越过 stage-done 屏障并清除它之前
就加载它；活跃路径依赖该不变量。

## 9. 图执行（Graph Execution）—— HBG 独有的复合任务

Graph Execution **仅在 HBG 可用**（`docs/GRAPH_EXECUTION.md`）。一个 Graph 是一个复合 incore 任务：
像 AIC/AIV/MIX/SPMD 任务一样被提交并完成一次，但内部包含一个被录制的任务 DAG。

### 9.1 录制与缓存

- 每次调用在 host 任务窗口里只放**恰好一个** `GRAPH` 任务。
- **首次调用**在 ring 之外录制 DAG：内部 submit 只构建 host-only 节点元数据、预留 scratch 输出缓冲，
  而不消耗 task-window slot，随后从新建的 Definition 发射外层 `GRAPH` 任务。
- **后续调用**复用缓存的 Definition，直接发射同一个 `GRAPH` 任务。
- 设备 Scheduler 展开保存的拓扑并派发内部节点；Host 编排器从不把这些内部节点作为 ring 任务提交。
- 命中不支持构造的录制会被丢弃，函数体回退到普通 task-submit 路径，确保该次工作仍然被提交。

### 9.2 Definition（POD 拓扑镜像）

录制使用 host-only C++ 状态（`std::vector` / `std::unordered_map` / `std::unique_ptr`）。
在 `graph_end` 时，录制被压缩成一个连续、无指针的 POD Definition，包含：

- 节点顺序与 AIC/AIV/MIX/SPMD kernel 元数据；
- `root_indices` + 双向不可变拓扑（fanin CSR 与 fanout CSR）；
- 每个节点一个 packed-heap offset；
- 每个节点的 ChipTensor 来源：`BOUNDARY_EXACT` / `BOUNDARY_VIEW` / `INTERNAL` / `OWN_OUTPUT`；
- 固定标量值 + boundary-scalar 源索引；
- 固定 boundary 签名与 alias 代表。

header 还携带完整 Definition 镜像的内容哈希。设备执行池要求 **哈希、Graph key、节点数三者全部匹配**
才能复用一个已驻留的 Definition（key-only 复用不安全，因为新 run 可能在同一函数身份下录制不同元数据）。
所有引用都是相对 Definition 基址的 32-bit offset。

缓存最多存 **16 个 Definition**，每个按其实际序列化大小分配；不存在按最大尺寸复制的固定录制数组。
缓存是 per-run 且从空开始，因此目前没有持久化 schema 版本。

### 9.3 缓存命中与内存

缓存命中时，Host 编排器：校验固定 boundary 契约 → 预留一个 task-window slot → 预留一块足够大的堆块
（容纳所有内部中间量）→ 只计算外部 fanin 与 boundary tensormap 效应 → 发射一个外层 `GRAPH` 任务 →
staging 精确大小的 POD 提交镜像 → 向 host 运行时申请一个对齐执行块（大小来自录制节点数、
Tensor 地址与标量 patch 容量、Definition 字节数），把该设备地址写进提交 wire 镜像。

**内部节点不消耗 ring task-window slot**——它们的 descriptor/payload/slot state 建在 host 拥有的 GM 中。
运行时为每个 `(pipeline slot, Graph key, occurrence index)` 保留一个 grow-only 块；重复 run 在同一 slot
复用分配，同一 key 在一次 run 内多次使用得到不同块，两个 pipeline slot 永不共享活跃块。
块的前序 Definition key 与内容哈希匹配时，保留本地 Definition、静态节点字段、Tensor 地址与标量 patch 表
（graph-affine replay 跳过拓扑绑定、分类、校验、静态字段存储等），只刷新动态状态。

### 9.4 设备侧 Scheduler 流程

Graph 被放进两个独立控制流：`graph_prepare_queue`（即使外部 fanin 未就绪也可物化保存节点）与
`graph_ready_queue`（外层 Graph 的外部 fanin 就绪信号）。核持有 Scheduler 线程每轮循环各 pop 最多一项；
一次 prepare 最多展开 4 个节点并重新入队，把图展开与普通调度交错。准备与外部就绪在同一个原子激活门里
设两个 bit，**谁设第二个 bit 就恰好一次地激活保存的根节点**。

内部依赖就绪借用完成状态轮询思想，但依赖接线仍是编排器职责：

- 录制构造 fanin/fanout CSR；首次物化构建静态 runnable 节点状态 + 紧凑 Tensor 地址/标量 patch 表；
- 物化把每个非根节点注册到其保存 fanin CSR 中选出的一个 producer 上；
- 节点的 release/acquire `task_state` 是其 Graph-local 完成标志，内部节点既不需要 ring 完成标志也不需要 task-window slot；
- producer 完成只关闭并 drain 当前 wake-list，而非遍历保存的 fanout CSR；
- 被 wake 的消费者扫描其保存的 fanin CSR，进入形状队列或注册到下一个未完成 producer；
- `WAKE_LIST_SENTINEL` 关闭完成/注册竞争：注册失败者观测到完成并立即重扫。

最终内部完成会完成唯一的外层 Graph 任务、发布外层 ring 完成标志、唤醒外部消费者，并为 host 可见完成计数贡献 1。

## 10. 两帧流水线与准备-激活协议（Two-Frame Staging）

直连 A2/A3 chip 端点在协商深度 ≥ 2 时使用两个 task frame，并通告 `supports_frame_staging`
（`docs/task-flow.md` §"Two-frame endpoint staging lane"）。一个 `WorkerThread` 拥有两帧，
通过非阻塞 progress 接口驱动它们；子进程也只有一个循环服务控制流量、两个 task frame 与有界的
active/prepared native 生命周期——**没有每帧一线程**。

```text
IDLE -> TASK_READY    -> FRAME_STAGED -> TASK_LAUNCHED -> TASK_DONE | TASK_FAILED
IDLE -> PREPARE_READY -> FRAME_STAGED -> ACTIVATE -> TASK_LAUNCHED
                                              -> TASK_DONE | TASK_FAILED
```

- `FRAME_STAGED` 表示子进程持有一个不可变 frame 快照，**它本身不区分**「仅校验 staging」与「已完成 native 准备」。
- 对一个 HBG successor，若其配置与活跃 predecessor 都非 diagnostic，子进程在 predecessor 执行期间
  于租用的非活跃 arena bank 中构建一个 generation-bound native run；若 predecessor 已 active，
  该准备在 `FRAME_STAGED` 发布前完成。
- 若 successor 在任何 predecessor 认领 active claim 之前到达子进程，则只发布「仅校验」状态，
  使父进程可以无死锁激活它；native prepare 在激活之后或后续 predecessor claim 时再进行，
  无需再次 mailbox 状态翻转。
- 临近 diagnostic 状态的 HBG 任务与所有 TRB 任务也把 `FRAME_STAGED` 用作「仅校验」状态：
  它们的共享 diagnostic/设备 scratch 状态在早期重写不安全，native prepare 必须等 predecessor 的完整设备 fence。

HBG successor 的 prepared token 在 `ACTIVATE` 之前**保持未 launch、未 accept**，而激活仍要等 predecessor
poll 完成并 finalize 后才能 launch 它；sticky acceptance word 因此在整个准备期间保持为 0。
shutdown、过期激活、pre-launch 失败会在 frame 终止前**恰好一次** finalize 该 token。

调度器只对 prepared FIFO successor 中**第一个**合格的单一 NEXT_LEVEL 任务做 staging；活跃 run 的任务只用
活跃 lane，因此第二帧不会造成同设备执行重叠。Remote、SUB、A5、simulation、nested-worker 与单帧端点保留
阻塞兼容路径。

> 注：**A5 HBG 不发布 pipeline 契约，停留在深度 1**；A5 TRB 虽为 whole-run 准入发布深度 2，
> 但本地端点仍只有一个 mailbox frame、设备执行仍串行——第二预约可 build 成 `PREPARED`，
> 但不会得到第二个并发可执行设备帧。

## 11. 依赖生成（dep_gen_host_graph）

`dep_gen_host_graph.h/.cpp` 是 HBG 专属的依赖图捕获机制。因为 HBG 在 Host 进程内把整段编排
运行到完成（任何 `submit_task` 与 tensormap 命中都按序、在进程内可见），图可以从**真实依赖路径**
（`compute_task_fanin` 的 emit/annotate 钩子）记录，而非像 TRB 那样从捕获的 submit 流回放重建。

捕获接口（编排侧）：

```text
begin_capture()         — 每次 orchestration 一次
begin_task()            — 每次 submit 一次，在其依赖步骤之前
add_explicit_edge()     — STEP 1，每个声明显式依赖
add_creator_edge()      — STEP 3A，每个 creator-retention producer
add_tensormap_edge()    — STEP 3B，每个 tensormap producer
end_task()              — 关闭该任务
```

控制接口（device runner 调用）：`set_enabled()` / `active()` / `emit()`。

特性：

- 图存于**线程局部状态**，捕获无锁；emit 读取调用线程状态，因此 run 的编排（构图）与其 drain（emit）
  必须落在同一线程。子进程 progress loop 是单线程，满足此不变量；若被破坏 emit 返回 -3。
- per-task producer 去重镜像 `PTO2FaninBuilder`（按 (ring, slot) 键），这里按 producer task id 键。
  二者一致仅因 HBG 是全图驻留、build 期不复用 task slot。
- 输出为 `deps.json`，schema 与 TRB replay 相同（见 `docs/dfx/dep-gen.md`），所有下游消费者
  （deps viewer、swimlane join）对两种运行时输出一视同仁。

## 12. 目录结构与关键文件

HBG 在 `a2a3` 与 `a5` 两套架构下是近乎完全相同的近邻复制。以 a2a3 为例：

```text
src/a2a3/runtime/host_build_graph/
├── build_config.py                 # 四个编译目标与源目录归属
├── common/                         # intrinsic、pto_runtime_status
├── orchestration/                  # 编进 host/aicpu/orchestration 三端的共享源
│   ├── pto_orchestration_api.h     # 编排 .so 的唯一对外头：ops 函数指针表 + rt_* 内联包装
│   ├── pto_arg_with_deps.h         # CoreTaskArgsWithDeps<N> 便利层
│   └── common.cpp
├── host/                           # Host 侧：DSO 物化、构图、张量访问
│   ├── runtime_maker.cpp           # 编译/链接运行时，dlopen 编排 .so
│   ├── dep_gen_host_graph.cpp      # Host 侧依赖图捕获实现
│   ├── host_tensor_access.cpp      # get/set_tensor_data
│   ├── runtime_compile_info.cpp
│   └── host_orch_compat_stubs.cpp
├── aicore/aicore_executor.cpp      # AICore 执行器（每个 func_id 一个核）
├── aicpu/aicpu_executor.cpp        # AICPU 执行器（boot、分类、派发）
├── runtime/                        # 运行时核心
│   ├── runtime.h / common.h        # POD 共享内存结构、PTO2TaskDescriptor/Payload/SlotState
│   ├── pto_runtime2.h / *.cpp      # PTO2Runtime 完整定义、运行时主体
│   ├── pto_orchestrator.h          # 编排器：submit_task、Ring、TensorMap、Scope
│   ├── pto_tensormap.h             # 张量区域 → producer 映射
│   ├── pto_ring_buffer.h           # 任务环 + per-scope 堆分配器（带 back-pressure）
│   ├── pto_dep_compute.h           # compute_task_fanin 依赖计算
│   ├── pto_completion_token.h      # 完成标志 / wake-list
│   ├── graph_cache.h / graph_execution.h / graph_host_state.h   # 图执行：Definition 缓存与物化
│   ├── dep_gen_host_graph.h
│   ├── scheduler/                  # 设备调度器（HBG/TRB 共享代码）
│   │   ├── pto_scheduler.{h,cpp}
│   │   ├── graph_execution.cpp     # GRAPH 任务的设备端展开
│   │   ├── scheduler_dispatch.cpp / scheduler_completion.cpp / scheduler_cold_path.cpp
│   │   └── scheduler_types.h / scheduler_context.h
│   ├── orchestrator_core/          # 编排核心实现（submit/scope/tensormap 主循环）
│   │   ├── pto_orchestrator.cpp / pto_ring_buffer.cpp / pto_runtime2.cpp
│   ├── shared/                     # 共享初始化/共享内存/tensormap/runtime 主体
│   ├── backend/sdma/               # SDMA 完成核/调度器
│   └── aicore_completion_mailbox*.h
└── docs/                           # RUNTIME_LOGIC / GRAPH_EXECUTION / SCALAR_DATA_ACCESS /
                                    # SUBMIT_BY_CLUSTER / profiling_levels / device_log_profiling
```

## 13. 与 TRB 的对照小结

| 维度 | HBG（host_build_graph） | TRB（tensormap_and_ringbuffer） |
| ---- | ----------------------- | ------------------------------- |
| 编排器位置 | Host CPU，一次性运行完成 | AICPU 线程 N-1，设备上运行 |
| 设备线程角色 | 0..N-1 **全部**调度 | 0..N-2 调度，N-1 编排 |
| 编排↔调度关系 | 串行：先构图后下发 | 设备上并发 |
| dep_gen 来源 | 真实依赖路径就地无损记录 | 捕获 submit 流后回放重建 |
| 临时张量 staging | per-run `device_malloc/free` | 每个 pipeline slot 保留的 retained buffer（始终开启） |
| pipeline_depth 契约 | A5 不发布（深度 1）；A2/A3 协商 ≥ 2 时支持两帧 staging | A5 发布深度 2 但仍串行执行 |
| early producer 传播 | **关闭**（boot 分类 + wake list 为就绪路径） | 保留 early staging |
| Graph Execution | **支持**（独有复合任务） | 不支持 |
| 共享部分 | 同一 Scheduler、Ring、共享内存布局、TensorMap、编排 `.so` | 同左 |

## 14. 错误与诊断

运行时把编排与调度错误 latch 进共享内存，并映射为 host 观察到的负 run 状态（`docs/RUNTIME_LOGIC.md` §9）：

- 非法参数 `-5`；sync-start 驻留违规 `-7`；tensor 等待超时 `-8`；scheduler 超时 `-100`。

设备日志只含调度器记录；host 构图诊断保留在 host 侧。详情见 `docs/device_log_profiling.md`。
profiling 命名中 `HBG_` 前缀即代表 host_build_graph 运行时（`docs/dfx/profiling-config-naming.md`）。

HBG 与 TRB 共用同一套 `507018`/分配器 fatal 的分类与超时 backstop（`PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES`，
约 500 ms），详见 `.claude/rules/running-onboard.md` 与 `docs/troubleshooting/device-error-codes.md`。

## 15. 验证方法

运行时 C++ 改动需要重建 editable 包，再跑两种仿真变体（`docs/RUNTIME_LOGIC.md` §10）：

```bash
pip install --no-build-isolation -e .
pytest examples tests/st --platform a2a3sim --runtime host_build_graph
pytest examples tests/st --platform a5sim   --runtime host_build_graph
```

纯调度器/core-tracker 与生命周期原语另有 `tests/ut/cpp` 下的 C++ 单测；图执行场景位于
`tests/st/a2a3/host_build_graph/graph_execution`（AIV fanin/fanout DAG、Qwen 风格 AIV/AIC
decoder-layer DAG、三槽 multi-block MIX/SPMD Graph、手动 Qwen3-14B 三层 decode）。

---

### 参考文档

- `docs/chip-level-arch.md` —— L2 单芯片三程序模型
- `docs/hierarchical-level-runtime.md` —— L0–L6 层级模型与组件组合
- `docs/task-flow.md` —— Callable / TaskArgs / CallConfig 数据流、两帧流水线
- `docs/orchestrator.md` / `docs/scheduler.md` / `docs/worker-manager.md` —— 三大引擎组件内部
- `src/{a2a3,a5}/runtime/host_build_graph/docs/RUNTIME_LOGIC.md` —— HBG 运行时设计（本文主要依据）
- `src/{a2a3,a5}/runtime/host_build_graph/docs/GRAPH_EXECUTION.md` —— 图执行机制
- `src/{a2a3,a5}/runtime/host_build_graph/build_config.py` —— 编译目标与「何时运行编排器」的权威注释
