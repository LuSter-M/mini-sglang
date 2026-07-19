# Chapter 03：Scheduler 初始化与主循环

Scheduler 是 mini-sglang 的在线推理中枢。本章先看 Scheduler 初始化了哪些管理器，再看默认 overlap loop 如何驱动一次又一次 forward。

## 1. 本章要回答的问题

1. Scheduler 初始化时创建哪些核心组件？
2. overlap scheduling 的循环结构是什么？
3. `UserMsg` 如何进入 prefill 队列？
4. 一轮 batch forward 前后 Scheduler 分别做什么？

## 2. Scheduler 初始化

`Scheduler` 定义在 `python/minisgl/scheduler/scheduler.py:45`，初始化入口是 `python/minisgl/scheduler/scheduler.py:46`。

它首先创建 `Engine(config)`，之后会创建几个管理器：

- `TableManager`：`python/minisgl/scheduler/scheduler.py:58`。
- `CacheManager`：`python/minisgl/scheduler/scheduler.py:59`。
- `DecodeManager`：`python/minisgl/scheduler/scheduler.py:62`。
- `PrefillManager`：`python/minisgl/scheduler/scheduler.py:63`。

这几个 manager 可以理解成 Scheduler 把复杂状态拆开的几个“小账本”：

| Manager | 管什么 | 为什么需要它 |
| --- | --- | --- |
| `TableManager` | 管 request table slot，以及每个 slot 对应的 `page_table` / `token_pool` 行。 | 每个请求都需要一个稳定的行号来保存 token ids 和逻辑位置到物理 KV cache 位置的映射。 |
| `CacheManager` | 管 KV cache 的空闲 pages、prefix cache 匹配、prefix 插入和 eviction。 | Scheduler 不直接操作大块 K/V tensor，而是通过它给请求分配或回收物理 KV 空间。 |
| `DecodeManager` | 管已经完成 prefill、正在 decode 的 running requests。 | decode 阶段每轮需要把所有还没结束的请求组成 batch，并估算它们未来还会占用多少 cache。 |
| `PrefillManager` | 管还没完成 prompt prefill 的 pending requests，包括 chunked prefill 请求。 | 新请求不能直接进入 decode，必须先完成 prompt 的 KV cache 构建；长 prompt 还可能被拆成多轮 prefill。 |

它们之间的关系可以粗略理解为：

```text
UserMsg
  ↓
PrefillManager：先进入 pending list，等待 prefill
  ↓
TableManager：为请求分配 table slot
  ↓
CacheManager：匹配 prefix cache，并为未缓存部分分配 KV pages
  ↓
Engine forward 完成 prefill
  ↓
DecodeManager：如果请求还没结束，加入 running set，后续继续 decode
```

更具体一点：

- `TableManager` 关心的是“这个请求在表里的哪一行”。这一行会同时用于 `token_pool` 和 `page_table`：前者保存 token id，后者保存每个 token 位置对应的物理 KV cache index。
- `CacheManager` 关心的是“这个请求的 token 应该写到哪些 KV cache 物理位置”。它会先尝试复用 prefix cache，复用不了的部分再分配新 pages。
- `PrefillManager` 关心的是“哪些请求还没把 prompt 算完”。它会根据 prefill token budget 选择本轮能处理多少请求、每个请求处理多少 token。
- `DecodeManager` 关心的是“哪些请求已经可以逐 token 生成”。它维护 running 请求集合，并在没有 prefill batch 时组成 decode batch。

所以 Scheduler 本身不把所有状态揉在一个大对象里，而是把请求生命周期拆成几类资源账本：table slot、KV cache、prefill 队列、decode 集合。后面看到 `_schedule_next_batch()`、`_prepare_batch()`、`_process_last_data()` 时，基本都是在更新这些账本。

还会加载 tokenizer 并记录 eos：

- tokenizer：`python/minisgl/scheduler/scheduler.py:69`。
- eos token：`python/minisgl/scheduler/scheduler.py:70`。

最后调用 `SchedulerIOMixin` 初始化通信，位置在 `python/minisgl/scheduler/scheduler.py:76`。

## 3. 为什么 Scheduler 需要自己的 stream

初始化时 Scheduler 创建了一个独立 CUDA stream，位置在 `python/minisgl/scheduler/scheduler.py:51`。

可以这样理解：

```text
Scheduler stream：准备 metadata、处理上一轮结果
Engine stream：执行模型 forward
```

默认 overlap 模式会让 CPU/metadata 处理与 GPU 计算尽量重叠。

## 4. run_forever：选择 normal loop 或 overlap loop

`run_forever()` 定义在 `python/minisgl/scheduler/scheduler.py:120`。

如果环境变量禁用 overlap scheduling，则不断调用 `normal_loop()`；否则默认调用 `overlap_loop()`：

```text
if DISABLE_OVERLAP_SCHEDULING:
    while True: normal_loop()
else:
    data = None
    while True: data = overlap_loop(data)
```

`normal_loop()` 更直观，定义在 `python/minisgl/scheduler/scheduler.py:108`；`overlap_loop()` 是默认优化路径，定义在 `python/minisgl/scheduler/scheduler.py:83`。

两者做的事情本质上相同，都是：收消息、调度 batch、执行 forward、处理输出。区别在于这些步骤是否重叠：

| loop | 执行方式 | 特点 |
| --- | --- | --- |
| `normal_loop()` | 本轮调度、forward、结果处理按顺序串行完成。 | 控制流更直观，适合理解和调试；但 GPU forward 期间，CPU 侧调度和结果处理更难被隐藏。 |
| `overlap_loop()` | 本轮先调度并发起当前 batch 的 forward，然后处理上一轮 batch 的结果。 | 用流水线方式重叠 CPU metadata 处理和 GPU 计算，吞吐更好；但理解时需要区分“当前 batch”和“上一轮 batch”。 |

可以用一张简化图理解：

```text
normal_loop:
  schedule N → forward N → process N

overlap_loop:
  iteration N:   schedule N   → launch forward N   → process N-1
  iteration N+1: schedule N+1 → launch forward N+1 → process N
```

所以阅读源码时可以先看 `normal_loop()` 建立顺序心智模型，再看 `overlap_loop()` 理解它如何把同样的步骤改造成流水线。

## 5. overlap_loop 的五步

`overlap_loop(last_data)` 做五件事：

1. 从 tokenizer / 其他 rank 接收消息：`python/minisgl/scheduler/scheduler.py:95`。
2. 处理消息：`python/minisgl/scheduler/scheduler.py:96`。
3. 调度下一批 batch：`python/minisgl/scheduler/scheduler.py:98`。
4. 在 engine stream 发起 forward：`python/minisgl/scheduler/scheduler.py:101` 到 `python/minisgl/scheduler/scheduler.py:103`。
5. 处理上一轮 forward 的结果：`python/minisgl/scheduler/scheduler.py:105`。

这也是“overlap”的关键：当前 batch 的 GPU forward 发出去后，Scheduler 再处理上一轮 batch 的 token、释放资源、发送结果。

## 6. UserMsg 进入系统

`UserMsg` 是 tokenizer 发给 scheduler 的“已完成编码的用户请求”。它已经不再是字符串，而是：

```text
UserMsg
  ├─ uid：请求 ID
  ├─ input_ids：prompt 对应的 token ids
  └─ sampling_params：max_tokens / temperature / top_k / top_p / ignore_eos 等生成参数
```

对应消息处理入口是 `Scheduler._process_one_msg()`，位置在 `python/minisgl/scheduler/scheduler.py:169`。

Scheduler 收到 `UserMsg` 后，并不会立刻把它送进模型。它会先做一层入口检查和请求登记：

```text
UserMsg
  ↓
检查 prompt 长度是否还能放进模型上下文
  ↓
根据剩余上下文修正 max_tokens
  ↓
交给 PrefillManager.add_one_req
  ↓
进入 prefill pending list
```

这里最重要的点是：**`max_tokens` 可能会被截断。** 如果 prompt 已经占用了大部分上下文，Scheduler 会把请求级别的最大生成长度限制在模型剩余上下文以内，避免后续写 KV cache 时越界。

这段逻辑在 `python/minisgl/scheduler/scheduler.py:175` ：先检查 `UserMsg`，再计算剩余输出长度，最后调用 `prefill_manager.add_one_req(msg)`。

所以 `UserMsg` 进入 Scheduler 后，本质上完成的是“从外部请求变成内部待调度请求”的转换。

## 7. 一轮 batch 的准备与执行

当 Scheduler 决定本轮要跑一个 batch 时，它要把“请求对象”整理成 Engine 和 attention backend 能直接消费的张量。这一步可以分成两个阶段：**选 batch** 和 **准备 batch**。

### 7.1 选 batch：本轮跑 prefill 还是 decode

当前策略是 prefill 优先：

```text
如果 PrefillManager 里有可运行请求：
    组成 prefill batch
否则：
    从 DecodeManager 里组成 decode batch
```

对应入口是 `Scheduler._schedule_next_batch()`，位置在 `python/minisgl/scheduler/scheduler.py:219`。它先尝试 `prefill_manager.schedule_next_batch(...)`，失败后再尝试 `decode_manager.schedule_next_batch()`。

prefill batch 和 decode batch 都会被包装成 `Batch`，但含义不同：

| batch 类型 | 包含哪些请求 | 本轮要计算什么 |
| --- | --- | --- |
| prefill batch | 还没完成 prompt prefill 的请求 | prompt 中尚未缓存的一段 token |
| decode batch | 已完成 prefill、仍在生成中的请求 | 每个请求当前最后一个 token |

### 7.2 准备 batch：把请求状态变成模型输入

选出 batch 后，Scheduler 会准备几类关键数据：

| 数据 | 作用 |
| --- | --- |
| `positions` | 每个输入 token 在原请求里的位置，用于 RoPE / position 相关计算。 |
| `input_mapping` | 告诉 Scheduler 应该从 `token_pool` 的哪些位置取本轮输入 token。 |
| `write_mapping` | 告诉 Scheduler 采样出的 next token 应该写回 `token_pool` 的哪些位置。 |
| `batch.out_loc` | 本轮输入 token 的 K/V 应该写入哪些物理 KV cache 位置。 |
| attention metadata | attention backend 运行需要的 batch 边界、page table、sequence length 等信息。 |

这些数据主要在 `Scheduler._prepare_batch()` 中生成，位置在 `python/minisgl/scheduler/scheduler.py:204`。其中 `positions`、`input_mapping`、`write_mapping`、`batch.out_loc` 和 attention metadata 会在同一段准备逻辑里完成。

这一步是 Scheduler 和 Engine 之间的关键接口。Scheduler 不直接调用模型层，而是把所有运行期信息打包成 `ForwardInput`：

```text
ForwardInput
  ├─ batch：本轮请求集合和 batch 类型
  ├─ sample_args：采样参数张量
  ├─ input_tuple：从 token_pool 取输入 token 的索引
  └─ write_tuple：把 next token 写回 token_pool 的索引
```

### 7.3 执行 forward：token_pool → Engine → token_pool

真正执行时，Scheduler 会先根据 `input_mapping` 从 `token_pool` 取出本轮输入 token，填到 `batch.input_ids`，然后调用 Engine：

```text
token_pool[input_mapping] → batch.input_ids
  ↓
engine.forward_batch(batch, sample_args)
  ↓
next_tokens_gpu 写回 token_pool[output_mapping]
  ↓
decode_manager.filter_reqs
```

这段执行逻辑对应 `Scheduler._forward()`，位置在 `python/minisgl/scheduler/scheduler.py:227`。

这里有一个容易混淆的点：

- `batch.out_loc` 是给 attention backend 写 KV cache 用的。
- `write_mapping` 是给 Scheduler 写 next token 到 `token_pool` 用的。

也就是说，本轮 forward 同时产生两类“写回”：一类是模型内部每层 attention 写 K/V 到 KV cache；另一类是 Scheduler 把采样出的 token id 写回 `token_pool`，供下一轮 decode 使用。

## 8. 处理 forward 结果

Engine 返回后，Scheduler 还不能只把 token 发出去就结束。它还要更新请求状态、判断是否结束、维护 prefix cache，并把输出交给 detokenizer。

结果处理入口是 `Scheduler._process_last_data()`，位置在 `python/minisgl/scheduler/scheduler.py:138`。

处理流程可以理解为：

```text
等待 next_tokens_cpu 可用
  ↓
遍历 batch.reqs
  ↓
跳过 ChunkedReq
  ↓
把 next_token 追加到 req 的 host-side input_ids
  ↓
判断请求是否 finished
  ↓
生成 DetokenizeMsg(uid, next_token, finished)
  ↓
根据 finished / prefill 状态更新资源
  ↓
send_result 发给 detokenizer
```

这里有几个关键细节。

### 8.1 为什么要跳过 ChunkedReq

`ChunkedReq` 表示这个请求的 prompt 还没 prefill 完。本轮 forward 只是计算了 prompt 的一段，并不能把采样结果当成真正的生成 token 返回给用户。

所以 chunked prefill 的请求在结果处理阶段会被跳过：它要等后续几轮把完整 prompt prefill 完，才会进入正常生成流程。

代码上可以对照 `python/minisgl/scheduler/scheduler.py:147` 到 `python/minisgl/scheduler/scheduler.py:149`：遍历 batch 请求时遇到 `ChunkedReq` 会直接 `continue`。

### 8.2 finished 怎么判断

一个请求结束通常有两类原因：

```text
达到 max_tokens
  或
生成 eos，且 ignore_eos=False
```

`Req.can_decode` 负责表达“是否还有生成预算”；eos 则来自 tokenizer 的结束 token。只要满足结束条件，Scheduler 就会把这次 `DetokenizeMsg` 标成 `finished=True`，让 detokenizer 和 frontend 知道这条流可以结束。

对应判断在 `python/minisgl/scheduler/scheduler.py:153` 到 `python/minisgl/scheduler/scheduler.py:156`。

### 8.3 资源怎么更新

结果处理时资源状态分两种：

| 场景 | Scheduler 做什么 |
| --- | --- |
| 请求 finished | 从 decode set 移除，释放 table slot，并把可复用前缀交给 CacheManager 处理。 |
| prefill 完成但未 finished | 把新完成的 prefix 写入 prefix cache，并继续保留请求资源，后续进入 decode。 |

这里的“释放”并不等于简单清空所有 KV cache。对于已经形成可复用前缀的部分，CacheManager 会尝试放进 prefix cache；真正不能复用或已经不需要的尾部 KV pages 才会回到 free list。

资源更新可以对照 `python/minisgl/scheduler/scheduler.py:159` ，释放请求资源的封装函数是 `_free_req_resources()`，位置在 `python/minisgl/scheduler/scheduler.py:200`。

### 8.4 为什么最后发 DetokenizeMsg

Scheduler 产出的是 token id，不直接产出字符串。它会把每个请求的新 token 包装成 `DetokenizeMsg` 发给 detokenizer：

```text
DetokenizeMsg
  ├─ uid：属于哪个用户请求
  ├─ next_token：本轮生成的 token id
  └─ finished：这个请求是否结束
```

后续 detokenizer 会把 token id 变成增量文本，再交给 frontend 返回用户。

`DetokenizeMsg` 的构造和发送分别可以对照 `python/minisgl/scheduler/scheduler.py:156` 和 `python/minisgl/scheduler/scheduler.py:167`。

## 9. 本章小结

Scheduler 的主线可以记成：

```text
初始化：Engine + TableManager + CacheManager + PrefillManager + DecodeManager

循环：
receive_msg
  ↓
process UserMsg / Abort / Exit
  ↓
schedule prefill or decode
  ↓
prepare batch metadata
  ↓
engine.forward_batch
  ↓
process next token and resource state
```

下一章具体看 prefill / decode batch 是如何被选出来的。
