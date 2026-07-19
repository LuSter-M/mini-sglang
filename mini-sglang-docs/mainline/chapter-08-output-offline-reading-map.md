# Chapter 08：结果回传、离线 LLM 与源码阅读地图

前面七章已经把一次请求完整走了一遍：前端接收 HTTP、tokenizer 切成 token、Scheduler 排进 prefill / decode、KV cache 管理显存、Engine 执行模型 forward、attention 计算注意力、LM head 输出 logits、sampler 选出 next token。

还剩最后一段：这个 next token 如何变回用户看到的文字。本章补齐这段回传链路，并说明离线 `LLM.generate()` 为什么能复用同一套 Scheduler，最后给出一份继续阅读源码的地图。

## 1. 本章要回答的问题

1. Engine 采样出的 next token 以什么形式返回给 Scheduler？
2. 同一个 token 为什么要同时写进 GPU 的 `token_pool` 和 CPU 的 `Req.input_ids`？
3. 一个请求在什么时候被判定为 finished？
4. finished 之后，table slot、KV pages、prefix cache 分别如何处理？
5. Detokenizer 如何把零散的 token id 拼成增量文本？
6. 离线 `LLM.generate()` 和在线服务究竟共用了哪些代码？
7. 若要继续精读源码，按什么顺序读最高效？

## 2. 结果回传的整体链路

先把整条链路铺开，避免只盯着某一行代码：

```text
Engine.forward_batch()
  ↓
Sampler.sample(logits)
  ↓
ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
  ↓
Scheduler._forward(): 写回 token_pool，供下一轮 decode 读取
  ↓
Scheduler._process_last_data(): 写回 Req.input_ids，判断 finished，构造 DetokenizeMsg
  ↓
Tokenizer worker: DetokenizeManager.detokenize()
  ↓
UserReply(incremental_output, finished)
  ↓
FrontendManager.listen() / wait_for_ack()
  ↓
HTTP streaming response
```

整条链路里最容易被忽略的一点：**next token 会被写往两个方向，用途完全不同**。

- 写往 GPU（`token_pool`）：让下一轮 decode 直接从显存里读到上一步的输出 token 作为输入。
- 写往 CPU（`next_tokens_cpu`）：拷回内存、追加到请求状态、发给 detokenizer，最终变成用户看到的文字。

所以 `next_tokens_gpu` 和 `next_tokens_cpu` 不是同一份数据复制两遍，而是一个服务于"继续推理"、一个服务于"返回用户"，各走各的路径。

## 3. Engine 返回 ForwardOutput

Engine 只负责一件事：在 GPU 上执行完当前 batch 的 forward 和采样。它不处理用户请求状态，执行完就把一个轻量的 `ForwardOutput` 返回给 Scheduler。

`ForwardOutput` 定义在 `python/minisgl/engine/engine.py:23`，三个字段：

| 字段 | 作用 |
| --- | --- |
| `next_tokens_gpu` | 保留在 GPU 上的采样结果，随即写回 `token_pool` 支撑下一轮 decode。 |
| `next_tokens_cpu` | 异步拷到 CPU 的同一批结果，供 Scheduler 更新 req 状态、回传 tokenizer。 |
| `copy_done_event` | 一个 CUDA event，用来告诉 Scheduler：CPU 那份拷贝在什么时候真正完成。 |

执行入口是 `Engine.forward_batch()`，位于 `python/minisgl/engine/engine.py:191`。它的步骤不多，但每一步都有讲究：

- **进入 `Context.forward_batch(batch)`**：把当前 batch 挂到全局 Context 上，之后模型层、attention backend、KV cache 写入逻辑都依赖它获取本轮 metadata。
- **在 CUDA Graph replay 和普通 `model.forward()` 之间二选一**：batch 形状匹配就重放提前 capture 好的图，省去 Python 调度和 kernel launch 的开销；否则走普通 forward。
- **对每个 req 调用 `req.complete_one()`**：把 `cached_len` 推进到 `device_len`，再让 `device_len += 1`。含义是：本轮进入模型的 token 已经算完，可以作为历史上下文，同时为下一个 token 预留位置。
- **`sampler.sample(...)`**：按各自的采样参数，为每个请求选出 next token。
- **异步把 next token 拷到 CPU**：GPU 那份继续供 decode 使用，CPU 这份留给 Scheduler 判断结束、发送 detokenize 消息。
- **记录一个 `copy_done_event`**：标记异步拷贝的完成时刻，overlap loop 依据它在安全的时间点读取 CPU 结果。
- **打包成 `ForwardOutput` 返回**。

## 4. Scheduler 写回 token_pool

拿到 Engine 的结果后，Scheduler 做的第一件事不是发给用户，而是先把 GPU 上的 next token 写回 `token_pool`。

`token_pool` 是 GPU 侧的一块 token 存储区，贯穿整个生命周期：

- prefill：从 prompt 里取出待处理 token，放进模型输入；
- decode：每个请求新生成的 token 写回它自己序列里的位置；
- 下一轮 decode 组 batch 时，再从这个位置读出 token 作为输入。

正因为读和写发生在同一块区域，`Scheduler._forward()`（`python/minisgl/scheduler/scheduler.py:227`）里才有两个方向相反的 mapping：

```text
input_mapping  : 从 token_pool 的哪些位置读 token，作为本轮模型输入
output_mapping : 把本轮采样得到的 next token 写回哪些位置
```

`_forward()` 的四步很直白：按 `input_mapping` 从 `token_pool` 取出 token 拼成 `batch.input_ids` → 交给 `engine.forward_batch()` 在 GPU 上执行 → 按 `output_mapping` 把 `next_tokens_gpu` 写回去 → 最后把已经没有生成预算的请求从 decode 候选中剔除。

`output_mapping` 来自 `_make_write_tuple()`（`python/minisgl/scheduler/scheduler.py:262`），规则只有一句话：还能 decode 的请求，写到 `req.device_len` 对应的位置；不能 decode 的，写到 `-1` 这个 sentinel，表示这个 token 到此为止，不再作为后续模型输入。

需要提醒的是：写 `token_pool` 是为后续 GPU 推理服务的，与用户可见的输出不是一回事。用户可见的文本要等 CPU 拷贝完成、再经过 detokenizer 才会产生。

## 5. Scheduler 处理上一轮结果

overlap scheduling 的核心手法，是让"当前 batch 在 GPU 上 forward"和"上一轮 batch 在 CPU 上收尾"两件事并行进行，用后者的时间把前者的 CPU 开销（结果处理、detokenize 消息构造、资源回收）隐藏掉。

主循环在 `Scheduler.overlap_loop()`（`python/minisgl/scheduler/scheduler.py:83`），一轮的流程如下：

```text
接收新消息，更新 prefill / decode 队列
  ↓
选出下一批 batch
  ↓
在 Engine stream 上启动这一批的 forward
  ↓
回过头处理上一轮 batch 的 next token 和 finished 状态
  ↓
把当前 batch 的 ForwardData 传给下一轮
```

所以 overlap 不是两个 batch 在同一个 Engine 里同时 forward，而是**当前 batch 的 GPU 计算**和**上一轮 batch 的 CPU 收尾**在时间上重叠。

收尾逻辑在 `_process_last_data()`（`python/minisgl/scheduler/scheduler.py:138`）。它先用 `copy_done.synchronize()` 等 CPU 那份 token 落地，然后遍历上一轮 `batch.reqs`：ChunkedReq 直接跳过（它只负责补 KV，不产出用户 token）；其余请求把 next token 追加进 CPU 侧 `req.input_ids`，判断是否 finished，并累积一条 `DetokenizeMsg`。收尾动作视情况而定：finished 的请求从 decode 队列移除、回收资源；prefill 且未结束的，把 prefix 放入 prefix cache。最后统一调用 `send_result(reply)` 全部发出。

```text
等 next_tokens_cpu 完成拷贝
  ↓
遍历上一轮 batch.reqs（跳过 ChunkedReq）
  ↓
next_token 追加到 CPU 侧 req.input_ids
  ↓
判断 finished，累积一条 DetokenizeMsg
  ↓
finished → 出队 + 释放资源；prefill 未 finished → prefix 入 cache
  ↓
send_result(reply)
```

## 6. finished 判断与资源释放

一个请求会 finished，无非两种原因：

1. **长度耗尽**：`Req.can_decode` 为 false，已经没有剩余生成预算。
2. **命中 EOS**：采样到 `eos_token_id`，且请求没有设置 `ignore_eos`。

`Req.can_decode`（`python/minisgl/core.py:60`）本质上就是检查 `remain_len > 0`。

finished 之后走 `_free_req_resources()`（`python/minisgl/scheduler/scheduler.py:200`），做两件事：

1. 让 `TableManager` 收回这个请求占用的 `table_idx`，把运行态槽位让给新请求。
2. 处理 KV cache，把可复用的有效前缀插入 prefix cache，再释放不再需要的尾部 pages。

这里有个常见误解需要澄清：finished **不等于**把它占用的 KV pages 全部立即清空。mini-sglang 会尽量保留可复用的前缀：

```text
请求结束
  ↓
收回运行态 table slot
  ↓
可缓存 prefix 插入 radix tree
  ↓
只释放不可复用 / 尾部多余的 pages
```

这样做的回报是：后续请求若命中相同前缀，就能直接复用已有 KV cache，显著减少 prefill 的计算量。

## 7. Detokenizer 到 Frontend

Scheduler 发出的不是文本，而是 `DetokenizeMsg`。

`DetokenizeMsg`（`python/minisgl/message/tokenizer.py:27`）只有三个字段：`uid`（属于哪个请求）、`next_token`（本轮的 token id）、`finished`（这个 token 之后请求是否结束）。

detokenizer worker 收到消息后调用 `DetokenizeManager.detokenize()`（`python/minisgl/tokenizer/detokenize.py:70`）。这一步的难点不在 `decode([next_token])`，恰恰相反，逐个 token 单独 decode 反而会出错。它真正要做的是为每个请求维护一份流式 decode 状态：

- 每个 `uid` 对应一个 `DecodeStatus`，记录这个请求已 decode 到哪里、已经发出多少文本。
- 新 token 追加进 `decoded_ids`，以便 tokenizer 基于更完整的序列做 decode。**唯一例外**：如果这个 token 是终止 EOS（`finished and next_token == eos_token_id`），则不追加，避免把 EOS 也 decode 成可见文本。
- 对一段 token 做 `batch_decode`，而不是只 decode 单个，因为单 token decode 会把词边界切错。
- 过滤不可打印或不完整的文本，避免输出半个词、半个汉字或残缺的 Unicode 替换字符。
- 只计算 `incremental_output`，也就是本轮新增的那段文本，专供 streaming 使用。
- 请求结束后删除这份 decode 状态。

worker 把结果包装成 `UserReply`（`python/minisgl/tokenizer/server.py:71`）发回前端。前端的 `FrontendManager.listen()`（`python/minisgl/server/api_server.py:116`）持续接收 `UserReply`，收到后放入 `ack_map[uid]` 并唤醒等待该请求的 async generator。随后 `wait_for_ack()` 逐条 yield 回复；streaming 接口再由 `stream_generate()` 把增量文本包装成 SSE 数据返回。

整体可以记为：

```text
DetokenizeMsg(token id)
  ↓
DetokenizeManager(token id → 增量文本)
  ↓
UserReply(incremental_output)
  ↓
Frontend ack_map / event_map
  ↓
HTTP streaming chunk
```

## 8. 在线模式与离线 LLM.generate

在线服务和离线 `LLM.generate()` 的差异并不在推理主线，而在 I/O。主线部分（调度、KV cache、Engine、模型、采样）两者完全一致。

在线：

```text
HTTP request → FrontendManager → tokenizer worker → Scheduler
  → Engine / Model / KV cache → detokenizer worker → FrontendManager → HTTP response
```

离线：

```text
LLM.generate(prompts) → pending_requests → offline_receive_msg() → Scheduler
  → Engine / Model / KV cache → offline_send_result() → tokenizer.decode(output_ids)
  → return [{"text", "token_ids"}]
```

`LLM`（`python/minisgl/llm/llm.py:28`）直接继承 `Scheduler`：初始化时用一份 `offline_mode=True` 的单卡 TP `SchedulerConfig` 启动，无需拉起完整的服务端进程拓扑；同时在对象内存里维护 `pending_requests` 和 `status_map` 两个队列，保存待处理请求和生成状态。

`offline_mode=True` 会整体替换 Scheduler 的输入输出：

- `offline_receive_msg()` 从 `pending_requests` 取出 prompt、转成 `UserMsg`：替代在线 tokenizer worker 的输入链路。
- `offline_send_result()` 接收 `DetokenizeMsg`、把 token id 追加进内存中的 request 状态：替代在线 detokenizer / frontend 的回传链路。

这个替换发生在 `SchedulerIOMixin` 初始化阶段（`python/minisgl/scheduler/io.py:30`）：一旦 `offline_mode=True`，`receive_msg` 就指向 `offline_receive_msg`，`send_result` 指向 `offline_send_result`，其余保持不变。

`LLM.generate()`（`python/minisgl/llm/llm.py:77`）本身很简短，流程如下：

1. 先清空上一轮的 pending / status，避免残留状态干扰本次调用。
2. 把 prompt 和 `SamplingParams` 放入 `pending_requests`。
3. 调用 `run_forever()` 复用 Scheduler 主循环，让离线请求同样经历 prefill、decode、forward、采样。
4. 当所有请求处理完毕后，`offline_receive_msg()` 会在没有新请求且处于 blocking 时抛出 `RequestAllFinished`，`generate()` 捕获后退出循环。
5. 最后对累积的 `output_ids` 做 `tokenizer.decode`，返回 `{"text", "token_ids"}`。

一句话概括两者关系：

- **复用**：Scheduler 主循环、prefill / decode 调度、TableManager / CacheManager / prefix cache、Engine、模型 forward、attention backend、sampler。
- **替换**：不走 ZMQ、不启动独立的 tokenizer / detokenizer worker、不走 HTTP streaming，输入输出全部在 `LLM` 对象内存中完成。

## 9. 推荐源码阅读顺序

若要继续深入源码，建议不要按文件名顺序读，而是按"请求主线"读，效率更高。下面分六遍，每一遍都有明确目标和"读完应能回答什么"。

### 9.1 第一遍：在线请求闭环

目标：弄清一个 HTTP 请求如何变成 token，又如何变回 HTTP 响应。

- `python/minisgl/__main__.py`：服务启动入口，命令行启动后进入的是哪条主线。
- `python/minisgl/server/launch.py`：服务端如何把 frontend、tokenizer、scheduler 组装起来。
- `python/minisgl/server/api_server.py`：HTTP 请求如何进入系统，streaming response 如何返回用户。
- `python/minisgl/message/*.py`：建立一张消息类型地图，看清组件之间传递的对象形态。
- `python/minisgl/tokenizer/server.py`：tokenizer worker 如何同时处理 tokenize、detokenize、abort 三种消息。
- `python/minisgl/tokenizer/tokenize.py`、`detokenize.py`：文本与 token id 的双向转换。

读完应能回答：`TokenizeMsg`、`UserMsg`、`DetokenizeMsg`、`UserReply` 分别在哪些组件之间流动。

### 9.2 第二遍：Scheduler 主循环

目标：请求进入后，如何进入 prefill / decode 队列，又如何组成 batch。

- `python/minisgl/scheduler/scheduler.py` 的初始化：Scheduler 如何一次性创建 Engine、TableManager、CacheManager、PrefillManager、DecodeManager。
- `scheduler.py` 的主循环：请求接收、batch 调度、Engine forward、上一轮结果处理如何串成一圈。
- `python/minisgl/scheduler/prefill.py`：prompt 请求如何进入 prefill 队列，chunked prefill 如何切分。
- `python/minisgl/scheduler/decode.py`：完成 prefill 的请求如何进入 decode 队列持续生成。
- `python/minisgl/core.py`：`Req`、`Batch`、`SamplingParams`、`Context` 这几个核心数据结构。

读完应能回答：`Req.cached_len`、`device_len`、`extend_len`、`remain_len` 各是什么，以及 prefill batch 和 decode batch 的输入形态区别何在。

### 9.3 第三遍：Table、KV cache 与 prefix cache

目标：理清 token、page table、KV cache 三者的关系。

- `python/minisgl/scheduler/table.py`：每个请求如何获得 `table_idx`，`token_pool` / `page_table` 的外层管理形态。
- `python/minisgl/scheduler/cache.py`：Scheduler 侧如何分配、释放 pages，如何维护 prefix cache。
- `python/minisgl/kvcache/*.py`：Engine / attention backend 真正读写的 KV cache pool 结构。
- 回到 `cache.py` 看 radix cache：可复用 prefix 如何插入 radix tree，以及为什么要考虑 page 对齐。

读完应能回答：`token_pool` 存什么、`page_table` 存什么、`batch.out_loc` 为什么能指导 attention backend 写 KV cache、prefix cache 为什么通常只缓存 page 对齐的前缀。

### 9.4 第四遍：Engine 与模型执行

目标：Scheduler 选出的 batch 如何真正运行到 GPU 模型里。

- `python/minisgl/engine/engine.py`：Engine 如何初始化 GPU 执行环境、如何执行 `forward_batch()`。
- `python/minisgl/engine/graph.py`：哪些 decode batch 能走 CUDA Graph，graph replay 如何减少调度开销。
- `python/minisgl/engine/sample.py`：logits 如何经过 greedy / top-k / top-p / temperature 变成 next token。
- `python/minisgl/core.py` 里的 Context：模型 forward 为什么能通过 global context 拿到当前 batch 和运行时资源。
- `python/minisgl/models/qwen2.py`、`qwen3.py`：embedding、decoder layer、attention、MLP、norm、LM head 的具体结构。
- `python/minisgl/models/utils.py`：权重加载、TP 切分及各类模型工具函数。

读完应能回答：模型 forward 为什么无需层层传递 `batch` 参数，而是通过 global context 直接拿到 `batch.input_ids`、`positions`、`out_loc` 和 attention backend。

### 9.5 第五遍：Attention backend、TP 与 kernel

目标：厘清性能相关的路径。

- `python/minisgl/layers/attention.py`：模型层如何把 q/k/v、RoPE、KV cache metadata 交给 backend。
- `python/minisgl/attention/base.py`：backend 的统一接口和 metadata 抽象。
- `python/minisgl/attention/fa.py`：FlashAttention backend 的 prefill / decode 处理。
- `python/minisgl/attention/fi.py`：FlashInfer backend 如何做 paged attention。
- `python/minisgl/attention/trtllm.py`：TRT-LLM backend 的接入方式。
- `python/minisgl/distributed/*.py`：TP rank、进程组、通信、all-reduce 等分布式基础。
- `python/minisgl/kernel/*.py`：项目自定义 kernel 各自优化的性能路径。

读完应能回答：prefill 和 decode 为什么使用不同的 attention 计算形态，以及 TP rank 之间在什么时候需要通信。

### 9.6 第六遍：离线模式

目标：把在线和离线放在一起对照。

- `python/minisgl/llm/llm.py`：离线 `LLM.generate()` 如何构造请求、运行 Scheduler、收集输出。
- `python/minisgl/scheduler/io.py`：`offline_mode=True` 时，输入输出如何被替换成离线路径。
- 回到 `python/minisgl/scheduler/scheduler.py`：对照确认两者复用的是同一套调度和推理主线。

读完应能回答：为什么 `LLM.generate()` 无需启动 HTTP / tokenizer worker / detokenizer worker，也能复用同一套主线。

## 10. 主线检查问题

整套文档读完后，下面这些问题至少应能顺利回答：

1. `TokenizeMsg`、`UserMsg`、`DetokenizeMsg`、`UserReply` 分别在哪些进程或组件之间流动？
2. 为什么每个 TP rank 都要有独立的 Scheduler 和 Engine？
3. `PrefillManager` 和 `DecodeManager` 各自维护什么状态？
4. `Req.cached_len`、`device_len`、`extend_len`、`remain_len` 分别是什么？
5. `token_pool` 和 `page_table` 分别保存什么？
6. `batch.out_loc` 是如何被 attention backend 用于写 KV cache 的？
7. Radix prefix cache 为什么只插入 page 对齐的前缀？
8. `Engine.forward_batch()` 为什么必须先设置 `Context.forward_batch(batch)`？
9. 模型 forward 为什么能从 global context 中获取 `batch.input_ids`？
10. LM head 和 sampler 在普通 prefill、chunked prefill、decode 中的结果分别如何使用？
11. `next_tokens_gpu` 和 `next_tokens_cpu` 分别服务于哪条路径？
12. 在线模式和离线 `LLM.generate()` 复用了什么、又替换了哪些 I/O？

## 11. 本章小结

这一章补齐了最后一段链路：

```text
logits
  ↓
sampler
  ↓
next_tokens_gpu / next_tokens_cpu
  ↓
token_pool 写回，支撑下一轮 decode
  ↓
Req.input_ids 写回，支撑状态更新和回传
  ↓
DetokenizeMsg → incremental_output → UserReply → Frontend streaming
```

至此，mini-sglang 一次请求的主线完整闭环：

```text
用户输入
  ↓
tokenize
  ↓
Scheduler 组织 prefill / decode batch
  ↓
Table / KV cache / prefix cache 管理运行态资源
  ↓
Engine 在 TP rank 的 GPU 上执行模型 forward
  ↓
attention backend 读写 KV cache
  ↓
LM head + sampler 产出 next token
  ↓
Scheduler 写回状态、判断 finished、回收资源
  ↓
detokenize
  ↓
用户看到增量输出
```

若只需记住一句话：**在线和离线的差异几乎全在 I/O 层；调度、KV cache、Engine、模型 forward 和采样这条主线，两者是同一套代码。**
