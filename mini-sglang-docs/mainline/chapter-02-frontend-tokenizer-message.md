# Chapter 02：Frontend、消息协议与 Tokenizer/Detokenizer

本章聚焦请求进入系统后的第一段：HTTP 请求如何变成内部消息，tokenizer 如何编码，detokenizer 如何增量返回文本。

## 1. 本章要回答的问题

1. mini-sglang 内部有哪些核心消息类型？
2. `/generate` 和 `/v1/chat/completions` 如何构造请求？
3. tokenizer worker 如何把 `TokenizeMsg` 变成 `UserMsg`？
4. detokenizer 如何把 token 增量 decode 成字符串？

## 2. 消息类型总览

主线里会遇到四类核心消息：

| 消息 | 方向 | 定义位置 | 含义 |
| --- | --- | --- | --- |
| `TokenizeMsg` | Frontend → Tokenizer | `python/minisgl/message/tokenizer.py:34` | 待编码的 prompt/messages 和采样参数 |
| `UserMsg` | Tokenizer → Scheduler | `python/minisgl/message/backend.py:32` | 已编码的 `input_ids` 和采样参数 |
| `DetokenizeMsg` | Scheduler → Detokenizer | `python/minisgl/message/tokenizer.py:27` | 新生成 token 和 finished 标记 |
| `UserReply` | Detokenizer → Frontend | `python/minisgl/message/frontend.py:25` | 增量输出文本和 finished 标记 |

还有 abort 相关消息：`AbortMsg` 在 `python/minisgl/message/tokenizer.py:41`，`AbortBackendMsg` 在 `python/minisgl/message/backend.py:39`。

## 3. Frontend 构造 TokenizeMsg

`FrontendManager` 是在线 serving 模式下的前端状态管理器。它不直接做模型推理，也不直接做 tokenize / detokenize，而是负责把一个 HTTP 请求变成后端可处理的消息，并把后端不断返回的增量文本重新路由回对应的 HTTP 响应。

可以先把它理解成 frontend 侧的“请求路由表 + 等待队列”：

| 字段 | 含义 | 主线作用 |
| --- | --- | --- |
| `send_tokenizer` | 发往 tokenizer / detokenizer worker 的 ZMQ push queue。 | 把 `TokenizeMsg`、`AbortMsg` 发出去。 |
| `recv_tokenizer` | 从 detokenizer worker 接收 `UserReply` 的 ZMQ pull queue。 | 接收模型生成后的增量文本。 |
| `uid_counter` | 单调递增的请求 ID 计数器。 | 为每个 HTTP 请求分配唯一 `uid`，后续所有消息都靠它关联。 |
| `ack_map` | `uid -> List[UserReply]`。 | 暂存某个请求已经返回、但还没被 HTTP streaming 消费的增量回复。 |
| `event_map` | `uid -> asyncio.Event`。 | 当某个请求有新回复时，唤醒正在等待的 streaming coroutine。 |
| `initialized` | 是否已经启动后台 listen task。 | 确保接收 detokenizer 回复的监听任务只启动一次。 |

它的主线动作是：

```text
HTTP handler
  ↓ new_user(): 分配 uid，并初始化 ack_map / event_map
  ↓ send_one(TokenizeMsg): 把请求发给 tokenizer
  ↓ wait_for_ack(uid): 等待 detokenizer 回来的 UserReply
  ↓ stream_generate / stream_chat_completions: 把增量文本写回用户
```

所以 `FrontendManager` 的定位不是“业务逻辑层”，而是 frontend 和 tokenizer/detokenizer 之间的异步桥接层：它把同步语义的 HTTP 请求，转换成后端多轮异步返回的流式响应。

`/generate` endpoint 定义在 `python/minisgl/server/api_server.py:228`。它会：

```text
state.new_user()
  ↓
TokenizeMsg(uid, prompt, SamplingParams)
  ↓
state.send_one(...)
  ↓
StreamingResponse
```

其中 `SamplingParams` 来自 `python/minisgl/core.py:15`，包含 `temperature`、`top_k`、`top_p`、`ignore_eos`、`max_tokens`。

OpenAI 风格入口 `/v1/chat/completions` 定义在 `python/minisgl/server/api_server.py:255`。如果请求中有 `messages`，它会保留 messages 列表，让 tokenizer 后续应用 chat template；否则使用普通 `prompt`。

## 4. Tokenizer worker 主循环

`tokenize_worker()` 定义在 `python/minisgl/tokenizer/server.py:30`。

它是 tokenizer / detokenizer 进程里的消息分发循环，作用是把前端、后端之间的文本转换工作统一收口：

- 从 ZMQ 收到前端或后端发来的 tokenizer 侧 `BaseTokenizerMsg` 类型的消息。
- 把不同类型的消息交给对应的处理器。
- 把处理结果重新包装成下一跳需要的消息类型。
- 再通过 ZMQ 发给 scheduler 或 frontend。

主线里可以把它理解成三条转换通道：

```text
DetokenizeMsg → DetokenizeManager.detokenize → UserReply
TokenizeMsg   → TokenizeManager.tokenize     → UserMsg
AbortMsg      → AbortBackendMsg
```

这三条通道的职责分别是：

- `TokenizeMsg` 通道：把用户输入的 prompt 或 chat messages 编码成 `input_ids`，再包装成 `UserMsg` 发给 scheduler。
- `DetokenizeMsg` 通道：把 scheduler 生成的新 token 增量解码成文本，再包装成 `UserReply` 发给 frontend。
- `AbortMsg` 通道：把前端取消请求转换成 scheduler 能识别的 `AbortBackendMsg`。

所以 `tokenize_worker()` 是连接前端文本世界和后端 token 世界的边界层：进入模型前，它负责 text → token ids；模型生成后，它负责 token ids → incremental text。

## 5. TokenizeManager：prompt/messages → input_ids

`TokenizeManager` 定义在 `python/minisgl/tokenizer/tokenize.py:10`。

核心逻辑：

- 如果 `msg.text` 是 list，认为是 chat messages，调用 `tokenizer.apply_chat_template(...)`，位置在 `python/minisgl/tokenizer/tokenize.py:18`。
- 否则直接把 `msg.text` 当作 prompt。
- 最后调用 `tokenizer.encode(prompt, return_tensors="pt")`，位置在 `python/minisgl/tokenizer/tokenize.py:27`。
- 输出会 reshape 成一维并转为 `torch.int32`，位置在 `python/minisgl/tokenizer/tokenize.py:30`。

输出 `UserMsg` 的 `input_ids` 是 CPU 1D int32 tensor，定义在 `python/minisgl/message/backend.py:32`。

## 6. DetokenizeManager：next_token → incremental_output

`DetokenizeManager` 定义在 `python/minisgl/tokenizer/detokenize.py:63`。

它为每个 uid 维护一个 `DecodeStatus`：

- `decoded_ids`：已接收的 token ids。
- `decoded_str`：当前已稳定 decode 的字符串。
- `read_offset` / `surr_offset` / `sent_offset`：用于避免重复输出或输出半个 token/乱码字符。

这三个 offset 可以先这样理解：

| 字段 | 记录的是什么 | 解决什么问题 |
| --- | --- | --- |
| `read_offset` | 已经被读入并合并到 `decoded_str` 的 token 边界。 | 避免每次都从头 decode 全部 token，同时知道哪些 token 已经成为稳定文本。 |
| `surr_offset` | 本轮 decode 时保留的上下文 token 起点，通常会比 `read_offset` 更早一点。 | 给 tokenizer 留一小段前文上下文，避免 BPE / unicode 边界导致新 token 单独 decode 不准确。 |
| `sent_offset` | 已经发送给 frontend 的字符串长度。 | 每轮只返回 `output_str[sent_offset:]`，避免重复发送已经流式输出过的文本。 |

换句话说，`read_offset` 和 `surr_offset` 主要服务于“怎么稳定 decode”，`sent_offset` 主要服务于“怎么只返回增量文本”。

每次收到 `DetokenizeMsg` 后，它会：

```text
append next_token 到 decoded_ids
  ↓
tokenizer.batch_decode(read_ids)
  ↓
计算 new_text
  ↓
用 find_printable_text 过滤不完整片段
  ↓
返回 incremental_output
```

核心 decode 在 `python/minisgl/tokenizer/detokenize.py:88`，增量输出切片在 `python/minisgl/tokenizer/detokenize.py:105`。

## 7. Frontend 如何返回用户

`FrontendManager.listen()` 会持续接收 detokenizer 的 `UserReply`，位置在 `python/minisgl/server/api_server.py:116`。

`FrontendManager.wait_for_ack()` 按 uid 等待并 yield 增量 ack，位置在 `python/minisgl/server/api_server.py:134`。

这里的 ack 可以理解成 **后端对某个用户请求的一次增量回复确认**。它不是底层网络协议里的 ACK，而是 mini-sglang 在 frontend 内部使用的 `UserReply` 对象：

```text
UserReply
  ├─ uid：这段输出属于哪个用户请求
  ├─ incremental_output：本轮新增的文本片段
  └─ finished：这个请求是否已经结束
```

Frontend 需要 ack 的原因是：一个 HTTP 请求发出去后，模型不是一次性返回完整文本，而是每生成一小段就由 detokenizer 发回一个 `UserReply`。`FrontendManager.listen()` 负责把这些 `UserReply` 按 `uid` 放进对应请求的等待队列；`wait_for_ack()` 再把队列里的增量结果逐个 yield 给 HTTP streaming 逻辑。

可以把它理解成：

```text
detokenizer 发回 UserReply
  ↓
FrontendManager.listen 按 uid 收集
  ↓
wait_for_ack 逐个取出 ack
  ↓
stream_generate / stream_chat_completions 写给用户
  ↓
遇到 ack.finished=True 后结束流式响应
```

流式 `/generate` 使用 `stream_generate()`，位置在 `python/minisgl/server/api_server.py:152`；OpenAI chat stream 使用 `stream_chat_completions()`，位置在 `python/minisgl/server/api_server.py:160`。

## 8. 本章小结

请求 I/O 层可以压缩成：

```text
HTTP request
  ↓
TokenizeMsg
  ↓
TokenizeManager.tokenize
  ↓
UserMsg
  ↓
Scheduler / Engine
  ↓
DetokenizeMsg
  ↓
DetokenizeManager.detokenize
  ↓
UserReply
  ↓
Frontend SSE / JSON
```

下一章进入 Scheduler，看 `UserMsg` 进入后端后如何进入调度循环。
