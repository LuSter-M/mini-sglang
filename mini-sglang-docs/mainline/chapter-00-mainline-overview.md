# Chapter 00：mini-sglang 主线总览

本文用于快速建立 mini-sglang 的整体地图：它不是 nano-vLLM 那种单进程薄封装，而是更接近真实在线 serving 系统的多进程架构。

## 1. 一句话主线

一次在线生成请求的核心流程是：

```text
HTTP / OpenAI API request
  ↓
FrontendManager 分配 uid，发送 TokenizeMsg
  ↓
Tokenizer worker: prompt/messages → input_ids → UserMsg
  ↓
Scheduler rank0 收 UserMsg，并广播给其他 TP rank
  ↓
PrefillManager / DecodeManager 组成 Batch
  ↓
Scheduler 准备 page_table、token_pool、positions、attention metadata
  ↓
Engine.forward_batch: model.forward → sampler
  ↓
Scheduler 处理 next_token、判断结束、更新 cache，发送 DetokenizeMsg
  ↓
Detokenizer 增量 decode 成 UserReply
  ↓
Frontend 以 SSE / JSON 返回用户
```

源码入口可从 `python/minisgl/__main__.py:1` 和 `python/minisgl/server/launch.py:40` 开始看。

## 2. 系统组件地图

在线模式的进程拓扑大致是：

```text
API Server / Frontend
  ├─ tokenizer worker(s)
  ├─ detokenizer worker
  └─ scheduler worker × TP size
       └─ Engine
            ├─ model
            ├─ KV cache pool
            ├─ page_table
            ├─ attention backend
            ├─ sampler
            └─ CUDA graph runner
```

`launch_server()` 会启动这些子进程：scheduler 进程在 `python/minisgl/server/launch.py:59` 创建，detokenizer 在 `python/minisgl/server/launch.py:73` 创建，tokenizer workers 在 `python/minisgl/server/launch.py:88` 创建。

## 3. 主线对象职责

| 对象 | 主线职责 | 所在模块 |
| --- | --- | --- |
| `FrontendManager` | 维护用户 `uid`、请求 ack、SSE 输出和 abort。 | `python/minisgl/server/api_server.py` |
| `TokenizeManager` | 把 prompt 或 chat messages 转成 token ids。 | `python/minisgl/tokenizer/tokenize.py` |
| `DetokenizeManager` | 维护每个 uid 的增量 decode 状态，把 token 流转成文本增量。 | `python/minisgl/tokenizer/detokenize.py` |
| `Scheduler` | 收消息、调度 prefill/decode、准备 batch、调用 engine、处理结果。 | `python/minisgl/scheduler/scheduler.py` |
| `PrefillManager` | 维护待 prefill 请求，处理 prefix cache 命中和 chunked prefill。 | `python/minisgl/scheduler/prefill.py` |
| `DecodeManager` | 维护 running decode 请求，并组成 decode batch。 | `python/minisgl/scheduler/decode.py` |
| `CacheManager` | 管理空闲 KV pages、prefix cache 匹配、插入和 eviction。 | `python/minisgl/scheduler/cache.py` |
| `TableManager` | 管理 request table slot、`page_table` 和 `token_pool`。 | `python/minisgl/scheduler/table.py` |
| `Engine` | 初始化模型、KV cache、attention backend、sampler 和 CUDA graph，并执行 forward。 | `python/minisgl/engine/engine.py` |

## 4. 最核心的 Scheduler 闭环

主线里最值得反复看的循环是 `Scheduler.overlap_loop()`：

```text
receive_msg
  ↓
_process_one_msg
  ↓
_schedule_next_batch
  ↓
_forward
  ↓
_process_last_data
```

对应代码在 `python/minisgl/scheduler/scheduler.py:83`。默认 overlap scheduling 会让本轮 GPU forward 和上一轮 CPU 结果处理重叠，以隐藏调度开销。

如果关闭 overlap，则走 `normal_loop()`，入口在 `python/minisgl/scheduler/scheduler.py:108`。

## 5. Prefill 与 Decode 的区别

| 阶段 | 请求来源 | 本轮输入 | 主要目标 |
| --- | --- | --- | --- |
| prefill | `PrefillManager.pending_list` | prompt 中尚未缓存的一段 token | 建立 KV cache，并为第一枚输出 token 计算 logits |
| decode | `DecodeManager.running_reqs` | 每个请求当前最后一个 token | 继续逐 token 生成 |

调度入口在 `python/minisgl/scheduler/scheduler.py:219`。当前策略是 prefill 优先：先调用 `PrefillManager.schedule_next_batch()`，没有 prefill batch 时才调用 `DecodeManager.schedule_next_batch()`。

## 6. mini-sglang 与 nano-vLLM 主线的关键差异

mini-sglang 的主线更“serving 化”：

1. 多进程：frontend、tokenizer、detokenizer、多个 TP scheduler 分离。
2. 消息协议：`TokenizeMsg`、`UserMsg`、`DetokenizeMsg`、`UserReply` 贯穿系统。
3. Paged KV：用 `page_table` 保存 request 逻辑位置到物理 KV token index 的映射。
4. Radix Cache：用 radix tree 保存可复用 prefix，而不是只做简单 block hash。
5. Overlap Scheduling：CPU metadata 处理和 GPU forward 可重叠。
6. 多 attention backend：FlashAttention、FlashInfer、TensorRT-LLM backend 可组合。

## 7. 完整主线图

```text
python -m minisgl
  ↓
launch_server
  ├─ scheduler process × TP size
  ├─ tokenizer worker(s)
  ├─ detokenizer worker
  └─ API server / shell
  ↓
POST /generate or /v1/chat/completions
  ↓
FrontendManager.new_user
  ↓
TokenizeMsg
  ↓
TokenizeManager.tokenize
  ↓
UserMsg
  ↓
Scheduler._process_one_msg
  ↓
PrefillManager.add_one_req
  ↓
while running:
  ├─ Scheduler._schedule_next_batch
  │    ├─ prefill first
  │    └─ decode otherwise
  ├─ Scheduler._prepare_batch
  │    ├─ graph padding
  │    ├─ CacheManager.allocate_paged
  │    ├─ positions / input_mapping / write_mapping
  │    └─ attn_backend.prepare_metadata
  ├─ Scheduler._forward
  │    ├─ token_pool → batch.input_ids
  │    ├─ Engine.forward_batch
  │    │    ├─ ctx.forward_batch
  │    │    ├─ model.forward or CUDA graph replay
  │    │    ├─ req.complete_one
  │    │    └─ sampler.sample
  │    └─ next_token 写回 token_pool
  └─ Scheduler._process_last_data
       ├─ append_host
       ├─ 判断 eos / max_tokens
       ├─ cache prefix or free resource
       └─ DetokenizeMsg
  ↓
DetokenizeManager.detokenize
  ↓
UserReply
  ↓
Frontend streaming / JSON response
```
