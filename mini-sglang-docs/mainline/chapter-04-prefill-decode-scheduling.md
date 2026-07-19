# Chapter 04：Prefill / Decode 调度

本章聚焦 Scheduler 内部的调度策略：新请求如何 prefill，长 prompt 如何 chunked prefill，已经完成 prompt 的请求如何进入 decode。

## 1. 本章要回答的问题

1. prefill pending list 保存什么？
2. prefix cache 命中如何影响 prefill？
3. chunked prefill 在哪里发生？
4. decode batch 如何组成？
5. 当前调度策略为什么是 prefill 优先？

## 2. PendingReq 与 Req

新请求从 `UserMsg` 进入 `PrefillManager.add_one_req()`，位置在 `python/minisgl/scheduler/prefill.py:123`。

`add_one_req()` 会把 `UserMsg` 包装成 `PendingReq`，放入 `pending_list`。

真正进入 forward 的是 `Req`，定义在 `python/minisgl/core.py:28`。它保存：

- `input_ids`：CPU tensor，代表当前已经进入该请求视野的 token。
- `table_idx`：该请求在 page table / token pool 中的行号。
- `cached_len`：前多少 token 已经有可用 KV cache。
- `device_len`：当前本轮需要让模型可见的长度。
- `output_len`：最多生成长度。
- `cache_handle`：prefix cache 命中的 handle。

`Req.extend_len` 定义在 `python/minisgl/core.py:48`，等于 `device_len - cached_len`，表示本轮真正要新计算的 token 数。

## 3. PrefillManager.schedule_next_batch

`PrefillManager` 定义在 `python/minisgl/scheduler/prefill.py:116`。

`schedule_next_batch()` 的职责是从 `pending_list` 里挑出本轮可以做 prefill 的请求，组成一个 `Batch(phase="prefill")` 交给 Scheduler 后续准备 metadata 和 forward，并非直接执行模型。

整体流程可以拆成 5 步：

```text
pending_list 为空？
  ├─ 是：返回 None
  └─ 否：创建 PrefillAdder
          ↓
        从 pending_list 头部按顺序尝试加入请求
          ↓
        对每个请求检查 token budget / table slot / KV cache 空间
          ↓
        生成 Req 或 ChunkedReq
          ↓
        更新 pending_list，并返回 Batch(phase="prefill")
```

### 3.1 空队列直接返回

如果当前没有待 prefill 请求，`schedule_next_batch()` 会直接返回 `None`，让 Scheduler 有机会继续尝试 decode batch。

代码入口：`python/minisgl/scheduler/prefill.py:127`。

### 3.2 创建 PrefillAdder：把本轮资源约束收口

如果 `pending_list` 非空，它会先创建一个 `PrefillAdder`：

```text
token_budget = prefill_budget
reserved_size = decode_manager.inflight_tokens
```

位置在 `python/minisgl/scheduler/prefill.py:130`。

这里有两个关键约束：

| 字段 | 含义 | 作用 |
| --- | --- | --- |
| `token_budget` | 本轮最多允许 prefill 多少个新 token。 | 控制一次 prefill batch 不要过大，并支持长 prompt 分块。 |
| `reserved_size` | 已有 decode 请求未来还可能需要的 KV cache 空间估计。 | 避免 prefill 把 KV cache 全部占满，导致正在 decode 的请求后续无法继续生成。 |

其中 `reserved_size` 来自 `DecodeManager.inflight_tokens`，它会把 running decode 请求剩余可能生成的 token 数，以及每个请求额外预留的 page 空间算进去。

### 3.3 顺序扫描 pending_list：能加就加，不能加就停

接下来它从 `pending_list` 的头部开始遍历，每个 `PendingReq` 调用一次 `adder.try_add_one()`。

```text
for pending_req in pending_list:
    req = adder.try_add_one(pending_req)
    if req 可以加入:
        放入本轮 reqs
    else:
        break
```

代码入口：`python/minisgl/scheduler/prefill.py:139`。

这里的 `break` 很重要：一旦队首附近某个请求因为资源不足无法加入，本轮就不再继续扫描后面的请求。这样做牺牲了一些“跳过大请求找小请求”的机会，但保持了 pending 队列的顺序语义，也让调度逻辑更简单。

### 3.4 try_add_one：把 PendingReq 变成 Req / ChunkedReq

`adder.try_add_one()` 是真正判断“这个 pending 请求本轮能不能进 batch”的地方，定义在 `python/minisgl/scheduler/prefill.py:92`。

它分两种情况：

| 情况 | 处理方式 |
| --- | --- |
| 这是上轮没 prefill 完的 chunked 请求 | 复用之前已经分配好的 `cache_handle` 和 `table_idx`，继续从上次位置往后 prefill。 |
| 这是第一次被调度的新请求 | 先尝试匹配 prefix cache、检查 KV cache 空间、分配 table slot，再构造本轮 `Req`。 |

对应代码在 `python/minisgl/scheduler/prefill.py:96`。

如果是新请求，它还会走 `_try_allocate_one()`，这个函数负责做资源准入：

1. 检查是否还有 request table slot。
2. 通过 `cache_manager.match_req(req)` 查询 prefix cache 命中长度。
3. 估算本请求需要的新 KV 空间：`extend_len + output_len`。
4. 加上 `reserved_size` 后检查 CacheManager 是否还有足够空间。
5. lock 命中的 prefix cache handle，防止本轮执行期间被淘汰。
6. 分配 table slot；如果 prefix cache 命中，把已命中的 token ids 和 KV page indices 写到 `token_pool` / `page_table` 对应位置。

代码入口：`python/minisgl/scheduler/prefill.py:39`。

真正把请求加入本轮 batch 的是 `_add_one_req()`。它会根据剩余 prompt 长度和 `token_budget` 决定本轮 prefill 多长：

```text
remain_len = pending_req.input_len - cached_len
chunk_size = min(token_budget, remain_len)
is_chunked = chunk_size < remain_len
```

如果 `chunk_size` 覆盖不了完整剩余 prompt，就返回 `ChunkedReq`；否则返回普通 `Req`。同时，它会把本轮要 prefill 的 token ids 写入 `token_pool`，但不会在这里分配新的 KV pages；新的 page 分配会在 Scheduler `_prepare_batch()` 里统一完成。

代码入口：`python/minisgl/scheduler/prefill.py:65` ；KV pages 后续分配入口是 `python/minisgl/scheduler/scheduler.py:206` 和 `python/minisgl/scheduler/cache.py:42`。

### 3.5 更新 pending_list 并返回 prefill batch

遍历结束后，如果一个请求都没加进去，说明当前资源不足或 token budget 不足，函数返回 `None`。

代码入口：`python/minisgl/scheduler/prefill.py:148`。

如果本轮形成了 `reqs`，它会更新 `pending_list`：

```text
pending_list = 本轮未完成的 chunked 请求 + 原 pending_list 中尚未扫描到的剩余请求
```

也就是说：

- 已经完整 prefill 完的请求会从 `pending_list` 移除，后续 forward 结束后进入 decode。
- 没有完整 prefill 完的长 prompt 会以 `chunked_req` 形式放回队首，下一轮继续 prefill。
- 本轮还没来得及扫描到的请求继续留在队列后面。

最后返回 `Batch(reqs=reqs, phase="prefill")`。

## 4. Prefix cache 匹配与资源检查

`PrefillAdder._try_allocate_one()` 定义在 `python/minisgl/scheduler/prefill.py:39`。

这一段可以理解成新请求进入 prefill batch 前的“资源准入闸门”：它先看请求有没有可复用 prefix，再判断当前 table slot 和 KV cache 空间是否足够。只有通过检查的请求，才会被真正加入本轮 prefill batch。

流程如下：

```text
检查 request table 是否有空行，没有则放弃
  ↓
匹配 prefix cache，得到 cached_len
  ↓
估算 extend_len + output_len，并结合 reserved_size 检查 KV 空间
  ↓
lock 命中的 prefix cache handle，避免本轮使用前被淘汰
  ↓
再次检查空间，失败则 unlock 并放弃
  ↓
分配 table_idx，并把已命中的 prefix 写入 token_pool / page_table
```

`cached_len` 表示 prompt 前面有多少 token 已经命中 prefix cache，不需要重新计算 KV。剩下真正要算的是：

```text
extend_len = input_len - cached_len
estimated_len = extend_len + output_len
```

这里把 `output_len` 也算进去，是为了提前为后续 decode 留出空间；再加上 `reserved_size`，是为了不抢占已有 running decode 请求未来要用的 KV cache。

通过资源检查后，请求会拿到一个 `table_idx`。如果 prefix cache 有命中，命中的 token id 会写入 `token_pool`，命中的物理 KV index 会写入 `page_table`。注意，这里只处理“已命中 prefix”的映射；未缓存的新 token 还没有真正分配 KV pages，后面会由 Scheduler `_prepare_batch()` 统一分配。

## 5. Chunked Prefill

`PrefillAdder._add_one_req()` 定义在 `python/minisgl/scheduler/prefill.py:65`。

核心逻辑是：

```text
remain_len = input_len - cached_len
chunk_size = min(token_budget, remain_len)
is_chunked = chunk_size < remain_len
```

位置在 `python/minisgl/scheduler/prefill.py:72`。

如果 `chunk_size` 不能覆盖完整 prompt，就会创建 `ChunkedReq`。它不会进入 decode，因为 `can_decode` 固定返回 `False`，位置在 `python/minisgl/scheduler/prefill.py:27`。

chunked 请求会被放回 pending list 前部，等待下一轮继续 prefill。这个逻辑在 `python/minisgl/scheduler/prefill.py:142`。

## 6. DecodeManager

`DecodeManager` 定义在 `python/minisgl/scheduler/decode.py:10`。

它维护 `running_reqs`，也就是 prompt 已经 prefill 完、可以继续 decode 的请求。

`filter_reqs()` 定义在 `python/minisgl/scheduler/decode.py:14`，会把本轮 batch 中仍然 `can_decode` 的请求加入 running set。

`schedule_next_batch()` 定义在 `python/minisgl/scheduler/decode.py:32`，当前逻辑很简单：如果有 running 请求，就按 uid 排序组成 `Batch(reqs=..., phase="decode")`。

按 uid 排序的位置在 `python/minisgl/scheduler/decode.py:35`，这有助于保持多 rank 或多轮调度中的稳定顺序。

## 7. prefill 优先策略

Scheduler 的 `_schedule_next_batch()` 定义在 `python/minisgl/scheduler/scheduler.py:219`。

当前策略是：

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

位置在 `python/minisgl/scheduler/scheduler.py:221`。

含义是：只要有可调度的 prefill batch，就优先跑 prefill；否则才跑 decode。

## 8. 本章小结

Prefill / Decode 调度可以总结为：

```text
UserMsg
  ↓
PendingReq in PrefillManager.pending_list
  ↓
match prefix cache + allocate table slot
  ↓
按 token_budget 形成 Req 或 ChunkedReq
  ↓
Batch(phase="prefill")
  ↓
forward 后非 chunk 请求进入 DecodeManager.running_reqs
  ↓
Batch(phase="decode")
```

下一章继续看这些 Req 如何映射到 `page_table`、`token_pool` 和真实 KV cache。
