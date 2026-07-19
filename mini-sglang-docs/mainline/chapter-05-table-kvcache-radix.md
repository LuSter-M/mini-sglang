# Chapter 05：Table、Paged KV Cache 与 Radix Prefix Cache

本章聚焦请求状态和 KV cache 管理：mini-sglang 如何用 `page_table`、`token_pool`、KV cache pool 和 radix tree 把逻辑 token 映射到物理 K/V 存储。

## 1. 本章要回答的问题

1. `table_idx`、`page_table`、`token_pool` 分别是什么？
2. KV cache 的真实 tensor 形状是什么？
3. `CacheManager.allocate_paged()` 什么时候分配 page？
4. Radix prefix cache 如何匹配、插入和淘汰？
5. mini-sglang 的 radix prefix cache 和 vLLM / nano-vLLM 的 block hash prefix cache 有什么差异？
6. 一个请求从进入到结束，`page_table` 和 `token_pool` 是怎么一步步变化的？

## 2. TableManager：请求表与 token_pool

`TableManager` 定义在 `python/minisgl/scheduler/table.py:4`。

它维护：

- `_free_slots`：可用 request table 行号，初始化在 `python/minisgl/scheduler/table.py:7`。
- `page_table`：Engine 创建的二维张量，按 request table 行保存每个逻辑位置对应的物理 KV token index。
- `token_pool`：和 page table 同形状的 int32 tensor，用于保存每个 request 每个位置的具体的 token id。

可以理解为：

```text
table_idx = 某个请求在系统里的行号

token_pool[table_idx, pos] = 这个请求 pos 位置的 token id
page_table[table_idx, pos] = 这个请求 pos 位置对应的物理 KV token index
```

## 3. Engine 创建 page_table

`Engine` 在初始化时创建 `page_table`，位置在 `python/minisgl/engine/engine.py:65`。

形状是：

```text
(max_running_req + 1, aligned_max_seq_len)
```

多出来的 1 行用于 dummy request，CUDA graph padding 时会用到。

## 4. KV cache pool

`MHAKVCache` 定义在 `python/minisgl/kvcache/mha_pool.py:10`。

真实 K/V buffer 形状在 `python/minisgl/kvcache/mha_pool.py:28`：

```text
(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
```

含义：

- `2`：K 和 V。
- `num_layers`：模型层数。
- `num_pages`：物理 KV pages 数。
- `page_size`：每个 page 存多少 token。
- `local_kv_heads`：当前 TP rank 上的 KV heads。
- `head_dim`：每个 head 的维度。

写入 K/V 的接口是 `store_kv()`，定义在 `python/minisgl/kvcache/mha_pool.py:45`。它会调用 CUDA kernel `store_cache()`，根据 `out_loc` 把当前 token 的 k/v 写入物理位置。

## 5. CacheManager：分配与释放 pages

`CacheManager` 定义在 `python/minisgl/scheduler/cache.py:15`。

它初始化空闲 page：

```text
free_slots = [0, page_size, 2*page_size, ...]
```

位置在 `python/minisgl/scheduler/cache.py:20`。注意这里保存的是 **page 起始 token index**，而不是 page id。

`allocate_paged()` 定义在 `python/minisgl/scheduler/cache.py:42`，它根据每个 req 的 `cached_len` 和 `device_len` 计算需要补充哪些 page：

```text
first_page = ceil(cached_len / page_size)
last_page  = ceil(device_len / page_size)
如果 last_page > first_page，则需要分配新 page
```

分配完成后调用 `_write_page_table()` 把物理 token indices 写入 `page_table`，位置在 `python/minisgl/scheduler/cache.py:127`。

## 6. Prefix cache 匹配

`CacheManager.match_req()` 定义在 `python/minisgl/scheduler/cache.py:27`。

它会对 `req.input_ids[: input_len - 1]` 做 prefix cache 匹配，位置在 `python/minisgl/scheduler/cache.py:30`。

为什么不匹配最后一个 token？因为 prefill 需要基于完整上下文计算“下一个 token”，最后一个 prompt token 通常还需要在本轮参与计算，不能直接把整个 prompt 都当成已缓存后跳过。

## 7. RadixPrefixCache：树形 prefix 复用

Radix 实现在 `python/minisgl/kvcache/radix_cache.py`。

RadixPrefixCache 是本章最值得重点看的结构之一。它体现了 SGLang 系统里非常核心的 serving 思路：**很多请求不是完全无关的，它们经常共享一段长前缀；如果能把这段前缀的 KV cache 复用起来，就可以显著减少重复 prefill。**

最典型的场景是：

- 多个请求使用同一段 system prompt。
- agent / workflow 里反复带着相同工具描述、角色设定、历史上下文。
- 多轮对话或批量任务中，前半段 prompt 高度相同，只有最后的问题不同。

RadixPrefixCache 的核心思想是：**把这些共享 prompt 前缀组织成一棵压缩前缀树，也就是 radix tree / compressed trie。**

它和普通 trie 的区别是：树上的一条边不是单个 token，而是一段连续 token。这样可以避免一个 token 一个节点带来的过多节点开销。

例如多个请求有共同系统提示词：

```text
请求 A：system prompt + question A
请求 B：system prompt + question B

radix tree：
root
  └─ system prompt 的 KV
       ├─ question A 的 KV
       └─ question B 的 KV
```

这样 `system prompt` 的 KV 只需要保存一份。后续请求命中后，可以直接拿到这段 prefix 对应的物理 KV indices，然后只 prefill 后面没有命中的部分。

### 7.1 节点里保存什么

核心类：

- `RadixTreeNode`：树节点。
- `RadixCacheHandle`：匹配结果 handle。
- `RadixPrefixCache`：prefix cache 主体。

一个 `RadixTreeNode` 可以理解成一段 prefix cache 片段：

| 字段 | 含义 |
| --- | --- |
| `_key` | 这一段 prefix 的 token ids。 |
| `_value` | 这一段 token 对应的 physical KV token indices。 |
| `children` | 后续可能分叉的子前缀。 |
| `parent` | 父节点，用于从命中节点回溯整条 prefix。 |
| `ref_count` | 当前是否被请求引用，用于 lock / eviction。 |
| `timestamp` | 最近访问时间，用于近似 LRU 淘汰。 |

这里最关键的是 `_key` 和 `_value` 的对应关系：

```text
_key   = [token ids of this prefix segment]
_value = [physical KV indices of these tokens]
```

也就是说，radix tree 不是只记录“哪些 token 前缀出现过”，它还记录这些 token 的 KV 实际存在哪里。命中 prefix cache 后，Scheduler 才能把 matched indices 写进 `page_table`，让 attention backend 读到已有 KV。

### 7.2 key_fn：如何选择下一条树边

`key_fn` 是 radix tree 的子节点索引函数，用来从一段 token 序列里取出 `children` 字典的 key。

代码里 `children` 是一个 dict：

```python
children: Dict[Any, RadixTreeNode]
```

所以沿树匹配时，需要先根据当前剩余 token 生成一个 key，再去找对应的 child。`key_fn` 做的就是这件事：

```text
当前剩余 token
  ↓
取前 page_size 个 token 作为 key
  ↓
用这个 key 查询 children
  ↓
决定 radix tree 下一步走哪条边
```

如果 `page_size = 4`，当前剩余 token 是：

```text
[10, 11, 12, 13, 20, 21, 22, 23]
```

那么 `key_fn` 生成的 key 就是：

```text
(10, 11, 12, 13)
```

这里使用 page 粒度作为 key，是因为 mini-sglang 的 radix prefix cache 本身按 page size 对齐插入和匹配。这样树的分叉粒度和 KV page 管理粒度保持一致。

### 7.3 match：沿树找最长可复用前缀

匹配入口是 `match_prefix()`，定义在 `python/minisgl/kvcache/radix_cache.py:132`。它会调用 `_tree_walk()`，定义在 `python/minisgl/kvcache/radix_cache.py:205`。

匹配过程可以理解成：

```text
从 root 开始
  ↓
根据当前剩余 token 的前几个 token 找 child
  ↓
比较 child._key 和请求 token 是否连续相同
  ↓
如果完整匹配，继续往下走
  ↓
如果部分匹配，split 节点并返回当前最长前缀
```

返回结果是一个 `RadixCacheHandle`。它保存两个信息：

```text
cached_len：命中了多少 token
node：命中到 radix tree 的哪个节点
```

后续 `handle.get_matched_indices()` 会从命中节点一路回溯到 root，把路径上的 `_value` 拼起来，得到完整命中 prefix 的 physical KV indices。入口在 `python/minisgl/kvcache/radix_cache.py:91`。

### 7.4 insert：把新完成的前缀挂回树上

插入入口是 `insert_prefix()`，定义在 `python/minisgl/kvcache/radix_cache.py:136`。它会按 page size 对齐：

```python
insert_len = align_down(len(input_ids), self.page_size)
```

位置在 `python/minisgl/kvcache/radix_cache.py:137`。

这意味着 radix cache 只缓存 page 对齐的完整前缀。

插入时也会先走一遍 `_tree_walk()`：

- 如果整段前缀已经存在，就不重复插入。
- 如果只有一部分存在，就从第一个不匹配的位置开始新建节点。
- 如果在某个节点中间发生部分匹配，会调用 `split_at()` 把节点切开，定义在 `python/minisgl/kvcache/radix_cache.py:69`。

这个 `split_at()` 是 radix tree 的关键操作。它保证公共前缀可以被单独提出来，后面不同请求的后缀再从这个公共节点分叉。

可以把 RadixPrefixCache 的主线记成：

```text
cache_req：把已完成 prefix 插入 radix tree
  ↓
match_req：新请求沿 radix tree 找最长命中
  ↓
matched indices 写入 page_table
  ↓
prefill 只计算未命中的后缀
```

这也是它区别于普通 block hash prefix cache 的地方：block hash 更关注“某个完整 block 是否已经存在”；radix tree 更关注“请求前缀在树上能连续命中多长”。

## 8. lock、unlock 与 eviction

`lock_handle()` 定义在 `python/minisgl/kvcache/radix_cache.py:113`。

它沿 handle 所在节点一路向 root 调整 `ref_count`：

- lock：把节点从 evictable 移到 protected。
- unlock：如果引用计数归零，把节点重新变成 evictable。

淘汰入口是 `evict()`，定义在 `python/minisgl/kvcache/radix_cache.py:148`。它收集 leaf nodes 并用 timestamp 小根堆做类似 LRU 的淘汰。

`CacheManager._allocate()` 在空闲 pages 不够时触发 prefix cache eviction，位置在 `python/minisgl/scheduler/cache.py:106`。

## 9. cache_req：把请求写回 prefix cache

`cache_req()` 是请求资源状态转换的关键入口，定义在 `python/minisgl/scheduler/cache.py:55`。它的作用是把请求已经算出来的 KV 尽可能转成可复用 prefix cache，同时把不能继续使用的物理页还回 free list。

它处理的是这段有效 KV 区间：

```text
[0, req.cached_len)
```

也就是当前请求已经完成计算、attention kernel 可以读取的历史 KV。`cache_req()` 会拿到这段 token ids 和它们在 `page_table` 里的 physical KV indices，然后调用 radix prefix cache 尝试插入。

插入后，资源会被分成几类：

| 区间 | 含义 | 处理方式 |
| --- | --- | --- |
| 已经命中过旧 prefix cache 的部分 | 请求进入时复用的旧前缀。 | 解锁 old handle。 |
| 被其他请求抢先插入 prefix cache 的部分 | 当前请求算过，但插入时发现树上已经有了。 | 释放当前请求持有的重复物理页，避免泄漏。 |
| 新插入 prefix cache 的部分 | 当前请求贡献出来的新可复用前缀。 | 留在 radix tree 中，后续请求可以命中。 |
| 不能 page 对齐插入的尾部 | 末尾不足一个 page，无法进入 prefix cache。 | 如果请求 finished，就释放；如果还要继续 decode，就继续保留给该请求。 |

所以 `finished` 参数决定的是尾部资源如何处理：

- `finished=True`：请求结束，尾部不再需要，直接释放。
- `finished=False`：请求还会继续 decode，尾部必须保留，并把 `req.cache_handle` 更新为新的 prefix cache handle，避免后续被 eviction。

可以把这一步记成：

```text
请求已有 KV
  ↓
尽量插入 radix prefix cache
  ↓
可复用前缀留在 cache
  ↓
重复页 / 无用尾部页释放
  ↓
未结束请求继续 lock 新 handle
```

## 10. 完整例子：一个请求的生命周期

前面几节分别讲了 TableManager、CacheManager、page_table、token_pool、prefix cache 等概念。本节用一个完整的数值例子，把这些概念串起来，跟踪一个请求从进入到结束的全过程，看 page_table 和 token_pool 是如何一步步变化的。

### 10.1 初始参数

为了便于展示，用一组简化但合理的参数：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `max_running_reqs` | 4 | 最多同时跑 4 个请求 |
| `num_pages` | 32 | KV cache 共 32 页 |
| `page_size` | 16 | 每页存 16 个 token 的 K/V |
| `max_seq_len` | 128 | 每个请求最长 128 token |
| 前缀缓存 | 空 | 初始没有任何缓存 |

初始状态下：

- **TableManager**：`_free_slots = [0, 1, 2, 3]`，4 个槽位全空闲。
- **CacheManager**：`free_slots = [0, 16, 32, 48, ..., 496]`，共 32 页，每页起始位置 = 页号 × page_size。
- **page_table**：形状 `[5, 128]`（+1 是 dummy request），全 0。
- **token_pool**：形状 `[5, 128]`，全 0。

### 10.2 请求进入：分配 slot 与匹配前缀

请求 R1：`input_ids` 共 40 个 token，`output_len = 30`。

**Step 1：TableManager 分配槽位**

`PrefillAdder._try_allocate_one()` 首先检查 TableManager 是否有空闲 slot，定义在 `python/minisgl/scheduler/prefill.py:39`。

```text
table_idx = table_manager.allocate()  # = 3
_free_slots: [0, 1, 2]  # 弹出 3
```

**Step 2：CacheManager 匹配前缀**

调用 `CacheManager.match_req()`，定义在 `python/minisgl/scheduler/cache.py:27`。前缀缓存为空，所以：

```text
cached_len = 0
handle.cached_len = 0
```

**Step 3：估算空间并 lock**

```text
extend_len = 40 - 0 = 40
estimated_len = 40 + 30 = 70 tokens ≈ 5 页
available_size = 32 页 × 16 = 512 tokens ✓
```

调用 `cache_manager.lock(handle)` 锁住前缀（防止被 evict），定义在 `python/minisgl/scheduler/cache.py:36`。

**Step 4：写入 token_pool**

`PrefillAdder._add_one_req()` 把 input_ids 写入 `token_pool[table_idx][cached_len:cached_len+chunk_size]`，定义在 `python/minisgl/scheduler/prefill.py:79`。

假设 prefill_budget 足够，一次跑完 40 个 token：

```text
token_pool[3][0:40] = [101, 205, 307, ..., 999]  # 40 个 token id
```

此时 `page_table[3]` 还是全 0——新页的分配要等到 `allocate_paged()` 才做。

### 10.3 Batch 准备：allocate_paged 分配物理页

`Scheduler._prepare_batch()` 调用 `CacheManager.allocate_paged()`，定义在 `python/minisgl/scheduler/cache.py:42`。

Req 状态：`cached_len=0, device_len=40`

```text
first_page = ceil(0 / 16) = 0
last_page  = ceil(40 / 16) = 3   # 第 0,1,2 页，共 3 页
needed_pages = 3 - 0 = 3
```

从 `free_slots` 分配 3 页：

```text
分配: [0, 16, 32]  →  对应第 0, 1, 2 页
剩余 free_slots: [48, 64, 80, ..., 496]
```

通过 `_write_page_table()` 写入 `page_table[3]`，定义在 `python/minisgl/scheduler/cache.py:127`：

```text
page_table[3][0:48] = [0,1,2,...,15, 16,17,...,31, 32,33,...,47]
                      └── 第 0 页 ──┘ └── 第 1 页 ──┘ └── 第 2 页 ──┘
```

> 注意：分配了 3 页 = 48 个位置，但请求只用前 40 个。第 40~47 个位置暂时空着，留给后续 decode 阶段用。

**此时的分工**：

- TableManager：已经分配了 slot 3，token_pool 里填好了 token id。
- CacheManager：分配了 3 个物理页，把物理地址写进了 page_table。
- 两者通过 `page_table[table_idx]` 这一行连接起来。

### 10.4 Prefill Forward + 缓存前缀

**Forward 过程**：

1. 从 `token_pool` 按 `input_mapping` 取出 input_ids（40 个）。
2. Attention 层算出 Q/K/V，把 K/V 通过 `kv_cache.store_kv()` 写入物理位置 0~39。
3. 采样得到第 1 个输出 token。
4. 把新 token 写回 `token_pool[3][40]`。
5. `req.complete_one()`：`cached_len: 0 → 40`，`device_len: 40 → 41`。

**缓存前缀**：

Prefill 完成后，`Scheduler._process_last_data()` 调用 `CacheManager.cache_req(req, finished=False)`，定义在 `python/minisgl/scheduler/cache.py:55`。

把 40 个 token 的前缀插入 radix tree（按 page_size 对齐，实际插入 32 个，即 2 页）：

```text
insert_len = align_down(40, 16) = 32
cached_len = 32, new_handle.cached_len = 32
```

- 解锁旧 handle（空前缀）。
- 前 32 个 token 进入前缀缓存，被 new_handle 引用并 lock 住。
- 第 32~39 个 token（第 2 页的前 8 个）：还没到 page 对齐，暂时作为请求的"私有尾部"保留。

### 10.5 Decode 阶段：逐步推进

此时 req 状态：`cached_len = 40, device_len = 41, remain_len = 29`

注意：decode 阶段不会每一步都重新去 radix tree 做 prefix match。请求已经在进入 prefill 时完成过一次匹配，prefill 完成后也通过 `cache_req(finished=False)` 更新了 handle；后续 decode 主要基于当前请求已有的 `page_table`、`token_pool` 和 `cache_handle` 继续推进。

**第 1 步 decode：**

- `extend_len = 1`，只输入第 40 个位置的 token。
- 不需要分配新页（第 2 页有 16 个位置，只用了 9 个，还剩 7 个）。
- Forward 算出新的 K/V，写入物理位置 40。
- 采样得到下一个 token，写回 `token_pool[3][41]`。
- `cached_len: 40 → 41`，`device_len: 41 → 42`。

... 重复 7 次后 ...

**第 9 步 decode（需要新页）：**

前 8 步 decode 用完了第 2 页（位置 32~47）。第 9 步 decode 时：

```text
cached_len = 48, device_len = 49
first_page = ceil(48 / 16) = 3   # cached_len 之前的页已分配
last_page  = ceil(49 / 16) = 4   # device_len 对齐到页
needed_pages = 4 - 3 = 1 页
```

`allocate_paged()` 再分配 1 页（第 3 页，起始位置 48），写入 `page_table[3][48:64]`。

之后继续 decode，直到请求结束。

### 10.6 请求结束：释放资源

`Scheduler._free_req_resources()` 做两件事，定义在 `python/minisgl/scheduler/scheduler.py:200`：

```python
self.table_manager.free(req.table_idx)           # 释放槽位
self.cache_manager.cache_req(req, finished=True)  # 缓存完整前缀 + 释放尾部
```

**TableManager**：把 `table_idx = 3` 放回 `_free_slots`，槽位可被新请求复用。

**CacheManager**：调用 `cache_req(finished=True)`：
- 把完整序列插入 radix tree，尽可能多地缓存前缀（按 page 对齐）。
- 已经被其他请求共享的页 → 释放私有引用（这些页归前缀缓存管了）。
- 尾部不能被缓存的页 → 直接释放回 `free_slots`。

**最终资源去向**：

- Slot 3 → 归还 TableManager。
- 能被前缀缓存的页 → 留在 radix tree 中，供后续请求匹配命中。
- 不能被缓存的尾部页 → 归还 free_slots，可被重新分配。

### 10.7 小结

这个例子展示了 TableManager 和 CacheManager 清晰的分工边界：

| 阶段 | TableManager 做什么 | CacheManager 做什么 |
| --- | --- | --- |
| 请求进入 | 分配 table_idx（行） | 匹配前缀、lock、估算空间 |
| Batch 准备 | （token 数据已写入 token_pool） | 分配物理页、写入 page_table |
| Prefill 完成 | — | 插入前缀缓存、更新 handle |
| Decode 阶段 | — | 按需分配新页 |
| 请求结束 | 释放 table_idx | 插入完整前缀、释放尾部页 |

一句话：**TableManager 管"行"（请求槽位 + token 数据），CacheManager 管"页"（物理 KV 内存 + 前缀缓存），两者通过 page_table 这张二维表连接。**

## 11. Prefix cache 与 KV 映射：和 vLLM / nano-vLLM 的对比

这一章是理解 mini-sglang 和 vLLM 系主线差异的关键：两者都在做 paged KV cache，也都要解决“逻辑 token 如何找到物理 KV”的问题，但抽象边界不一样。

可以先用一张表记住：

| 维度 | mini-sglang | vLLM / nano-vLLM |
| --- | --- | --- |
| 请求状态 | `Req.table_idx` 指向全局 request table 的一行。 | `Sequence.block_table` 直接挂在 sequence 上。 |
| token 记录 | `token_pool[table_idx, pos]` 保存每个逻辑位置的 token id。 | token ids 主要保存在 sequence 自身。 |
| KV 映射 | `page_table[table_idx, pos]` 记录 token 级别的物理 KV token index。 | `block_table` 记录逻辑 block 到物理 block id 的映射。 |
| KV 管理对象 | `TableManager` 管请求行，`CacheManager` 管 KV pages 和 prefix cache。 | `BlockManager` 同时管理 block 分配、block_table 复用和 prefix cache。 |
| prefix cache 结构 | radix tree 保存可复用 prefix，并返回 matched physical indices。 | block hash / 链式 hash 保存完整 block 的复用关系。 |
| attention 视角 | 通过 `page_table` 得到更直接的 token-level KV index。 | 通过 `block_tables`、`slot_mapping`、`context_lens` 找到历史 KV。 |

所以两者不是“一个有 paged KV，一个没有 paged KV”的区别；它们都有 paged KV。真正的差异在于：

```text
vLLM / nano-vLLM：Sequence → logical block → physical block
mini-sglang：Req.table_idx → token position → physical KV token index
```

vLLM / nano-vLLM 更像“以 sequence 为中心”：sequence 自己带着 `block_table`，`BlockManager` 负责分配、复用和引用计数。

mini-sglang 更像“以全局表为中心”：请求只拿一个 `table_idx`，具体 token ids 和 KV 物理位置都写进全局 `token_pool` / `page_table`。`TableManager` 负责 request table 行分配，`CacheManager.allocate_paged()` 负责补齐 page 映射。

prefix cache 的组织方式也不同：nano-vLLM 用完整 block 的链式 hash 做命中，重点是“完整 block 是否可复用”；mini-sglang 用 radix tree 表示前缀关系，重点是“请求 token 前缀能在树上匹配到多长”。

从阅读主线看，可以这样记：

```text
vLLM / nano-vLLM：Sequence.block_table → BlockManager → slot_mapping / block_tables
mini-sglang：Req.table_idx → token_pool / page_table → CacheManager → RadixPrefixCache
```

抓住这个差异后，再看后面的 Engine prepare batch 和 attention backend，就不会把 `block_table`、`page_table`、`token_pool` 混在一起。

## 12. 本章小结

KV/cache 主线可以记成：

```text
Req.table_idx
  ↓
token_pool[table_idx, pos] 保存 token id
page_table[table_idx, pos] 保存物理 KV token index
  ↓
batch.out_loc = page_table[input_mapping]
  ↓
attention backend 调用 kv_cache.store_kv(k, v, out_loc, layer_id)
  ↓
prefix cache 保存可复用前缀 token ids → physical indices
```

下一章进入 Engine，看模型、KV cache、attention backend 和 sampler 是如何装配起来并执行 forward 的。
