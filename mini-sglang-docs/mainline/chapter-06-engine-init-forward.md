# Chapter 06：Engine 初始化与 forward_batch

Engine 是每个 TP rank 内真正贴近 GPU 的执行组件。本章讲它如何初始化模型、KV cache、attention backend、sampler 和 CUDA graph，以及一轮 `forward_batch()` 如何执行。

## 1. 本章要回答的问题

1. Engine 初始化顺序是什么？
2. KV cache pages 数量如何根据显存估算？
3. `Context` 在 Engine 和模型之间起什么作用？
4. `forward_batch()` 如何从 batch 到 logits 再到 token？

## 2. Engine 的职责和定位

在 mini-sglang 主线里，Scheduler 负责“决定下一步跑什么”，Engine 负责“真正把这个 batch 跑起来”。它是每个 TP rank 内最贴近 GPU 的执行组件。

可以把 Engine 理解成 Scheduler 和模型之间的执行层：

```text
Scheduler
  ├─ 选择 prefill / decode batch
  ├─ 准备 input_ids / positions / page_table / out_loc
  ↓
Engine
  ├─ 持有模型 shard
  ├─ 持有 KV cache pool
  ├─ 持有 attention backend / MoE backend
  ├─ 持有 sampler
  ├─ 管理 CUDA stream / CUDA graph runner
  ↓
Model / Attention / Sampler
```

Engine 的核心职责可以分成两类：

| 阶段 | Engine 负责什么 |
| --- | --- |
| 初始化阶段 | 绑定 CUDA device，初始化 TP 通信，创建模型，加载权重，估算 KV cache pages，创建 KV cache pool / page_table / backend / sampler / CUDA graph runner。 |
| 运行阶段 | 接收 Scheduler 准备好的 `Batch`，挂载到 `Context`，调用模型 forward 或 CUDA graph replay，采样 next token，并把结果返回给 Scheduler。 |

所以 Engine 不负责请求排队、prefix cache 调度、prefill/decode batch 选择；这些属于 Scheduler。Engine 关注的是：**给定一个已经准备好的 batch，如何在当前 TP rank 的 GPU 上高效执行一轮 forward。**

## 3. Engine 初始化入口

`Engine` 定义在 `python/minisgl/engine/engine.py:29`。

主流程：

```text
设置 TP 信息
  ↓
调整配置
  ↓
设置 CUDA device / stream / dtype
  ↓
创建 Context 并 set_global_ctx
  ↓
初始化分布式通信
  ↓
创建模型并加载权重
  ↓
根据显存估算 KV pages
  ↓
创建 KV cache pool
  ↓
创建 page_table
  ↓
创建 attention / MoE backend
  ↓
创建 Sampler
  ↓
初始化 CUDA graph runner
```

`Context` 创建在 `python/minisgl/engine/engine.py:41`。

## 4. 分布式通信

`_init_communication()` 定义在 `python/minisgl/engine/engine.py:112`。

这一节处理的是 TP rank 之间的通信初始化。mini-sglang 里每个 TP rank 对应一个 Scheduler / Engine 进程，每个进程绑定一张 CUDA device。模型权重和计算被切到不同 GPU 后，forward 过程中就需要跨 rank 做 collective communication，例如 all-reduce / all-gather，才能得到完整结果。

几个概念可以先这样理解：

| 概念 | 简单解释 | 在这里的作用 |
| --- | --- | --- |
| CUDA device | 当前进程绑定的 GPU。 | 每个 TP rank 在自己的 GPU 上执行模型 shard。 |
| NCCL | NVIDIA 的 GPU 间通信库。 | 用于高性能 GPU collective communication。 |
| Gloo | PyTorch 的通用分布式通信 backend，偏 CPU 控制面。 | 用于初始化、同步和 CPU 侧 group 管理。 |
| PyNCCL | Python 侧封装的 NCCL 通信路径。 | 在部分配置下接管 GPU 通信，让 mini-sglang 自己控制通信 buffer 和调用方式。 |
| process group | 一组参与通信的 rank。 | 定义哪些进程需要互相通信，以及用哪个 backend 通信。 |

当前代码有两条路径：

```text
tp_size == 1 或 use_pynccl=True
  ↓
初始化 gloo group
  ↓
调用 enable_pynccl_distributed()

否则
  ↓
初始化 NCCL process group
  ↓
额外创建 gloo CPU group
```

第一条路径的入口在 `python/minisgl/engine/engine.py:113`。即使是单卡，代码也会初始化一个 gloo group，这样后续逻辑可以统一按“有分布式 group”的方式写，不需要到处特殊判断单卡。启用 PyNCCL 时，也需要先有 CPU 侧 group 做 rank 协调，再让 PyNCCL 建立 GPU 通信能力。

第二条路径的入口在 `python/minisgl/engine/engine.py:127`。如果不走 PyNCCL，就直接用 PyTorch NCCL backend 做 GPU 间通信；同时额外创建一个 gloo CPU group，给 CPU 侧控制、同步或元信息交换使用。

为什么要区分 NCCL 和 Gloo？简单说：

```text
NCCL：更适合 GPU tensor 的高速通信
Gloo：更适合 CPU 侧控制面和通用同步
```

所以这里是在为两类通信准备不同通道：GPU 数据面走 NCCL / PyNCCL，CPU 控制面保留 gloo group。

## 5. 模型创建与权重加载

模型创建发生在 meta device 上：

```python
with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)
```

入口在 `python/minisgl/engine/engine.py:50`。

随后加载权重：`python/minisgl/engine/engine.py:52`。

如果使用 dummy weight，会随机生成权重；否则调用 `load_weight()` 从模型目录加载，入口在 `python/minisgl/engine/engine.py:139`。

## 6. KV pages 数量估算

`_determine_num_pages()` 定义在 `python/minisgl/engine/engine.py:148`。

它会估算每个 page 的 KV cache 字节数：

```text
2 * head_dim * local_kv_heads * page_size * dtype.itemsize * num_layers
```

入口在 `python/minisgl/engine/engine.py:150`。

如果用户没有显式指定 `num_page_override`，会根据 `memory_ratio`、加载模型前后的显存差计算可用于 KV cache 的内存，入口在 `python/minisgl/engine/engine.py:158`。

## 7. Context 的作用

`Context` 定义在 `python/minisgl/core.py:100`。

它保存 Engine 和模型 forward 之间共享的一组运行期对象。这样模型内部不需要层层传递 `batch`、`page_table`、`kv_cache` 等参数，而是通过 `get_global_ctx()` 读取当前上下文。

主要字段如下：

| 字段 | 作用 |
| --- | --- |
| `page_size` | KV cache 的 page 粒度，很多 page 对齐、prefix cache 插入、page table 写入都会依赖它。 |
| `page_table` | 请求逻辑 token 位置到物理 KV token index 的映射表。attention backend 根据它找到历史 token 的 K/V 存储位置。 |
| `attn_backend` | 当前使用的 attention backend，例如 FlashAttention / FlashInfer / TensorRT-LLM backend。模型 attention 层通过它执行具体 kernel。 |
| `moe_backend` | MoE 模型使用的专家路由 / expert 执行 backend。非 MoE 模型通常不会用到。 |
| `kv_cache` | 真正保存每层 K/V tensor 的 KV cache pool。attention 层会把新 token 的 K/V 写进去，也会从里面读历史 K/V。 |
| `_batch` / `batch` | 当前正在 forward 的 batch。里面包含本轮 reqs、input mapping、out loc、phase 等运行时信息。 |

这里的 `page_table` 和 Chapter 05 里 TableManager / CacheManager 维护的是同一张表，只是视角不同：

```text
Engine 创建 page_table
  ↓
Context 持有 page_table，供模型 forward / attention backend 读取
  ↓
TableManager 按 table_idx 管理请求所在行，并配套维护 token_pool
  ↓
CacheManager 分配物理 KV pages，并把物理 KV token index 写入 page_table
```

所以 Chapter 05 主要讲 Scheduler 侧如何填充和维护 `page_table`；这里则强调同一张 `page_table` 如何通过 `Context` 暴露给模型和 attention backend 使用。

`Context.forward_batch()` 定义在 `python/minisgl/core.py:115`，用于在模型 forward 期间设置当前 batch。

它是一个 context manager，进入 forward 时设置 active batch，退出时清空：

```text
with ctx.forward_batch(batch):
    model.forward()
```

这样可以保证同一时刻只有一个 active batch，避免模型内部读取到错误的 batch 状态。

这里的“同一时刻只有一个 active batch”并不表示只能处理一个请求。mini-sglang 的并行粒度是 batch：多个请求会先被 Scheduler 合成一个 `Batch`，再由一次 `Engine.forward_batch(batch)` 一起送进模型。Context 中挂的是这个 batch，而不是某个单独请求。

所以可以这样理解：

```text
batch 之间：同一个 Engine 进程里串行 forward
batch 内部：多个 req 通过 GPU batch 计算并行执行
```

overlap scheduling 也不会让两个 batch 同时覆盖同一个 `Context._batch`；它重叠的是“当前 batch 的 GPU forward”和“上一 batch 的 CPU 结果处理”，不是两个 `model.forward()` 同时进入同一个 Context。

## 8. forward_batch 主链路

`Engine.forward_batch()` 定义在 `python/minisgl/engine/engine.py:191`。

`forward_batch()` 接收的是 Scheduler 已经准备好的一个 batch。这个 batch 里可能包含多个请求：prefill batch 里可能有多个新请求或 chunked 请求，decode batch 里通常有多个正在逐 token 生成的请求。Engine 不会为每个请求单独调用一次模型，而是把整个 batch 作为一次 GPU 计算提交。

主流程：

```text
with self.ctx.forward_batch(batch):
    if graph_runner.can_use_cuda_graph(batch):
        logits = graph_runner.replay(batch)
    else:
        logits = self.model.forward()

for req in batch.reqs:
    req.complete_one()

next_tokens_gpu = sampler.sample(logits[:batch.size], args)
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
record copy_done_event
return ForwardOutput
```

每一步的作用可以这样理解：

| 步骤 | 作用 |
| --- | --- |
| 进入 `ctx.forward_batch(batch)` | 把当前 batch 挂到全局 Context 上，让模型内部、attention backend、KV cache 写入逻辑都能读到本轮 batch 的 metadata。 |
| 判断是否使用 CUDA graph | 如果 batch 形状适合 CUDA graph，就复用提前 capture 好的 graph，减少 Python 调度和 kernel launch 开销。 |
| `graph_runner.replay(batch)` / `model.forward()` | 真正执行模型 forward，得到 logits。prefill 通常走普通 forward，稳定形状的 decode batch 更可能走 graph replay。 |
| `req.complete_one()` | 更新每个请求的长度状态：本轮进入模型的 token 已经完成计算，后续可以作为历史上下文使用。 |
| `sampler.sample(...)` | 根据 logits 和采样参数选出每个请求的下一个 token。 |
| GPU → CPU 异步拷贝 | 把采样出来的 token 从 GPU 拷回 CPU，方便 Scheduler 后续处理、判断结束条件、发送 detokenize 消息。 |
| 记录 copy done event | 标记 CPU 拷贝什么时候完成，便于 overlap loop 在合适时机安全读取结果。 |
| 返回 `ForwardOutput` | 把 next tokens 和拷贝完成事件交还给 Scheduler。 |

`Req.complete_one()` 定义在 `python/minisgl/core.py:52`，它会把 `cached_len` 更新到 `device_len`，并让 `device_len += 1`，为下一个 decode token 做准备。

从执行顺序看，同一个 Engine 上的 `model.forward()` 是 batch 级串行的：本轮 batch forward 完成后，才会进入下一轮 batch forward。但 batch 内部的 req 会被整理成连续 tensor 和 metadata，在 attention / MLP / sampler 等 GPU kernel 中并行处理。

## 9. CUDA Graph 的位置

Engine 初始化最后会创建 `GraphRunner`，位置在 `python/minisgl/engine/engine.py:99`。

在 forward 时，只有满足 `graph_runner.can_use_cuda_graph(batch)` 的 batch 才会走 replay。一般来说这是 decode 阶段的小 batch；prefill 因为长度变化大，通常直接 `model.forward()`。

## 10. 本章小结

Engine 可以理解为每个 TP rank 的 GPU 执行器：

```text
Engine 初始化阶段：
设置 CUDA device / stream / dtype
  ↓
初始化 TP 通信 group
  ↓
创建模型并加载权重
  ↓
根据剩余显存估算 KV pages
  ↓
创建 KV cache pool + page_table
  ↓
创建 attention / MoE backend + sampler
  ↓
初始化 CUDA graph runner


Engine 运行阶段：
Scheduler 准备 Batch
  ↓
Engine.forward_batch
  ↓
Context 挂载当前 active batch
  ↓
选择 graph replay 或普通 model.forward
  ↓
模型内部通过 Context 读取 batch / page_table / kv_cache / backend
  ↓
attention 写入或读取 KV cache
  ↓
sampler.sample
  ↓
next token 异步拷贝回 CPU
  ↓
ForwardOutput 返回 Scheduler
```

下一章进入模型内部，看 forward、attention backend 和 sampler 如何协作。
