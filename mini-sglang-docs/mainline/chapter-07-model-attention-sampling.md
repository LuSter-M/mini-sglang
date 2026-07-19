# Chapter 07：模型 Forward、Attention Backend 与采样

本章沿着 `Engine.forward_batch()` 继续向模型内部走：以 Qwen2 为例，看 embedding、decoder layer、attention、KV cache 写入、LM head 和 sampler 的主链路。

## 1. 本章要回答的问题

1. 模型 forward 从哪里拿 input ids？
2. Qwen2 的 forward 层级是什么？
3. HF 权重如何加载到 TP rank 对应的模型 shard？
4. AttentionLayer 如何使用 global Context？
5. attention backend 什么时候写 KV cache？
6. Sampler 如何根据 logits 生成 next token？

## 2. 模型注册与创建

Engine 通过 `create_model(config.model_config)` 创建模型，调用位置在 `python/minisgl/engine/engine.py:51`。

模型注册逻辑在 `python/minisgl/models/register.py`，支持 Llama、Mistral、Qwen2、Qwen3、Qwen3MoE 等模型。

权重加载入口由 Engine 调用 `load_weight()`，位置在 `python/minisgl/engine/engine.py:146`。权重加载和 TP 切分逻辑主要在 `python/minisgl/models/weight.py`。

## 3. 权重加载与 TP 切分

模型创建完成后，Engine 会调用 `load_weight()` 读取 HuggingFace checkpoint，并把权重转换成当前 TP rank 实际需要的 runtime 权重。

这一步之所以值得单独看，是因为 **checkpoint 里的权重形态** 和 **mini-sglang runtime 里的权重形态** 不一定一一对应：

```text
HF checkpoint 原始权重
  ↓
按当前 TP rank 切分
  ↓
合并 runtime 中的 fused projection
  ↓
加载到当前 rank 的模型 shard
```

`load_weight()` 是 streaming loader：它逐个 safetensors 文件读取，每次处理一个 tensor，处理完就 yield 给 `load_state_dict()`。这样做的好处是降低加载阶段的峰值内存，不需要一次性把所有权重都放在 CPU 内存里。

TP 切分规则可以先记成：

| 权重类型 | 切分方式 | 直觉 |
| --- | --- | --- |
| `q_proj / k_proj / v_proj` | 按输出维切分 | 每个 rank 负责一部分 attention heads。 |
| `gate_proj / up_proj` | 按输出维切分 | MLP 中间维度被切到不同 rank。 |
| `o_proj / down_proj` | 按输入维切分 | 前一层已经被切分，输出投影需要接收当前 rank 的局部 hidden。 |
| `embed_tokens / lm_head` | 按 vocab 维切分 | 每个 rank 负责一段词表。 |
| 其他权重 | 通常不切 | 例如 norm 等小参数每个 rank 保留完整副本。 |

这里的 `gate_proj / up_proj / down_proj` 是 Transformer MLP 里的三个投影。以 Qwen / Llama 常见的 gated MLP 为例，它不是简单的两层 FFN，而是类似：

```text
hidden states
  ├─ gate_proj → activation
  ├─ up_proj
  ↓
activation(gate_proj(x)) * up_proj(x)
  ↓
down_proj
  ↓
回到 hidden size
```

三个投影的含义是：

| 投影 | 作用 | 直觉 |
| --- | --- | --- |
| `gate_proj` | 从 hidden size 投影到 intermediate size，并经过激活函数。 | 生成“门控”，决定哪些中间维度更重要。 |
| `up_proj` | 从 hidden size 投影到 intermediate size。 | 提供被门控分支调制的内容。 |
| `down_proj` | 从 intermediate size 投影回 hidden size。 | 把 MLP 处理后的结果投回模型主干维度。 |

所以 `gate_proj` 和 `up_proj` 都是“升维到中间层”，适合按输出维切分；`down_proj` 是“从中间层降回 hidden”，适合按输入维切分。

加载时还有两个重要的结构转换：

1. **Q/K/V 合并**  
   HF checkpoint 里通常是独立的 `q_proj`、`k_proj`、`v_proj`；mini-sglang 的 attention wrapper 里使用 fused `qkv_proj`。所以加载时会先分别切分，再把 q/k/v 拼成 runtime 需要的 `qkv_proj`。

2. **Gate/Up 合并**  
   MLP 里 `gate_proj` 和 `up_proj` 也会被合并成 `gate_up_proj`，减少 runtime 中的投影调用和参数管理复杂度。

如果是 MoE 模型，还会把多个 expert 的权重 stack 到一个 packed tensor 里。这样 forward 时可以按 packed expert tensor 执行，而不是每个 expert 单独维护一套分散参数。

所以，Chapter 07 后面看到的 `qkv_proj`、`gate_up_proj`、TP shard 后的 embedding / lm_head，并不一定和 HF checkpoint 里的原始参数名完全一致。权重加载层负责完成这层“checkpoint 表达 → runtime 表达”的转换。

## 4. Qwen2ForCausalLM.forward

以 Qwen2 为例，`Qwen2ForCausalLM` 定义在 `python/minisgl/models/qwen2.py:66`。

forward 入口在 `python/minisgl/models/qwen2.py:77`：

```python
output = self.model.forward(get_global_ctx().batch.input_ids)
logits = self.lm_head.forward(output)
return logits
```

注意这里没有显式传 `input_ids` 参数，而是从 `get_global_ctx().batch.input_ids` 读取。这个 `batch.input_ids` 是 Scheduler 在 `_forward()` 中从 `token_pool[input_mapping]` 取出来的，位置在 `python/minisgl/scheduler/scheduler.py:229`。

## 5. Qwen2Model.forward

`Qwen2Model` 定义在 `python/minisgl/models/qwen2.py:44`，forward 在 `python/minisgl/models/qwen2.py:58`。

主线：

```text
input_ids
  ↓
VocabParallelEmbedding
  ↓
Qwen2DecoderLayer × num_layers
  ↓
final RMSNorm
  ↓
hidden states
```

每一步的含义是：

| 步骤 | 作用 |
| --- | --- |
| `input_ids` | 本轮模型真正要处理的 token ids。prefill 时可能是一段 prompt token；decode 时通常是每个请求最新的一个 token。 |
| `VocabParallelEmbedding` | 把离散 token id 查表成连续 hidden 向量。TP 场景下，词表 embedding 可以按 vocab 维切到不同 rank。 |
| `Qwen2DecoderLayer × num_layers` | 逐层执行 Transformer decoder block。每层都会做 self-attention、MLP 和残差更新，让 token 表示不断融合上下文信息。 |
| `final RMSNorm` | 所有 decoder layer 结束后做一次归一化，让输出 hidden states 的尺度更稳定，方便后续 LM head 计算 logits。 |
| `hidden states` | 模型主干最终输出的 token 表示，会继续送入 `lm_head` 映射到词表 logits。 |

decoder layer 定义在 `python/minisgl/models/qwen2.py:18`，forward 在 `python/minisgl/models/qwen2.py:33`。

其中，一层 decoder 的结构是：

```text
RMSNorm
  ↓
self_attn
  ↓
RMSNorm
  ↓
MLP
```

这一层里，第一段 `RMSNorm → self_attn` 负责让 token 读取上下文信息；第二段 `RMSNorm → MLP` 负责对每个 token 的 hidden 表示做非线性变换。两段之间通过 residual 连接保留原始信息，避免深层网络训练和推理时表示退化。

## 6. RopeAttn 与 AttentionLayer

通用 attention wrapper `RopeAttn` 定义在 `python/minisgl/models/utils.py:79`。

从 `Qwen2Model.forward()` 到真正进入 attention runtime 的调用链是：

```text
Qwen2Model.forward()
  ↓
Qwen2DecoderLayer.forward()
  ↓
self_attn.forward()
  ↓
RopeAttn.forward()
  ↓
AttentionLayer.forward()
```

每一层的职责可以这样看：

| 层级 | 作用 |
| --- | --- |
| `Qwen2Model.forward()` | 模型主干入口：做 embedding，循环执行所有 decoder layer，最后做 RMSNorm。 |
| `Qwen2DecoderLayer.forward()` | 单层 Transformer decoder：先做 self-attention，再做 MLP，并通过 norm / residual 维持表示稳定。 |
| `self_attn.forward()` | 当前 decoder layer 的 self-attention 入口。在 Qwen2 里，它实际指向 `RopeAttn.forward()`。 |
| `RopeAttn.forward()` | self-attention wrapper：负责 QKV 投影、调用 `AttentionLayer`、再做输出投影。 |
| `AttentionLayer.forward()` | 模型 attention 和 serving runtime 的交界：处理 q/k/v、RoPE、KV cache 读写，并调用具体 attention backend。 |

所以 `RopeAttn` 不是 decode 阶段专属模块，而是一层 decoder layer 里的 self-attention wrapper。prefill 和 decode 都会走到这里；区别是后面的 attention backend 会根据 batch phase 选择不同执行路径。

更准确地说，它先用 `qkv_proj` 把 hidden states 投影成 Q/K/V，再交给 `AttentionLayer` 处理 RoPE、KV cache 读写和具体 attention backend，最后用 `o_proj` 把 attention 输出投回 hidden size。

它在初始化时创建三个核心组件：

| 组件 | 作用 |
| --- | --- |
| `qkv_proj` | fused Q/K/V 投影，一次线性层输出 q、k、v 三部分。 |
| `AttentionLayer` | 负责 RoPE、q/k/v 拆分、调用 attention backend，并处理 KV cache 写入和 paged attention。 |
| `o_proj` | output projection，把 attention 输出投影回 hidden size，交还给 decoder layer 后续残差 / MLP。 |

`RopeAttn.forward()` 的主线是：

```text
x
  ↓ qkv_proj
qkv
  ↓ AttentionLayer.forward
o
  ↓ o_proj
```

每一步的含义是：

| 步骤 | 作用 |
| --- | --- |
| `x` | 当前 batch token 的 hidden states。 |
| `qkv_proj` | 把 hidden states 一次性投影成拼接后的 Q/K/V。这样比三个独立 projection 更适合 fused runtime。 |
| `qkv` | 包含 query、key、value 的合并张量，后续会被拆成 q、k、v。 |
| `AttentionLayer.forward` | 拆分 q/k/v，应用 RoPE，调用 attention backend，完成 KV cache 写入和 attention 计算。 |
| `o` | attention 计算后的输出，还处在 attention 子模块的输出空间。 |
| `o_proj` | 把 attention 输出投回模型 hidden size，使它可以回到 decoder layer 主干。 |

`AttentionLayer` 是连接模型层和 serving runtime 的关键位置。

它会：

| 步骤 | 作用 |
| --- | --- |
| 读取 global context | 拿到当前 batch、attention backend、KV cache、page table 等运行期对象。 |
| split q/k/v | 把 fused `qkv` 拆成 attention 需要的 query、key、value。 |
| 可选 q/k norm | 部分模型会对 q/k 做额外归一化，提高 attention 数值稳定性。 |
| 用 `ctx.batch.positions` 做 RoPE | 给 q/k 注入位置信息，保证 token 的相对/绝对位置能被 attention 感知。 |
| 调用 `ctx.attn_backend.forward(...)` | 进入具体 attention backend，完成 KV cache 写入、历史 KV 读取和 attention kernel 执行。 |

## 7. Attention Backend

attention backend 抽象定义在 `python/minisgl/attention/base.py:18`。

它有两个关键方法：

- `prepare_metadata(batch)`：在 Scheduler `_prepare_batch()` 中调用，用 batch 信息准备 kernel metadata。
- `forward(q, k, v, layer_id, batch)`：在模型 attention 层中调用，执行实际 attention。

mini-sglang 这里把 attention backend 做成可替换组件，是因为 prefill 和 decode 的 attention 形态差异很大：

| 阶段 | 输入形态 | attention 重点 |
| --- | --- | --- |
| prefill | 每个请求可能有一段 prompt token，`extend_len` 通常大于 1。 | 要处理一段 query tokens，并把这段 token 的 K/V 写入 KV cache；如果有 prefix cache，还要让新 token 看到已缓存前缀。 |
| decode | 每个请求通常只输入最新 1 个 token。 | 当前 token 的 Q 去读取完整历史 KV cache，生成下一个 token；更依赖 paged KV 的高效随机读取。 |

所以同一个 `AttentionLayer.forward()` 最后都会调用：

```text
ctx.attn_backend.forward(q, k, v, layer_id, batch)
```

但 backend 内部会根据 batch phase 选择不同实现：

```text
batch.is_prefill → prefill_backend
batch.is_decode  → decode_backend
```

这就是 Hybrid backend 的作用：模型层不用关心当前是 prefill 还是 decode，只把 q/k/v 和 batch 交给 attention backend；具体走 prefill kernel 还是 decode kernel，由 backend 根据 `batch.is_prefill / batch.is_decode` 决定。

以 FlashInfer backend 为例，它内部会准备两类 wrapper：

| wrapper | 用途 |
| --- | --- |
| `BatchPrefillWithPagedKVCacheWrapper` | 处理 prefill / extend prefill，一次处理多个 query tokens，并支持 paged KV cache。 |
| `BatchDecodeWithPagedKVCacheWrapper` | 处理 decode，每个请求通常一个 query token，重点是从 paged KV cache 中读历史 K/V。 |

无论 prefill 还是 decode，backend 都会先准备 metadata，再在 forward 中把当前 token 的 K/V 写入 KV cache，最后调用具体 attention kernel。差异主要在 metadata 形态和 kernel wrapper：prefill 需要描述每个请求的 query 长度和 context 长度；decode 则更像“每个请求 1 个 query + 一段历史 KV”。

## 8. KV cache 写入发生在哪里

以 FlashInfer backend 为例，forward 会先准备 metadata，然后把本层当前 token 的 k/v 写入 KV cache，再调用 wrapper 执行 attention。

从主线角度看，写入关系是：

```text
Scheduler: batch.out_loc = page_table[input_mapping]
  ↓
AttentionLayer: ctx.attn_backend.forward(q, k, v, layer_id, batch)
  ↓
Backend: kv_cache.store_kv(k, v, batch.out_loc, layer_id)
  ↓
MHAKVCache.store_kv → store_cache kernel
```

`MHAKVCache.store_kv()` 定义在 `python/minisgl/kvcache/mha_pool.py:45`。

## 9. LM Head 与 logits

`Qwen2ForCausalLM.forward()` 在模型输出 hidden states 后调用 `self.lm_head.forward(output)`，位置在 `python/minisgl/models/qwen2.py:79`。

LM head 可以理解成语言模型最后一层“分类器”：它把每个 token 位置的 hidden state 映射到整个词表空间，得到每个候选 token 的分数。

```text
hidden state
  ↓ LM Head
logits[vocab_size]
```

这里的 `logits` 还不是概率，而是每个词表 token 的原始分数。后续 sampler 会根据采样参数，对 logits 做 argmax、softmax、top-k、top-p 等处理，最终选出 next token。

举个例子，如果词表里有：

```text
["我", "你", "是", "猫", "狗", ...]
```

LM head 输出的 logits 可以理解成：

```text
"我" -> 1.2
"你" -> 0.8
"是" -> 5.6
"猫" -> 2.1
"狗" -> 1.9
```

分数越高，表示模型越倾向于把它作为下一个 token。TP 场景下，embedding / LM head 通常会按 vocab 或 hidden 维切分，并通过分布式通信完成必要聚合。

需要注意的是，prefill 阶段也会经过 LM head 和 sampler。普通 prefill 一次处理完整 prompt，sampler 产出的 token 就是第一个生成 token。decode 阶段则是每轮基于最新 token 继续采样下一个 token。

chunked prefill 要单独看：中间 chunk 虽然也会走统一的模型 forward / LM head / sampler 工程路径，但这个阶段的目标只是补 KV cache，采样结果不会被当成用户可见输出使用。只有最后一个 chunk 把完整 prompt prefill 完后，sampler 产出的 token 才会进入真正的生成流程。

可以这样记：

| 场景 | sampler 结果是否作为生成 token 使用 |
| --- | --- |
| 普通 prefill | 是，第一个生成 token。 |
| chunked prefill 中间 chunk | 否，只是在补 KV cache。 |
| chunked prefill 最后一个 chunk | 是，第一个生成 token。 |
| decode | 是，后续生成 token。 |

## 10. Sampler

`Sampler` 定义在 `python/minisgl/engine/sample.py:48`。

在 Scheduler `_prepare_batch()` 中会先调用 `sampler.prepare(batch)`，位置在 `python/minisgl/scheduler/scheduler.py:214`。`Sampler.prepare()` 会把每个 req 的 `SamplingParams` 变成 batch-level tensors，入口在 `python/minisgl/engine/sample.py:53`。

采样主要受每个请求的 `SamplingParams` 控制：

| 参数 | 作用 |
| --- | --- |
| `temperature` | 控制分布平滑程度。越低越确定，越高越随机；`temperature <= 0` 会走 greedy。 |
| `top_k` | 只在概率最高的前 k 个 token 里采样；`top_k = 1` 等价于 greedy 倾向。 |
| `top_p` | nucleus sampling，只在累计概率达到 p 的候选集合里采样；`top_p = 1.0` 表示不做 top-p 截断。 |

`Sampler.prepare()` 会把 batch 内每个请求的这些参数整理成 GPU tensor。这样同一个 batch 里不同请求可以有不同采样参数。

真正采样在 `Sampler.sample()`，定义在 `python/minisgl/engine/sample.py:70`：

- 如果所有请求都是 greedy，则直接 `torch.argmax(logits, dim=-1)`，位置在 `python/minisgl/engine/sample.py:73`。
- 否则调用 `sample_impl()`，先用 temperature 对 logits 做 softmax，再根据是否设置 top-k / top-p 选择普通采样、top-k 采样、top-p 采样或 top-k + top-p 组合采样，入口在 `python/minisgl/engine/sample.py:24`。

Engine 调用采样的位置在 `python/minisgl/engine/engine.py:202`。

## 11. 本章小结

模型侧主线可以总结为：

```text
batch.input_ids
  ↓
Qwen2ForCausalLM.forward
  ↓
Qwen2Model: embedding → decoder layers → norm
  ↓
RopeAttn: qkv_proj → AttentionLayer → o_proj
  ↓
AttentionLayer: RoPE + ctx.attn_backend.forward
  ↓
attention backend: store KV + paged attention
  ↓
LM Head
  ↓
logits
  ↓
Sampler.sample
  ↓
next token
```

下一章回到系统边界，看 next token 如何回传给 detokenizer，以及离线 LLM 入口如何复用同一套 Scheduler/Engine。
