# Q / K / V 与 KV Cache：概念矫正

这篇文档把前面讨论中涉及的 Q / K / V 概念和推理引擎里的 KV cache 做一次集中梳理，避免几个常见混淆。

---

## 1. Q / K / V 是什么

Transformer attention 里，每个 token 在每一层都会产生三个向量：

| 符号 | 全称 | 角色 | 一句话理解 |
| --- | --- | --- | --- |
| Q | Query | 查询向量 | 我现在想找什么信息 |
| K | Key | 键向量 | 我这里有什么特征，能不能被匹配上 |
| V | Value | 值向量 | 如果你关注我，我真正提供给你的内容 |

attention 的计算分两步：

1. **匹配**：用当前 token 的 Q 去和所有 token 的 K 做点积，得到注意力权重。
2. **聚合**：用注意力权重对所有 token 的 V 做加权求和，得到当前 token 吸收上下文后的表示。

```text
attention_output = softmax(Q · Kᵀ / √d) · V
```

### 举例

句子：

```text
小明 把 苹果 放进 书包
```

处理到“书包”这个 token 时：

- “书包”产生一个 Q，相当于在问：和我有关的上下文在哪里？
- 前面每个 token 都有自己的 K / V：
  - 小明：K = 人物特征；V = “小明”这个内容
  - 苹果：K = 物品特征；V = “苹果”这个内容
  - 放进：K = 动作特征；V = “放进”这个内容
- Q 和 K 匹配后，得到对各个历史 token 的关注程度。
- 再用这个权重去加权 V，得到“书包”这个位置融合了上下文的表示。

---

## 2. Q / K / V 和推理引擎 KV cache 的关系

推理引擎里说的 **KV cache**，缓存的就是 Transformer 每一层、每个历史 token 计算出来的 **K 和 V**。

为什么只缓存 K / V，不缓存 Q？

因为自回归生成时：

```text
每生成一个新 token：
  新 token 的 Q 是新的，必须现算
  历史 token 的 K / V 不变，可以复用
```

所以 KV cache 的作用是：

> 把已经算过的历史 token 的 K / V 保存下来，避免每生成一个新 token 都从头重新计算整段 prompt 的 K / V。

一句话对应：

```text
Transformer 的 K/V：attention 层里的数学概念
推理引擎的 KV cache：把历史 K/V 张量保存下来的工程实现
```

---

## 3. Q / K / V 的基本单位是 token，不是 block

这是一个常见混淆点。

**Q / K / V 的基本语义单位是 token**：

```text
每一层 Transformer
每一个 token
都会产生自己的 Q / K / V 向量
```

vLLM / sglang 里的 **block / page** 不是 Q/K/V 的语义单位，而是**推理引擎为了管理 KV cache 显存，把多个 token 的 K/V 打包存储的内存管理单位**。

对比一下：

| 概念 | 基本单位 | 作用 |
| --- | --- | --- |
| Q / K / V | token | attention 的数学计算单位 |
| KV cache | token 的 K/V | 保存历史 token 的 K/V，避免重复计算 |
| block / page | 一组连续 token 的 K/V | 显存分页 / 分配 / 复用 / 调度管理单位 |

### 举例

假设 block size = 16。

不是说“16 个 token 共享一个 K/V”，而是：

```text
token 0 有自己的 K/V
token 1 有自己的 K/V
...
token 15 有自己的 K/V

这 16 个 token 的 K/V 被放在同一个 physical block 里管理
```

类比文件系统：

```text
文件内容的基本单位：字节
磁盘分配的基本单位：block
```

KV cache 也类似：

```text
attention 计算的基本单位：token 的 K/V
显存管理的基本单位：block / page
```

---

## 4. Decode 阶段新 token 的 Q / K / V 怎么处理

### 一个常见误解

先澄清一个最容易困惑的点：

> **decode 阶段每次只输入 1 个新 token，不是把所有历史 token 重新喂一遍。**

直觉上你可能以为，模型要"看到全部上下文"就得把全部 token 都输进去。但有了 KV cache 之后不是这样的：

- **输入侧**：只有最新 1 个 token 的 embedding 进入模型（因为只有它需要被算出 Q/K/V）
- **attention 侧**：这 1 个 token 的 Q 会去和 KV cache 里**全部历史 token 的 K/V** 做 attention

所以"看到全部上下文"这件事，是通过 KV cache 实现的，不是通过重新输入全部 token 实现的。历史 token 的 K/V 已经算好了、存在 cache 里了，不需要再算一遍。

这也是 KV cache 最核心的价值：把 decode 每一步的计算量从 O(n²) 降到 O(n)（n 是当前序列长度）。

### Decode 的完整流程

decode 阶段是一个循环：

```text
上一步 sample 出新 token（token id）
  ↓
下一轮把这个 token id 作为模型输入
  ↓
embedding + Transformer layers
  ↓
每一层为这个 token 计算 Q_new / K_new / V_new
  ↓
K_new / V_new 写入 KV cache
  ↓
Q_new 和「历史 KV cache + 当前 K/V」做 attention
  ↓
得到 logits
  ↓
sample 出下一个 token
```

几个关键点：

1. **sample 出的新 token 本身没有 Q/K/V**  
   它只是一个 token id，需要在下一轮 forward 中经过模型计算后才会有 Q/K/V。

2. **K_new / V_new 会写入 KV cache**  
   因为这个 token 之后会变成“历史 token”，供后续生成的 token 读取。

3. **Q_new 只用于当前这一步 attention**  
   它拿来查询历史上下文，不需要长期缓存。

4. **attention 读取的是历史 K/V + 当前 K/V**  
   这样当前 token 可以看到之前所有 token，并生成下一个 token 的 logits。

一句话总结：

> decode 时 sample 出的新 token，会在下一轮 forward 中被计算出 Q/K/V；其中 K/V 被追加写入 KV cache，Q 只用于本轮查询历史上下文，然后丢弃。

---

## 5. 一张总表

| 问题 | 答案 |
| --- | --- |
| Q/K/V 是什么？ | Transformer attention 层里每个 token 每层都会生成的三个向量。 |
| KV cache 是什么？ | 把历史 token 的 K/V 缓存起来，避免重复计算。 |
| 为什么只缓存 K/V，不缓存 Q？ | Q 每步都是新的，必须现算；历史 K/V 不变，可以复用。 |
| Q/K/V 的基本单位？ | token。每个 token 有自己的 Q/K/V。 |
| block / page 是什么？ | KV cache 的显存管理单位，把多个 token 的 K/V 放在一起管理。 |
| decode 时新 token 的 K/V 怎么来的？ | 下一轮 forward 中模型算出来的，算完后写入 KV cache。 |
| decode 时新 token 的 Q 怎么来的？ | 同样是下一轮 forward 中模型算出来的，只用于本轮 attention，不缓存。 |
