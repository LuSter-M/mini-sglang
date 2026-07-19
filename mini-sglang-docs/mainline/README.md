# mini-sglang 主线学习路线 Chapters

这个目录整理 mini-sglang 源码的一条“主线阅读路径”。目标不是覆盖每个 kernel 或每个模型细节，而是先把一次请求从进入系统到返回文本的生命周期串起来。

推荐阅读顺序：

0. [Chapter 00](./chapter-00-mainline-overview.md)：mini-sglang 主线总览
1. [Chapter 01](./chapter-01-entry-and-process-topology.md)：入口、启动流程与多进程拓扑
2. [Chapter 02](./chapter-02-frontend-tokenizer-message.md)：Frontend、消息协议与 Tokenizer/Detokenizer
3. [Chapter 03](./chapter-03-scheduler-loop.md)：Scheduler 初始化与主循环
4. [Chapter 04](./chapter-04-prefill-decode-scheduling.md)：Prefill / Decode 调度
5. [Chapter 05](./chapter-05-table-kvcache-radix.md)：Table、Paged KV Cache 与 Radix Prefix Cache
6. [Chapter 06](./chapter-06-engine-init-forward.md)：Engine 初始化与 forward_batch
7. [Chapter 07](./chapter-07-model-attention-sampling.md)：模型 Forward、Attention Backend 与采样
8. [Chapter 08](./chapter-08-output-offline-reading-map.md)：结果回传、离线 LLM 与源码阅读地图

如果只想快速建立全局图，先读 Chapter 00；如果目标是理解一次在线 `/v1/chat/completions` 请求生命周期，建议按 Chapter 01 到 Chapter 08 顺序阅读。

主线核心循环可以先记成：

```text
Frontend/API
  ↓
Tokenizer
  ↓
Scheduler: receive → schedule → prepare → forward → postprocess
  ↓
Engine: model forward → sampler
  ↓
Detokenizer
  ↓
Frontend streaming response
```

