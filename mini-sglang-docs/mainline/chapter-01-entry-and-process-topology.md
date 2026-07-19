# Chapter 01：入口、启动流程与多进程拓扑

本章从 `python -m minisgl` 开始，梳理 mini-sglang 如何启动在线服务，以及每类进程承担什么职责。

## 1. 本章要回答的问题

1. CLI 入口在哪里？
2. API server、scheduler、tokenizer、detokenizer 是如何启动的？
3. TP rank 和 scheduler 进程是什么关系？
4. shell 模式与在线 server 模式共享哪些后端组件？

## 2. CLI 入口

Python 模块入口是 `python/minisgl/__main__.py:1`，它最终调用 `launch_server()`。

`launch_server()` 定义在 `python/minisgl/server/launch.py:40`，主要做三件事：

```text
parse_args
  ↓
定义 start_subprocess
  ↓
run_api_server(server_args, start_subprocess, run_shell)
```

`run_api_server()` 位于 `python/minisgl/server/api_server.py:411`，它会创建 `FrontendManager`，调用 `start_backend()` 启动后端，再运行 uvicorn 或 shell。

## 3. Scheduler 进程：每个 TP rank 一个

`launch_server()` 中会根据 `world_size = server_args.tp_info.size` 启动多个 scheduler 进程：

```text
for i in range(world_size):
    new_args.tp_info = DistributedInfo(i, world_size)
    Process(target=_run_scheduler, ...)
```

位置：`python/minisgl/server/launch.py:54` 到 `python/minisgl/server/launch.py:69`。

每个 scheduler 进程都会执行 `_run_scheduler()`，在其中创建 `Scheduler(args)` 并调用 `scheduler.run_forever()`，入口在 `python/minisgl/server/launch.py:16`。

这意味着：

```text
TP size = N
  ↓
N 个 Scheduler 进程
  ↓
每个 Scheduler 内部有自己的 Engine、模型 shard、KV cache shard
```

rank 0 是 primary rank，负责和 tokenizer / detokenizer 直接交互；其他 rank 通过 scheduler I/O 和分布式通信同步请求与执行。

## 4. Tokenizer 与 Detokenizer 进程

detokenizer 固定启动 1 个，位置在 `python/minisgl/server/launch.py:73`。

tokenizer worker 可以有多个，启动位置在 `python/minisgl/server/launch.py:88`。

这个设计背后的直觉是：**tokenize 更像请求入口侧的并行 CPU 预处理，detokenize 更像输出侧的集中式状态维护**。

tokenizer worker 可以有多个，主要因为：

- tokenize 面对的是用户新请求，多个请求之间基本独立；不同 prompt / chat messages 可以被不同 worker 并行编码。
- tokenize 可能包含 chat template 渲染、字符串处理和 tokenizer 编码，都是 CPU 侧工作；高并发入口下容易成为前端侧瓶颈。
- tokenize 完成后只需要产出 `UserMsg(uid, input_ids, sampling_params)` 发给 scheduler，不需要长期保存复杂状态。

detokenizer 固定 1 个，主要因为：

- 输出 token 只由 primary scheduler 持续产生，通常是每轮 decode 给每个活跃请求一个新 token，入口相对集中。
- detokenize 需要维护每个 `uid` 的增量 decode 状态，例如已 decode token、已发送文本偏移、用于避免输出半个字符或乱码的缓冲状态；集中在一个进程里更简单。
- 单个请求的输出必须按 token 生成顺序增量返回；如果拆成多个 detokenizer，需要额外做 per-uid 路由、状态同步和顺序保证。
- 相比模型 forward，detokenize 通常不是主要性能瓶颈；先用一个进程可以降低系统复杂度。

所以可以粗略理解为：

```text
tokenize：入口多请求、相互独立、可横向扩展
detokenize：出口状态机、按 uid 维护增量状态、集中处理更简单
```

另外，默认 `num_tokenizer=0` 时，mini-sglang 会让 tokenizer 地址和 detokenizer 地址共享：也就是只启动一个 `tokenize_worker()`，同时处理 `TokenizeMsg` 和 `DetokenizeMsg`。只有当用户配置了额外 tokenizer 数量时，才会启动多个独立 tokenizer worker，而 detokenizer 仍然保持 1 个。

两者复用同一个函数 `tokenize_worker()`，定义在 `python/minisgl/tokenizer/server.py:30`。它内部会同时创建：

- `TokenizeManager`：处理 `TokenizeMsg`。
- `DetokenizeManager`：处理 `DetokenizeMsg`。

区别主要来自监听地址和在拓扑中的角色：

```text
Frontend → tokenizer worker → Scheduler
Scheduler → detokenizer worker → Frontend
```

## 5. Frontend / API Server

`run_api_server()` 会初始化全局 `FrontendManager`，位置在 `python/minisgl/server/api_server.py:430` 到 `python/minisgl/server/api_server.py:443`。

`FrontendManager` 负责：

- 给请求分配 uid：`python/minisgl/server/api_server.py:109`。
- 发送 tokenizer 消息：`python/minisgl/server/api_server.py:130`。
- 监听 detokenizer 返回：`python/minisgl/server/api_server.py:116`。
- 等待用户增量输出：`python/minisgl/server/api_server.py:134`。
- 输出 SSE：`python/minisgl/server/api_server.py:152` 和 `python/minisgl/server/api_server.py:160`。

两个主要 HTTP 入口：

- `/generate`：定义在 `python/minisgl/server/api_server.py:228`。
- `/v1/chat/completions`：定义在 `python/minisgl/server/api_server.py:255`。

## 6. Shell 模式

如果用户传入 `--shell`，仍然会启动同一套后端组件，只是不运行 uvicorn，而是在 `run_api_server()` 里调用 `asyncio.run(shell())`，位置在 `python/minisgl/server/api_server.py:449` 到 `python/minisgl/server/api_server.py:452`。

shell 的请求构造逻辑在 `python/minisgl/server/api_server.py:351`，它会维护多轮对话历史，然后复用 `shell_completion()` 发起一次 chat completion 请求。

所以 shell 模式不是另一套推理引擎，它只是前端交互方式不同。

## 7. 本章小结

启动阶段可以总结为：

```text
python -m minisgl
  ↓
launch_server
  ↓
run_api_server
  ├─ 初始化 FrontendManager
  ├─ start_backend
  │    ├─ Scheduler × TP size
  │    ├─ tokenizer worker(s)
  │    └─ detokenizer worker
  └─ uvicorn or shell
```

下一章进入请求 I/O 层，看 HTTP request 如何变成内部消息，又如何被 tokenizer/detokenizer 转换。
