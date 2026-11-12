# 接口设计

本文是首版的接口设计，还没有实现。定位、分层和首版范围见 `CLAUDE.md`，本文只写接口和行为。名字和语义尽量对齐 Pi 的 `agent` 包，方便对照 Pi 的文档和 pi-python 的测试。

## 用起来是什么样

```python
import os

from agent_looplite import (
    ChatCompletionsClient,
    TextDelta,
    Tool,
    ToolOutput,
    UserMessage,
    run,
)


async def search_knowledge(args: dict, ctx) -> ToolOutput:
    hits = await kb.search(args["query"])
    return ToolOutput(content=render(hits), details=hits)


search = Tool(
    name="search_knowledge",
    description="检索知识库，返回带编号的原文片段。",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    execute=search_knowledge,
)

model = ChatCompletionsClient(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
)

handle = run(
    model=model,
    system="只依据检索到的资料回答。",
    messages=[UserMessage.of("今年 1 至 9 月集装箱吞吐量是多少？")],
    tools=[search],
)
async for event in handle:
    if isinstance(event, TextDelta):
        print(event.text, end="")
result = await handle.wait()
store.save(result.new_messages)
```

使用者只写三样东西：模型连接器的配置、工具、怎样消费事件。循环本身不用碰。

## 消息

消息是不可变的 dataclass，字段用元组，不用列表。循环每次追加消息时生成新的元组，钩子拿到的消息不会被循环改掉，所以不用深拷贝。

| 类型 | 字段 | 说明 |
|---|---|---|
| `UserMessage` | `content: tuple[TextContent, ...]` | `UserMessage.of(text)` 是便捷写法。首版只有文字 |
| `AssistantMessage` | `content: tuple[TextContent \| ThinkingContent \| ToolCall, ...]`，`stop_reason`，`usage`，`model` | 一次模型请求的完整回复 |
| `ToolResultMessage` | `tool_call_id`，`tool_name`，`content: str`，`is_error`，`details` | `content` 发给模型；`details` 给界面和日志，不发给模型 |
| `CustomMessage` | `kind: str`，`data` | 项目自己的消息，例如界面卡片。默认不发给模型，要发就在 `transform_context` 里转成别的消息 |

内容块：

| 类型 | 字段 | 说明 |
|---|---|---|
| `TextContent` | `text` | |
| `ThinkingContent` | `text`，`replay: dict \| None` | `replay` 存厂商要求原样回传的数据，例如 DeepSeek 的 `reasoning_content`、Anthropic 的签名。连接器负责写入和回传，循环不读 |
| `ToolCall` | `id`，`name`，`arguments: dict`，`arguments_error: str \| None` | 参数不是合法 JSON 时，`arguments` 为空，`arguments_error` 写明原因和模型发来的原文 |

系统提示词不是消息，是 `run()` 的参数。这样压缩、会话存储都不用特殊处理第一条消息。

`stop_reason` 的取值：`stop`（模型给出回答），`tool_use`（模型要调工具），`length`（输出达到上限被截断）。请求失败和被取消的回复不进消息列表，见“错误”和“取消”两节。

`Usage` 记录 `input`、`output`、`cache_read`、`cache_write`、`reasoning` 五项 token 数。不算费用，价格由使用者自己算。

序列化：`encode_messages(messages) -> list[dict]` 和 `decode_messages(data) -> tuple[Message, ...]`。每条带 `schema_version`，格式变化时升版本号，`decode_messages` 负责读旧版本。

## 事件

`run()` 返回的句柄可以 `async for` 迭代事件。事件只带增量，不带拼好的部分消息。

| 事件 | 字段 | 何时发出 |
|---|---|---|
| `RunStart` | | 开始 |
| `TurnStart` | `index` | 每次请求模型前 |
| `TextDelta` | `text` | 模型正文的一段 |
| `ThinkingDelta` | `text` | 模型思考的一段 |
| `ToolCallStart` | `call_id`，`name` | 模型开始写某个工具调用 |
| `ToolCallDelta` | `call_id`，`arguments` | 工具参数的一段原文 |
| `MessageEnd` | `message: AssistantMessage` | 一次回复结束，带完整消息和本次用量 |
| `ToolStart` | `call: ToolCall` | 开始执行工具 |
| `ToolProgress` | `call_id`，`data` | 工具报告进度 |
| `ToolEnd` | `result: ToolResultMessage`，`duration_ms` | 工具执行完，按完成顺序发出 |
| `TurnEnd` | `message`，`tool_results` | 本轮结束，`tool_results` 按调用顺序排列 |
| `RunEnd` | `result: RunResult` | 最后一个事件 |

连接器走整轮返回时，`TextDelta`、`ThinkingDelta`、`ToolCallStart`、`ToolCallDelta` 一个都不发，直接发 `MessageEnd`。界面要同时兼容两种情况，以 `MessageEnd` 里的完整消息为准。

事件放进一个无界队列，由后台任务产生。使用者不迭代也不会卡住循环。

## 模型连接器

连接器实现 `ModelClient` 协议：

```python
class ModelClient(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...
```

`ModelRequest` 包含 `system`、`messages`（已经过 `transform_context`，自定义消息已去掉）、`tools`（名字、说明、参数的 JSON Schema）。

`ModelEvent` 是 `TextDelta`、`ThinkingDelta`、`ToolCallStart`、`ToolCallDelta`，最后一个必须是 `ModelDone(message: AssistantMessage)`。失败时抛 `ModelError(message, retryable)`。

连接器负责的事，循环都不管：

- 把流式分片拼成完整的 `AssistantMessage`，工具 ID 不在第一片、参数分很多片都在这里处理。
- 参数拼完不是合法 JSON 时，写进 `arguments_error`，不抛异常。
- 重试。可以重试的错误在连接器里重试完，重试用尽再抛 `ModelError`。
- 识别网关错误，包括 HTTP 200 但正文是错误的情况。
- 出网前检查：生成最终请求体后、发出前，调用 `on_payload`。

内置的 `ChatCompletionsClient` 的参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `base_url`，`api_key`，`model` | 必填 | |
| `headers` | 无 | 额外的请求头。要求密钥不带 `Bearer` 的网关，`api_key` 留空，把密钥写在这里 |
| `stream` | `True` | `False` 时整轮返回 |
| `reasoning_field` | `reasoning_content` | 思考内容在响应里的字段名，有的网关叫 `reasoning` |
| `replay_reasoning` | `False` | 是否把思考内容随历史回传。DeepSeek 在工具调用的轮次要求回传 |
| `on_payload` | 无 | `async (body: dict) -> dict`，返回的请求体才会被发出。抛异常则这次请求失败，不重试 |
| `check_body` | 无 | `(body: dict) -> str \| None`，返回错误说明时当作请求失败。用来识别 HTTP 200 包着的网关错误 |
| `retry` | 3 次，指数退避 | 只重试 429、5xx 和连接错误 |
| `timeout` | 120 秒 | 单次请求的超时 |
| `settings` | 无 | 原样放进请求体的其他字段，例如 `temperature`、`max_tokens` |

接一个协议特殊的平台，例如只收发文本的政务百炼，就自己实现 `ModelClient`。

## 工具

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    execute: Callable[[dict, ToolContext], Awaitable[ToolOutput | str]]
    sequential: bool = False
    timeout: float | None = None  # 不填用 Limits.tool_timeout
```

`execute` 返回字符串时，当作 `ToolOutput(content=字符串)`。

```python
@dataclass(frozen=True)
class ToolOutput:
    content: str  # 发给模型
    details: Any = None  # 给界面和日志
    is_error: bool = False
    terminate: bool = False  # 本轮所有工具都带它时，循环不再请求模型
```

`ToolContext` 提供 `call_id` 和 `progress(data)`。`progress` 发出 `ToolProgress` 事件。工具需要的数据库连接、可见范围等，由使用者用闭包或 `functools.partial` 绑定，不经过 `ToolContext`。

以下情况都生成 `is_error=True` 的工具结果回传给模型，不结束运行：

| 情况 | 回传内容 |
|---|---|
| 工具名不存在 | 说明没有这个工具，列出可用的工具名 |
| `arguments_error` 不为空 | 连接器写的原因 |
| 参数不符合 JSON Schema | 第一处不符合的位置和原因 |
| 工具抛异常 | 异常类型和消息 |
| 超时 | 超时秒数 |
| `before_tool_call` 拦截 | 钩子给的原因 |

一轮里有多个工具调用时，默认并发执行，并发数受 `Limits.max_concurrency` 限制。只要其中一个工具声明了 `sequential=True`，这一轮全部按顺序执行。`ToolEnd` 按完成顺序发出；写回消息列表时按调用顺序排列，保证同样的输入得到同样的历史。

参数校验和从函数签名生成工具，取决于是否依赖 pydantic，见文末“待定”。

## 钩子

钩子都是可选的 async 函数，放在 `Hooks` 里传给 `run()`。

| 钩子 | 签名 | 用途 |
|---|---|---|
| `transform_context` | `(messages) -> messages` | 每次请求模型前调用。压缩上下文、把自定义消息转成模型能读的消息。返回值只用于这次请求，不改消息列表 |
| `before_tool_call` | `(call, args) -> Block \| None` | 参数校验通过后、执行前调用。返回 `Block(reason, terminate=False)` 则不执行 |
| `after_tool_call` | `(call, output) -> ToolOutput \| None` | 执行后调用。返回新的 `ToolOutput` 则替换原结果 |
| `finish_turn` | `(turn) -> "continue" \| "end" \| None` | 本轮工具都执行完、发 `TurnEnd` 前调用。`None` 表示按默认规则 |

出网前检查不是钩子，是连接器的 `on_payload` 参数。原因见 `CLAUDE.md`“分层”。

## 循环

```python
def run(
    *,
    model: ModelClient,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[Tool] = (),
    hooks: Hooks = Hooks(),
    limits: Limits = Limits(),
) -> RunHandle: ...
```

`run()` 立即返回句柄，后台任务开始执行。句柄提供：

- `async for event in handle`：迭代事件，直到 `RunEnd`。
- `await handle.wait() -> RunResult`：等运行结束。
- `handle.cancel()`：取消。

每一轮：

1. 检查上限：请求数已到 `max_requests`，结束。
2. `transform_context`，去掉自定义消息，交给连接器。
3. 转发连接器的增量事件；收到 `ModelDone` 后，把回复追加进消息列表，发 `MessageEnd`。
4. 回复里没有工具调用：调用 `finish_turn`，返回 `continue` 则回到第 1 步，否则以 `completed` 结束。
5. 执行工具，结果按调用顺序追加进消息列表。工具调用总数会超过 `max_tool_calls` 时，只执行还在额度内的调用，其余每个补一条说明上限已到的错误结果，然后以 `max_tool_calls` 结束。
6. 调用 `finish_turn`。返回 `end` 则结束；返回 `continue` 或本轮结果不全带 `terminate`，回到第 1 步；全带 `terminate` 则以 `terminated` 结束。

`RunResult`：

| 字段 | 说明 |
|---|---|
| `new_messages` | 本次运行追加的消息，按顺序。调用方存这一份即可 |
| `reason` | 结束原因，见下表 |
| `error` | `reason` 为 `model_error` 时的错误说明 |
| `usage` | 本次所有模型请求的用量合计 |
| `requests`，`tool_calls` | 实际发生的次数 |

结束原因：

| `reason` | 含义 |
|---|---|
| `completed` | 模型给出回答，没有再调工具 |
| `terminated` | 本轮所有工具结果都带 `terminate` |
| `finish_turn` | `finish_turn` 返回了 `end` |
| `max_requests`，`max_tool_calls`，`run_timeout` | 触发了对应的上限 |
| `cancelled` | 调用了 `cancel()` |
| `model_error` | 连接器抛了 `ModelError`，重试已用尽 |

`Limits` 的默认值：

| 字段 | 默认值 |
|---|---|
| `max_requests` | 50 |
| `max_tool_calls` | 200 |
| `max_concurrency` | 8 |
| `tool_timeout` | 60 秒 |
| `run_timeout` | 600 秒 |

默认值偏宽，只防止失控。业务按需要覆盖。

## 错误

连接器抛 `ModelError` 时，这次请求已经收到的部分回复丢弃，不进消息列表，运行以 `model_error` 结束。已经执行完的工具结果保留。这样消息列表始终可以直接用于下一次 `run()`。

钩子抛异常时，不吞掉，运行以异常结束：不发 `RunEnd`，迭代事件和 `handle.wait()` 都抛出同一个异常。钩子是使用者自己的代码，出错说明程序有问题，不该当作模型能修正的错误。

## 取消

`cancel()` 或 `run_timeout` 到期时：

- 正在请求模型：丢弃这次回复的部分内容。
- 正在执行工具：取消工具任务。已经完成的结果保留；没完成的，每个补一条 `is_error=True`、内容说明已取消的结果。
- 然后发 `RunEnd`，`reason` 为 `cancelled` 或 `run_timeout`。

工具结果补齐后，工具调用和结果在历史里一定成对，用户可以接着问。工具被取消时副作用是否已经发生，循环无法知道。需要保证不重复执行的工具，自己处理幂等。

## 测试支持

`agent_looplite.testing.ScriptedModel` 按预先写好的回复依次返回，可以选择流式还是整轮返回。本库自己的测试和使用者的测试都用它，不调真实模型。

和 pi-python 对照时，把真实模型的响应录下来，用同一份录制分别回放给两边，比较事件序列、工具执行顺序和 CPU 开销。

## 模块划分

| 模块 | 内容 | 依赖 |
|---|---|---|
| `messages.py` | 消息、内容块、`Usage`、序列化 | 无 |
| `events.py` | 事件 | `messages` |
| `model.py` | `ModelClient` 协议、`ModelRequest`、`ModelEvent`、`ModelError` | `messages`、`events` |
| `tools.py` | `Tool`、`ToolOutput`、`ToolContext`、参数校验、执行 | `messages`、`events` |
| `hooks.py` | `Hooks`、`Block` | `messages`、`tools` |
| `loop.py` | `run()`、`RunHandle`、`RunResult`、`Limits` | 以上全部 |
| `chat_completions.py` | `ChatCompletionsClient` | `model`、`messages`、httpx |
| `testing.py` | `ScriptedModel` | `model`、`messages` |

依赖只往上指，没有互相导入。运行依赖只有 httpx，pydantic 视下面的决定而定。

## 待定

| 事项 | 选项 | 影响 |
|---|---|---|
| 参数校验和函数式工具是否依赖 pydantic | 依赖：从函数的类型注解生成 JSON Schema 并校验，校验错误信息清楚。不依赖：使用者手写 JSON Schema，本库只校验必填字段和基本类型 | 决定 `@tool` 装饰器能不能有，以及校验的严格程度 |
| 首版是否提供 `Agent` 类和同步调用 | 提供：写脚本更方便。不提供：只有 `run()` | 不影响循环，随时可以加 |
| 取消和模型出错时，已收到的部分正文是否保留 | 丢弃（当前设计）：历史简单，一定合法。保留没有工具调用的部分正文：用户能看到断在哪里 | 影响 `stop_reason` 是否要加 `aborted`，以及界面怎样展示中断 |
