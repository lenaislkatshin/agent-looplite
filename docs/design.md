# 接口设计

本文是首版的接口设计，还没有实现。定位、分层和首版范围见 `CLAUDE.md`，本文只写接口和行为。

接口的结构、名字和语义对齐 Pi TS 原版的 `agent` 包，Python 侧的写法参考 pi-python。名字改成 Python 的 snake_case，例如 Pi 的 `beforeToolCall` 写作 `before_tool_call`，事件 `tool_execution_end` 对应类 `ToolExecutionEnd`。和 Pi 不一样的地方都列在文末“与 Pi 的差异”，没列出的就和 Pi 相同。

## 两层入口

和 Pi 一样分两层：

| 层 | 接口 | 对应 Pi | 用途 |
|---|---|---|---|
| 底层 | `agent_loop()`、`agent_loop_continue()` | `agentLoop()`、`agentLoopContinue()` | 无状态。调用方自己保存消息，适合服务端按请求运行 |
| 上层 | `Agent` | `Agent` | 持有系统提示词、模型、工具和消息，适合脚本和长期对话 |

两层用同一个循环实现，`Agent` 内部调用 `agent_loop()`。

```python
import os

from agent_looplite import (
    Agent,
    AgentToolResult,
    ChatCompletionsClient,
    MessageUpdate,
    TextContent,
    tool,
)

model = ChatCompletionsClient(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
)


@tool
async def search_knowledge(query: str) -> AgentToolResult:
    """检索知识库，返回带编号的原文片段。"""
    hits = await kb.search(query)
    return AgentToolResult(content=(TextContent(render(hits)),), details=hits)


agent = Agent(
    model=model, system_prompt="只依据检索到的资料回答。", tools=[search_knowledge]
)
agent.subscribe(
    lambda event: (
        print(event.delta, end="")
        if isinstance(event, MessageUpdate) and event.kind == "text_delta"
        else None
    )
)
await agent.prompt("今年 1 至 9 月集装箱吞吐量是多少？")
```

`@tool` 需要安装可选依赖 `agent-looplite[pydantic]`。不装时直接构造 `AgentTool`，手写参数的 JSON Schema，见“工具”。

## 消息

消息是不可变的 dataclass，字段用元组。循环追加消息时生成新的元组，钩子拿到的消息不会被循环改掉，所以不用深拷贝。

| 类型 | 字段 | 对应 Pi |
|---|---|---|
| `SystemMessage` | `content: str` | `SystemMessage`。可以出现在历史的任何位置，用来在运行中插入上下文 |
| `UserMessage` | `content: tuple[TextContent, ...]` | `UserMessage` |
| `AssistantMessage` | `content: tuple[TextContent \| ThinkingContent \| ToolCall, ...]`，`stop_reason`，`usage`，`model`，`error_message` | `AssistantMessage` |
| `ToolResultMessage` | `tool_call_id`，`tool_name`，`content: tuple[TextContent, ...]`，`details`，`is_error` | `ToolResultMessage`。`details` 给界面和日志，不发给模型 |
| `CustomMessage` | `kind: str`，`data` | Pi 用 TypeScript 的声明合并扩展消息类型，Python 用一个通用类型代替。由 `convert_to_llm` 决定发不发给模型 |

每条消息都有 `timestamp`，单位毫秒，和 Pi 一致。

内容块：

| 类型 | 字段 | 说明 |
|---|---|---|
| `TextContent` | `text` | |
| `ThinkingContent` | `thinking`，`signature`，`redacted` | `signature` 存厂商要求原样回传的数据，例如 Anthropic 的签名。连接器负责写入和回传，循环不读 |
| `ToolCall` | `id`，`name`，`arguments: dict`，`arguments_error: str \| None` | 参数不是合法 JSON 时，`arguments` 为空，`arguments_error` 写明原因和模型发来的原文 |

`stop_reason` 的取值和 Pi 相同：

| 取值 | 含义 |
|---|---|
| `stop` | 模型给出回答 |
| `tool_use` | 模型要调工具 |
| `length` | 输出达到上限被截断 |
| `error` | 请求失败，`error_message` 写明原因 |
| `aborted` | 被取消 |

`error` 和 `aborted` 的回复照样记进历史，保留中断前收到的正文，方便界面显示断在哪里。这类回复里的工具调用会被删掉，所以历史里的工具调用和结果总是成对的。连接器整理历史发给模型时，跳过这两类回复。

`Usage` 记录 `input`、`output`、`cache_read`、`cache_write`、`reasoning`、`total_tokens`。不算费用，Pi 的费用依赖它的模型目录，本库没有模型目录。

序列化：`encode_messages(messages) -> list[dict]` 和 `decode_messages(data) -> tuple[AgentMessage, ...]`。每条带 `schema_version`，格式变化时升版本号，`decode_messages` 负责读旧版本。

## 事件

事件的种类和顺序与 Pi 相同：

| 事件 | 字段 | 何时发出 |
|---|---|---|
| `AgentStart` | | 开始 |
| `AgentEnd` | `messages`，`reason` | 最后一个事件。`messages` 是本次新增的消息 |
| `TurnStart` | | 每次请求模型前 |
| `TurnEnd` | `message`，`tool_results` | 本轮结束，`tool_results` 按调用顺序排列 |
| `MessageStart` | `message` | 任何一条消息开始：传入的提示、模型回复、工具结果 |
| `MessageUpdate` | `kind`，`content_index`，`delta` | 只用于模型回复，带一段增量 |
| `MessageEnd` | `message` | 消息完成，带完整消息 |
| `ToolExecutionStart` | `tool_call_id`，`tool_name`，`args` | 开始执行工具 |
| `ToolExecutionUpdate` | `tool_call_id`，`tool_name`，`args`，`partial_result` | 工具报告进度 |
| `ToolExecutionEnd` | `tool_call_id`，`tool_name`，`result`，`is_error`，`duration_ms` | 工具执行完，按完成顺序发出 |

`MessageUpdate.kind` 取 `text_start`、`text_delta`、`text_end`、`thinking_start`、`thinking_delta`、`thinking_end`、`toolcall_start`、`toolcall_delta`、`toolcall_end`，和 Pi 的 `assistantMessageEvent` 相同。

`MessageUpdate` 只带增量，不带拼好的部分消息，这是和 Pi 的一处差异，原因见文末。需要部分消息的界面，用 `agent_looplite.accumulate()` 把增量拼起来。

连接器走整轮返回时，不发 `MessageUpdate`，`MessageStart` 之后直接是 `MessageEnd`。界面以 `MessageEnd` 里的完整消息为准，就能同时兼容两种情况。

## 模型连接器

连接器对应 Pi 的 `StreamFn`，实现 `ModelClient` 协议：

```python
class ModelClient(Protocol):
    def stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]: ...
```

`ModelContext` 包含已经过 `convert_to_llm` 的消息和工具声明（名字、说明、参数的 JSON Schema）。

`AssistantMessageEvent` 和 Pi 的同名类型相同，去掉了 `partial` 字段：`start`，`text_start` 到 `toolcall_end` 这九种增量，最后是 `done(reason, message)` 或 `error(reason, message)`。请求失败时不抛异常，发 `error` 事件，`message` 是一条 `stop_reason` 为 `error` 或 `aborted`、带着已收到内容的回复。这和 Pi 一致。

连接器负责的事，循环都不管：

- 把流式分片拼成完整的 `AssistantMessage`。工具 ID 不在第一片、参数分很多片，都在这里处理。
- 参数拼完不是合法 JSON 时，写进 `arguments_error`，不抛异常。
- 整理历史：跳过 `error` 和 `aborted` 的回复，按厂商要求回传思考内容。
- 重试。可以重试的错误在连接器里重试完，用尽后发 `error` 事件。
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
| `on_payload` | 无 | `async (body: dict) -> dict`，返回的请求体才会被发出。抛异常则这次请求以 `error` 结束，不重试 |
| `check_body` | 无 | `(body: dict) -> str \| None`，返回错误说明时当作请求失败。用来识别 HTTP 200 包着的网关错误 |
| `retry` | 3 次，指数退避 | 只重试 429、5xx 和连接错误 |
| `timeout` | 120 秒 | 单次请求的超时 |
| `settings` | 无 | 原样放进请求体的其他字段，例如 `temperature`、`max_tokens` |

接一个协议特殊的平台，例如只收发文本的政务百炼，就自己实现 `ModelClient`。

## 工具

```python
@dataclass(frozen=True)
class AgentTool:
    name: str
    label: str
    description: str
    parameters: dict  # JSON Schema
    execute: Callable[[str, dict, ToolUpdate], Awaitable[AgentToolResult]]
    execution_mode: Literal["parallel", "sequential"] | None = None


@dataclass(frozen=True)
class AgentToolResult:
    content: tuple[TextContent, ...]  # 发给模型
    details: Any = None  # 给界面和日志
    is_error: bool = False
    terminate: bool = False
```

`execute` 的三个参数依次是工具调用 ID、校验过的参数、报告进度的函数，对应 Pi 的 `execute(toolCallId, params, signal, onUpdate)`。取消靠 asyncio 取消工具任务，所以没有 `signal` 参数。`label` 是给界面显示的名字。工具需要的数据库连接、可见范围等，由使用者用闭包或 `functools.partial` 绑定。

参数用 `jsonschema` 按 `parameters` 校验，这和 pi-python 相同。`@tool` 装饰器从函数的类型注解和 docstring 生成 `AgentTool`，用 pydantic 生成 schema，相当于 Pi 里 TypeBox 的角色。pydantic 是可选依赖，核心不需要它。

以下情况都生成 `is_error=True` 的工具结果回传给模型，不结束运行。回传的文字和 Pi 一致，方便和 pi-python 对照：

| 情况 | 说明 |
|---|---|
| 工具名不存在 | |
| `arguments_error` 不为空 | 回传连接器写的原因 |
| 参数不符合 JSON Schema | 回传校验错误 |
| 工具抛异常 | 回传异常消息 |
| 超时 | 超过 `RunLimits.tool_timeout` |
| `before_tool_call` 拦截 | 回传钩子给的原因 |

执行方式和 Pi 相同：`AgentLoopConfig.tool_execution` 默认是 `parallel`。一轮中只要有一个工具声明了 `execution_mode="sequential"`，这一轮全部按顺序执行。并行时，`ToolExecutionEnd` 按完成顺序发出，工具结果消息和 `TurnEnd.tool_results` 按调用顺序排列。

## 钩子

钩子是 `AgentLoopConfig` 的字段，都是可选的 async 函数，和 Pi 相同：

| 钩子 | 返回值 | 用途 |
|---|---|---|
| `transform_context(messages)` | 新的消息列表 | 每次请求前，在 `convert_to_llm` 之前调用。压缩、补充外部上下文 |
| `convert_to_llm(messages)` | 发给模型的消息列表 | 去掉只给界面看的消息，把 `CustomMessage` 转成模型能读的消息。不填时去掉全部 `CustomMessage` |
| `before_tool_call(ctx)` | `BeforeToolCallResult(block, reason, terminate)` 或 `None` | 参数校验通过后、执行前调用，可以拦截 |
| `after_tool_call(ctx)` | `AfterToolCallResult(content, details, is_error, terminate)` 或 `None` | 执行后调用，填了的字段替换原结果，没填的保留 |
| `finish_turn(turn)` | `"continue"`、`"end"` 或 `None` | 本轮结束、发 `TurnEnd` 前调用。`None` 按默认规则 |

`transform_context` 和 `convert_to_llm` 的返回值只用于这次请求，不改消息列表。

`before_tool_call` 和 `after_tool_call` 拿到的上下文包含发起调用的回复、这次工具调用、校验过的参数和当前消息列表，`after_tool_call` 还有执行结果。

`finish_turn` 的语义和 Pi 相同：模型正常回复时，`continue` 保证再请求一次，`end` 在 `TurnEnd` 后立即结束；回复是 `error` 或 `aborted` 时，返回值被忽略，运行一定结束。

钩子抛异常时不吞掉，运行以异常结束：不发 `AgentEnd`，迭代事件和取结果都抛出同一个异常。钩子是使用者自己的代码，出错说明程序有问题，不该当作模型能修正的错误。

## 底层循环

```python
def agent_loop(
    prompts: Sequence[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
) -> EventStream: ...


def agent_loop_continue(
    context: AgentContext, config: AgentLoopConfig
) -> EventStream: ...
```

- `AgentContext`：`messages`（含系统消息）和 `tools`，和 Pi 相同。
- `AgentLoopConfig`：`model: ModelClient`，上面的钩子，`tool_execution`，`limits: RunLimits`。
- `agent_loop` 把 `prompts` 追加到历史末尾后开始；`agent_loop_continue` 从现有历史继续，最后一条必须是用户消息或工具结果，用于重试。

返回的 `EventStream` 和 Pi 一样，可以 `async for` 迭代事件，另外提供：

- `await stream.result() -> RunResult`：等运行结束。
- `stream.cancel()`：取消。

和 Pi 一样，底层的事件流只负责通知：循环不等使用者处理完事件就继续往下执行。需要“处理完这条消息再执行工具”这种先后保证的，用 `Agent`。

每一轮：

1. 检查请求数上限。
2. `transform_context`，`convert_to_llm`，交给连接器。
3. 转发连接器的增量事件；回复结束后追加进消息列表。
4. 回复是 `error` 或 `aborted`：调用 `finish_turn`，发 `TurnEnd`，结束。
5. 回复没有工具调用：调用 `finish_turn`，发 `TurnEnd`。返回 `continue` 则回到第 1 步，否则以 `completed` 结束。
6. 执行工具，结果按调用顺序追加进消息列表。工具调用总数会超过 `max_tool_calls` 时，只执行还在额度内的调用，其余每个补一条说明上限已到的错误结果，发 `TurnEnd` 后以 `max_tool_calls` 结束。
7. 调用 `finish_turn`，发 `TurnEnd`。返回 `end` 则结束；本轮结果全带 `terminate` 且没有返回 `continue`，以 `terminated` 结束；否则回到第 1 步。

`RunResult`：

| 字段 | 说明 |
|---|---|
| `messages` | 本次新增的消息，按顺序，包括传入的 `prompts`。调用方存这一份即可 |
| `reason` | 结束原因，见下表 |
| `usage` | 本次所有模型请求的用量合计 |

结束原因：

| `reason` | 含义 |
|---|---|
| `completed` | 模型给出回答，没有再调工具 |
| `terminated` | 本轮所有工具结果都带 `terminate` |
| `finish_turn` | `finish_turn` 返回了 `end` |
| `error` | 模型请求失败，重试已用尽，最后一条回复带 `error_message` |
| `aborted` | 被取消 |
| `max_requests`，`max_tool_calls`，`run_timeout` | 触发了对应的上限 |

`RunLimits` 的字段名沿用 pi-python，默认值是本库加的：

| 字段 | 默认值 |
|---|---|
| `max_model_requests` | 50 |
| `max_tool_calls` | 200 |
| `max_concurrency` | 8 |
| `tool_timeout` | 60 秒 |
| `run_timeout` | 600 秒 |

默认值偏宽，只防止失控。业务按需要覆盖。

## 取消

`cancel()`、`Agent.abort()` 或 `run_timeout` 到期时，行为和 Pi 相同：

- 正在请求模型：这次回复以 `aborted` 记进历史，保留已收到的正文，删掉其中的工具调用。
- 正在执行工具：取消工具任务。已经完成的结果保留；没完成的，每个补一条 `is_error=True`、内容为 `Operation aborted` 的结果。
- 然后发 `TurnEnd` 和 `AgentEnd`。

这样工具调用和结果在历史里一定成对，用户可以接着问。工具被取消时副作用是否已经发生，循环无法知道。需要保证不重复执行的工具，自己处理幂等。

## Agent

```python
agent = Agent(
    model=model,
    system_prompt="……",
    tools=[...],
    messages=[...],
    transform_context=...,
    convert_to_llm=...,
    before_tool_call=...,
    after_tool_call=...,
    finish_turn=...,
    tool_execution="parallel",
    limits=RunLimits(),
)
```

| 方法和属性 | 对应 Pi | 说明 |
|---|---|---|
| `await prompt(text_or_message)` | `prompt()` | 追加一条用户消息并运行，结束后返回 `RunResult` |
| `await continue_run()` | `continue()` | 从现有历史继续。Python 里 `continue` 是关键字，名字沿用 pi-python |
| `abort()` | `abort()` | 取消当前运行 |
| `await wait_for_idle()` | `waitForIdle()` | 等当前运行结束 |
| `subscribe(listener)` | `subscribe()` | 订阅事件，返回取消订阅的函数 |
| `state` | `state` | `system_prompt`、`model`、`tools`、`messages`、`is_streaming`、`pending_tool_calls`、`error_message` |
| `reset()` | `reset()` | 清空消息 |

订阅的监听函数可以是 async 的，和 Pi 一样按注册顺序依次等待完成。模型回复的 `MessageEnd` 处理完之后才开始执行工具，所以 `before_tool_call` 看到的消息列表里已经有这条回复。

同一时间只能有一次运行。运行中再调用 `prompt()` 会报错。

## 测试支持

`agent_looplite.testing.ScriptedModel` 按预先写好的回复依次返回，可以选择流式还是整轮返回。本库自己的测试和使用者的测试都用它，不调真实模型。

和 pi-python 对照时，把真实模型的响应录下来，用同一份录制分别回放给两边，比较事件序列、工具执行顺序和 CPU 开销。

## 模块划分

| 模块 | 内容 | 依赖 |
|---|---|---|
| `messages.py` | 消息、内容块、`Usage`、序列化 | 无 |
| `events.py` | 事件、`accumulate()` | `messages` |
| `model.py` | `ModelClient` 协议、`ModelContext`、`AssistantMessageEvent` | `messages` |
| `tools.py` | `AgentTool`、`AgentToolResult`、参数校验、执行 | `messages`、`events`、jsonschema |
| `loop.py` | `agent_loop()`、`agent_loop_continue()`、`EventStream`、`AgentLoopConfig`、`RunLimits`、`RunResult`、钩子的类型 | 以上全部 |
| `agent.py` | `Agent` | `loop` |
| `chat_completions.py` | `ChatCompletionsClient` | `model`、`messages`、httpx |
| `function_tool.py` | `@tool` | `tools`、pydantic（可选） |
| `testing.py` | `ScriptedModel` | `model`、`messages` |

依赖只往上指，没有互相导入。运行依赖是 httpx 和 jsonschema，pydantic 可选。

## 与 Pi 的差异

| 地方 | Pi | 本库 | 原因 |
|---|---|---|---|
| `MessageUpdate` | 带增量和拼好的部分消息 | 只带增量，另给 `accumulate()` | Python 里每个增量都生成一份部分消息，开销随回复长度平方增长。pi-python 在 50 路并发时占单核 35%，原因就在这里 |
| 运行上限 | 没有 | `RunLimits`，有默认值 | 没有上限时，模型可能一直调用工具，成本和时延失控 |
| 取消 | 传入 `AbortSignal` | `stream.cancel()`、`Agent.abort()`，内部用 asyncio 取消 | Python 没有 `AbortSignal`，asyncio 的取消是标准做法 |
| 结束原因 | `agent_end` 只带消息 | `AgentEnd` 和 `RunResult` 带 `reason` | 有了运行上限，调用方需要知道是正常结束还是触发了上限 |
| 模型 | 模型目录加 `streamFn`，模型目录管费用和上下文长度 | 只有 `ModelClient` | 本库不维护模型目录 |
| 出网前检查 | `onPayload` 作为请求选项，由循环传给 `streamFn` | `on_payload` 是连接器的构造参数 | 换连接器时检查跟着连接器走，调用方不会忘传 |
| 工具超时 | 没有 | `RunLimits.tool_timeout` | 一个卡住的工具不应拖住整次运行 |
| `prepare_request` | 有 | 首版不做 | 用途可以由 `transform_context` 覆盖 |
| 插话和排队 | `steer()`、`followUp()` | 首版不做 | 见 `CLAUDE.md` |
| 多模态、思考等级、`thinkingBudgets`、`sessionId`、`getApiKey` | 有 | 首版不做 | 首版只做 Chat Completions，这些由连接器的 `settings` 和 `headers` 承担 |
