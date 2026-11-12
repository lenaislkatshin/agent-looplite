# 接口设计

定位、分层和首版范围见 `CLAUDE.md`，本文写接口和行为。

接口的结构、名字和语义对齐 Pi TS 原版的 `agent` 包，Python 侧的写法参考 pi-python。名字改成 Python 的 snake_case，例如 Pi 的 `beforeToolCall` 写作 `before_tool_call`，事件 `tool_execution_end` 对应类 `ToolExecutionEnd`。和 Pi 不一样的地方都列在文末“与 Pi 的差异”，没列出的就和 Pi 相同。

## 两层入口

和 Pi 一样分两层：

| 层 | 接口 | 对应 Pi | 用途 |
|---|---|---|---|
| 底层 | `agent_loop()`、`agent_loop_continue()` | `agentLoop()`、`agentLoopContinue()` | 无状态。调用方自己保存消息，适合服务端按请求运行 |
| 上层 | `Agent` | `Agent` | 持有系统提示词、模型、工具和消息，适合脚本和长期对话 |

两层共用 `loop.AgentRun`，区别只在事件怎样交出去：底层放进队列，`Agent` 依次等监听函数处理完。

```python
import os

from agent_looplite import Agent, ChatCompletionsClient, MessageUpdate, tool

model = ChatCompletionsClient(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
)


@tool
async def search_knowledge(query: str) -> str:
    """检索知识库，返回带编号的原文片段。"""
    return render(await kb.search(query))


def print_text(event):
    if isinstance(event, MessageUpdate) and event.kind == "text_delta":
        print(event.delta, end="")


agent = Agent(model=model, system_prompt="只依据检索到的资料回答。", tools=[search_knowledge])
agent.subscribe(print_text)
await agent.prompt("今年 1 至 9 月集装箱吞吐量是多少？")
```

`@tool` 需要安装可选依赖 `agent-looplite[pydantic]`。不装时直接构造 `AgentTool`，手写参数的 JSON Schema，见“工具”。

`scripts/e2e.py` 是一个用真实模型跑的完整例子。

## 消息

消息是不可变的 dataclass（`messages.py`），内容用元组。循环追加消息时不改已有的对象，钩子拿到的消息不会被循环改掉，所以不用深拷贝。

| 类型 | 字段 | 对应 Pi |
|---|---|---|
| `SystemMessage` | `content: str` | `SystemMessage`。可以出现在历史的任何位置 |
| `UserMessage` | `content: tuple[TextContent, ...]` | `UserMessage`。`UserMessage.of(text)` 是便捷写法 |
| `AssistantMessage` | `content`，`stop_reason`，`usage`，`model`，`error_message` | `AssistantMessage`。`text` 和 `tool_calls` 两个属性方便读取 |
| `ToolResultMessage` | `tool_call_id`，`tool_name`，`content: tuple[TextContent, ...]`，`details`，`is_error` | `ToolResultMessage`。`details` 给界面和日志，不发给模型 |
| `CustomMessage` | `kind: str`，`data` | Pi 用 TypeScript 的声明合并扩展消息类型，Python 用一个通用类型代替。由 `convert_to_llm` 决定发不发给模型 |

每条消息都有 `timestamp`，单位毫秒，和 Pi 一致。

内容块：

| 类型 | 字段 | 说明 |
|---|---|---|
| `TextContent` | `text` | |
| `ThinkingContent` | `thinking`，`signature`，`redacted` | `signature` 存厂商要求原样回传的数据。连接器负责写入和回传，循环不读 |
| `ToolCall` | `id`，`name`，`arguments: dict`，`arguments_error: str \| None` | 参数不是合法 JSON 对象时，`arguments` 为空，`arguments_error` 写明原因和模型发来的原文 |

`stop_reason` 的取值和 Pi 相同：

| 取值 | 含义 |
|---|---|
| `stop` | 模型给出回答 |
| `tool_use` | 模型要调工具 |
| `length` | 输出达到上限被截断 |
| `error` | 请求失败，`error_message` 写明原因 |
| `aborted` | 被取消 |

`error` 和 `aborted` 的回复照样记进历史，保留中断前收到的正文，方便界面显示断在哪里。这类回复里的工具调用会被删掉，所以历史里的工具调用和结果总是成对的。连接器整理历史发给模型时，跳过这两类回复。

`Usage` 记录 `input`、`output`、`cache_read`、`cache_write`、`reasoning`、`total_tokens`，可以相加。`input` 不含命中缓存的部分。不算费用。

序列化：`encode_messages(messages) -> list[dict]` 和 `decode_messages(data) -> tuple[AgentMessage, ...]`。每条带 `schema_version`，现在是 1。格式变化时升版本号，`decode_messages` 负责读旧版本。`details` 和 `CustomMessage.data` 原样写入，调用方保证它们能转成 JSON。

## 事件

事件的种类和顺序与 Pi 相同（`events.py`），每个类的 `type` 是 Pi 的事件名。一次运行的顺序是：`AgentStart`，`TurnStart`，传入提示的 `MessageStart` 和 `MessageEnd`，模型回复的 `MessageStart`、若干 `MessageUpdate`、`MessageEnd`，工具执行和工具结果消息，`TurnEnd`；之后每轮从 `TurnStart` 开始；最后是 `AgentEnd`。工具结果消息什么时候发，见“工具”。

| 事件 | 字段 | 何时发出 |
|---|---|---|
| `AgentStart` | | 开始 |
| `AgentEnd` | `messages`，`reason` | 最后一个事件。`messages` 是本次新增的消息 |
| `TurnStart` | | 每次请求模型前 |
| `TurnEnd` | `message`，`tool_results` | 本轮结束，`tool_results` 按调用顺序排列 |
| `MessageStart` | `role`，`message` | 任何一条消息开始。模型回复开始时还没有内容，`message` 为 `None` |
| `MessageUpdate` | `kind`，`content_index`，`delta`，`tool_call_id`，`tool_name` | 只用于模型回复，带一段增量。`toolcall_start` 带上已知的工具调用 ID 和名字 |
| `MessageEnd` | `message` | 消息完成，带完整消息 |
| `ToolExecutionStart` | `tool_call_id`，`tool_name`，`args` | 开始执行工具 |
| `ToolExecutionUpdate` | `tool_call_id`，`tool_name`，`args`，`partial_result` | 工具报告进度 |
| `ToolExecutionEnd` | `tool_call_id`，`tool_name`，`result`，`is_error`，`duration_ms` | 工具执行完，按完成顺序发出。工具没有真正执行时 `duration_ms` 为 `None` |

`MessageUpdate.kind` 取 `text_start`、`text_delta`、`text_end`、`thinking_start`、`thinking_delta`、`thinking_end`、`toolcall_start`、`toolcall_delta`、`toolcall_end`，和 Pi 的 `assistantMessageEvent` 相同。

`MessageUpdate` 只带增量，不带拼好的部分消息，这是和 Pi 的一处差异。需要部分消息的界面，用 `PartialMessage` 把增量拼起来：`apply()` 只追加，读 `text`、`thinking` 或 `content()` 时才拼接字符串。

连接器走整轮返回时，不发 `MessageUpdate`，`MessageStart` 之后直接是 `MessageEnd`。界面以 `MessageEnd` 里的完整消息为准，就能同时兼容两种情况。

## 模型连接器

连接器对应 Pi 的 `StreamFn`，实现 `ModelClient` 协议（`model.py`）：

```python
class ModelClient(Protocol):
    def stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]: ...
```

`ModelContext` 包含已经过 `transform_context` 和 `convert_to_llm` 的消息，以及工具声明（名字、说明、参数的 JSON Schema）。

连接器发出的事件：

| 事件 | 对应 Pi | 说明 |
|---|---|---|
| `StreamDelta(kind, content_index, delta, tool_call_id, tool_name)` | `text_start` 到 `toolcall_end` 九种 | 去掉了 Pi 的 `partial` 字段，循环原样转成 `MessageUpdate` |
| `StreamDone(message)` | `done` | 最后一个事件。`message.stop_reason` 是 `stop`、`tool_use` 或 `length` |
| `StreamError(message)` | `error` | 最后一个事件。`message.stop_reason` 是 `error` 或 `aborted`，带着出错前已收到的内容 |

请求失败时不抛异常，发 `StreamError`，这和 Pi 一致。连接器的事件流在没有 `StreamDone` 或 `StreamError` 时就结束，循环把它当作 `error`。循环只执行 `StreamDone` 回复里的工具调用：`StreamError` 回复里的工具调用会被删掉，不执行。

连接器负责的事，循环都不管：

- 把流式分片拼成完整的 `AssistantMessage`。工具 ID 和名字不在第一片、参数分很多片，都在这里处理。
- 参数拼完不是合法 JSON 对象时，写进 `arguments_error`，不抛异常。`NaN`、`Infinity` 也算不合法：它们不是 JSON，回传后再发给模型会出错。
- 整理历史：跳过 `error` 和 `aborted` 的回复，按需要回传思考内容。用户消息和工具结果的内容拼成一个字符串发送；工具参数紧凑序列化，和 Pi 相同。
- 分块：一条回复最多一个正文块和一个思考块，工具调用按分片里的 `index` 或 ID 归到各自的块。工具调用的每个分片都发一个 `toolcall_delta`，没有参数时增量为空字符串。所有块在流结束时按顺序结束。这些都和 Pi 相同。
- 重试。还没收到任何内容时，连接错误、超时、408、409、429 和 5xx 会重试；已经发出增量后出错，不重试，发 `StreamError`。
- 识别网关错误，包括 HTTP 200 但正文是错误的情况。
- 判定成功：只有明确的结束原因才算成功，规则同 Pi 的 `mapStopReason`。`stop`、`end` 是正常回答，`length` 是被截断，`tool_calls`、`function_call` 是调用工具；`network_error`、`content_filter` 等其他原因都是失败。响应没有结束原因，包括流正常关闭却没发结束原因，也算失败：这时已经拼出的工具参数可能看起来合法，执行它们可能产生实际副作用。确实不发结束原因的网关，设 `supports_finish_reason=False`。
- 出网前检查：生成最终请求体后、发出前，调用 `on_payload`。

内置的 `ChatCompletionsClient`（`chat_completions.py`）的参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `base_url`，`model` | 必填 | `base_url` 写到版本号为止，请求发到 `{base_url}/chat/completions` |
| `api_key` | 无 | 填了就发 `Authorization: Bearer <api_key>` |
| `headers` | 无 | 额外的请求头，覆盖同名的头。要求密钥不带 `Bearer` 的网关，`api_key` 留空，把密钥写在这里 |
| `stream` | `True` | `False` 时整轮返回 |
| `stream_usage` | `True` | 流式时请求 `stream_options.include_usage`，拿到用量。不支持这个字段的网关设为 `False` |
| `reasoning_field` | `reasoning_content` | 思考内容在响应里的字段名，有的网关叫 `reasoning` |
| `supports_finish_reason` | `True` | 网关会发结束原因。设为 `False` 时，没有结束原因的响应按内容判定：有工具调用算 `tool_use`，否则算 `stop`。对应 Pi 的 `compat.supportsFinishReason` |
| `replay_reasoning` | `False` | 是否把思考内容随历史回传。DeepSeek 在工具调用的轮次要求回传 |
| `on_payload` | 无 | `async (body: dict) -> dict`，返回的请求体才会被发出。抛异常则这次请求以 `error` 结束，不重试 |
| `check_body` | 无 | `(body: dict) -> str \| None`，返回错误说明时当作请求失败。整轮的响应体和流式的每个分片都会检查。没填时，只识别没有 `choices` 却有 `error` 字段的响应 |
| `max_retries` | 3 | 重试次数，不含第一次 |
| `retry_base_delay` | 1 秒 | 第 n 次重试前等待 `retry_base_delay * 2**(n-1)` 秒 |
| `timeout` | 120 秒 | 单次请求的超时 |
| `settings` | 无 | 最后合并进请求体的其他字段，例如 `temperature`、`max_tokens` |
| `http_client` | 无 | 传入自己的 `httpx.AsyncClient`，例如测试时用 `MockTransport`。不传时自己创建，`aclose()` 或 `async with` 时关闭 |

接一个协议特殊的平台，例如只收发文本的政务百炼，就自己实现 `ModelClient`。

出网前检查是连接器的职责：内置连接器在发送前调用已配置的 `on_payload`；自定义连接器自己负责出网检查，循环不会替它做。有强制出网检查要求的项目，接入时核实实际使用的连接器在发送前做了检查。

## 工具

```python
@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[[str, dict[str, Any], ToolUpdate], Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["parallel", "sequential"] | None = None


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    content: tuple[TextContent, ...]  # 发给模型
    details: Any = None  # 给界面和日志
    is_error: bool = False
    terminate: bool = False
```

`execute` 的三个参数依次是工具调用 ID、校验过的参数、报告进度的 async 函数，对应 Pi 的 `execute(toolCallId, params, signal, onUpdate)`。`await on_update(partial_result)` 发出 `ToolExecutionUpdate`；工具结束后再调用会被忽略。取消靠 asyncio 取消工具任务，所以没有 `signal` 参数。`label` 是给界面显示的名字。工具需要的数据库连接、可见范围等，由使用者用闭包或 `functools.partial` 绑定。`AgentToolResult.of(text, details)` 是便捷写法。

一次运行开始前，循环检查每个工具的 `parameters` 本身是不是合法的 JSON Schema，工具名有没有重复，有问题直接抛异常。运行中用 `jsonschema` 按 `parameters` 校验参数，这和 pi-python 相同。

`@tool`（`function_tool.py`）从 async 函数的类型注解和 docstring 生成 `AgentTool`。参数格式由 pydantic 生成，不允许多余的字段；执行前再用 pydantic 转换一次，所以参数可以是 pydantic 模型。函数返回字符串时当作 `AgentToolResult.of(字符串)`。可以用 `@tool(name=..., label=..., description=..., execution_mode=...)` 覆盖默认值。pydantic 在这里相当于 Pi 里 TypeBox 的角色，是可选依赖，核心不需要它。

以下情况都生成 `is_error=True` 的工具结果回传给模型，不结束运行。回传的文字和 Pi、pi-python 一致：

| 情况 | 回传内容 |
|---|---|
| 工具名不存在 | `Tool {name} not found` |
| `arguments_error` 不为空 | `Invalid arguments for tool "{name}": {原因}` |
| 参数不符合 JSON Schema | `Validation failed for tool "{name}":`，列出最多 5 处问题和收到的参数 |
| 工具抛异常 | 异常消息 |
| 超时 | `Tool timed out` |
| `before_tool_call` 拦截 | 钩子给的原因，没给时是 `Tool execution was blocked` |
| 回复因输出上限被截断（`stop_reason` 是 `length`） | 其中的工具调用都不执行，回传参数可能不完整、请重新调用的说明。这条规则来自 Pi |
| 这一轮的调用数会超过 `max_tool_calls` | 整轮都不执行，回传 `Entire tool batch exceeds remaining budget`。这条规则来自 pi-python |
| 被取消 | `Operation aborted` |

执行方式和 Pi 相同：`AgentLoopConfig.tool_execution` 默认是 `parallel`，并发数受 `RunLimits.max_concurrency` 限制。一轮中只要有一个工具声明了 `execution_mode="sequential"`，这一轮全部按顺序执行。

- 顺序执行：每个调用依次校验、调用 `before_tool_call`、执行，结束后立即发出它的工具结果消息，再开始下一个。
- 并行执行：先按顺序做完每个调用的校验和 `before_tool_call`，再并发执行。`ToolExecutionEnd` 按完成顺序发出；全部结束后，工具结果消息按调用顺序发出。
- 回复被截断或超过工具调用上限时，每个调用依次发 `ToolExecutionStart`、`ToolExecutionEnd` 和结果消息。

两种方式下，写进历史的工具结果和 `TurnEnd.tool_results` 都按调用顺序排列。

每个调用的参数复制一份交给钩子和工具，和 Pi 相同。`before_tool_call` 修改参数会影响工具拿到的值，但历史里的工具调用仍是模型发来的原始参数。

## 钩子

钩子是 `AgentLoopConfig` 的字段，都是可选的 async 函数，和 Pi 相同：

| 钩子 | 返回值 | 用途 |
|---|---|---|
| `transform_context(messages)` | 新的消息列表 | 每次请求前，在 `convert_to_llm` 之前调用。压缩、补充外部上下文 |
| `convert_to_llm(messages)` | 发给模型的消息列表 | 把 `CustomMessage` 转成模型能读的消息。不填时去掉全部 `CustomMessage`。返回的列表里不能再有 `CustomMessage` |
| `before_tool_call(ctx)` | `BeforeToolCallResult(block, reason, terminate)` 或 `None` | 参数校验通过后、执行前调用，可以拦截 |
| `after_tool_call(ctx)` | `AfterToolCallResult(content, details, is_error, terminate)` 或 `None` | 执行后调用，填了的字段替换原结果，没填的保留。`details` 可以替换成 `None`，所以没填用 `UNSET` 表示 |
| `finish_turn(turn)` | `"continue"`、`"end"` 或 `None` | 本轮结束、发 `TurnEnd` 前调用。`None` 按默认规则 |

`transform_context` 和 `convert_to_llm` 的返回值只用于这次请求，不改消息列表。

`before_tool_call` 和 `after_tool_call` 的上下文（`BeforeToolCallContext`、`AfterToolCallContext`）包含发起调用的回复、这次工具调用、校验过的参数和当前消息列表，`after_tool_call` 还有执行结果。`finish_turn` 的 `TurnContext` 包含本轮的回复、工具结果和当前消息列表。

`finish_turn` 的语义和 Pi 相同：模型正常回复时，`continue` 保证再请求一次，`end` 在 `TurnEnd` 后立即结束；回复是 `error` 或 `aborted`，或者工具执行被取消时，返回值被忽略，运行一定结束。

钩子或 `Agent` 的监听函数抛异常时不吞掉，运行以异常结束：不发 `TurnEnd` 和 `AgentEnd`，迭代事件和取结果都抛出同一个异常。钩子是使用者自己的代码，出错说明程序有问题，不该当作模型能修正的错误。抛出前，给已写入的回复里没有结果的工具调用补上 `is_error=True` 的结果，写明运行因什么异常中断，所以历史仍可直接用于下一次运行。补结果时再出错，只向外抛最初的异常。

## 底层循环

```python
def agent_loop(
    prompts: Sequence[AgentMessage], context: AgentContext, config: AgentLoopConfig
) -> EventStream: ...


def agent_loop_continue(context: AgentContext, config: AgentLoopConfig) -> EventStream: ...
```

- `AgentContext`：`messages`（含系统消息）和 `tools`，和 Pi 相同。
- `AgentLoopConfig`：`model: ModelClient`，上面的钩子，`tool_execution`，`limits: RunLimits`。
- `agent_loop` 把 `prompts` 追加到历史末尾后开始；`agent_loop_continue` 从现有历史继续，用于重试，最后一条标准消息必须是用户消息或工具结果，否则立即抛 `ValueError`。
- 两者都必须在事件循环里调用，调用后运行立即开始。

返回的 `EventStream` 可以 `async for` 迭代事件，另外提供：

- `await stream.result() -> RunResult`：等运行结束。等待的一方被取消不会取消运行。
- `stream.cancel()`：取消。

和 Pi 一样，底层的事件流只负责通知：事件放进无界队列，循环不等使用者处理完就继续执行，使用者不迭代也不会卡住循环。只能有一个迭代者。需要“处理完这条消息再执行工具”这种先后保证的，用 `Agent`。

每一轮：

1. 请求数到了 `max_model_requests` 则结束。
2. 发 `TurnStart`。第一轮在这里写入传入的 `prompts`。
3. `transform_context`，`convert_to_llm`，交给连接器，转发增量事件；回复结束后追加进消息列表。已经取消时不发请求，直接记一条 `aborted` 回复。
4. 回复是 `error` 或 `aborted`：调用 `finish_turn`，发 `TurnEnd`，结束。
5. 回复没有工具调用：调用 `finish_turn`，发 `TurnEnd`。返回 `continue` 则回到第 1 步，否则结束。
6. 执行工具，结果按调用顺序追加进消息列表。回复被截断，或这一轮的调用数会超过 `max_tool_calls` 时，整轮不执行，见“工具”。
7. 调用 `finish_turn`，发 `TurnEnd`。被取消或超过了 `max_tool_calls` 则结束；返回 `end` 则结束；本轮结果全带 `terminate` 且没有返回 `continue`，结束；否则回到第 1 步。

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
| `error` | 模型请求失败，最后一条回复带 `error_message` |
| `aborted` | 被取消 |
| `max_model_requests`，`max_tool_calls`，`run_timeout` | 触发了对应的上限 |

`RunLimits` 的字段名沿用 pi-python，默认值是本库加的，`None` 表示不限：

| 字段 | 默认值 |
|---|---|
| `max_model_requests` | 50 |
| `max_tool_calls` | 200 |
| `max_concurrency` | 8 |
| `tool_timeout` | 60 秒 |
| `run_timeout` | 600 秒 |

默认值偏宽，只防止失控。业务按需要覆盖。

## 取消

`stream.cancel()`、`Agent.abort()` 或 `run_timeout` 到期时，行为和 Pi 相同。取消可能落在一轮中的任何位置，收尾按当时的进度处理：

- 还没有模型回复，包括运行还没开始执行、正在调用 `transform_context` 时：不再发请求，记一条 `aborted` 回复。
- 正在请求模型：这次回复以 `aborted` 记进历史，保留已收到的正文和思考，删掉其中的工具调用。
- 回复已写入，工具还没开始或正在执行：取消工具任务。已经完成的结果保留；没有结果的调用，每个补一条 `is_error=True`、内容为 `Operation aborted` 的结果。
- 然后发 `TurnEnd` 和 `AgentEnd`，`reason` 是 `aborted` 或 `run_timeout`。

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
| `await prompt(text_or_messages)` | `prompt()` | 追加一条或几条消息并运行，结束后返回 `RunResult`。传字符串时当作一条用户消息 |
| `await continue_run()` | `continue()` | 从现有历史继续。Python 里 `continue` 是关键字，名字沿用 pi-python |
| `abort()` | `abort()` | 取消当前运行 |
| `await wait_for_idle()` | `waitForIdle()` | 等当前运行结束 |
| `subscribe(listener)` | `subscribe()` | 订阅事件，返回取消订阅的函数 |
| `state` | `state` | `system_prompt`、`model`、`tools`、`messages`、`is_streaming`、`pending_tool_calls`、`error_message` |
| `reset()` | `reset()` | 清空消息 |

`system_prompt` 不存进 `state.messages`。每次运行时，如果历史的第一条不是系统消息，就在最前面加一条。

监听函数可以是同步或 async 的，和 Pi 一样按注册顺序依次等待完成。模型回复的 `MessageEnd` 处理完之后才开始执行工具，所以 `before_tool_call` 看到的消息列表里已经有这条回复。

同一时间只能有一次运行，运行中再调用 `prompt()` 抛 `AgentBusyError`。工具配置有错（schema 不合法、工具名重复）时，`prompt()` 直接抛异常，`Agent` 不进入运行状态。调用 `prompt()` 的一方被取消时，按 `abort()` 处理：运行照常收尾，消息完整写回 `state` 后再向外抛出取消。

## 测试支持

`agent_looplite.testing.ScriptedModel` 按预先写好的回复依次返回，可以选择流式还是整轮返回，`chunk_size` 和 `delay` 控制增量的切分和间隔。`contexts` 记录每次收到的请求。`text_reply()`、`tool_reply()`、`call()`、`error_reply()` 用来写回复。

## 模块划分

| 模块 | 内容 | 依赖 |
|---|---|---|
| `messages.py` | 消息、内容块、`Usage`、序列化 | 无 |
| `model.py` | `ModelClient` 协议、`ModelContext`、`StreamDelta`、`StreamDone`、`StreamError` | `messages` |
| `events.py` | 事件、`PartialMessage` | `messages`、`model` |
| `tools.py` | `AgentTool`、`AgentToolResult`、参数校验、执行 | `messages`、`model`、jsonschema |
| `loop.py` | `agent_loop()`、`agent_loop_continue()`、`EventStream`、`AgentRun`、`AgentLoopConfig`、`RunLimits`、`RunResult`、钩子的类型 | 以上全部 |
| `agent.py` | `Agent` | `loop` |
| `chat_completions.py` | `ChatCompletionsClient` | `model`、`messages`、httpx |
| `function_tool.py` | `@tool` | `tools`、pydantic（可选） |
| `testing.py` | `ScriptedModel` | `model`、`messages` |

依赖只往上指，没有互相导入。运行依赖是 httpx 和 jsonschema，pydantic 可选：`agent_looplite.tool` 在第一次用到时才导入 pydantic。

## 与 Pi 的差异

| 地方 | Pi | 本库 | 原因 |
|---|---|---|---|
| `MessageUpdate` | 带增量和拼好的部分消息 | 只带增量，另给 `PartialMessage` | Python 里每个增量都生成一份部分消息，开销随回复长度增长。实测见 `docs/benchmark.md` |
| `MessageStart` | 模型回复也带部分消息 | 模型回复的 `message` 为 `None` | 同上，回复开始时还没有内容 |
| 运行上限 | 没有 | `RunLimits`，有默认值 | 没有上限时，模型可能一直调用工具，成本和时延失控 |
| 取消 | 传入 `AbortSignal` | `stream.cancel()`、`Agent.abort()`，内部用 asyncio 取消 | Python 没有 `AbortSignal`，asyncio 的取消是标准做法 |
| 工具的进度回调 | 同步函数 | async 函数 | `Agent` 要等监听函数处理完事件，同步回调里没法等 |
| 结束原因 | `agent_end` 只带消息 | `AgentEnd` 和 `RunResult` 带 `reason` | 有了运行上限，调用方需要知道是正常结束还是触发了上限 |
| 模型 | 模型目录加 `streamFn`，模型目录管费用和上下文长度 | 只有 `ModelClient` | 本库不维护模型目录 |
| 出网前检查 | `onPayload` 作为请求选项，由循环传给 `streamFn` | `on_payload` 是内置连接器的构造参数，自定义连接器自己负责 | 各网关的请求体格式不同，检查放在生成请求体的地方最准确 |
| 钩子抛异常 | `before_tool_call` 抛异常时记为失败的工具结果，循环继续 | 运行以异常结束，抛出前补齐工具结果 | 钩子是使用者的代码，出错应让调用方知道，而不是交给模型处理 |
| 继续运行的末条消息 | 自定义消息经 `convertToLlm` 转成用户消息后，可以作为末条 | 跳过自定义消息，最后一条标准消息必须是用户消息或工具结果 | 首版没有需要自定义末条消息的场景，严格检查能更早发现调用错误 |
| 工具清单的变化 | 在系统消息里记录工具清单的变化，随历史保存 | 工具清单不随历史保存，每次运行由调用方传入 | 首版的使用方式是每次运行传入固定的工具 |
| 工具超时 | 没有 | `RunLimits.tool_timeout` | 一个卡住的工具不应拖住整次运行 |
| 超过工具调用上限 | 没有上限 | 整轮不执行，规则来自 pi-python | 只执行一部分时，执行哪些由调用顺序偶然决定 |
| `prepare_request` | 有 | 首版不做 | 用途可以由 `transform_context` 覆盖 |
| 插话和排队 | `steer()`、`followUp()` | 首版不做 | 见 `CLAUDE.md` |
| 多模态、思考等级、`thinkingBudgets`、`sessionId`、`getApiKey` | 有 | 首版不做 | 首版只做 Chat Completions，这些由连接器的 `settings` 和 `headers` 承担 |
| Chat Completions 请求体 | 用户消息发内容块数组；按模型声明发 `max_completion_tokens`；发 `store: false` | 用户消息发字符串；最大输出长度和其他字段由使用者在 `settings` 里写 | 私有网关对字符串内容和最少字段的兼容性最好；`store` 是 OpenAI 专用字段 |
