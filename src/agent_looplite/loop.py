"""底层循环，对应 Pi 的 `agentLoop` 和 `agentLoopContinue`。

请求模型，执行模型要求的工具，把结果写回消息，再请求模型，直到模型给出回答。取消、运行
上限和结束原因都在这里处理。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from .events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    PartialMessage,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    Message,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from .model import ModelClient, ModelContext, StreamDelta, StreamDone, StreamError
from .tools import ABORTED_TEXT, AgentTool, AgentToolResult, ToolSet, error_result, run_tool

TurnDecision = Literal["continue", "end"] | None


@dataclass(frozen=True, slots=True)
class RunLimits:
    """默认值偏宽，只防止失控。`None` 表示不限。"""

    max_model_requests: int | None = 50
    max_tool_calls: int | None = 200
    max_concurrency: int = 8
    tool_timeout: float | None = 60.0
    run_timeout: float | None = 600.0


@dataclass(frozen=True, slots=True)
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    messages: tuple[AgentMessage, ...]


@dataclass(frozen=True, slots=True)
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: dict[str, Any]
    result: AgentToolResult
    is_error: bool
    messages: tuple[AgentMessage, ...]


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()


@dataclass(frozen=True, slots=True)
class AfterToolCallResult:
    """填了的字段替换原结果，没填的保留。`details` 可以替换成 `None`，所以用 `UNSET` 表示没填。"""

    content: tuple[TextContent, ...] | None = None
    details: Any = UNSET
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass(frozen=True, slots=True)
class TurnContext:
    message: AssistantMessage
    tool_results: tuple[ToolResultMessage, ...]
    messages: tuple[AgentMessage, ...]


MessagesHook = Callable[[tuple[AgentMessage, ...]], Awaitable[Sequence[AgentMessage]]]
BeforeToolCall = Callable[[BeforeToolCallContext], Awaitable[BeforeToolCallResult | None]]
AfterToolCall = Callable[[AfterToolCallContext], Awaitable[AfterToolCallResult | None]]
FinishTurn = Callable[[TurnContext], Awaitable[TurnDecision]]
Emit = Callable[[AgentEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AgentLoopConfig:
    """`convert_to_llm` 不填时去掉全部 `CustomMessage`。"""

    model: ModelClient
    transform_context: MessagesHook | None = None
    convert_to_llm: MessagesHook | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    finish_turn: FinishTurn | None = None
    tool_execution: Literal["parallel", "sequential"] = "parallel"
    limits: RunLimits = RunLimits()


@dataclass(frozen=True, slots=True)
class AgentContext:
    messages: Sequence[AgentMessage] = ()
    tools: Sequence[AgentTool] = ()


@dataclass(frozen=True, slots=True)
class RunResult:
    """`messages` 是本次新增的消息，包括传入的提示。"""

    messages: tuple[AgentMessage, ...]
    reason: str
    usage: Usage


TRUNCATED_TOOL_CALL = (
    'Tool call "{name}" was not executed: the response hit the output token limit, so its '
    "arguments may be truncated. Re-issue the tool call with complete arguments."
)
BATCH_OVER_LIMIT = "Entire tool batch exceeds remaining budget"
STREAM_WITHOUT_RESULT = "Model stream ended without a final message"


class AgentRun:
    """一次运行。

    `agent_loop` 和 `Agent` 都用它，区别只在 `emit`：前者放进队列，后者依次等监听函数。
    """

    def __init__(
        self,
        prompts: Sequence[AgentMessage],
        context: AgentContext,
        config: AgentLoopConfig,
        emit: Emit,
    ) -> None:
        self._prompts = tuple(prompts)
        self._messages: list[AgentMessage] = list(context.messages)
        self._tools = ToolSet(context.tools)
        self._config = config
        self._emit = emit
        self._new: list[AgentMessage] = []
        self._usage = Usage()
        self._abort_reason: str | None = None
        self._started = False
        self._finished = False
        self._task: asyncio.Task[RunResult] | None = None

    def start(self) -> asyncio.Task[RunResult]:
        self._task = asyncio.get_running_loop().create_task(self._run())
        return self._task

    def abort(self, reason: str = "aborted") -> None:
        """任务还没开始执行时只记下原因：取消一个还没开始的协程，它的收尾代码一行都不会执行。"""
        if self._finished or self._abort_reason is not None:
            return
        self._abort_reason = reason
        if self._started and self._task is not None:
            self._task.cancel()

    def _absorb_abort(self) -> None:
        """在 `except CancelledError` 里调用。取消来自 `abort()` 时，撤销任务的取消状态，让收尾代码
        能继续等待；其他来源的取消由调用方重新抛出。"""
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()

    async def _run(self) -> RunResult:
        self._started = True
        limits = self._config.limits
        timer = None
        if limits.run_timeout is not None:
            timer = asyncio.get_running_loop().call_later(
                limits.run_timeout, self.abort, "run_timeout"
            )
        try:
            await self._emit(AgentStart())
            for prompt in self._prompts:
                await self._add(prompt)
            reason = await self._loop()
        except asyncio.CancelledError:
            if self._abort_reason is None:
                raise
            self._absorb_abort()
            reason = self._abort_reason
        finally:
            if timer is not None:
                timer.cancel()
        self._finished = True
        await self._emit(AgentEnd(tuple(self._new), reason))
        return RunResult(tuple(self._new), reason, self._usage)

    async def _add(self, message: AgentMessage) -> None:
        await self._emit(MessageStart(message.role, message))
        self._messages.append(message)
        self._new.append(message)
        await self._emit(MessageEnd(message))

    async def _loop(self) -> str:
        limits = self._config.limits
        requests = 0
        tool_calls = 0
        while True:
            if self._abort_reason is not None:
                return self._abort_reason
            if limits.max_model_requests is not None and requests >= limits.max_model_requests:
                return "max_model_requests"
            await self._emit(TurnStart())
            assistant = await self._request()
            requests += 1
            self._usage += assistant.usage

            if assistant.stop_reason in ("error", "aborted"):
                await self._finish_turn(assistant, ())
                await self._emit(TurnEnd(assistant, ()))
                if assistant.stop_reason == "error":
                    return "error"
                return self._abort_reason or "aborted"

            calls = assistant.tool_calls
            if not calls:
                decision = await self._finish_turn(assistant, ())
                await self._emit(TurnEnd(assistant, ()))
                if decision == "continue":
                    continue
                return "finish_turn" if decision == "end" else "completed"

            over_limit = (
                limits.max_tool_calls is not None
                and tool_calls + len(calls) > limits.max_tool_calls
            )
            if assistant.stop_reason == "length":
                outcomes = await self._fail_all(
                    calls, lambda call: TRUNCATED_TOOL_CALL.format(name=call.name)
                )
                aborted = False
            elif over_limit:
                outcomes = await self._fail_all(calls, lambda call: BATCH_OVER_LIMIT)
                aborted = False
            else:
                outcomes, aborted = await self._execute(assistant, calls)
                tool_calls += len(calls)

            results = await self._add_tool_results(calls, outcomes)
            decision = await self._finish_turn(assistant, results)
            await self._emit(TurnEnd(assistant, results))
            if aborted:
                return self._abort_reason or "aborted"
            if over_limit:
                return "max_tool_calls"
            if decision == "end":
                return "finish_turn"
            if decision != "continue" and all(outcome.terminate for outcome in outcomes):
                return "terminated"

    async def _request(self) -> AssistantMessage:
        context = tuple(self._messages)
        if self._config.transform_context is not None:
            context = tuple(await self._config.transform_context(context))
        if self._config.convert_to_llm is not None:
            llm_messages = tuple(await self._config.convert_to_llm(context))
        else:
            llm_messages = tuple(m for m in context if not isinstance(m, CustomMessage))
        for message in llm_messages:
            if not isinstance(message, Message):
                raise TypeError(f"发给模型的消息必须是标准消息，收到 {type(message).__name__}")

        await self._emit(MessageStart("assistant", None))
        partial = PartialMessage()
        message: AssistantMessage | None = None
        stream = self._config.model.stream(ModelContext(llm_messages, self._tools.declarations()))
        try:
            async for event in stream:
                if isinstance(event, StreamDelta):
                    partial.apply(event)
                    await self._emit(
                        MessageUpdate(
                            event.kind,
                            event.content_index,
                            event.delta,
                            event.tool_call_id,
                            event.tool_name,
                        )
                    )
                elif isinstance(event, StreamDone):
                    message = event.message
                    break
                elif isinstance(event, StreamError):
                    message = event.message
                    if message.stop_reason not in ("error", "aborted"):
                        message = replace(message, stop_reason="error")
                    break
        except asyncio.CancelledError:
            if self._abort_reason is None:
                raise
            self._absorb_abort()
            message = AssistantMessage(
                content=partial.content(),
                stop_reason="aborted",
                error_message=f"Request aborted ({self._abort_reason})",
            )
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        if message is None:
            message = AssistantMessage(
                content=partial.content(), stop_reason="error", error_message=STREAM_WITHOUT_RESULT
            )
        if message.stop_reason in ("error", "aborted") and message.tool_calls:
            message = replace(
                message,
                content=tuple(b for b in message.content if not isinstance(b, ToolCall)),
            )
        self._messages.append(message)
        self._new.append(message)
        await self._emit(MessageEnd(message))
        return message

    async def _finish_turn(
        self, message: AssistantMessage, results: tuple[ToolResultMessage, ...]
    ) -> TurnDecision:
        """回复是 `error` 或 `aborted` 时也调用，但返回值被忽略，运行一定结束。"""
        if self._config.finish_turn is None:
            return None
        return await self._config.finish_turn(TurnContext(message, results, tuple(self._messages)))

    async def _fail_all(
        self, calls: tuple[ToolCall, ...], text: Callable[[ToolCall], str]
    ) -> list[AgentToolResult]:
        outcomes = []
        for call in calls:
            await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
            result = error_result(text(call))
            await self._emit(ToolExecutionEnd(call.id, call.name, result, True, None))
            outcomes.append(result)
        return outcomes

    async def _execute(
        self, assistant: AssistantMessage, calls: tuple[ToolCall, ...]
    ) -> tuple[list[AgentToolResult], bool]:
        """返回按调用顺序排列的结果，以及是否被取消。"""
        limits = self._config.limits
        semaphore = asyncio.Semaphore(limits.max_concurrency)
        started: set[int] = set()
        outcomes: dict[int, AgentToolResult] = {}

        async def finalize(index: int, result: AgentToolResult, duration_ms: float | None) -> None:
            outcomes[index] = result
            call = calls[index]
            await self._emit(
                ToolExecutionEnd(call.id, call.name, result, result.is_error, duration_ms)
            )

        async def preflight(index: int) -> bool:
            call = calls[index]
            started.add(index)
            await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
            rejected = self._tools.check_arguments(call)
            if rejected is None and self._config.before_tool_call is not None:
                decision = await self._config.before_tool_call(
                    BeforeToolCallContext(assistant, call, call.arguments, tuple(self._messages))
                )
                if decision is not None and decision.block:
                    rejected = replace(
                        error_result(decision.reason or "Tool execution was blocked"),
                        terminate=decision.terminate,
                    )
            if rejected is not None:
                await finalize(index, rejected, None)
                return False
            return True

        async def execute(index: int) -> None:
            call = calls[index]
            tool = self._tools.get(call.name)
            assert tool is not None
            settled = False

            async def on_update(partial_result: Any) -> None:
                if not settled:
                    await self._emit(
                        ToolExecutionUpdate(call.id, call.name, call.arguments, partial_result)
                    )

            async with semaphore:
                result, duration_ms = await run_tool(tool, call, on_update, limits.tool_timeout)
            settled = True
            if self._config.after_tool_call is not None:
                override = await self._config.after_tool_call(
                    AfterToolCallContext(
                        assistant,
                        call,
                        call.arguments,
                        result,
                        result.is_error,
                        tuple(self._messages),
                    )
                )
                if override is not None:
                    result = _merge(result, override)
            await finalize(index, result, duration_ms)

        sequential = self._config.tool_execution == "sequential" or any(
            (tool := self._tools.get(call.name)) is not None and tool.execution_mode == "sequential"
            for call in calls
        )
        aborted = False
        try:
            if sequential:
                for index in range(len(calls)):
                    if await preflight(index):
                        await execute(index)
            else:
                runnable = [index for index in range(len(calls)) if await preflight(index)]
                try:
                    async with asyncio.TaskGroup() as group:
                        for index in runnable:
                            group.create_task(execute(index))
                except BaseExceptionGroup as group_error:
                    raise group_error.exceptions[0] from None
        except asyncio.CancelledError:
            if self._abort_reason is None:
                raise
            self._absorb_abort()
            aborted = True
            for index, call in enumerate(calls):
                if index in outcomes:
                    continue
                if index not in started:
                    await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
                await finalize(index, error_result(ABORTED_TEXT), None)
        return [outcomes[index] for index in range(len(calls))], aborted

    async def _add_tool_results(
        self, calls: tuple[ToolCall, ...], outcomes: list[AgentToolResult]
    ) -> tuple[ToolResultMessage, ...]:
        results = []
        for call, result in zip(calls, outcomes, strict=True):
            message = ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=result.content,
                details=result.details,
                is_error=result.is_error,
            )
            await self._add(message)
            results.append(message)
        return tuple(results)


def _merge(result: AgentToolResult, override: AfterToolCallResult) -> AgentToolResult:
    return AgentToolResult(
        content=result.content if override.content is None else override.content,
        details=result.details if override.details is UNSET else override.details,
        is_error=result.is_error if override.is_error is None else override.is_error,
        terminate=result.terminate if override.terminate is None else override.terminate,
    )


_END = object()


class EventStream:
    """`async for` 迭代事件，`await result()` 取结果，`cancel()` 取消。

    事件只负责通知：循环不等使用者处理完事件就继续执行。只能有一个迭代者。
    """

    def __init__(
        self, prompts: Sequence[AgentMessage], context: AgentContext, config: AgentLoopConfig
    ) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._run = AgentRun(prompts, context, config, self._push)
        self._task = self._run.start()
        self._task.add_done_callback(lambda _: self._queue.put_nowait(_END))

    async def _push(self, event: AgentEvent) -> None:
        self._queue.put_nowait(event)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[AgentEvent]:
        while True:
            item = await self._queue.get()
            if item is _END:
                if not self._task.cancelled() and self._task.exception() is not None:
                    raise self._task.exception()  # type: ignore[misc]
                return
            yield item

    async def result(self) -> RunResult:
        return await asyncio.shield(self._task)

    def cancel(self) -> None:
        self._run.abort("aborted")


def agent_loop(
    prompts: Sequence[AgentMessage], context: AgentContext, config: AgentLoopConfig
) -> EventStream:
    """把 `prompts` 追加到历史末尾后开始运行。必须在事件循环里调用。"""
    return EventStream(prompts, context, config)


def agent_loop_continue(context: AgentContext, config: AgentLoopConfig) -> EventStream:
    """从现有历史继续，用于重试。最后一条标准消息必须是用户消息或工具结果。"""
    check_continuable(context.messages)
    return EventStream((), context, config)


def check_continuable(messages: Sequence[AgentMessage]) -> None:
    last = next((m for m in reversed(messages) if not isinstance(m, CustomMessage)), None)
    if last is None or isinstance(last, (AssistantMessage, SystemMessage)):
        raise ValueError("继续运行时，最后一条消息必须是用户消息或工具结果")
