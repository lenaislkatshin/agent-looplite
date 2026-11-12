"""底层循环，对应 Pi 的 `agentLoop` 和 `agentLoopContinue`。

请求模型，执行模型要求的工具，把结果写回消息，再请求模型，直到模型给出回答。取消、运行
上限和结束原因都在这里处理。
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
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
    ContentBlock,
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


@dataclass
class _Turn:
    """当前一轮的进度。取消或出错时据此补齐工具结果，保证历史里的工具调用和结果成对。"""

    assistant: AssistantMessage | None = None
    started: set[int] = field(default_factory=set)
    outcomes: dict[int, AgentToolResult] = field(default_factory=dict)
    results: dict[int, ToolResultMessage] = field(default_factory=dict)
    ended: bool = False

    def ordered_results(self) -> tuple[ToolResultMessage, ...]:
        return tuple(self.results[index] for index in sorted(self.results))


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
        self._turn: _Turn | None = None
        self._abort_reason: str | None = None
        self._interrupt_sent = False
        self._started = False
        self._finished = False
        self._task: asyncio.Task[RunResult] | None = None

    def start(self) -> asyncio.Task[RunResult]:
        self._task = asyncio.get_running_loop().create_task(self._run())
        return self._task

    def abort(self, reason: str = "aborted") -> None:
        """和 Pi 的取消信号一样，先记下原因，循环在请求模型前、执行每个工具前检查它。

        从别的任务调用时，再取消运行任务，打断正在等待的网络请求和工具。在运行任务内部调用时，
        例如同步监听函数里，只记原因：任务要到下一次挂起才收到取消，如果在那之前就结束，整次
        运行会被当成已取消，结果丢失。任务还没开始执行时也只记原因：取消一个还没开始的协程，
        它的收尾代码一行都不会执行。
        """
        if self._finished or self._interrupt_sent:
            return
        if self._abort_reason is None:
            self._abort_reason = reason
        if self._started and self._task is not None and asyncio.current_task() is not self._task:
            # 只打断一次，后续取消不能中断工具结果的补写。
            self._interrupt_sent = True
            self._task.cancel()

    def _resume_after_abort(self) -> None:
        """在 `except CancelledError` 里、确认取消来自 `abort()` 后调用。

        撤销任务的取消状态，收尾代码才能继续等待钩子和监听函数。
        """
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
            reason = await self._loop()
        except asyncio.CancelledError:
            if self._abort_reason is None:
                raise
            self._resume_after_abort()
            reason = self._abort_reason
            await self._close_turn()
        except Exception as error:
            await self._repair_history(error)
            raise
        finally:
            if timer is not None:
                timer.cancel()
        self._finished = True
        await self._emit(AgentEnd(tuple(self._new), reason))
        return RunResult(tuple(self._new), reason, self._usage)

    async def _add(self, message: AgentMessage, result_index: int | None = None) -> None:
        """`result_index` 是工具结果在本轮的调用序号。

        写入历史和标记“已写入”之间没有 await，取消或异常不会只做了其中一半。
        """
        await self._emit(MessageStart(message.role, message))
        self._messages.append(message)
        self._new.append(message)
        if result_index is not None:
            assert self._turn is not None
            self._turn.results[result_index] = message
        await self._emit(MessageEnd(message))

    async def _loop(self) -> str:
        limits = self._config.limits
        requests = 0
        tool_calls = 0
        prompts = self._prompts
        while True:
            if limits.max_model_requests is not None and requests >= limits.max_model_requests:
                return "max_model_requests"
            turn = self._turn = _Turn()
            await self._emit(TurnStart())
            for prompt in prompts:
                await self._add(prompt)
            prompts = ()
            assistant = await self._request()
            requests += 1
            self._usage += assistant.usage

            if assistant.stop_reason in ("error", "aborted"):
                await self._finish_turn()
                await self._end_turn()
                if assistant.stop_reason == "error":
                    return "error"
                return self._abort_reason or "aborted"

            calls = assistant.tool_calls
            if not calls:
                decision = await self._finish_turn()
                await self._end_turn()
                if decision == "continue":
                    continue
                return "finish_turn" if decision == "end" else "completed"

            over_limit = (
                limits.max_tool_calls is not None
                and tool_calls + len(calls) > limits.max_tool_calls
            )
            aborted = False
            if assistant.stop_reason == "length":
                await self._fail_all(calls, lambda call: TRUNCATED_TOOL_CALL.format(name=call.name))
            elif over_limit:
                await self._fail_all(calls, lambda call: BATCH_OVER_LIMIT)
            else:
                aborted = await self._execute(assistant, calls)
                tool_calls += len(calls)

            decision = await self._finish_turn()
            await self._end_turn()
            if aborted:
                return self._abort_reason or "aborted"
            if over_limit:
                return "max_tool_calls"
            if decision == "end":
                return "finish_turn"
            if decision != "continue" and all(o.terminate for o in turn.outcomes.values()):
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
        if self._abort_reason is not None:
            # 和 Pi 一样，已经取消时不发请求，直接记一条 aborted 回复。
            message = self._aborted_reply(())
        else:
            message = await self._stream_reply(llm_messages)
        if message.stop_reason in ("error", "aborted") and message.tool_calls:
            message = replace(
                message,
                content=tuple(b for b in message.content if not isinstance(b, ToolCall)),
            )
        assert self._turn is not None
        self._turn.assistant = message
        self._messages.append(message)
        self._new.append(message)
        await self._emit(MessageEnd(message))
        return message

    async def _stream_reply(self, llm_messages: tuple[Message, ...]) -> AssistantMessage:
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
                    if self._abort_reason is not None:
                        message = self._aborted_reply(partial.content())
                        break
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
            self._resume_after_abort()
            message = self._aborted_reply(partial.content())
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        if message is None:
            message = AssistantMessage(
                content=partial.content(), stop_reason="error", error_message=STREAM_WITHOUT_RESULT
            )
        return message

    def _aborted_reply(self, content: tuple[ContentBlock, ...]) -> AssistantMessage:
        return AssistantMessage(
            content=content,
            stop_reason="aborted",
            error_message=f"Request aborted ({self._abort_reason})",
        )

    async def _finish_turn(self) -> TurnDecision:
        """回复是 `error` 或 `aborted` 时也调用，但返回值被忽略，运行一定结束。"""
        turn = self._turn
        assert turn is not None and turn.assistant is not None
        if self._config.finish_turn is None:
            return None
        return await self._config.finish_turn(
            TurnContext(turn.assistant, turn.ordered_results(), tuple(self._messages))
        )

    async def _end_turn(self) -> None:
        turn = self._turn
        assert turn is not None and turn.assistant is not None
        turn.ended = True
        await self._emit(TurnEnd(turn.assistant, turn.ordered_results()))

    async def _close_turn(self) -> None:
        """取消后收尾：取消可能落在一轮中的任何位置，包括回复已写入、工具还没开始的时候。

        还没有回复就补一条 aborted 回复，没有结果的工具调用补上 `Operation aborted`，
        再发 `TurnEnd`。
        """
        turn = self._turn
        if turn is None or turn.ended:
            return
        if turn.assistant is None:
            turn.assistant = self._aborted_reply(())
            await self._add(turn.assistant)
        await self._complete_tool_results(ABORTED_TEXT)
        await self._end_turn()

    async def _repair_history(self, error: Exception) -> None:
        """钩子或监听函数抛异常时，运行到此为止，但历史要能直接用于下一次运行。

        给已写入的回复里没有结果的工具调用补上错误结果。补的事件仍然发给监听函数，但它们再抛的
        异常不打断补写：出错的往往就是同一个监听函数，一中断，后面的结果就补不上了。向外只抛最初
        的异常。
        """
        turn = self._turn
        if turn is None or turn.ended or turn.assistant is None:
            return
        text = f"Tool call was not completed: the run failed with {type(error).__name__}: {error}"
        emit = self._emit

        async def emit_ignoring_errors(event: AgentEvent) -> None:
            with contextlib.suppress(Exception):
                await emit(event)

        self._emit = emit_ignoring_errors
        try:
            await self._complete_tool_results(text)
        finally:
            self._emit = emit

    async def _complete_tool_results(self, text: str) -> None:
        turn = self._turn
        assert turn is not None and turn.assistant is not None
        for index, call in enumerate(turn.assistant.tool_calls):
            if index in turn.results:
                continue
            if index not in turn.outcomes:
                if index not in turn.started:
                    turn.started.add(index)
                    await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
                await self._finalize(index, call, error_result(text), None)
            await self._add_result(index, call)

    async def _finalize(
        self, index: int, call: ToolCall, result: AgentToolResult, duration_ms: float | None
    ) -> None:
        assert self._turn is not None
        self._turn.outcomes[index] = result
        await self._emit(ToolExecutionEnd(call.id, call.name, result, result.is_error, duration_ms))

    async def _add_result(self, index: int, call: ToolCall) -> None:
        assert self._turn is not None
        result = self._turn.outcomes[index]
        message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            details=result.details,
            is_error=result.is_error,
        )
        await self._add(message, index)

    async def _fail_all(self, calls: tuple[ToolCall, ...], text: Callable[[ToolCall], str]) -> None:
        assert self._turn is not None
        for index, call in enumerate(calls):
            self._turn.started.add(index)
            await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
            await self._finalize(index, call, error_result(text(call)), None)
            await self._add_result(index, call)

    async def _execute(self, assistant: AssistantMessage, calls: tuple[ToolCall, ...]) -> bool:
        """执行一轮工具，返回是否被取消。

        和 Pi 一样：顺序执行时，每个工具结束后立即写入它的结果消息；并行执行时，先按顺序做完
        校验和 `before_tool_call`，再并发执行，全部结束后按调用顺序写入结果消息。
        """
        turn = self._turn
        assert turn is not None
        limits = self._config.limits
        semaphore = asyncio.Semaphore(limits.max_concurrency)
        # 每个调用复制一份参数交给钩子和工具，它们改参数不会改到模型发来的原始消息。
        arguments: dict[int, dict[str, Any]] = {}

        async def preflight(index: int) -> bool:
            call = calls[index]
            turn.started.add(index)
            await self._emit(ToolExecutionStart(call.id, call.name, call.arguments))
            rejected = self._tools.check_arguments(call)
            if rejected is None:
                arguments[index] = copy.deepcopy(call.arguments)
                if self._config.before_tool_call is not None:
                    decision = await self._config.before_tool_call(
                        BeforeToolCallContext(
                            assistant, call, arguments[index], tuple(self._messages)
                        )
                    )
                    if decision is not None and decision.block:
                        rejected = replace(
                            error_result(decision.reason or "Tool execution was blocked"),
                            terminate=decision.terminate,
                        )
            if rejected is None and self._abort_reason is not None:
                rejected = error_result(ABORTED_TEXT)
            if rejected is not None:
                await self._finalize(index, call, rejected, None)
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
                if self._abort_reason is not None:
                    await self._finalize(index, call, error_result(ABORTED_TEXT), None)
                    return
                result, duration_ms = await run_tool(
                    tool, call.id, arguments[index], on_update, limits.tool_timeout
                )
            settled = True
            if self._config.after_tool_call is not None:
                override = await self._config.after_tool_call(
                    AfterToolCallContext(
                        assistant,
                        call,
                        arguments[index],
                        result,
                        result.is_error,
                        tuple(self._messages),
                    )
                )
                if override is not None:
                    result = _merge(result, override)
            await self._finalize(index, call, result, duration_ms)

        sequential = self._config.tool_execution == "sequential" or any(
            (tool := self._tools.get(call.name)) is not None and tool.execution_mode == "sequential"
            for call in calls
        )
        try:
            if sequential:
                for index, call in enumerate(calls):
                    if await preflight(index):
                        await execute(index)
                    await self._add_result(index, call)
            else:
                runnable = [index for index in range(len(calls)) if await preflight(index)]
                try:
                    async with asyncio.TaskGroup() as group:
                        for index in runnable:
                            group.create_task(execute(index))
                except BaseExceptionGroup as group_error:
                    raise group_error.exceptions[0] from None
                for index, call in enumerate(calls):
                    await self._add_result(index, call)
        except asyncio.CancelledError:
            if self._abort_reason is None:
                raise
            self._resume_after_abort()
            await self._complete_tool_results(ABORTED_TEXT)
            return True
        return self._abort_reason is not None


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
