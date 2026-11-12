"""上层入口，对应 Pi 的 `Agent`：持有系统提示词、模型、工具和消息。

和底层的事件流不同，监听函数按注册顺序依次等待完成后，循环才继续。所以模型回复的
`MessageEnd` 处理完之后才开始执行工具。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .events import AgentEvent, MessageEnd, ToolExecutionEnd, ToolExecutionStart
from .loop import (
    AfterToolCall,
    AgentContext,
    AgentLoopConfig,
    AgentRun,
    BeforeToolCall,
    FinishTurn,
    MessagesHook,
    RunLimits,
    RunResult,
    check_continuable,
)
from .messages import AgentMessage, AssistantMessage, SystemMessage, UserMessage
from .model import ModelClient
from .tools import AgentTool

Listener = Callable[[AgentEvent], Awaitable[None] | None]


class AgentBusyError(RuntimeError):
    """同一时间只能有一次运行。"""


@dataclass
class AgentState:
    system_prompt: str
    model: ModelClient
    tools: list[AgentTool]
    messages: list[AgentMessage]
    is_streaming: bool = False
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None


class Agent:
    def __init__(
        self,
        *,
        model: ModelClient,
        system_prompt: str = "",
        tools: Sequence[AgentTool] = (),
        messages: Sequence[AgentMessage] = (),
        transform_context: MessagesHook | None = None,
        convert_to_llm: MessagesHook | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        finish_turn: FinishTurn | None = None,
        tool_execution: Literal["parallel", "sequential"] = "parallel",
        limits: RunLimits = RunLimits(),
    ) -> None:
        self.state = AgentState(system_prompt, model, list(tools), list(messages))
        self.transform_context = transform_context
        self.convert_to_llm = convert_to_llm
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.finish_turn = finish_turn
        self.tool_execution = tool_execution
        self.limits = limits
        self._listeners: list[Listener] = []
        self._run: AgentRun | None = None
        self._task: asyncio.Task[RunResult] | None = None

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """监听函数可以是同步或 async 的。返回取消订阅的函数。"""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    async def prompt(self, message: str | AgentMessage | Sequence[AgentMessage]) -> RunResult:
        if isinstance(message, str):
            prompts: Sequence[AgentMessage] = (UserMessage.of(message),)
        elif isinstance(message, Sequence):
            prompts = message
        else:
            prompts = (message,)
        return await self._start(prompts)

    async def continue_run(self) -> RunResult:
        """从现有历史继续，用于重试。最后一条标准消息必须是用户消息或工具结果。"""
        check_continuable(self.state.messages)
        return await self._start(())

    def abort(self) -> None:
        if self._run is not None:
            self._run.abort("aborted")

    async def wait_for_idle(self) -> None:
        if self._task is not None:
            await asyncio.wait({self._task})

    def reset(self) -> None:
        if self.state.is_streaming:
            raise AgentBusyError("运行中不能清空消息")
        self.state.messages.clear()
        self.state.error_message = None

    async def _start(self, prompts: Sequence[AgentMessage]) -> RunResult:
        if self.state.is_streaming:
            raise AgentBusyError("上一次运行还没结束")
        history: list[AgentMessage] = []
        if self.state.system_prompt and not (
            self.state.messages and isinstance(self.state.messages[0], SystemMessage)
        ):
            history.append(SystemMessage(self.state.system_prompt))
        history.extend(self.state.messages)
        config = AgentLoopConfig(
            model=self.state.model,
            transform_context=self.transform_context,
            convert_to_llm=self.convert_to_llm,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            finish_turn=self.finish_turn,
            tool_execution=self.tool_execution,
            limits=self.limits,
        )
        # 先构造运行再标记忙：工具配置有错时构造就会失败，Agent 不能因此一直处于忙的状态。
        self._run = AgentRun(
            prompts, AgentContext(history, self.state.tools), config, self._dispatch
        )
        self.state.is_streaming = True
        self.state.error_message = None
        self._task = self._run.start()
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # 调用方不再等待时按取消处理，让运行照常收尾，消息完整写回 state。
            self._run.abort("aborted")
            await asyncio.wait({self._task})
            raise
        finally:
            self.state.is_streaming = False
            self.state.pending_tool_calls.clear()

    async def _dispatch(self, event: AgentEvent) -> None:
        if isinstance(event, MessageEnd):
            self.state.messages.append(event.message)
            if isinstance(event.message, AssistantMessage) and event.message.error_message:
                self.state.error_message = event.message.error_message
        elif isinstance(event, ToolExecutionStart):
            self.state.pending_tool_calls.add(event.tool_call_id)
        elif isinstance(event, ToolExecutionEnd):
            self.state.pending_tool_calls.discard(event.tool_call_id)
        for listener in list(self._listeners):
            outcome = listener(event)
            if inspect.isawaitable(outcome):
                await outcome
