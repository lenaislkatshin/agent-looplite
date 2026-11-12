from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_looplite import (
    Agent,
    AgentBusyError,
    AssistantMessage,
    MessageEnd,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_looplite.testing import ScriptedModel, call, text_reply, tool_reply

from .helpers import echo_tool


async def test_prompt_runs_and_keeps_history() -> None:
    model = ScriptedModel(
        [tool_reply(call("echo", {"text": "a"})), text_reply("完成"), text_reply("再答")]
    )
    agent = Agent(model=model, system_prompt="系统", tools=[echo_tool()])
    result = await agent.prompt("问")
    assert result.reason == "completed"
    assert [m.role for m in agent.state.messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert isinstance(model.contexts[0].messages[0], SystemMessage)

    await agent.prompt("再问")
    assert len(agent.state.messages) == 6
    assert [m.role for m in model.contexts[-1].messages][:2] == ["system", "user"]


async def test_listeners_are_awaited_before_tools_run() -> None:
    order: list[str] = []

    async def listener(event: Any) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            await asyncio.sleep(0.01)
            order.append("listener")

    async def before(ctx: Any) -> None:
        order.append("before_tool_call")
        assert isinstance(ctx.messages[-1], AssistantMessage)

    model = ScriptedModel([tool_reply(call("echo", {"text": "a"})), text_reply("好")])
    agent = Agent(model=model, tools=[echo_tool()], before_tool_call=before)
    agent.subscribe(listener)
    await agent.prompt("问")
    assert order[:2] == ["listener", "before_tool_call"]


async def test_abort_and_busy() -> None:
    model = ScriptedModel([tool_reply(call("slow", {"text": "a"}))])
    agent = Agent(model=model, tools=[echo_tool("slow", delay=5)])
    task = asyncio.create_task(agent.prompt("问"))
    while not agent.state.pending_tool_calls:
        await asyncio.sleep(0.005)
    with pytest.raises(AgentBusyError):
        await agent.prompt("再问")
    agent.abort()
    result = await task
    assert result.reason == "aborted"
    assert isinstance(agent.state.messages[-1], ToolResultMessage)
    assert not agent.state.is_streaming


async def test_continue_run_and_reset() -> None:
    model = ScriptedModel([text_reply("重试成功")])
    agent = Agent(model=model, messages=[UserMessage.of("问")])
    await agent.continue_run()
    assert agent.state.messages[-1].text == "重试成功"
    with pytest.raises(ValueError):
        await agent.continue_run()
    agent.reset()
    assert agent.state.messages == []


async def test_caller_cancellation_aborts_run_cleanly() -> None:
    model = ScriptedModel([tool_reply(call("slow", {"text": "a"}))])
    agent = Agent(model=model, tools=[echo_tool("slow", delay=5)])
    task = asyncio.create_task(agent.prompt("问"))
    while not agent.state.pending_tool_calls:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.state.messages[-1].content[0].text == "Operation aborted"
    assert not agent.state.is_streaming
