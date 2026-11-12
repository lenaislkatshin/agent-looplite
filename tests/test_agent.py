from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_looplite import (
    Agent,
    AgentBusyError,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    MessageEnd,
    RunLimits,
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


async def test_abort_after_reply_before_tools_keeps_pairs() -> None:
    model = ScriptedModel([tool_reply(call("echo", {"text": "a"}, id="c1"))])
    agent = Agent(model=model, tools=[echo_tool()])
    seen: list[str] = []

    async def listener(event: Any) -> None:
        seen.append(event.type)
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            agent.abort()
            await asyncio.sleep(0)

    agent.subscribe(listener)
    result = await agent.prompt("问")
    assert result.reason == "aborted"
    assert [m.role for m in agent.state.messages] == ["user", "assistant", "tool_result"]
    assert agent.state.messages[-1].content[0].text == "Operation aborted"
    assert seen[-2:] == ["turn_end", "agent_end"]


@pytest.mark.parametrize(
    ("interrupted", "role"),
    [
        ("agent_start", None),
        ("turn_start", None),
        ("message_start", "user"),
        ("message_start", "assistant"),
    ],
)
async def test_abort_inside_a_listener_keeps_the_prompt(interrupted: str, role: str | None) -> None:
    agent = Agent(model=ScriptedModel([text_reply("答")]))
    entered = asyncio.Event()
    starts: list[str] = []

    def record(event: Any) -> None:
        if event.type == "message_start":
            starts.append(event.role)

    async def listener(event: Any) -> None:
        # 每次收到这个事件都等很久：收尾时要是把它再发一次，运行就结束不了。
        if event.type == interrupted and getattr(event, "role", None) == role:
            entered.set()
            await asyncio.sleep(5)

    agent.subscribe(record)
    agent.subscribe(listener)
    task = asyncio.create_task(agent.prompt("问题"))
    await entered.wait()
    agent.abort()
    result = await asyncio.wait_for(task, 1)
    assert result.reason == "aborted"
    assert starts == ["user", "assistant"]
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert [m.role for m in agent.state.messages] == ["user", "assistant"]
    assert agent.state.messages[0].content[0].text == "问题"
    assert agent.state.messages[1].stop_reason == "aborted"


async def test_config_error_does_not_leave_agent_busy() -> None:
    tool = echo_tool()
    agent = Agent(model=ScriptedModel([text_reply("好")]), tools=[tool, tool])
    with pytest.raises(ValueError, match="重复"):
        await agent.prompt("问")
    assert not agent.state.is_streaming
    agent.state.tools = [tool]
    result = await agent.prompt("问")
    assert result.reason == "completed"


async def test_hook_exception_keeps_agent_history_reusable() -> None:
    async def before(ctx: Any) -> None:
        raise RuntimeError("审计服务不可用")

    model = ScriptedModel([tool_reply(call("echo", {"text": "a"})), text_reply("继续")])
    agent = Agent(model=model, tools=[echo_tool()], before_tool_call=before)
    with pytest.raises(RuntimeError):
        await agent.prompt("问")
    assert isinstance(agent.state.messages[-1], ToolResultMessage)
    agent.before_tool_call = None
    result = await agent.prompt("再问")
    assert result.reason == "completed"


def counting_tool(runs: list[str]) -> AgentTool:
    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        runs.append(call_id)
        return AgentToolResult.of(f"ok:{call_id}")

    return AgentTool("t", "", {"type": "object"}, execute)


def assert_pairs_sent(context: Any) -> None:
    """下一轮发给模型的每个工具调用，都紧跟着按顺序配对的结果。"""
    messages = list(context.messages)
    for position, message in enumerate(messages):
        if isinstance(message, AssistantMessage) and message.tool_calls:
            following = messages[position + 1 : position + 1 + len(message.tool_calls)]
            assert [m.tool_call_id for m in following] == [c.id for c in message.tool_calls]


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
async def test_sync_listener_abort_runs_no_tools(mode: str) -> None:
    runs: list[str] = []
    model = ScriptedModel([tool_reply(call("t", id="a"), call("t", id="b")), text_reply("再答")])
    agent = Agent(model=model, tools=[counting_tool(runs)], tool_execution=mode)

    def listener(event: Any) -> None:
        if isinstance(event, MessageEnd) and isinstance(event.message, AssistantMessage):
            agent.abort()

    unsubscribe = agent.subscribe(listener)
    result = await agent.prompt("问")
    assert result.reason == "aborted"
    assert runs == []
    unsubscribe()
    await agent.prompt("再问")
    assert_pairs_sent(model.contexts[-1])


@pytest.mark.parametrize("interrupt", ["timeout", "external_abort"])
async def test_pending_tool_interrupted_after_listener_abort(interrupt: str) -> None:
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        await on_update("等待查询")
        waiting.set()
        await release.wait()
        return AgentToolResult.of("完成")

    agent = Agent(
        model=ScriptedModel([tool_reply(call("lookup", id="a"))]),
        tools=[AgentTool("lookup", "", {"type": "object"}, execute)],
        tool_execution="sequential",
        limits=RunLimits(tool_timeout=None, run_timeout=0.05 if interrupt == "timeout" else None),
    )

    def listener(event: Any) -> None:
        if event.type == "tool_execution_update":
            agent.abort()

    agent.subscribe(listener)
    task = asyncio.create_task(agent.prompt("查询"))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        if interrupt == "external_abort":
            agent.abort()
            agent.abort()
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done
    finally:
        release.set()
        result = await task
    assert result.reason == "aborted"
    assert [(m.tool_call_id, m.is_error) for m in result.messages if m.role == "tool_result"] == [
        ("a", True)
    ]


async def test_sync_listener_abort_stops_streaming() -> None:
    model = ScriptedModel([text_reply("abcdefghij")], chunk_size=2)
    agent = Agent(model=model)
    deltas = 0

    def listener(event: Any) -> None:
        nonlocal deltas
        if event.type == "message_update" and event.kind == "text_delta":
            deltas += 1
            if deltas == 2:
                agent.abort()

    agent.subscribe(listener)
    result = await agent.prompt("问")
    assert result.reason == "aborted"
    assert agent.state.messages[-1].stop_reason == "aborted"
    assert agent.state.messages[-1].text == "abcd"


async def test_listener_exception_while_streaming_ends_the_run() -> None:
    # 连接器的异常记成 error 回复，监听函数的异常不算在内，照常向外抛。
    agent = Agent(model=ScriptedModel([text_reply("abcdef")], chunk_size=2))

    def listener(event: Any) -> None:
        if event.type == "message_update" and event.kind == "text_delta":
            raise RuntimeError("界面推送失败")

    agent.subscribe(listener)
    with pytest.raises(RuntimeError, match="界面推送失败"):
        await agent.prompt("问")
    assert not agent.state.is_streaming


@pytest.mark.parametrize("action", ["abort", "external_abort", "raise"])
async def test_interrupt_at_first_tool_result_keeps_every_result(action: str) -> None:
    runs: list[str] = []
    model = ScriptedModel([tool_reply(call("t", id="a"), call("t", id="b")), text_reply("再答")])
    agent = Agent(model=model, tools=[counting_tool(runs)])
    entered = asyncio.Event()
    starts: list[str] = []

    def record(event: Any) -> None:
        if event.type == "message_start" and isinstance(event.message, ToolResultMessage):
            starts.append(event.message.tool_call_id)

    async def listener(event: Any) -> None:
        message = getattr(event, "message", None)
        if (
            event.type == "message_start"
            and isinstance(message, ToolResultMessage)
            and message.tool_call_id == "a"
        ):
            if action == "abort":
                agent.abort()
            elif action == "external_abort":
                entered.set()
                await asyncio.sleep(5)
            else:
                raise RuntimeError("监听函数出错")

    agent.subscribe(record)
    unsubscribe = agent.subscribe(listener)
    if action == "abort":
        await agent.prompt("问")
    elif action == "external_abort":
        task = asyncio.create_task(agent.prompt("问"))
        await entered.wait()
        agent.abort()
        await asyncio.wait_for(task, 1)
    else:
        with pytest.raises(RuntimeError):
            await agent.prompt("问")
    unsubscribe()
    # 被打断的那条结果直接写完，开始事件不发第二次。
    assert starts == ["a", "b"]
    results = [m for m in agent.state.messages if isinstance(m, ToolResultMessage)]
    assert [m.tool_call_id for m in results] == ["a", "b"]
    assert results[0].content[0].text == "ok:a"
    await agent.prompt("再问")
    assert_pairs_sent(model.contexts[-1])
