from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_looplite import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    BeforeToolCallResult,
    CustomMessage,
    MessageEnd,
    MessageUpdate,
    RunLimits,
    StreamDelta,
    StreamDone,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResultMessage,
    UserMessage,
    agent_loop,
    agent_loop_continue,
)
from agent_looplite.testing import ScriptedModel, call, error_reply, text_reply, tool_reply

from .helpers import ECHO_SCHEMA, echo_tool, run_events, types


async def test_text_reply_event_sequence() -> None:
    events, result = await run_events(ScriptedModel([text_reply("你好")]))
    # 顺序同 Pi：先 turn_start，再发首条提示消息。
    assert types(events) == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert result.reason == "completed"
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert result.messages[1].text == "你好"


async def test_tool_call_then_answer() -> None:
    model = ScriptedModel([tool_reply(call("echo", {"text": "a"}, id="c1")), text_reply("完成")])
    events, result = await run_events(model, [echo_tool()])
    assert result.reason == "completed"
    roles = [m.role for m in result.messages]
    assert roles == ["user", "assistant", "tool_result", "assistant"]
    tool_result = result.messages[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.tool_call_id == "c1"
    assert tool_result.content == (TextContent("echo:a"),)
    assert tool_result.details == {"call_id": "c1"}
    second_request = model.contexts[1].messages
    assert isinstance(second_request[-1], ToolResultMessage)
    assert "tool_execution_start" in types(events)
    assert types(events).index("tool_execution_end") < types(events).index("turn_end")


async def test_non_streaming_model_sends_no_updates() -> None:
    events, result = await run_events(ScriptedModel([text_reply("整轮")], stream=False))
    assert not any(isinstance(e, MessageUpdate) for e in events)
    assert result.messages[-1].text == "整轮"


async def test_parallel_results_follow_call_order() -> None:
    slow, fast = echo_tool("slow", delay=0.05), echo_tool("fast")
    model = ScriptedModel(
        [
            tool_reply(call("slow", {"text": "1"}, id="a"), call("fast", {"text": "2"}, id="b")),
            text_reply("好"),
        ]
    )
    events, result = await run_events(model, [slow, fast])
    ends = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEnd)]
    assert ends == ["b", "a"]
    results = [m.tool_call_id for m in result.messages if isinstance(m, ToolResultMessage)]
    assert results == ["a", "b"]


async def test_sequential_tool_forces_whole_batch_sequential() -> None:
    order: list[str] = []

    def tracked(name: str, delay: float, mode: Any = None) -> AgentTool:
        async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
            order.append(f"start:{name}")
            await asyncio.sleep(delay)
            order.append(f"end:{name}")
            return AgentToolResult.of(name)

        return AgentTool(name, name, ECHO_SCHEMA, execute, execution_mode=mode)

    model = ScriptedModel(
        [
            tool_reply(call("a", {"text": "x"}), call("b", {"text": "y"})),
            text_reply("好"),
        ]
    )
    await run_events(model, [tracked("a", 0.02, "sequential"), tracked("b", 0.0)])
    assert order == ["start:a", "end:a", "start:b", "end:b"]


@pytest.mark.parametrize(
    ("bad_call", "expected"),
    [
        (call("missing", {"text": "x"}), "Tool missing not found"),
        (call("echo", {"text": ["x"]}), 'Validation failed for tool "echo"'),
        (
            ToolCall("c", "echo", {}, arguments_error="Expecting value"),
            'Invalid arguments for tool "echo": Expecting value',
        ),
    ],
)
async def test_bad_calls_become_error_results(bad_call: ToolCall, expected: str) -> None:
    model = ScriptedModel([tool_reply(bad_call), text_reply("改正了")])
    _, result = await run_events(model, [echo_tool()])
    tool_result = result.messages[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.is_error
    assert expected in tool_result.content[0].text
    assert result.reason == "completed"


async def test_tool_exception_and_timeout_become_error_results() -> None:
    async def boom(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        raise RuntimeError("数据库连不上")

    async def hang(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        await asyncio.sleep(10)
        raise AssertionError

    tools = [AgentTool("boom", "", ECHO_SCHEMA, boom), AgentTool("hang", "", ECHO_SCHEMA, hang)]
    model = ScriptedModel(
        [
            tool_reply(call("boom", {"text": "x"}), call("hang", {"text": "y"})),
            text_reply("好"),
        ]
    )
    _, result = await run_events(model, tools, limits=RunLimits(tool_timeout=0.05))
    texts = [m.content[0].text for m in result.messages if isinstance(m, ToolResultMessage)]
    assert texts == ["数据库连不上", "Tool timed out"]


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
async def test_tool_raising_cancelled_error_becomes_error_result(mode: str) -> None:
    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        # 运行没有被取消，工具等待的另一个任务被取消了。
        other = asyncio.get_running_loop().create_future()
        other.cancel()
        await other
        raise AssertionError

    model = ScriptedModel([tool_reply(call("t", {"text": "x"}, id="a")), text_reply("换个办法")])
    _, result = await run_events(
        model, [AgentTool("t", "", ECHO_SCHEMA, execute)], tool_execution=mode
    )
    assert result.reason == "completed"
    tool_result = result.messages[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert tool_result.tool_call_id == "a" and tool_result.is_error
    assert tool_result.content[0].text == "CancelledError"


async def test_before_tool_call_blocks_and_after_tool_call_overrides() -> None:
    async def before(ctx: Any) -> BeforeToolCallResult | None:
        assert ctx.tool_call in ctx.assistant_message.tool_calls
        if ctx.tool_call.name == "blocked":
            return BeforeToolCallResult(block=True, reason="不允许")
        return None

    async def after(ctx: Any) -> AfterToolCallResult:
        return AfterToolCallResult(content=(TextContent("改写"),), details=None)

    model = ScriptedModel(
        [
            tool_reply(call("blocked", {"text": "x"}), call("echo", {"text": "y"})),
            text_reply("好"),
        ]
    )
    _, result = await run_events(
        model,
        [echo_tool("blocked"), echo_tool()],
        before_tool_call=before,
        after_tool_call=after,
    )
    blocked, rewritten = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert blocked.is_error and blocked.content[0].text == "不允许"
    assert rewritten.content[0].text == "改写" and rewritten.details is None


async def test_terminate_only_when_every_result_terminates() -> None:
    async def submit(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        return AgentToolResult(content=(TextContent("已提交"),), terminate=True)

    tools = [AgentTool("submit", "", ECHO_SCHEMA, submit), echo_tool()]
    only_submit = ScriptedModel([tool_reply(call("submit", {"text": "x"}))])
    _, result = await run_events(only_submit, tools)
    assert result.reason == "terminated"

    mixed = ScriptedModel(
        [tool_reply(call("submit", {"text": "x"}), call("echo", {"text": "y"})), text_reply("好")]
    )
    _, result = await run_events(mixed, tools)
    assert result.reason == "completed"


async def test_finish_turn_continue_and_end() -> None:
    calls = 0

    async def finish(turn: Any) -> str | None:
        nonlocal calls
        calls += 1
        return "continue" if calls == 1 else None

    model = ScriptedModel([text_reply("第一次"), text_reply("第二次")])
    _, result = await run_events(model, finish_turn=finish)
    assert [m.text for m in result.messages if isinstance(m, AssistantMessage)] == [
        "第一次",
        "第二次",
    ]

    async def end(turn: Any) -> str:
        return "end"

    model = ScriptedModel([tool_reply(call("echo", {"text": "x"}))])
    _, result = await run_events(model, [echo_tool()], finish_turn=end)
    assert result.reason == "finish_turn"


async def test_max_model_requests() -> None:
    model = ScriptedModel([tool_reply(call("echo", {"text": str(i)})) for i in range(5)])
    _, result = await run_events(model, [echo_tool()], limits=RunLimits(max_model_requests=2))
    assert result.reason == "max_model_requests"
    assert len(model.contexts) == 2
    assert isinstance(result.messages[-1], ToolResultMessage)


async def test_max_tool_calls_rejects_whole_batch() -> None:
    model = ScriptedModel(
        [
            tool_reply(call("echo", {"text": "1"})),
            tool_reply(call("echo", {"text": "2"}), call("echo", {"text": "3"})),
        ]
    )
    _, result = await run_events(model, [echo_tool()], limits=RunLimits(max_tool_calls=2))
    assert result.reason == "max_tool_calls"
    last_two = [m for m in result.messages if isinstance(m, ToolResultMessage)][-2:]
    assert all(m.is_error for m in last_two)
    assert "exceeds remaining budget" in last_two[0].content[0].text


async def test_truncated_reply_does_not_execute_tools() -> None:
    executed = False

    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        nonlocal executed
        executed = True
        return AgentToolResult.of("x")

    model = ScriptedModel(
        [tool_reply(call("t", {"text": "x"}), stop_reason="length"), text_reply("重发")]
    )
    _, result = await run_events(model, [AgentTool("t", "", ECHO_SCHEMA, execute)])
    assert not executed
    assert "output token limit" in result.messages[2].content[0].text


async def test_model_error_is_recorded_and_ends_run() -> None:
    model = ScriptedModel([error_reply("HTTP 500", text="写了一半")])
    _, result = await run_events(model)
    assert result.reason == "error"
    last = result.messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "error" and last.error_message == "HTTP 500"
    assert last.text == "写了一半"


async def test_connector_exception_becomes_error_reply() -> None:
    class DropsMidStream:
        def stream(self, context: Any) -> Any:
            return self._stream()

        async def _stream(self) -> Any:
            yield StreamDelta("text_start", 0)
            yield StreamDelta("text_delta", 0, "写了一半")
            raise ConnectionError("网关断开")

    events, result = await run_events(DropsMidStream())
    assert result.reason == "error"
    last = result.messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "error" and last.error_message == "ConnectionError: 网关断开"
    assert last.text == "写了一半"
    assert types(events)[-2:] == ["turn_end", "agent_end"]

    class FailsBeforeStreaming:
        def stream(self, context: Any) -> Any:
            raise RuntimeError("没有配置密钥")

    _, result = await run_events(FailsBeforeStreaming())
    assert result.reason == "error"
    assert result.messages[-1].error_message == "RuntimeError: 没有配置密钥"


async def test_connector_exception_while_closing_keeps_the_reply() -> None:
    class FailsOnClose:
        def stream(self, context: Any) -> Any:
            return self._stream()

        async def _stream(self) -> Any:
            try:
                yield StreamDone(text_reply("答"))
            finally:
                raise RuntimeError("关闭连接失败")

    _, result = await run_events(FailsOnClose())
    assert result.reason == "completed"
    assert result.messages[-1].text == "答"


async def test_error_reply_drops_tool_calls() -> None:
    broken = AssistantMessage(
        (TextContent("前文"), ToolCall("c", "echo", {})), "error", error_message="断开"
    )
    _, result = await run_events(ScriptedModel([broken]), [echo_tool()])
    assert result.messages[-1].tool_calls == ()


async def test_cancel_while_streaming_keeps_partial_text() -> None:
    model = ScriptedModel([text_reply("abcdefghij")], chunk_size=2, delay=0.02)
    stream = agent_loop([UserMessage.of("问")], AgentContext(), AgentLoopConfig(model=model))
    seen = 0
    async for event in stream:
        if isinstance(event, MessageUpdate) and event.kind == "text_delta":
            seen += 1
            if seen == 2:
                stream.cancel()
    result = await stream.result()
    assert result.reason == "aborted"
    last = result.messages[-1]
    assert last.stop_reason == "aborted"
    assert last.text.startswith("abcd") and last.text != "abcdefghij"


async def test_cancel_while_tools_run_fills_missing_results() -> None:
    model = ScriptedModel(
        [tool_reply(call("fast", {"text": "1"}, id="a"), call("slow", {"text": "2"}, id="b"))]
    )
    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(tools=[echo_tool("fast"), echo_tool("slow", delay=5)]),
        AgentLoopConfig(model=model),
    )
    async for event in stream:
        if isinstance(event, ToolExecutionEnd) and event.tool_call_id == "a":
            stream.cancel()
    result = await stream.result()
    assert result.reason == "aborted"
    fast, slow = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert not fast.is_error
    assert slow.is_error and slow.content[0].text == "Operation aborted"


async def test_run_timeout() -> None:
    model = ScriptedModel([tool_reply(call("slow", {"text": "1"}))])
    _, result = await run_events(
        model, [echo_tool("slow", delay=5)], limits=RunLimits(run_timeout=0.05)
    )
    assert result.reason == "run_timeout"
    assert result.messages[-1].content[0].text == "Operation aborted"


async def test_cancel_before_start() -> None:
    stream = agent_loop(
        [UserMessage.of("问")], AgentContext(), AgentLoopConfig(model=ScriptedModel([]))
    )
    stream.cancel()
    events = [event.type async for event in stream]
    result = await stream.result()
    assert result.reason == "aborted"
    # 和 Pi 一样，已经取消时不发请求，记一条 aborted 回复，轮次照常结束。
    assert [m.role for m in result.messages] == ["user", "assistant"]
    assert result.messages[-1].stop_reason == "aborted"
    assert events[-2:] == ["turn_end", "agent_end"]


async def test_hook_exception_propagates_without_agent_end() -> None:
    async def broken(messages: Any) -> Any:
        raise ValueError("钩子写错了")

    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(),
        AgentLoopConfig(model=ScriptedModel([text_reply("x")]), transform_context=broken),
    )
    seen: list[str] = []
    with pytest.raises(ValueError, match="钩子写错了"):
        async for event in stream:
            seen.append(event.type)
    assert "agent_end" not in seen
    with pytest.raises(ValueError):
        await stream.result()


async def test_custom_messages_are_not_sent_by_default() -> None:
    model = ScriptedModel([text_reply("好")])
    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(messages=[SystemMessage("系统"), CustomMessage("card", {"x": 1})]),
        AgentLoopConfig(model=model),
    )
    await stream.result()
    assert [m.role for m in model.contexts[0].messages] == ["system", "user"]


async def test_transform_context_does_not_change_history() -> None:
    async def keep_last(messages: Any) -> Any:
        return messages[-1:]

    model = ScriptedModel([text_reply("好")])
    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(messages=[SystemMessage("系统"), UserMessage.of("早先")]),
        AgentLoopConfig(model=model, transform_context=keep_last),
    )
    result = await stream.result()
    assert [m.role for m in model.contexts[0].messages] == ["user"]
    assert [m.role for m in result.messages] == ["user", "assistant"]


async def test_tool_progress_events() -> None:
    async def slow(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        await on_update({"step": 1})
        return AgentToolResult.of("好")

    model = ScriptedModel([tool_reply(call("s", {"text": "x"})), text_reply("完")])
    events, _ = await run_events(model, [AgentTool("s", "", ECHO_SCHEMA, slow)])
    updates = [e for e in events if e.type == "tool_execution_update"]
    assert updates[0].partial_result == {"step": 1}


async def test_continue_requires_user_or_tool_result_tail() -> None:
    with pytest.raises(ValueError):
        agent_loop_continue(
            AgentContext(messages=[UserMessage.of("问"), text_reply("答")]),
            AgentLoopConfig(model=ScriptedModel([])),
        )
    model = ScriptedModel([text_reply("重试成功")])
    stream = agent_loop_continue(
        AgentContext(messages=[UserMessage.of("问")]), AgentLoopConfig(model=model)
    )
    result = await stream.result()
    assert [m.role for m in result.messages] == ["assistant"]


async def test_invalid_schema_fails_before_running() -> None:
    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        return AgentToolResult.of("x")

    bad = AgentTool("bad", "", {"type": "not-a-type"}, execute)
    with pytest.raises(Exception, match="not-a-type"):
        agent_loop(
            [UserMessage.of("问")],
            AgentContext(tools=[bad]),
            AgentLoopConfig(model=ScriptedModel([])),
        )


async def test_sequential_results_are_written_right_after_each_tool() -> None:
    model = ScriptedModel(
        [
            tool_reply(call("a", {"text": "1"}, id="a"), call("b", {"text": "2"}, id="b")),
            text_reply("好"),
        ]
    )
    events, _ = await run_events(
        model, [echo_tool("a"), echo_tool("b")], tool_execution="sequential"
    )
    trace = []
    for event in events:
        if isinstance(event, ToolExecutionStart | ToolExecutionEnd):
            trace.append(f"{event.type}:{event.tool_call_id}")
        elif isinstance(event, MessageEnd) and isinstance(event.message, ToolResultMessage):
            trace.append(f"message_end:{event.message.tool_call_id}")
    assert trace == [
        "tool_execution_start:a",
        "tool_execution_end:a",
        "message_end:a",
        "tool_execution_start:b",
        "tool_execution_end:b",
        "message_end:b",
    ]


async def test_parallel_results_are_written_after_all_tools() -> None:
    model = ScriptedModel(
        [
            tool_reply(call("a", {"text": "1"}, id="a"), call("b", {"text": "2"}, id="b")),
            text_reply("好"),
        ]
    )
    events, _ = await run_events(model, [echo_tool("a"), echo_tool("b")])
    kinds = [
        e.type
        for e in events
        if e.type == "tool_execution_end"
        or (e.type == "message_end" and isinstance(e.message, ToolResultMessage))
    ]
    assert kinds == ["tool_execution_end", "tool_execution_end", "message_end", "message_end"]


async def test_hooks_get_a_copy_of_the_arguments() -> None:
    seen: list[dict[str, Any]] = []

    async def before(ctx: Any) -> None:
        ctx.args["text"] = "改过"

    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        seen.append(params)
        params["extra"] = True
        return AgentToolResult.of("好")

    model = ScriptedModel([tool_reply(call("t", {"text": "原值"})), text_reply("完")])
    _, result = await run_events(
        model, [AgentTool("t", "", ECHO_SCHEMA, execute)], before_tool_call=before
    )
    assert seen[0]["text"] == "改过"
    assert result.messages[1].tool_calls[0].arguments == {"text": "原值"}


async def test_hook_exception_leaves_reusable_history() -> None:
    async def before(ctx: Any) -> None:
        raise RuntimeError("审计服务不可用")

    model = ScriptedModel([tool_reply(call("echo", {"text": "a"}, id="c1"))])
    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(tools=[echo_tool()]),
        AgentLoopConfig(model=model, before_tool_call=before),
    )
    ends: list[Any] = []
    with pytest.raises(RuntimeError, match="审计服务不可用"):
        async for event in stream:
            if event.type == "message_end":
                ends.append(event.message)
    assert isinstance(ends[-1], ToolResultMessage)
    assert ends[-1].tool_call_id == "c1" and ends[-1].is_error
    assert "RuntimeError: 审计服务不可用" in ends[-1].content[0].text
