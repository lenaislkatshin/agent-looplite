from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_looplite import (
    AgentToolResult,
    AssistantMessage,
    CustomMessage,
    PartialMessage,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    decode_messages,
    encode_messages,
    tool,
)
from agent_looplite.testing import ScriptedModel, call, text_reply, tool_reply

from .helpers import run_events


def test_messages_round_trip() -> None:
    messages = (
        SystemMessage("系统"),
        UserMessage.of("问"),
        AssistantMessage(
            (ThinkingContent("想", signature="sig"), ToolCall("c", "t", {"a": 1}, "坏")),
            "tool_use",
            Usage(input=1, output=2, total_tokens=3),
            "m",
        ),
        ToolResultMessage("c", "t", (TextContent("结果"),), {"rows": 3}, True),
        CustomMessage("card", {"x": 1}),
    )
    data = encode_messages(messages)
    assert all(item["schema_version"] == 1 for item in data)
    assert decode_messages(data) == messages


def test_decode_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="版本"):
        decode_messages([{"schema_version": 99, "role": "user"}])


def test_partial_message() -> None:
    from agent_looplite import MessageUpdate

    partial = PartialMessage()
    for update in [
        MessageUpdate("thinking_start", 0),
        MessageUpdate("thinking_delta", 0, "想"),
        MessageUpdate("text_start", 1),
        MessageUpdate("text_delta", 1, "答"),
        MessageUpdate("text_delta", 1, "案"),
        MessageUpdate("toolcall_start", 2, "", "c", "t"),
        MessageUpdate("toolcall_delta", 2, '{"a"'),
    ]:
        partial.apply(update)
    assert partial.text == "答案"
    assert partial.thinking == "想"
    assert partial.content() == (ThinkingContent("想"), TextContent("答案"))


class Point(BaseModel):
    x: int
    y: int


async def test_tool_decorator_builds_schema_and_validates() -> None:
    @tool
    async def locate(place: str, point: Point, zoom: int = 3) -> str:
        """在地图上定位。"""
        return f"{place}@{point.x},{point.y}x{zoom}"

    assert locate.name == "locate"
    assert locate.description == "在地图上定位。"
    assert locate.parameters["required"] == ["place", "point"]
    assert locate.parameters["additionalProperties"] is False

    model = ScriptedModel(
        [
            tool_reply(call("locate", {"place": "宁波", "point": {"x": 1, "y": 2}})),
            tool_reply(call("locate", {"place": "宁波", "point": {"x": "a", "y": 2}})),
            text_reply("好"),
        ]
    )
    _, result = await run_events(model, [locate])
    first, second = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert first.content[0].text == "宁波@1,2x3"
    assert second.is_error and "Validation failed" in second.content[0].text


def test_tool_decorator_options_and_errors() -> None:
    @tool(name="search_knowledge", execution_mode="sequential")
    async def search(query: str) -> AgentToolResult:
        return AgentToolResult.of(query)

    assert search.name == "search_knowledge" and search.execution_mode == "sequential"

    with pytest.raises(TypeError, match="async"):

        @tool
        def sync_tool(query: str) -> str:
            return query
