from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_looplite import AgentTool, AgentToolResult, ToolCall, ToolResultMessage, tool
from agent_looplite.testing import ScriptedModel, call, text_reply, tool_reply
from agent_looplite.tools import ToolSet

from .helpers import run_events


def prepare(schema: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        return AgentToolResult.of("")

    tools = ToolSet([AgentTool("t", "", schema, execute)])
    return tools.prepare_arguments(ToolCall("c", "t", arguments))


def obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


@pytest.mark.parametrize(
    ("schema", "sent", "expected"),
    [
        ({"type": "integer"}, "42", 42),
        ({"type": "integer"}, " -7 ", -7),
        ({"type": "integer"}, "3.0", 3),
        ({"type": "integer"}, "12345678901234567890", 12345678901234567890),
        ({"type": "number"}, "2.5", 2.5),
        ({"type": "number"}, "8", 8),
        ({"type": "boolean"}, "True", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "string"}, 2026, "2026"),
        ({"type": "string"}, True, "true"),
        ({"type": "array", "items": {"type": "integer"}}, "5", [5]),
        ({"type": "array", "items": {"type": "string"}}, ["a", 1], ["a", "1"]),
        ({"enum": [1, 2, 3]}, "2", 2),
        ({"const": 1}, "1", 1),
        ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "9", 9),
        ({"type": ["integer", "null"]}, "4", 4),
    ],
)
def test_values_are_coerced_to_the_schema_type(schema: Any, sent: Any, expected: Any) -> None:
    arguments, rejected = prepare(obj({"v": schema}, ("v",)), {"v": sent})
    assert rejected is None
    assert arguments == {"v": expected}
    assert type(arguments["v"]) is type(expected)


@pytest.mark.parametrize(
    ("schema", "sent"),
    [
        ({"type": "integer"}, "3.5"),
        ({"type": "integer"}, "abc"),
        ({"type": "number"}, "1e400"),
        ({"type": "number"}, "NaN"),
        ({"type": "boolean"}, "yes"),
        ({"type": "object"}, '{"a": 1}'),
        ({"type": "integer"}, None),
        ({"type": ["array", "null"], "items": {"type": "integer"}}, 5),
    ],
)
def test_values_that_cannot_be_coerced_are_reported(schema: Any, sent: Any) -> None:
    arguments, rejected = prepare(obj({"v": schema}, ("v",)), {"v": sent})
    assert arguments == {}
    assert rejected is not None and rejected.is_error
    text = rejected.content[0].text
    assert text.startswith('Validation failed for tool "t":')
    assert "Received arguments" in text


def test_null_for_optional_property_counts_as_omitted() -> None:
    schema = obj(
        {
            "page": {"type": "integer"},
            "note": {"type": ["string", "null"]},
            "q": {"type": "string"},
        },
        ("q",),
    )
    arguments, rejected = prepare(schema, {"q": "x", "page": None, "note": None})
    assert rejected is None
    assert arguments == {"q": "x", "note": None}

    _, rejected = prepare(schema, {"q": None})
    assert rejected is not None


def test_nested_objects_and_refs_are_coerced() -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "$defs": {"Item": obj({"n": {"type": "integer"}}, ("n",))},
    }
    arguments, rejected = prepare(schema, {"items": {"n": "1"}})
    assert rejected is None
    assert arguments == {"items": [{"n": 1}]}


def test_recursive_schema_does_not_loop() -> None:
    schema = {
        **obj({"tree": {"$ref": "#/$defs/Tree"}}, ("tree",)),
        "$defs": {"Tree": {"type": "array", "items": {"$ref": "#/$defs/Tree"}}},
    }
    arguments, rejected = prepare(schema, {"tree": 5})
    assert rejected is not None
    assert arguments == {}
    arguments, rejected = prepare(schema, {"tree": [[], [[]]]})
    assert rejected is None
    assert arguments == {"tree": [[], [[]]]}


async def test_tool_gets_coerced_arguments_and_history_keeps_the_original() -> None:
    received: list[dict[str, Any]] = []

    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        received.append(dict(params))
        params["page"] = 0
        return AgentToolResult.of("第 42 页")

    page_tool = AgentTool("read_page", "", obj({"page": {"type": "integer"}}, ("page",)), execute)
    model = ScriptedModel([tool_reply(call("read_page", {"page": "42"})), text_reply("读完了")])
    _, result = await run_events(model, [page_tool])
    assert received == [{"page": 42}]
    assert result.messages[1].tool_calls[0].arguments == {"page": "42"}
    tool_result = result.messages[2]
    assert isinstance(tool_result, ToolResultMessage) and not tool_result.is_error


class Filter(BaseModel):
    year: int
    ports: list[str] = []


async def test_tool_decorator_accepts_loosely_typed_arguments() -> None:
    seen: list[Any] = []

    @tool
    async def search(query: str, limit: int | None = None, scope: Filter | None = None) -> str:
        """检索。"""
        seen.append((query, limit, scope))
        return "好"

    model = ScriptedModel(
        [
            tool_reply(
                call(
                    "search",
                    {"query": 2026, "limit": "5", "scope": {"year": "2026", "ports": "宁波舟山港"}},
                )
            ),
            text_reply("完成"),
        ]
    )
    _, result = await run_events(model, [search])
    assert not result.messages[2].is_error
    assert seen == [("2026", 5, Filter(year=2026, ports=["宁波舟山港"]))]
