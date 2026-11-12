from __future__ import annotations

import asyncio
from typing import Any

from agent_looplite import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    UserMessage,
    agent_loop,
)

ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def echo_tool(name: str = "echo", *, delay: float = 0.0, **kwargs: Any) -> AgentTool:
    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        await asyncio.sleep(delay)
        return AgentToolResult.of(f"{name}:{params['text']}", details={"call_id": call_id})

    return AgentTool(name, f"返回 {name}:text", ECHO_SCHEMA, execute, **kwargs)


async def run_events(model: Any, tools: list[AgentTool] = (), prompt: str = "问题", **config: Any):
    stream = agent_loop(
        [UserMessage.of(prompt)],
        AgentContext(tools=tools),
        AgentLoopConfig(model=model, **config),
    )
    events = [event async for event in stream]
    return events, await stream.result()


def types(events: list[Any]) -> list[str]:
    return [event.type for event in events]
