"""本库一侧的回放：和 pi_replay.mjs 用同样的系统提示词、工具和问题，事件写成同样的格式。

    uv run python scripts/pi_parity/looplite_replay.py <端口> <结果目录>

角色名和结束原因换成 Pi 的写法，例如 `tool_result` 写作 `toolResult`，方便逐条比对。
"""

import asyncio
import json
import pathlib
import sys
from typing import Any

from agent_looplite import Agent, AgentEvent, AgentTool, AgentToolResult, ChatCompletionsClient

SYSTEM_PROMPT = "你是宁波市交通运输局的业务助手，用中文简洁作答。"
QUESTION = (
    "请用 lookup_rule 工具分别查询甬交规〔2026〕7号和甬交规〔2026〕9号的规定，"
    "然后回答两者的限速相差多少公里每小时。"
)
RULES = {
    "甬交规〔2026〕7号": "城区主干道限速 50 公里/小时",
    "甬交规〔2026〕9号": "城区次干道限速 40 公里/小时",
}
PI_ROLE = {"tool_result": "toolResult"}
PI_STOP_REASON = {"tool_use": "toolUse"}

executed: list[list[str]] = []


async def lookup(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
    executed.append([call_id, params["code"]])
    return AgentToolResult.of(RULES.get(params["code"], "没有找到"))


LOOKUP = AgentTool(
    "lookup_rule",
    "按文号查询交通规定的内容。",
    {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    lookup,
    label="查规定",
)


def describe(event: AgentEvent) -> str:
    if event.type == "message_update":
        if event.kind == "toolcall_start":
            extra = f"{event.tool_call_id} {event.tool_name}"
        elif event.kind.endswith("_delta"):
            extra = json.dumps(event.delta, ensure_ascii=False)
        else:
            extra = ""
        return f"message_update {event.kind}@{event.content_index} {extra}".strip()
    if event.type in ("message_start", "message_end"):
        message = event.message
        role = event.role if event.type == "message_start" else message.role
        line = f"{event.type} {PI_ROLE.get(role, role)}"
        if message is not None and message.role == "tool_result":
            line += f" {message.tool_call_id}"
        return line
    if event.type == "tool_execution_end":
        return f"{event.type} {event.tool_call_id} error={str(event.is_error).lower()}"
    if event.type.startswith("tool_execution"):
        return f"{event.type} {event.tool_call_id}"
    return event.type


def summarize(message: Any) -> dict[str, Any]:
    if message.role == "assistant":
        return {
            "role": "assistant",
            "text": message.text,
            "tool_calls": [[c.id, c.name, c.arguments] for c in message.tool_calls],
            "stop_reason": PI_STOP_REASON.get(message.stop_reason, message.stop_reason),
            "usage": {
                "input": message.usage.input,
                "output": message.usage.output,
                "cache_read": message.usage.cache_read,
                "total": message.usage.total_tokens,
            },
        }
    item: dict[str, Any] = {
        "role": message.role,
        "text": "".join(block.text for block in message.content),
    }
    if message.role == "tool_result":
        item["tool_call_id"] = message.tool_call_id
        item["is_error"] = message.is_error
    return item


async def main() -> None:
    port, out = sys.argv[1], pathlib.Path(sys.argv[2])
    model = ChatCompletionsClient(
        base_url=f"http://127.0.0.1:{port}/looplite/v1", model="deepseek-v4.1-flash-1017"
    )
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=[LOOKUP])
    events: list[str] = []
    agent.subscribe(lambda event: events.append(describe(event)))
    async with model:
        await agent.prompt(QUESTION)
    result = {
        "events": events,
        "executed": executed,
        "messages": [summarize(m) for m in agent.state.messages],
    }
    (out / "looplite-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"本库完成：事件 {len(events)} 个，工具执行 {len(executed)} 次")


asyncio.run(main())
