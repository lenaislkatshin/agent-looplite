"""用真实模型跑一遍：流式和整轮返回各一次，模型要连续调用工具才能回答。

    LOOPLITE_BASE_URL=... LOOPLITE_API_KEY=... LOOPLITE_MODEL=... uv run python scripts/e2e.py

只发送下面写死的示例问题和示例数据。
"""

from __future__ import annotations

import asyncio
import os
import time

from agent_looplite import (
    Agent,
    ChatCompletionsClient,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    tool,
)

THROUGHPUT = {"宁波舟山港": 3120, "上海港": 4180}


@tool
async def container_throughput(port: str) -> str:
    """查询港口 2026 年 1 至 9 月的集装箱吞吐量（示例数据），单位：万标准箱。"""
    if port not in THROUGHPUT:
        return f"没有 {port} 的数据，可查询：{'、'.join(THROUGHPUT)}"
    return f"{port}：{THROUGHPUT[port]} 万标准箱"


@tool
async def subtract(a: float, b: float) -> str:
    """计算 a - b。"""
    return str(a - b)


async def run_once(stream: bool) -> None:
    model = ChatCompletionsClient(
        base_url=os.environ["LOOPLITE_BASE_URL"],
        api_key=os.environ["LOOPLITE_API_KEY"],
        model=os.environ["LOOPLITE_MODEL"],
        stream=stream,
    )
    agent = Agent(
        model=model,
        system_prompt="用工具查数据并计算，最后用一句中文回答。",
        tools=[container_throughput, subtract],
    )
    deltas = 0

    def listen(event: object) -> None:
        nonlocal deltas
        if isinstance(event, MessageUpdate) and event.kind.endswith("_delta"):
            deltas += 1
        elif isinstance(event, ToolExecutionStart):
            print(f"  调用 {event.tool_name}({event.args})")
        elif isinstance(event, ToolExecutionEnd):
            print(f"  结果 {event.result.content[0].text}")

    agent.subscribe(listen)
    started = time.monotonic()
    async with model:
        result = await agent.prompt("宁波舟山港今年 1 至 9 月的集装箱吞吐量比上海港少多少？")
    elapsed = time.monotonic() - started
    print(f"  结束原因 {result.reason}，增量事件 {deltas} 个，用时 {elapsed:.1f} 秒")
    print(f"  用量 {result.usage}")
    print(f"  回答 {agent.state.messages[-1].text}")
    assert result.reason == "completed", agent.state.error_message


async def main() -> None:
    for stream in (True, False):
        print("流式" if stream else "整轮返回")
        await run_once(stream)


if __name__ == "__main__":
    asyncio.run(main())
