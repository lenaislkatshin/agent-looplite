"""用同一份模拟的流式响应，比较本库和 pi-python 处理增量的 CPU 开销。

    uv run python scripts/bench.py --deltas 4000 --runs 50
    uv run --with <pi_python_core 的 wheel> python scripts/bench.py --impl pi-python \
        --deltas 4000 --runs 50

响应由 httpx 的 `MockTransport` 返回，不走网络。每个增量是 5 个汉字，最后一片带 `finish_reason`。
两边都从 Chat Completions 连接器跑到 agent 循环，事件全部交给一个只计数的监听函数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable

import httpx

DELTA_TEXT = "港口吞吐量"


def sse_body(deltas: int) -> bytes:
    chunk = json.dumps({"choices": [{"delta": {"content": DELTA_TEXT}}]}, ensure_ascii=False)
    end = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    return (f"data: {chunk}\n\n" * deltas + f"data: {end}\n\ndata: [DONE]\n\n").encode()


def mock_client(body: bytes) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def looplite_runner(client: httpx.AsyncClient) -> Callable[[], Awaitable[int]]:
    from agent_looplite import Agent, ChatCompletionsClient

    model = ChatCompletionsClient(base_url="http://bench/v1", model="m", http_client=client)

    async def run() -> int:
        agent = Agent(model=model, system_prompt="s")
        events = 0

        def count(event: object) -> None:
            nonlocal events
            events += 1

        agent.subscribe(count)
        await agent.prompt("问")
        return events

    return run


def pi_python_runner(client: httpx.AsyncClient) -> Callable[[], Awaitable[int]]:
    from pi_python import Agent
    from pi_python.providers import OpenAICompletionsProvider
    from pi_python.providers.transport import HTTPTransport

    provider = OpenAICompletionsProvider(
        base_url="http://bench/v1", name="deepseek", api_key="k", transport=HTTPTransport(client)
    )
    model = provider.model("m", context_window=128000, max_tokens=8000, reasoning=True)

    async def run() -> int:
        agent = Agent(provider=provider, model=model, system_prompt="s")
        events = 0

        def count(event: object) -> None:
            nonlocal events
            events += 1

        agent.subscribe(count)
        await agent.prompt("问")
        return events

    return run


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["looplite", "pi-python"], default="looplite")
    parser.add_argument("--deltas", type=int, default=4000)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    client = mock_client(sse_body(args.deltas))
    factory = looplite_runner if args.impl == "looplite" else pi_python_runner
    run = factory(client)
    await run()
    cpu_start = time.process_time()
    events = await asyncio.gather(*(run() for _ in range(args.runs)))
    cpu = time.process_time() - cpu_start
    per_delta = cpu / (args.runs * args.deltas) * 1e6
    print(
        f"{args.impl}：{args.runs} 路并发，每路 {args.deltas} 个增量，每路事件 {events[0]} 个，"
        f"CPU {cpu:.2f} 秒，每个增量 {per_delta:.1f} 微秒"
    )


if __name__ == "__main__":
    asyncio.run(main())
