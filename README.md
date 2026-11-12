# agent-looplite

Pi `agent` 核心（`earendil-works/pi`）的轻量 Python 实现：把对话发给模型，执行模型要求的工具，把结果交回模型，直到模型给出回答。

```bash
uv add git+<本仓库地址>
uv add "agent-looplite[pydantic] @ git+<本仓库地址>"   # 要用 @tool 装饰器时
```

```python
from agent_looplite import Agent, ChatCompletionsClient, tool


@tool
async def add(a: float, b: float) -> str:
    """计算 a + b。"""
    return str(a + b)


model = ChatCompletionsClient(
    base_url="https://api.deepseek.com", api_key="...", model="deepseek-chat"
)
agent = Agent(model=model, system_prompt="用工具计算。", tools=[add])
result = await agent.prompt("1.5 加 2 等于多少？")
print(agent.state.messages[-1].text)
```

- 接口：`docs/design.md`
- 性能对比：`docs/benchmark.md`
- 定位和首版范围：`CLAUDE.md`

```bash
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest -q
```
