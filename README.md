# agent-looplite

通用的 Python agent 循环库：把对话发给模型，执行模型要求的工具，把结果交回模型，直到模型给出回答。

项目处于设计阶段，还没有可用的接口。设计约束见 `CLAUDE.md`。

```bash
uv sync
uv run ruff format . && uv run ruff check .
uv run pytest -q
```
