"""工具的定义、参数校验和执行。

工具出错一律变成 `is_error=True` 的结果回传给模型，不结束运行。回传的文字与 Pi 一致，方便和
pi-python 对照；只有参数校验里每处问题的说明来自 jsonschema。
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator

from .coercion import coerce_arguments
from .messages import TextContent, ToolCall
from .model import ToolDeclaration

ToolUpdate = Callable[[Any], Awaitable[None]]
ToolExecute = Callable[[str, dict[str, Any], ToolUpdate], Awaitable["AgentToolResult"]]

ABORTED_TEXT = "Operation aborted"
MAX_REPORTED_PROBLEMS = 5


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """`content` 发给模型；`details` 给界面和日志。

    本轮所有结果都带 `terminate` 时，循环不再请求模型。
    """

    content: tuple[TextContent, ...]
    details: Any = None
    is_error: bool = False
    terminate: bool = False

    @classmethod
    def of(cls, text: str, details: Any = None) -> AgentToolResult:
        return cls(content=(TextContent(text),), details=details)


@dataclass(frozen=True, slots=True)
class AgentTool:
    """`execute(tool_call_id, params, on_update)`。

    `params` 已按 `parameters` 校验，`on_update` 报告进度。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: ToolExecute
    label: str = ""
    execution_mode: Literal["parallel", "sequential"] | None = None

    def declaration(self) -> ToolDeclaration:
        return ToolDeclaration(self.name, self.description, self.parameters)


def error_result(text: str) -> AgentToolResult:
    return AgentToolResult(content=(TextContent(text),), is_error=True)


class ToolSet:
    """一次运行可用的工具。构造时检查 schema 本身是否合法，写错的 schema 在运行前就报出来。"""

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools: dict[str, tuple[AgentTool, Draft202012Validator]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"工具名重复：{tool.name}")
            Draft202012Validator.check_schema(tool.parameters)
            self._tools[tool.name] = (tool, Draft202012Validator(tool.parameters))

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(tool.declaration() for tool, _ in self._tools.values())

    def get(self, name: str) -> AgentTool | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def prepare_arguments(self, call: ToolCall) -> tuple[dict[str, Any], AgentToolResult | None]:
        """返回交给钩子和工具的参数，以及参数不能执行时要回传给模型的错误结果。

        和 Pi 一样，参数先复制一份，做类型宽容转换后再校验；历史里的工具调用保留模型发来的原始
        参数。不能执行时参数为空。
        """
        entry = self._tools.get(call.name)
        if entry is None:
            return {}, error_result(f"Tool {call.name} not found")
        if call.arguments_error is not None:
            return {}, error_result(
                f'Invalid arguments for tool "{call.name}": {call.arguments_error}'
            )
        validator = entry[1]
        arguments = coerce_arguments(copy.deepcopy(call.arguments), validator)
        problems = [
            f"  - {_error_path(error.json_path)}: {error.message}"
            for error in validator.iter_errors(arguments)
        ]
        if not problems:
            return arguments, None
        if len(problems) > MAX_REPORTED_PROBLEMS:
            more = len(problems) - MAX_REPORTED_PROBLEMS
            problems = [*problems[:MAX_REPORTED_PROBLEMS], f"  (and {more} more)"]
        received = json.dumps(call.arguments, ensure_ascii=False, indent=2)
        return {}, error_result(
            f'Validation failed for tool "{call.name}":\n'
            + "\n".join(problems)
            + f"\n\nReceived arguments:\n{received}"
        )


def _error_path(json_path: str) -> str:
    return "root" if json_path == "$" else json_path.removeprefix("$.")


async def run_tool(
    tool: AgentTool,
    tool_call_id: str,
    arguments: dict[str, Any],
    on_update: ToolUpdate,
    timeout: float | None,
) -> tuple[AgentToolResult, float]:
    """执行一个工具，返回结果和耗时（毫秒）。工具抛异常或超时都变成错误结果。

    运行被取消时，取消照常向外传。运行没被取消、工具自己抛出 `CancelledError` 时，例如它等待的
    另一个任务被取消，和其他异常一样变成错误结果。
    """
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout) as deadline:
            result = await tool.execute(tool_call_id, arguments, on_update)
    except TimeoutError as exc:
        result = error_result(
            "Tool timed out" if deadline.expired() else str(exc) or "TimeoutError"
        )
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        result = error_result(str(exc) or type(exc).__name__)
    except Exception as exc:
        result = error_result(str(exc) or type(exc).__name__)
    if not isinstance(result, AgentToolResult):
        result = error_result(
            f"Tool {tool.name} returned {type(result).__name__}, not AgentToolResult"
        )
    return result, (time.monotonic() - started) * 1000
