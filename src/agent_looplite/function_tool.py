"""`@tool`：从函数的类型注解和 docstring 生成 `AgentTool`，参数格式由 pydantic 生成。

pydantic 在这里的角色相当于 Pi 里的 TypeBox，是可选依赖：`pip install agent-looplite[pydantic]`。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal, overload

from pydantic import ConfigDict, create_model

from .tools import AgentTool, AgentToolResult, ToolUpdate

ToolFunction = Callable[..., Awaitable[AgentToolResult | str]]


@overload
def tool(function: ToolFunction, /) -> AgentTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    label: str | None = None,
    description: str | None = None,
    execution_mode: Literal["parallel", "sequential"] | None = None,
) -> Callable[[ToolFunction], AgentTool]: ...


def tool(
    function: ToolFunction | None = None,
    /,
    *,
    name: str | None = None,
    label: str | None = None,
    description: str | None = None,
    execution_mode: Literal["parallel", "sequential"] | None = None,
) -> AgentTool | Callable[[ToolFunction], AgentTool]:
    """函数必须是 async 的，返回 `AgentToolResult` 或字符串。说明默认取 docstring。"""

    def build(fn: ToolFunction) -> AgentTool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"{fn.__name__} 必须是 async 函数")
        signature = inspect.signature(fn, eval_str=True)
        fields: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError(f"{fn.__name__} 不能有 *args 或 **kwargs")
            if parameter.annotation is inspect.Parameter.empty:
                raise TypeError(f"{fn.__name__} 的参数 {parameter.name} 缺少类型注解")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (parameter.annotation, default)
        arguments_model = create_model(
            f"{fn.__name__}_arguments", __config__=ConfigDict(extra="forbid"), **fields
        )
        schema = arguments_model.model_json_schema()
        schema.pop("title", None)

        async def execute(
            tool_call_id: str, params: dict[str, Any], on_update: ToolUpdate
        ) -> AgentToolResult:
            parsed = arguments_model.model_validate(params)
            result = await fn(**{key: getattr(parsed, key) for key in fields})
            return AgentToolResult.of(result) if isinstance(result, str) else result

        tool_name = name or fn.__name__
        return AgentTool(
            name=tool_name,
            description=description or inspect.getdoc(fn) or "",
            parameters=schema,
            execute=execute,
            label=label or tool_name,
            execution_mode=execution_mode,
        )

    if function is not None:
        return build(function)
    return build
