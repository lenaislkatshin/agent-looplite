"""模型连接器的协议，对应 Pi 的 `StreamFn`。

连接器把流式分片或整轮返回的响应整理成同一种事件，最后一个事件必须是 `StreamDone` 或
`StreamError`。请求失败不抛异常，发 `StreamError`；连接器仍然抛出的异常，循环记成 `error`
回复。拼接分片、解析工具参数、重试、识别网关错误、出网前检查都是连接器的事，循环不管。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .messages import AssistantMessage, Message

DeltaKind = Literal[
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
]


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelContext:
    """`messages` 已经过 `transform_context` 和 `convert_to_llm`。"""

    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...]


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """`*_start` 和 `*_end` 的 `delta` 为空。`toolcall_start` 带上已知的工具调用 ID 和名字。"""

    kind: DeltaKind
    content_index: int
    delta: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class StreamDone:
    """`message.stop_reason` 是 `stop`、`tool_use` 或 `length`。"""

    message: AssistantMessage


@dataclass(frozen=True, slots=True)
class StreamError:
    """`message.stop_reason` 是 `error` 或 `aborted`，带着出错前已收到的内容。"""

    message: AssistantMessage


AssistantMessageEvent = StreamDelta | StreamDone | StreamError


class ModelClient(Protocol):
    def stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]: ...
