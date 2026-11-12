"""循环发出的事件，种类和顺序与 Pi 的 `AgentEvent` 相同。

`MessageUpdate` 只带增量，不带拼好的部分消息。需要部分消息时用 `PartialMessage` 自己拼。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from .messages import (
    AgentMessage,
    AssistantMessage,
    ContentBlock,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
)
from .model import DeltaKind, StreamDelta

if TYPE_CHECKING:
    from .tools import AgentToolResult


@dataclass(frozen=True, slots=True)
class AgentStart:
    type: ClassVar[str] = "agent_start"


@dataclass(frozen=True, slots=True)
class AgentEnd:
    """`messages` 是本次新增的消息，`reason` 是结束原因。"""

    type: ClassVar[str] = "agent_end"
    messages: tuple[AgentMessage, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class TurnStart:
    type: ClassVar[str] = "turn_start"


@dataclass(frozen=True, slots=True)
class TurnEnd:
    type: ClassVar[str] = "turn_end"
    message: AssistantMessage
    tool_results: tuple[ToolResultMessage, ...]


@dataclass(frozen=True, slots=True)
class MessageStart:
    """模型回复开始时还没有内容，`message` 为 `None`，`role` 是 `assistant`。"""

    type: ClassVar[str] = "message_start"
    role: str
    message: AgentMessage | None


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    type: ClassVar[str] = "message_update"
    kind: DeltaKind
    content_index: int
    delta: str = ""
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class MessageEnd:
    type: ClassVar[str] = "message_end"
    message: AgentMessage


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    type: ClassVar[str] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdate:
    type: ClassVar[str] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    partial_result: Any


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    """按完成顺序发出。`duration_ms` 在工具没有真正执行时为 `None`。"""

    type: ClassVar[str] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool
    duration_ms: float | None


AgentEvent = (
    AgentStart
    | AgentEnd
    | TurnStart
    | TurnEnd
    | MessageStart
    | MessageUpdate
    | MessageEnd
    | ToolExecutionStart
    | ToolExecutionUpdate
    | ToolExecutionEnd
)


class PartialMessage:
    """把一次回复的增量拼起来。每次 `apply` 只追加，取内容时才拼接字符串。"""

    def __init__(self) -> None:
        self._blocks: dict[int, tuple[str, list[str]]] = {}

    def apply(self, update: MessageUpdate | StreamDelta) -> None:
        kind = update.kind.split("_", 1)[0]
        block = self._blocks.setdefault(update.content_index, (kind, []))
        if update.delta:
            block[1].append(update.delta)

    @property
    def text(self) -> str:
        return "".join("".join(parts) for kind, parts in self._ordered() if kind == "text")

    @property
    def thinking(self) -> str:
        return "".join("".join(parts) for kind, parts in self._ordered() if kind == "thinking")

    def content(self) -> tuple[ContentBlock, ...]:
        """正文和思考按出现顺序返回；工具调用的参数可能还不完整，不返回。"""
        blocks: list[ContentBlock] = []
        for kind, parts in self._ordered():
            if kind == "text":
                blocks.append(TextContent("".join(parts)))
            elif kind == "thinking":
                blocks.append(ThinkingContent("".join(parts)))
        return tuple(blocks)

    def _ordered(self) -> list[tuple[str, list[str]]]:
        return [self._blocks[index] for index in sorted(self._blocks)]
