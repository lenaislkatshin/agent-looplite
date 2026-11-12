"""消息、内容块、用量和序列化。

消息不可变：循环追加消息时生成新的元组，钩子拿到的列表不会被循环改掉，所以不用深拷贝。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

SCHEMA_VERSION = 1

StopReason = Literal["stop", "tool_use", "length", "error", "aborted"]


def now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class TextContent:
    type: ClassVar[str] = "text"
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingContent:
    """`signature` 存厂商要求原样回传的数据，由连接器写入和回传，循环不读。"""

    type: ClassVar[str] = "thinking"
    thinking: str
    signature: str | None = None
    redacted: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """参数不是合法 JSON 对象时，`arguments` 为空，`arguments_error` 写明原因和收到的原文。"""

    type: ClassVar[str] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_error: str | None = None


ContentBlock = TextContent | ThinkingContent | ToolCall


@dataclass(frozen=True, slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            reasoning=self.reasoning + other.reasoning,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True, slots=True)
class SystemMessage:
    role: ClassVar[str] = "system"
    content: str
    timestamp: int = field(default_factory=now_ms)


@dataclass(frozen=True, slots=True)
class UserMessage:
    role: ClassVar[str] = "user"
    content: tuple[TextContent, ...]
    timestamp: int = field(default_factory=now_ms)

    @classmethod
    def of(cls, text: str) -> UserMessage:
        return cls(content=(TextContent(text),))


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    role: ClassVar[str] = "assistant"
    content: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: Usage = Usage()
    model: str = ""
    error_message: str | None = None
    timestamp: int = field(default_factory=now_ms)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCall))

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextContent))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """`content` 发给模型；`details` 给界面和日志，不发给模型。"""

    role: ClassVar[str] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: tuple[TextContent, ...]
    details: Any = None
    is_error: bool = False
    timestamp: int = field(default_factory=now_ms)


@dataclass(frozen=True, slots=True)
class CustomMessage:
    """项目自己的消息，例如界面卡片。默认不发给模型，由 `convert_to_llm` 决定怎样发。"""

    role: ClassVar[str] = "custom"
    kind: str
    data: Any = None
    timestamp: int = field(default_factory=now_ms)


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
AgentMessage = Message | CustomMessage


def text_of(content: tuple[TextContent, ...]) -> str:
    return "".join(block.text for block in content)


def encode_messages(messages: Any) -> list[dict[str, Any]]:
    """`details` 和 `CustomMessage.data` 原样写入，调用方保证它们能转成 JSON。"""
    return [_encode_message(message) for message in messages]


def decode_messages(data: list[dict[str, Any]]) -> tuple[AgentMessage, ...]:
    return tuple(_decode_message(item) for item in data)


def _encode_block(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingContent):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
            "redacted": block.redacted,
        }
    return {
        "type": "tool_call",
        "id": block.id,
        "name": block.name,
        "arguments": block.arguments,
        "arguments_error": block.arguments_error,
    }


def _decode_block(item: dict[str, Any]) -> ContentBlock:
    kind = item["type"]
    if kind == "text":
        return TextContent(item["text"])
    if kind == "thinking":
        return ThinkingContent(item["thinking"], item.get("signature"), item.get("redacted", False))
    if kind == "tool_call":
        return ToolCall(item["id"], item["name"], item["arguments"], item.get("arguments_error"))
    raise ValueError(f"未知的内容块类型：{kind}")


def _encode_usage(usage: Usage) -> dict[str, int]:
    return {
        "input": usage.input,
        "output": usage.output,
        "cache_read": usage.cache_read,
        "cache_write": usage.cache_write,
        "reasoning": usage.reasoning,
        "total_tokens": usage.total_tokens,
    }


def _encode_message(message: AgentMessage) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": message.role,
        "timestamp": message.timestamp,
    }
    if isinstance(message, SystemMessage):
        base["content"] = message.content
    elif isinstance(message, UserMessage):
        base["content"] = [_encode_block(block) for block in message.content]
    elif isinstance(message, AssistantMessage):
        base["content"] = [_encode_block(block) for block in message.content]
        base["stop_reason"] = message.stop_reason
        base["usage"] = _encode_usage(message.usage)
        base["model"] = message.model
        base["error_message"] = message.error_message
    elif isinstance(message, ToolResultMessage):
        base["tool_call_id"] = message.tool_call_id
        base["tool_name"] = message.tool_name
        base["content"] = [_encode_block(block) for block in message.content]
        base["details"] = message.details
        base["is_error"] = message.is_error
    elif isinstance(message, CustomMessage):
        base["kind"] = message.kind
        base["data"] = message.data
    else:
        raise TypeError(f"不是消息：{message!r}")
    return base


def _decode_message(item: dict[str, Any]) -> AgentMessage:
    version = item.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"不支持的消息格式版本：{version}")
    role = item["role"]
    timestamp = item["timestamp"]
    if role == "system":
        return SystemMessage(item["content"], timestamp)
    if role == "user":
        return UserMessage(tuple(_decode_block(b) for b in item["content"]), timestamp)
    if role == "assistant":
        return AssistantMessage(
            content=tuple(_decode_block(b) for b in item["content"]),
            stop_reason=item["stop_reason"],
            usage=Usage(**item["usage"]),
            model=item["model"],
            error_message=item["error_message"],
            timestamp=timestamp,
        )
    if role == "tool_result":
        return ToolResultMessage(
            tool_call_id=item["tool_call_id"],
            tool_name=item["tool_name"],
            content=tuple(_decode_block(b) for b in item["content"]),
            details=item["details"],
            is_error=item["is_error"],
            timestamp=timestamp,
        )
    if role == "custom":
        return CustomMessage(item["kind"], item["data"], timestamp)
    raise ValueError(f"未知的消息角色：{role}")
