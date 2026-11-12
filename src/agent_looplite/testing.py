"""测试用的模型：按预先写好的回复依次返回，不调真实模型。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from .messages import AssistantMessage, TextContent, ThinkingContent, ToolCall, Usage
from .model import AssistantMessageEvent, ModelContext, StreamDelta, StreamDone, StreamError

Response = AssistantMessage | Callable[[ModelContext], AssistantMessage]


def text_reply(
    text: str, *, thinking: str | None = None, usage: Usage = Usage()
) -> AssistantMessage:
    content: list[Any] = [ThinkingContent(thinking)] if thinking else []
    content.append(TextContent(text))
    return AssistantMessage(tuple(content), "stop", usage, "scripted")


def call(name: str, arguments: dict[str, Any] | None = None, *, id: str | None = None) -> ToolCall:
    return ToolCall(id or f"call_{uuid.uuid4().hex[:8]}", name, arguments or {})


def tool_reply(*calls: ToolCall, text: str = "", stop_reason: str = "tool_use") -> AssistantMessage:
    content: list[Any] = [TextContent(text)] if text else []
    content.extend(calls)
    return AssistantMessage(tuple(content), stop_reason, Usage(), "scripted")  # type: ignore[arg-type]


def error_reply(message: str, *, text: str = "") -> AssistantMessage:
    content = (TextContent(text),) if text else ()
    return AssistantMessage(content, "error", Usage(), "scripted", error_message=message)


class ScriptedModel:
    """每次请求取下一条回复。`stream=True` 时把正文按 `chunk_size` 切成多段增量发出。

    `contexts` 记录每次收到的请求，方便断言模型看到了什么。`delay` 是每段增量之前的等待秒数，
    用来测试流式过程中取消。
    """

    def __init__(
        self,
        responses: Sequence[Response],
        *,
        stream: bool = True,
        chunk_size: int = 0,
        delay: float = 0.0,
    ) -> None:
        self._responses = list(responses)
        self.streaming = stream
        self.chunk_size = chunk_size
        self.delay = delay
        self.contexts: list[ModelContext] = []

    def stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]:
        return self._stream(context)

    async def _stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]:
        self.contexts.append(context)
        if not self._responses:
            yield StreamError(error_reply("ScriptedModel has no more responses"))
            return
        response = self._responses.pop(0)
        message = response(context) if callable(response) else response
        if message.stop_reason in ("error", "aborted"):
            yield StreamError(message)
            return
        if self.streaming:
            for index, block in enumerate(message.content):
                async for event in self._block_events(index, block):
                    yield event
        await asyncio.sleep(self.delay)
        yield StreamDone(message)

    async def _block_events(self, index: int, block: Any) -> AsyncIterator[StreamDelta]:
        if isinstance(block, ToolCall):
            yield StreamDelta("toolcall_start", index, "", block.id, block.name)
            for piece in self._pieces(json.dumps(block.arguments, ensure_ascii=False)):
                await asyncio.sleep(self.delay)
                yield StreamDelta("toolcall_delta", index, piece)
            yield StreamDelta("toolcall_end", index)
            return
        kind = "thinking" if isinstance(block, ThinkingContent) else "text"
        text = block.thinking if isinstance(block, ThinkingContent) else block.text
        yield StreamDelta(f"{kind}_start", index)  # type: ignore[arg-type]
        for piece in self._pieces(text):
            await asyncio.sleep(self.delay)
            yield StreamDelta(f"{kind}_delta", index, piece)  # type: ignore[arg-type]
        yield StreamDelta(f"{kind}_end", index)  # type: ignore[arg-type]

    def _pieces(self, text: str) -> list[str]:
        if not self.chunk_size:
            return [text]
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
