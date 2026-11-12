"""OpenAI 兼容的 Chat Completions 连接器，同时支持流式和整轮返回。

拼接分片、解析工具参数、重试、识别网关错误、出网前检查都在这里，循环不管。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .messages import (
    AssistantMessage,
    ContentBlock,
    Message,
    StopReason,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    text_of,
)
from .model import AssistantMessageEvent, ModelContext, StreamDelta, StreamDone, StreamError

OnPayload = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
CheckBody = Callable[[dict[str, Any]], str | None]

RETRYABLE_STATUS = {408, 409, 429}
MAX_ERROR_BODY = 2000


class _Retryable(Exception):
    """还没收到任何内容时可以重试的失败：连接错误、超时、429 和 5xx。"""


class _Rejected(Exception):
    """不重试的 HTTP 错误。"""


class ChatCompletionsClient:
    """`base_url` 写到版本号为止，例如 `https://api.openai.com/v1`。

    请求发到 `{base_url}/chat/completions`。

    要求密钥不带 `Bearer` 的网关，`api_key` 留空，把密钥写进 `headers`。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        stream: bool = True,
        stream_usage: bool = True,
        reasoning_field: str = "reasoning_content",
        replay_reasoning: bool = False,
        on_payload: OnPayload | None = None,
        check_body: CheckBody | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        timeout: float = 120.0,
        settings: dict[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            **(headers or {}),
        }
        self.streaming = stream
        self.stream_usage = stream_usage
        self.reasoning_field = reasoning_field
        self.replay_reasoning = replay_reasoning
        self.on_payload = on_payload
        self.check_body = check_body
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout
        self.settings = settings or {}
        self._http = http_client
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> ChatCompletionsClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]:
        return self._stream(context)

    async def _stream(self, context: ModelContext) -> AsyncIterator[AssistantMessageEvent]:
        builder = _ResponseBuilder(self.reasoning_field, self.model)
        body = self.build_body(context)
        if self.on_payload is not None:
            try:
                body = await self.on_payload(body)
            except Exception as exc:
                yield builder.error(f"on_payload rejected the request: {exc}")
                return
        attempt = 0
        while True:
            try:
                async for event in self._attempt(body, builder):
                    yield event
                return
            except _Rejected as exc:
                yield builder.error(str(exc))
                return
            except _Retryable as exc:
                if builder.emitted or attempt >= self.max_retries:
                    yield builder.error(str(exc))
                    return
                await asyncio.sleep(self.retry_base_delay * 2**attempt)
                attempt += 1

    def build_body(self, context: ModelContext) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m for m in (self._convert(m) for m in context.messages) if m is not None],
        }
        if context.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in context.tools
            ]
        body["stream"] = self.streaming
        if self.streaming and self.stream_usage:
            body["stream_options"] = {"include_usage": True}
        body.update(self.settings)
        return body

    def _convert(self, message: Message) -> dict[str, Any] | None:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": text_of(message.content)}
        if isinstance(message, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": text_of(message.content),
            }
        if message.stop_reason in ("error", "aborted"):
            return None
        converted: dict[str, Any] = {"role": "assistant", "content": message.text or None}
        calls = message.tool_calls
        if calls:
            converted["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ]
        elif converted["content"] is None:
            converted["content"] = ""
        if self.replay_reasoning:
            thinking = "".join(
                b.thinking for b in message.content if isinstance(b, ThinkingContent)
            )
            if thinking:
                converted[self.reasoning_field] = thinking
        return converted

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._http

    async def _attempt(
        self, body: dict[str, Any], builder: _ResponseBuilder
    ) -> AsyncIterator[AssistantMessageEvent]:
        client = self._client()
        try:
            if not self.streaming:
                response = await client.post(self.url, json=body, headers=self.headers)
                await _raise_for_status(response)
                async for event in self._from_body(response.content, builder):
                    yield event
                return
            async with client.stream("POST", self.url, json=body, headers=self.headers) as response:
                await _raise_for_status(response)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    async for event in self._from_body(await response.aread(), builder):
                        yield event
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        yield builder.error(f"Malformed stream chunk: {payload[:MAX_ERROR_BODY]}")
                        return
                    problem = self._problem(chunk)
                    if problem is not None:
                        yield builder.error(problem)
                        return
                    for event in builder.feed(chunk):
                        yield event
                for event in builder.finish():
                    yield event
        except httpx.TransportError as exc:
            raise _Retryable(f"{type(exc).__name__}: {exc}") from exc

    async def _from_body(
        self, raw: bytes, builder: _ResponseBuilder
    ) -> AsyncIterator[AssistantMessageEvent]:
        try:
            data = json.loads(raw)
        except ValueError:
            yield builder.error(
                f"Response is not JSON: {raw[:MAX_ERROR_BODY].decode(errors='replace')}"
            )
            return
        problem = self._problem(data) if isinstance(data, dict) else "Response is not a JSON object"
        if problem is None and not data.get("choices"):
            problem = (
                f"Response has no choices: {json.dumps(data, ensure_ascii=False)[:MAX_ERROR_BODY]}"
            )
        if problem is not None:
            yield builder.error(problem)
            return
        for event in builder.from_full(data):
            yield event

    def _problem(self, data: dict[str, Any]) -> str | None:
        """网关常用 HTTP 200 包着错误。先交给 `check_body`，再识别通用的 `error` 字段。"""
        if self.check_body is not None:
            problem = self.check_body(data)
            if problem is not None:
                return problem
        if data.get("error") and not data.get("choices"):
            return (
                f"Provider error: {json.dumps(data['error'], ensure_ascii=False)[:MAX_ERROR_BODY]}"
            )
        return None


async def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    text = (await response.aread()).decode(errors="replace")[:MAX_ERROR_BODY]
    message = f"HTTP {response.status_code}: {text}"
    if response.status_code in RETRYABLE_STATUS or response.status_code >= 500:
        raise _Retryable(message)
    raise _Rejected(message)


@dataclass
class _Block:
    kind: str
    parts: list[str] = field(default_factory=list)
    id: str | None = None
    name: str | None = None


class _ResponseBuilder:
    """把一次响应拼成 `AssistantMessage`。

    流式分片和整轮返回走同一个出口，保证两种模式的结果一致。
    """

    def __init__(self, reasoning_field: str, model: str) -> None:
        self.reasoning_field = reasoning_field
        self.model = model
        self.blocks: list[_Block] = []
        self.open_index: int | None = None
        self.tool_blocks: dict[int, int] = {}
        self.usage = Usage()
        self.finish_reason: str | None = None
        self.emitted = False

    def feed(self, chunk: dict[str, Any]) -> list[StreamDelta]:
        events: list[StreamDelta] = []
        if chunk.get("usage"):
            self.usage = _parse_usage(chunk["usage"])
        if chunk.get("model"):
            self.model = chunk["model"]
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            reasoning = delta.get(self.reasoning_field)
            if reasoning:
                events.extend(self._append_text("thinking", reasoning))
            if delta.get("content"):
                events.extend(self._append_text("text", delta["content"]))
            for tool_delta in delta.get("tool_calls") or ():
                events.extend(self._append_tool(tool_delta))
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if events:
            self.emitted = True
        return events

    def _append_text(self, kind: str, text: str) -> list[StreamDelta]:
        events = []
        if self.open_index is None or self.blocks[self.open_index].kind != kind:
            events.extend(self._close_open())
            self.blocks.append(_Block(kind))
            self.open_index = len(self.blocks) - 1
            events.append(StreamDelta(f"{kind}_start", self.open_index))  # type: ignore[arg-type]
        self.blocks[self.open_index].parts.append(text)
        events.append(StreamDelta(f"{kind}_delta", self.open_index, text))  # type: ignore[arg-type]
        return events

    def _append_tool(self, tool_delta: dict[str, Any]) -> list[StreamDelta]:
        events = []
        position = tool_delta.get("index", 0)
        function = tool_delta.get("function") or {}
        if position not in self.tool_blocks:
            events.extend(self._close_open())
            self.blocks.append(_Block("toolcall"))
            self.tool_blocks[position] = len(self.blocks) - 1
            index = self.tool_blocks[position]
            block = self.blocks[index]
            block.id = tool_delta.get("id") or None
            block.name = function.get("name") or None
            events.append(StreamDelta("toolcall_start", index, "", block.id, block.name))
        index = self.tool_blocks[position]
        block = self.blocks[index]
        block.id = block.id or tool_delta.get("id") or None
        block.name = block.name or function.get("name") or None
        arguments = function.get("arguments")
        if arguments:
            block.parts.append(arguments)
            events.append(StreamDelta("toolcall_delta", index, arguments))
        return events

    def _close_open(self) -> list[StreamDelta]:
        if self.open_index is None:
            return []
        index, self.open_index = self.open_index, None
        return [StreamDelta(f"{self.blocks[index].kind}_end", index)]  # type: ignore[arg-type]

    def finish(self) -> list[AssistantMessageEvent]:
        events: list[AssistantMessageEvent] = list(self._close_open())
        events.extend(StreamDelta("toolcall_end", index) for index in self.tool_blocks.values())
        events.append(self._result())
        return events

    def from_full(self, data: dict[str, Any]) -> list[AssistantMessageEvent]:
        if data.get("usage"):
            self.usage = _parse_usage(data["usage"])
        self.model = data.get("model") or self.model
        choice = data["choices"][0]
        message = choice.get("message") or {}
        if message.get(self.reasoning_field):
            self.blocks.append(_Block("thinking", [message[self.reasoning_field]]))
        if message.get("content"):
            self.blocks.append(_Block("text", [message["content"]]))
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments) if arguments is not None else ""
            self.blocks.append(
                _Block("toolcall", [arguments], call.get("id"), function.get("name"))
            )
        self.finish_reason = choice.get("finish_reason")
        return [self._result()]

    def _content(self, include_tool_calls: bool) -> tuple[ContentBlock, ...]:
        content: list[ContentBlock] = []
        for block in self.blocks:
            text = "".join(block.parts)
            if block.kind == "text":
                content.append(TextContent(text))
            elif block.kind == "thinking":
                content.append(ThinkingContent(text))
            elif include_tool_calls:
                arguments, error = parse_arguments(text)
                content.append(
                    ToolCall(
                        id=block.id or f"call_{uuid.uuid4().hex[:24]}",
                        name=block.name or "",
                        arguments=arguments,
                        arguments_error=error,
                    )
                )
        return tuple(content)

    def _result(self) -> AssistantMessageEvent:
        if self.finish_reason == "content_filter":
            return self.error("Response blocked by the provider's content filter")
        content = self._content(include_tool_calls=True)
        has_calls = any(isinstance(block, ToolCall) for block in content)
        stop_reason: StopReason
        if self.finish_reason == "length":
            stop_reason = "length"
        elif has_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "stop"
        return StreamDone(AssistantMessage(content, stop_reason, self.usage, self.model))

    def error(self, message: str) -> StreamError:
        return StreamError(
            AssistantMessage(
                content=self._content(include_tool_calls=False),
                stop_reason="error",
                usage=self.usage,
                model=self.model,
                error_message=message,
            )
        )


def parse_arguments(text: str) -> tuple[dict[str, Any], str | None]:
    """严格解析完整的参数文本。

    `NaN`、`Infinity` 不是合法 JSON，回传后再发给模型会出错，所以也拒绝。
    """
    if not text.strip():
        return {}, None
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        return {}, f"{exc}\n\nReceived arguments:\n{text}"
    if not isinstance(value, dict):
        return {}, f"Tool arguments must be a JSON object\n\nReceived arguments:\n{text}"
    return value, None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _parse_usage(raw: dict[str, Any]) -> Usage:
    prompt = raw.get("prompt_tokens") or 0
    completion = raw.get("completion_tokens") or 0
    cached = (
        (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        or raw.get("prompt_cache_hit_tokens")
        or 0
    )
    reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return Usage(
        input=prompt - cached,
        output=completion,
        cache_read=cached,
        reasoning=reasoning,
        total_tokens=raw.get("total_tokens") or prompt + completion,
    )
