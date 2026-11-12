"""OpenAI 兼容的 Chat Completions 连接器，同时支持流式和整轮返回。

拼接分片、解析工具参数、重试、识别网关错误、出网前检查都在这里，循环不管。
"""

from __future__ import annotations

import asyncio
import json
import re
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

OnPayload = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]
CheckBody = Callable[[dict[str, Any]], str | None]

RETRYABLE_STATUS = {408, 409, 429}
SUCCESS_FINISH_REASONS = {"stop", "end", "length", "tool_calls", "function_call"}
MAX_ERROR_BODY = 2000
NO_TOOL_OUTPUT = "(no tool output)"
_VALID_ESCAPE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')
_SSE_LINE = re.compile(r"(?:data|event|id|retry)?:")


class _Retryable(Exception):
    """还没收到任何内容时可以重试的失败：连接错误、超时、408、409、429 和 5xx。"""


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
        supports_finish_reason: bool = True,
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
        self.supports_finish_reason = supports_finish_reason
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
        """和 Pi 一样，任何失败都记成一条 `error` 回复，不向循环抛异常。"""
        builder = _ResponseBuilder(self.reasoning_field, self.model, self.supports_finish_reason)
        try:
            body = self.build_body(context)
            if self.on_payload is not None:
                try:
                    checked = await self.on_payload(body)
                except Exception as exc:
                    yield builder.error(f"on_payload rejected the request: {exc}")
                    return
                if checked is not None:
                    body = checked
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
        except Exception as exc:
            yield builder.error(f"{type(exc).__name__}: {exc}")

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
        """返回 `None` 表示这条消息不发。助手回复和工具结果的规则同 Pi。"""
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": text_of(message.content)}
        if isinstance(message, ToolResultMessage):
            text = "\n".join(block.text for block in message.content)
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": text if text.strip() else NO_TOOL_OUTPUT,
            }
        if message.stop_reason in ("error", "aborted"):
            return None
        text = "".join(
            block.text
            for block in message.content
            if isinstance(block, TextContent) and block.text.strip()
        )
        calls = message.tool_calls
        if not text and not calls:
            # 有的网关不接受内容为空的消息。
            return None
        converted: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            converted["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
                for call in calls
            ]
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
                # 请求了流式，网关仍可能回整段 JSON，常见于用 HTTP 200 包着的错误。`content-type`
                # 不可靠，所以看内容：第一行有内容的行不是 SSE 的字段行或注释行，就收集整个响应。
                whole_body: list[str] | None = None
                sniffed = False
                async for line in _lines(response):
                    if not sniffed and line.strip():
                        sniffed = True
                        if not _SSE_LINE.match(line):
                            whole_body = []
                    if whole_body is not None:
                        whole_body.append(line)
                        continue
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
                    if not isinstance(chunk, dict):
                        # 和 Pi 一样跳过不是 JSON 对象的分片。
                        continue
                    problem = self._problem(chunk)
                    if problem is not None:
                        yield builder.error(problem)
                        return
                    for event in builder.feed(chunk):
                        yield event
                if whole_body is not None:
                    async for event in self._from_body("\n".join(whole_body).encode(), builder):
                        yield event
                    return
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


async def _lines(response: httpx.Response) -> AsyncIterator[str]:
    """按 SSE 规定的三种行结束符分行：`\\n`、`\\r\\n` 和单独的 `\\r`。

    不用 `response.aiter_lines()`：它还会在 U+2028、U+0085 这类字符处分行。这些字符可以原样出现
    在 JSON 字符串里，分片会被拆成两行，解析失败。
    """
    pending = ""
    async for text in response.aiter_text():
        pending += text
        # 末尾的 `\\r` 可能是 `\\r\\n` 的前一半，留到下一块再判断。
        held = "\r" if pending.endswith("\r") else ""
        normalized = pending.removesuffix("\r").replace("\r\n", "\n").replace("\r", "\n")
        *lines, pending = normalized.split("\n")
        pending += held
        for line in lines:
            yield line
    if pending.removesuffix("\r"):
        yield pending.removesuffix("\r")


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

    流式分片和整轮返回走同一个出口，保证两种模式的结果一致。分块规则同 Pi 的 Chat Completions
    连接器：一条回复最多一个正文块和一个思考块，工具调用按分片里的 `index` 或 ID 归到各自的块；
    所有块在流结束时按顺序结束。
    """

    def __init__(self, reasoning_field: str, model: str, supports_finish_reason: bool) -> None:
        self.reasoning_field = reasoning_field
        self.supports_finish_reason = supports_finish_reason
        self.model = model
        self.blocks: list[_Block] = []
        self.text_blocks: dict[str, int] = {}
        self.tools_by_position: dict[int, int] = {}
        self.tools_by_id: dict[str, int] = {}
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
        index = self.text_blocks.get(kind)
        if index is None:
            self.blocks.append(_Block(kind))
            index = self.text_blocks[kind] = len(self.blocks) - 1
            events.append(StreamDelta(f"{kind}_start", index))  # type: ignore[arg-type]
        self.blocks[index].parts.append(text)
        events.append(StreamDelta(f"{kind}_delta", index, text))  # type: ignore[arg-type]
        return events

    def _append_tool(self, tool_delta: dict[str, Any]) -> list[StreamDelta]:
        """每个工具调用分片都发一个 `toolcall_delta`，没有参数时增量为空字符串，和 Pi 一致。"""
        events = []
        position = tool_delta.get("index")
        call_id = tool_delta.get("id") or None
        function = tool_delta.get("function") or {}
        index = self.tools_by_position.get(position) if position is not None else None
        if index is None and call_id is not None:
            index = self.tools_by_id.get(call_id)
        if index is None:
            self.blocks.append(_Block("toolcall", id=call_id, name=function.get("name") or None))
            index = len(self.blocks) - 1
            block = self.blocks[index]
            events.append(StreamDelta("toolcall_start", index, "", block.id, block.name))
        block = self.blocks[index]
        if position is not None:
            self.tools_by_position.setdefault(position, index)
        block.id = block.id or call_id
        if block.id is not None:
            self.tools_by_id.setdefault(block.id, index)
        block.name = block.name or function.get("name") or None
        arguments = function.get("arguments") or ""
        block.parts.append(arguments)
        events.append(StreamDelta("toolcall_delta", index, arguments))
        return events

    def finish(self) -> list[AssistantMessageEvent]:
        events: list[AssistantMessageEvent] = [
            StreamDelta(f"{block.kind}_end", index)  # type: ignore[arg-type]
            for index, block in enumerate(self.blocks)
        ]
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
        """只有明确的成功结束原因才算成功，规则同 Pi 的 `mapStopReason`。

        `network_error` 这类原因、或者响应没有结束原因，都说明响应可能不完整。这时已经拼出的工具
        参数可能看起来合法，执行它们可能产生实际副作用，所以一律当作失败。
        """
        reason = self.finish_reason
        if reason is None and self.supports_finish_reason:
            return self.error("Response ended without finish_reason")
        if reason is not None and reason not in SUCCESS_FINISH_REASONS:
            return self.error(f"Provider finish_reason: {reason}")
        content = self._content(include_tool_calls=True)
        has_calls = any(isinstance(block, ToolCall) for block in content)
        stop_reason: StopReason
        if reason == "length":
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
    """解析完整的参数文本，规则同 Pi 的 `parseToolCallArguments`。

    解析失败时，先修复字符串里未转义的控制字符和不合法的反斜杠转义，再解析一次。模型在参数里
    写多行文本或正则时常出这两类错。`NaN`、`Infinity` 不是合法 JSON，回传后再发给模型会出错，
    所以也拒绝。
    """
    if not text.strip():
        return {}, None
    try:
        value = _loads_with_repair(text)
    except (ValueError, RecursionError) as exc:
        return {}, f"{exc}\n\nReceived arguments:\n{text}"
    if not isinstance(value, dict):
        return {}, f"Tool arguments must be a JSON object\n\nReceived arguments:\n{text}"
    return value, None


def _loads_with_repair(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        repaired = _repair_string_literals(text)
        if repaired == text:
            raise
        return json.loads(repaired, parse_constant=_reject_constant)


def _repair_string_literals(text: str) -> str:
    """把字符串里未转义的控制字符写成转义，把不合法转义前的反斜杠写成两个。字符串之外不动。"""
    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            repaired.append(char)
            in_string = char == '"'
        elif char == '"':
            repaired.append(char)
            in_string = False
        elif char == "\\":
            escape = _VALID_ESCAPE.match(text, index)
            if escape is None:
                repaired.append("\\\\")
            else:
                repaired.append(escape.group())
                index = escape.end() - 1
        elif char < " ":
            repaired.append(json.dumps(char)[1:-1])
        else:
            repaired.append(char)
        index += 1
    return "".join(repaired)


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
        input=max(0, prompt - cached),
        output=completion,
        cache_read=cached,
        reasoning=reasoning,
        total_tokens=raw.get("total_tokens") or prompt + completion,
    )
