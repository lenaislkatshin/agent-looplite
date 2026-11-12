from __future__ import annotations

import json
from typing import Any

import httpx

from agent_looplite import (
    AssistantMessage,
    ChatCompletionsClient,
    ModelContext,
    StreamDelta,
    StreamDone,
    StreamError,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDeclaration,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from agent_looplite.chat_completions import parse_arguments

CONTEXT = ModelContext(
    messages=(SystemMessage("系统"), UserMessage.of("问")),
    tools=(ToolDeclaration("search", "检索", {"type": "object"}),),
)


def sse(*chunks: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def client(handler: Any, **kwargs: Any) -> ChatCompletionsClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatCompletionsClient(
        base_url="https://gw.example/v1",
        model="m",
        api_key="k",
        http_client=http,
        retry_base_delay=0,
        **kwargs,
    )


async def collect(model: ChatCompletionsClient, context: ModelContext = CONTEXT) -> list[Any]:
    return [event async for event in model.stream(context)]


async def test_stream_assembles_thinking_text_and_split_tool_call() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = sse(
            {"choices": [{"delta": {"reasoning": "想一想"}}]},
            {"choices": [{"delta": {"content": "先查"}}]},
            # 工具 ID 和名字不在第一片里。
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]}}
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "search", "arguments": ': "港口"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    events = await collect(client(handler, reasoning_field="reasoning"))
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert requests[0]["tools"][0]["function"]["name"] == "search"
    kinds = [e.kind for e in events if isinstance(e, StreamDelta)]
    assert kinds == [
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
    ]
    done = events[-1]
    assert isinstance(done, StreamDone)
    message = done.message
    assert message.stop_reason == "tool_use"
    assert message.content[0] == ThinkingContent("想一想")
    assert message.content[1] == TextContent("先查")
    assert message.content[2] == ToolCall("call_1", "search", {"q": "港口"})
    assert message.usage == Usage(input=6, output=5, cache_read=4, total_tokens=15)


async def test_non_stream_matches_stream_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "答案",
                            "reasoning_content": "思考",
                            "tool_calls": [
                                {"id": "c", "function": {"name": "search", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    events = await collect(client(handler, stream=False))
    assert len(events) == 1 and isinstance(events[0], StreamDone)
    assert events[0].message.content == (
        ThinkingContent("思考"),
        TextContent("答案"),
        ToolCall("c", "search", {}),
    )


async def test_gateway_error_inside_http_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 401, "msg": "鉴权失败"})

    def check(body: dict[str, Any]) -> str | None:
        return f"网关鉴权失败：{body['msg']}" if body.get("code", 0) != 0 else None

    events = await collect(client(handler, check_body=check))
    assert isinstance(events[-1], StreamError)
    assert events[-1].message.error_message == "网关鉴权失败：鉴权失败"

    events = await collect(client(handler))
    assert "Response has no choices" in events[-1].message.error_message


async def test_retries_before_any_content_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(
            200,
            content=sse({"choices": [{"delta": {"content": "好"}, "finish_reason": "stop"}]}),
            headers={"content-type": "text/event-stream"},
        )

    events = await collect(client(handler))
    assert attempts == 3
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.text == "好"


async def test_gives_up_after_max_retries_and_does_not_retry_4xx() -> None:
    attempts = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="busy")

    events = await collect(client(unavailable, max_retries=2))
    assert attempts == 3
    assert events[-1].message.error_message == "HTTP 503: busy"

    attempts = 0

    def bad_request(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad")

    events = await collect(client(bad_request))
    assert attempts == 1
    assert isinstance(events[-1], StreamError)


async def test_on_payload_rewrites_and_rejects() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]}
        )

    async def mask(body: dict[str, Any]) -> dict[str, Any]:
        body["messages"][-1]["content"] = "已脱敏"
        return body

    await collect(client(handler, stream=False, on_payload=mask))
    assert seen[-1]["messages"][-1]["content"] == "已脱敏"

    async def reject(body: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("含身份证号")

    events = await collect(client(handler, stream=False, on_payload=reject))
    assert len(seen) == 1
    assert "含身份证号" in events[-1].message.error_message


async def test_history_conversion() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]}
        )

    history = (
        UserMessage.of("问"),
        AssistantMessage((ThinkingContent("想"), ToolCall("c1", "search", {"q": "x"})), "tool_use"),
        ToolResultMessage("c1", "search", (TextContent("结果"),)),
        AssistantMessage((TextContent("断了"),), "error", error_message="HTTP 500"),
        UserMessage.of("再问"),
    )
    await collect(
        client(handler, stream=False, replay_reasoning=True, headers={"user": "u1"}),
        ModelContext(history, ()),
    )
    messages = sent[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[1]["reasoning_content"] == "想"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    assert "tools" not in sent[0]


async def test_headers_and_settings() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatCompletionsClient(
        base_url="https://gw.example/v1/",
        model="m",
        headers={"Authorization": "raw-key", "user": "u1"},
        stream=False,
        settings={"temperature": 0.1},
        http_client=http,
    )
    await collect(model)
    request = seen[0]
    assert str(request.url) == "https://gw.example/v1/chat/completions"
    assert request.headers["authorization"] == "raw-key"
    assert request.headers["user"] == "u1"
    assert json.loads(request.content)["temperature"] == 0.1


async def test_connection_drop_mid_stream_keeps_partial_text() -> None:
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield 'data: {"choices": [{"delta": {"content": "前半"}}]}\n\n'.encode()
            raise httpx.ReadError("连接断开")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Broken(), headers={"content-type": "text/event-stream"})

    events = await collect(client(handler))
    error = events[-1]
    assert isinstance(error, StreamError)
    assert error.message.text == "前半"
    assert "ReadError" in error.message.error_message


def test_parse_arguments_rejects_non_json_values() -> None:
    assert parse_arguments('{"a": 1}') == ({"a": 1}, None)
    assert parse_arguments("") == ({}, None)
    for bad in ['{"a": NaN}', '{"a": 1', "[1]"]:
        arguments, error = parse_arguments(bad)
        assert arguments == {} and "Received arguments" in error


async def test_only_explicit_finish_reasons_count_as_success() -> None:
    tool_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "c", "function": {"name": "search", "arguments": "{}"}}
                    ]
                }
            }
        ]
    }

    def respond(*tail: dict[str, Any]) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=sse(tool_chunk, *tail), headers={"content-type": "text/event-stream"}
            )

        return handler

    events = await collect(
        client(respond({"choices": [{"delta": {}, "finish_reason": "network_error"}]}))
    )
    assert isinstance(events[-1], StreamError)
    assert events[-1].message.error_message == "Provider finish_reason: network_error"
    assert events[-1].message.tool_calls == ()

    events = await collect(client(respond()))
    assert isinstance(events[-1], StreamError)
    assert events[-1].message.error_message == "Response ended without finish_reason"

    events = await collect(client(respond(), supports_finish_reason=False))
    assert isinstance(events[-1], StreamDone)
    assert events[-1].message.stop_reason == "tool_use"


async def test_failed_response_never_runs_tools() -> None:
    from agent_looplite import (
        AgentContext,
        AgentLoopConfig,
        AgentTool,
        AgentToolResult,
        agent_loop,
    )

    executed: list[str] = []

    async def execute(call_id: str, params: dict[str, Any], on_update: Any) -> AgentToolResult:
        executed.append(call_id)
        return AgentToolResult.of("好")

    def handler(request: httpx.Request) -> httpx.Response:
        body = sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c",
                                    "function": {"name": "write", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "network_error",
                    }
                ]
            }
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    stream = agent_loop(
        [UserMessage.of("问")],
        AgentContext(tools=[AgentTool("write", "", {"type": "object"}, execute)]),
        AgentLoopConfig(model=client(handler)),
    )
    result = await stream.result()
    assert result.reason == "error"
    assert executed == []
