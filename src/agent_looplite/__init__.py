"""Pi `agent` 核心的轻量 Python 实现。

请求模型，执行模型要求的工具，把结果交回模型，直到模型给出回答。
"""

from typing import Any

from .agent import Agent, AgentBusyError, AgentState
from .chat_completions import ChatCompletionsClient
from .events import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    PartialMessage,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    TurnEnd,
    TurnStart,
)
from .loop import (
    UNSET,
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    EventStream,
    RunLimits,
    RunResult,
    TurnContext,
    agent_loop,
    agent_loop_continue,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
    decode_messages,
    encode_messages,
)
from .model import (
    AssistantMessageEvent,
    ModelClient,
    ModelContext,
    StreamDelta,
    StreamDone,
    StreamError,
    ToolDeclaration,
)
from .tools import AgentTool, AgentToolResult, ToolUpdate

__all__ = [
    "UNSET",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "Agent",
    "AgentBusyError",
    "AgentContext",
    "AgentEnd",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "AgentStart",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AssistantMessage",
    "AssistantMessageEvent",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "ChatCompletionsClient",
    "CustomMessage",
    "EventStream",
    "Message",
    "MessageEnd",
    "MessageStart",
    "MessageUpdate",
    "ModelClient",
    "ModelContext",
    "PartialMessage",
    "RunLimits",
    "RunResult",
    "StreamDelta",
    "StreamDone",
    "StreamError",
    "SystemMessage",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolDeclaration",
    "ToolExecutionEnd",
    "ToolExecutionStart",
    "ToolExecutionUpdate",
    "ToolResultMessage",
    "ToolUpdate",
    "TurnContext",
    "TurnEnd",
    "TurnStart",
    "Usage",
    "UserMessage",
    "agent_loop",
    "agent_loop_continue",
    "decode_messages",
    "encode_messages",
    "tool",
]


def __getattr__(name: str) -> Any:
    # `@tool` 依赖可选的 pydantic，用到时才导入，没装 pydantic 也能用核心。
    if name == "tool":
        from .function_tool import tool

        return tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
