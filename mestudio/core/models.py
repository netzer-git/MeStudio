"""Pydantic models for messages, tool calls, etc."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """A function call within a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Optional name for the message sender

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API message format."""
        d: dict[str, Any] = {"role": self.role}
        
        if self.content is not None:
            d["content"] = self.content
        
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        
        if self.name is not None:
            d["name"] = self.name
        
        return d

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> Message:
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class ToolResult(BaseModel):
    """Result of executing a tool."""

    tool_call_id: str
    content: str
    success: bool = True
    error: str | None = None


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two TokenUsage instances."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """Response from an LLM call."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    model: str | None = None


class StreamChunk(BaseModel):
    """A chunk from a streaming LLM response."""

    content: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class SessionMetadata(BaseModel):
    """Metadata for a saved session."""

    session_id: str
    label: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    total_tokens_used: int = 0
    compaction_count: int = 0
    tools_called: list[str] = Field(default_factory=list)
