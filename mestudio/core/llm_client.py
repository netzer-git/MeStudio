"""OpenAI SDK wrapper for LM Studio."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from mestudio.core.config import Settings, get_settings
from mestudio.core.models import (
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
    FunctionCall,
)
from mestudio.utils.logging import (
    log_llm_request,
    log_llm_response,
    log_llm_retry,
)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""


class LLMConnectionError(LLMClientError):
    """Connection to LM Studio failed."""


class LLMTimeoutError(LLMClientError):
    """Request to LM Studio timed out."""


class LMStudioClient:
    """Async client for LM Studio's OpenAI-compatible API.
    
    Wraps openai.AsyncOpenAI with retry logic, health checks,
    and token usage tracking.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client.
        
        Args:
            settings: Application settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._client = AsyncOpenAI(
            base_url=self.settings.lm_studio_url,
            api_key=self.settings.lm_studio_api_key,
            timeout=60.0,
        )
        
        # Cumulative token usage for the session
        self._session_usage = TokenUsage()
        self._call_count = 0

    @property
    def session_usage(self) -> TokenUsage:
        """Get cumulative token usage for this session."""
        return self._session_usage

    @property
    def call_count(self) -> int:
        """Get the number of LLM calls made this session."""
        return self._call_count

    def reset_session_stats(self) -> None:
        """Reset session statistics."""
        self._session_usage = TokenUsage()
        self._call_count = 0

    async def is_available(self) -> bool:
        """Check if LM Studio is running and responsive.
        
        Returns:
            True if LM Studio is available, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.settings.lm_studio_url}/models",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"LM Studio health check failed: {e}")
            return False

    async def get_models(self) -> list[str]:
        """Get list of available models from LM Studio.
        
        Returns:
            List of model IDs.
        """
        try:
            models = await self._client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> AsyncGenerator[StreamChunk, None] | LLMResponse:
        """Send a chat completion request.
        
        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tool definitions (OpenAI format).
            stream: Whether to stream the response.
            response_format: Optional response format (e.g., for JSON mode).
            max_retries: Maximum number of retries on connection/timeout errors.
        
        Returns:
            If stream=True: AsyncGenerator yielding StreamChunks.
            If stream=False: LLMResponse with the complete response.
        
        Raises:
            LLMConnectionError: If connection fails after all retries.
            LLMTimeoutError: If request times out after all retries.
        """
        if stream:
            return self._stream_chat(messages, tools, response_format, max_retries)
        else:
            return await self._complete_chat(messages, tools, response_format, max_retries)

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        max_retries: int,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion response."""
        openai_messages = [m.to_openai_dict() for m in messages]
        
        kwargs: dict[str, Any] = {
            "model": self.settings.lm_studio_model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.settings.response_budget,
        }
        
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        # Log the request
        log_llm_request(
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
        )

        last_error: Exception | None = None
        start_time = time.perf_counter()
        
        for attempt in range(max_retries):
            try:
                stream = await self._client.chat.completions.create(**kwargs)
                
                self._call_count += 1
                final_usage: TokenUsage | None = None
                tool_call_count = 0
                
                async for chunk in stream:
                    stream_chunk = self._parse_stream_chunk(chunk)
                    
                    # Track usage from the final chunk
                    if stream_chunk.usage:
                        self._session_usage = self._session_usage + stream_chunk.usage
                        final_usage = stream_chunk.usage
                    
                    # Count tool calls
                    if stream_chunk.tool_calls_delta:
                        for tc in stream_chunk.tool_calls_delta:
                            if tc.get("id"):  # New tool call
                                tool_call_count += 1
                    
                    yield stream_chunk
                
                # Log successful response
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_llm_response(
                    duration_ms=duration_ms,
                    prompt_tokens=final_usage.prompt_tokens if final_usage else 0,
                    completion_tokens=final_usage.completion_tokens if final_usage else 0,
                    tool_calls=tool_call_count,
                )
                
                return  # Success, exit retry loop
                
            except APIConnectionError as e:
                last_error = e
                backoff = 2 ** attempt
                log_llm_retry(attempt + 1, max_retries, str(e), backoff)
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            except APITimeoutError as e:
                last_error = e
                backoff = 2 ** attempt
                log_llm_retry(attempt + 1, max_retries, str(e), backoff)
                logger.warning(f"Timeout error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                # Don't retry on other errors
                logger.error(f"Unexpected error in chat stream: {e}")
                raise LLMClientError(str(e)) from e
            
            # Exponential backoff between retries
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if isinstance(last_error, APIConnectionError):
            raise LLMConnectionError(
                f"Failed to connect to LM Studio after {max_retries} attempts"
            ) from last_error
        elif isinstance(last_error, APITimeoutError):
            raise LLMTimeoutError(
                f"Request timed out after {max_retries} attempts"
            ) from last_error
        else:
            raise LLMClientError("Unknown error") from last_error

    async def _complete_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        max_retries: int,
    ) -> LLMResponse:
        """Send a non-streaming chat completion request."""
        openai_messages = [m.to_openai_dict() for m in messages]
        
        kwargs: dict[str, Any] = {
            "model": self.settings.lm_studio_model,
            "messages": openai_messages,
            "stream": False,
            "max_tokens": self.settings.response_budget,
        }
        
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        # Log the request
        log_llm_request(
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
        )

        last_error: Exception | None = None
        start_time = time.perf_counter()
        
        for attempt in range(max_retries):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                
                self._call_count += 1
                
                # Parse response
                choice = response.choices[0]
                message = choice.message
                
                # Extract tool calls if present
                tool_calls: list[ToolCall] | None = None
                tool_call_count = 0
                if message.tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=tc.id,
                            type="function",
                            function=FunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                        for tc in message.tool_calls
                    ]
                    tool_call_count = len(tool_calls)
                
                # Track usage
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                )
                self._session_usage = self._session_usage + usage
                
                # Log successful response
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                log_llm_response(
                    duration_ms=duration_ms,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    tool_calls=tool_call_count,
                )
                
                return LLMResponse(
                    content=message.content,
                    tool_calls=tool_calls,
                    usage=usage,
                    finish_reason=choice.finish_reason,
                    model=response.model,
                )
                
            except APIConnectionError as e:
                last_error = e
                backoff = 2 ** attempt
                log_llm_retry(attempt + 1, max_retries, str(e), backoff)
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            except APITimeoutError as e:
                last_error = e
                backoff = 2 ** attempt
                log_llm_retry(attempt + 1, max_retries, str(e), backoff)
                logger.warning(f"Timeout error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error in chat: {e}")
                raise LLMClientError(str(e)) from e
            
            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if isinstance(last_error, APIConnectionError):
            raise LLMConnectionError(
                f"Failed to connect to LM Studio after {max_retries} attempts"
            ) from last_error
        elif isinstance(last_error, APITimeoutError):
            raise LLMTimeoutError(
                f"Request timed out after {max_retries} attempts"
            ) from last_error
        else:
            raise LLMClientError("Unknown error") from last_error

    async def structured_output(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Get a structured JSON response matching a schema.
        
        Args:
            messages: List of messages in the conversation.
            schema: JSON schema for the expected response.
            max_retries: Maximum number of retries.
        
        Returns:
            Parsed JSON response matching the schema.
        
        Raises:
            LLMClientError: If the response cannot be parsed as JSON.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": schema,
        }
        
        response = await self._complete_chat(
            messages=messages,
            tools=None,
            response_format=response_format,
            max_retries=max_retries,
        )
        
        if not response.content:
            raise LLMClientError("Empty response from structured output request")
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {response.content}")
            raise LLMClientError(f"Invalid JSON response: {e}") from e

    def _parse_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Parse an OpenAI stream chunk into our StreamChunk model."""
        content: str | None = None
        tool_calls_delta: list[dict[str, Any]] | None = None
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            
            if delta.content:
                content = delta.content
            
            if delta.tool_calls:
                tool_calls_delta = [
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": getattr(tc.function, "name", None),
                            "arguments": getattr(tc.function, "arguments", None),
                        } if tc.function else None,
                    }
                    for tc in delta.tool_calls
                ]
            
            finish_reason = choice.finish_reason
        
        if chunk.usage:
            usage = TokenUsage(
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
                total_tokens=chunk.usage.total_tokens,
            )
        
        return StreamChunk(
            content=content,
            tool_calls_delta=tool_calls_delta,
            finish_reason=finish_reason,
            usage=usage,
        )
