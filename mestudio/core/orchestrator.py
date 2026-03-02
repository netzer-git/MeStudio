"""Main orchestrator agent loop."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from mestudio.agents import SubAgentSpawner
from mestudio.context.budget import CompactionLevel
from mestudio.context.manager import ContextManager
from mestudio.core.config import Settings, get_settings
from mestudio.core.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LMStudioClient,
)
from mestudio.core.models import (
    Message,
    StreamChunk,
    ToolCall,
    FunctionCall,
    TokenUsage,
)
from mestudio.planner import PlanTracker
from mestudio.tools.agent_tools import register_agent_handler
from mestudio.tools.registry import ToolRegistry
from mestudio.utils.logging import (
    transcript_user,
    transcript_assistant,
    transcript_tool_call,
    transcript_tool_result,
    transcript_system,
)


# System prompt for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """You are MeStudio Agent, a local AI assistant with tool-calling capabilities.

You can:
- Read, write, search, and edit files anywhere on the user's PC
- Search the web for information
- Create and track multi-step plans for complex tasks
- Delegate focused tasks to sub-agents via delegate_task

File Access:
- You have UNRESTRICTED file access on this PC
- Use get_environment_info ONCE at the start to discover user's home folder
- Common user folders: Documents, Desktop, Downloads, Dropbox, OneDrive
- Use SPECIFIC searches - don't scan the entire home with broad patterns

Search Strategy (IMPORTANT):
1. If user mentions a specific folder (like "Dropbox"), search THERE directly
2. Use targeted patterns: "**/RPG*" is better than "**/*"
3. AFTER finding a folder, use list_directory to see what's inside
4. If a folder doesn't exist, tell the user - don't keep searching
5. Limit max_results to 20-50 for faster results
6. PDF/binary files: search_files CANNOT read PDF content - identify items by FILENAME, then use web_search if you need descriptions

Guidelines:
- For complex multi-step tasks, create a plan first using create_plan
- For focused sub-tasks, use delegate_task(agent_type, task) to delegate:
  - "file" agent for file read/write/search/edit operations
  - "search" agent for web research
  - "summary" agent for condensing large text
- Be concise in your responses — context is precious
- When reading files, request only the lines you need
- When tool results are large, summarize the key findings before responding
- When you have completed a task, provide a clear summary of what was done

AVOID REDUNDANT CALLS:
- NEVER call get_environment_info more than once - results are cached
- NEVER call the same tool with the same arguments twice
- Once you have the information you need, STOP and answer the user
- If you've called a tool 3+ times without progress, summarize what you found and respond"""


class OutputHandler(Protocol):
    """Protocol for handling orchestrator output."""

    async def on_text_chunk(self, text: str) -> None:
        """Handle a streaming text chunk."""
        ...

    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle start of tool execution."""
        ...

    async def on_tool_result(self, name: str, result: str, success: bool) -> None:
        """Handle tool execution result."""
        ...

    async def on_compaction(self, level: CompactionLevel) -> None:
        """Handle context compaction event."""
        ...

    async def on_error(self, error: str) -> None:
        """Handle an error."""
        ...

    async def on_response_complete(self) -> None:
        """Handle completion of a response (flush any buffered output)."""
        ...


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    max_tool_iterations: int = 20  # Max tool call loops per user turn
    max_parallel_tools: int = 5  # Max concurrent tool executions
    serialize_file_writes: bool = True  # Serialize writes to same file
    system_prompt: str = ORCHESTRATOR_SYSTEM_PROMPT


@dataclass
class TurnResult:
    """Result of processing a user turn."""

    response: str
    tool_calls_made: list[str] = field(default_factory=list)
    tokens_used: TokenUsage = field(default_factory=TokenUsage)
    compaction_triggered: CompactionLevel = CompactionLevel.NONE
    error: str | None = None


class ConsoleOutputHandler:
    """Simple console output handler for testing."""

    async def on_text_chunk(self, text: str) -> None:
        print(text, end="", flush=True)

    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:3])
        print(f"\n[tool] {name}({args_str})")

    async def on_tool_result(self, name: str, result: str, success: bool) -> None:
        icon = "OK" if success else "X"
        preview = result[:100] + "..." if len(result) > 100 else result
        print(f"   {icon} {preview}")

    async def on_compaction(self, level: CompactionLevel) -> None:
        print(f"\n[!] Context compaction: {level.name}")

    async def on_error(self, error: str) -> None:
        print(f"\n[X] Error: {error}")

    async def on_response_complete(self) -> None:
        print()  # Newline after response


class NullOutputHandler:
    """Silent output handler that does nothing."""

    async def on_text_chunk(self, text: str) -> None:
        pass

    async def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        pass

    async def on_tool_result(self, name: str, result: str, success: bool) -> None:
        pass

    async def on_compaction(self, level: CompactionLevel) -> None:
        pass

    async def on_error(self, error: str) -> None:
        pass

    async def on_response_complete(self) -> None:
        pass


class Orchestrator:
    """Main orchestrator agent that coordinates LLM, tools, and sub-agents.
    
    The orchestrator is the central controller that:
    - Manages the conversation context
    - Sends requests to the LLM
    - Executes tool calls
    - Delegates to sub-agents
    - Handles context compaction
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config: OrchestratorConfig | None = None,
        output_handler: OutputHandler | None = None,
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            settings: Application settings.
            config: Orchestrator-specific configuration.
            output_handler: Handler for streaming output.
        """
        self._settings = settings or get_settings()
        self._config = config or OrchestratorConfig()
        self._output = output_handler or NullOutputHandler()
        
        # Core components
        self._llm_client = LMStudioClient(self._settings)
        self._tool_registry = ToolRegistry()
        self._plan_tracker = PlanTracker()
        
        # LLM semaphore - ensures only one LLM call at a time
        self._llm_semaphore = asyncio.Semaphore(1)
        
        # File write locks - prevents concurrent writes to same file
        self._file_locks: dict[str, asyncio.Lock] = {}
        
        # Context manager with system prompt
        self._context = ContextManager(
            settings=self._settings,
            system_prompt=self._config.system_prompt,
            llm_client=self._llm_client,
        )
        
        # Sub-agent spawner
        self._sub_agent_spawner = SubAgentSpawner(
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            llm_semaphore=self._llm_semaphore,
        )
        
        # Register sub-agent handlers
        self._register_sub_agent_handlers()
        
        # State
        self._initialized = False
        self._turn_count = 0

    def _register_sub_agent_handlers(self) -> None:
        """Register sub-agent handlers for the delegate_task tool."""
        for agent_type in self._sub_agent_spawner.get_agent_types():
            handler = self._create_agent_handler(agent_type)
            register_agent_handler(agent_type, handler)

    def _create_agent_handler(self, agent_type: str) -> Callable[[str], Any]:
        """Create a handler function for a sub-agent type."""
        async def handler(task: str) -> str:
            return await self._sub_agent_spawner.spawn(agent_type, task)
        return handler

    async def initialize(self) -> bool:
        """Initialize the orchestrator and check LM Studio availability.
        
        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True
        
        # Check LM Studio
        if not await self._llm_client.is_available():
            logger.error("LM Studio is not available")
            return False
        
        # Get available models
        models = await self._llm_client.get_models()
        if models:
            logger.info(f"LM Studio available with models: {models}")
        
        self._initialized = True
        logger.info("Orchestrator initialized")
        return True

    @property
    def context(self) -> ContextManager:
        """Get the context manager."""
        return self._context

    @property
    def plan_tracker(self) -> PlanTracker:
        """Get the plan tracker."""
        return self._plan_tracker

    @property
    def llm_client(self) -> LMStudioClient:
        """Get the LLM client."""
        return self._llm_client

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the tool registry."""
        return self._tool_registry

    def set_output_handler(self, handler: OutputHandler) -> None:
        """Set the output handler."""
        self._output = handler

    async def run(self, user_input: str) -> TurnResult:
        """Process a user input and generate a response.
        
        This is the main entry point for handling user messages.
        
        Args:
            user_input: The user's message.
        
        Returns:
            TurnResult with the response and metadata.
        """
        if not self._initialized:
            if not await self.initialize():
                return TurnResult(
                    response="",
                    error="Failed to initialize: LM Studio not available",
                )
        
        self._turn_count += 1
        result = TurnResult(response="")
        
        try:
            # Add user message to context
            self._context.add_message(Message.user(user_input))
            
            # Check and handle compaction
            compaction_level = self._context._budget.should_compact(
                self._context._usage.total
            )
            if compaction_level != CompactionLevel.NONE:
                await self._output.on_compaction(compaction_level)
                await self._context.trigger_compaction(compaction_level)
                result.compaction_triggered = compaction_level
            
            # Main tool loop
            iteration = 0
            accumulated_response = ""
            
            while iteration < self._config.max_tool_iterations:
                iteration += 1
                logger.debug(f"Tool loop iteration {iteration}")
                
                # Get messages for LLM
                messages = self._context.get_prompt_messages()
                
                # Get tools (with plan state if active)
                tools = self._tool_registry.get_openai_tools()
                
                # Call LLM with streaming
                async with self._llm_semaphore:
                    try:
                        stream = await self._llm_client.chat(
                            messages=messages,
                            tools=tools if tools else None,
                            stream=True,
                        )
                        
                        # Process stream
                        text_response, tool_calls = await self._process_stream(stream)
                        
                    except (LLMConnectionError, LLMTimeoutError) as e:
                        await self._output.on_error(str(e))
                        result.error = str(e)
                        return result
                
                accumulated_response += text_response
                
                # If no tool calls, we're done
                if not tool_calls:
                    # Add assistant response to context
                    self._context.add_message(Message.assistant(text_response))
                    break
                
                # Add assistant message with tool calls
                self._context.add_message(Message.assistant(
                    content=text_response if text_response else None,
                    tool_calls=tool_calls,
                ))
                
                # Execute tool calls
                tool_results = await self._execute_tool_calls(tool_calls)
                
                # Track tool calls
                for tc in tool_calls:
                    result.tool_calls_made.append(tc.function.name)
                
                # Add tool results to context
                for tc, tool_result in zip(tool_calls, tool_results):
                    self._context.add_message(
                        Message.tool_result(tc.id, tool_result)
                    )
                
                # Check compaction after tool results
                compaction_level = self._context._budget.should_compact(
                    self._context._usage.total
                )
                if compaction_level != CompactionLevel.NONE:
                    await self._output.on_compaction(compaction_level)
                    await self._context.trigger_compaction(compaction_level)
                    if result.compaction_triggered == CompactionLevel.NONE:
                        result.compaction_triggered = compaction_level
            
            # Check for max iterations
            if iteration >= self._config.max_tool_iterations:
                warning = f"\n\n[Warning: Reached maximum tool iterations ({self._config.max_tool_iterations})]"
                accumulated_response += warning
                await self._output.on_text_chunk(warning)
            
            # Log the full assistant response to transcript
            if accumulated_response:
                transcript_assistant(accumulated_response)
            
            result.response = accumulated_response
            result.tokens_used = self._llm_client.session_usage
            await self._output.on_response_complete()
            
        except Exception as e:
            logger.exception("Error in orchestrator run")
            await self._output.on_error(str(e))
            result.error = str(e)
        
        return result

    async def _process_stream(
        self,
        stream: AsyncGenerator[StreamChunk, None],
    ) -> tuple[str, list[ToolCall] | None]:
        """Process a streaming LLM response.
        
        Args:
            stream: Async generator of stream chunks.
        
        Returns:
            Tuple of (text_response, tool_calls).
        """
        text_parts: list[str] = []
        tool_calls_builder: dict[int, dict[str, Any]] = {}
        
        async for chunk in stream:
            # Handle text content
            if chunk.content:
                text_parts.append(chunk.content)
                await self._output.on_text_chunk(chunk.content)
            
            # Handle tool call deltas
            if chunk.tool_calls_delta:
                for tc_delta in chunk.tool_calls_delta:
                    idx = tc_delta.get("index", 0)
                    
                    if idx not in tool_calls_builder:
                        tool_calls_builder[idx] = {
                            "id": None,
                            "name": "",
                            "arguments": "",
                        }
                    
                    builder = tool_calls_builder[idx]
                    
                    if tc_delta.get("id"):
                        builder["id"] = tc_delta["id"]
                    
                    func = tc_delta.get("function") or {}
                    if func.get("name"):
                        builder["name"] = func["name"]
                    if func.get("arguments"):
                        builder["arguments"] += func["arguments"]
        
        # Build text response
        text_response = "".join(text_parts)
        
        # Build tool calls
        tool_calls: list[ToolCall] | None = None
        if tool_calls_builder:
            tool_calls = []
            for idx in sorted(tool_calls_builder.keys()):
                builder = tool_calls_builder[idx]
                if builder["id"] and builder["name"]:
                    tool_calls.append(ToolCall(
                        id=builder["id"],
                        type="function",
                        function=FunctionCall(
                            name=builder["name"],
                            arguments=builder["arguments"],
                        ),
                    ))
        
        # Fallback: try to extract tool calls from text if model outputs them as text
        # Some models output tool calls as text like: {"name":"tool_name","arguments":{...}}
        if not tool_calls and text_response:
            extracted = self._extract_tool_calls_from_text(text_response)
            if extracted:
                tool_calls = extracted
                # Remove the tool call JSON from text response
                text_response = self._clean_tool_call_text(text_response)
        
        return text_response, tool_calls
    
    def _extract_tool_calls_from_text(self, text: str) -> list[ToolCall] | None:
        """Try to extract tool calls from text response.
        
        Some models output tool calls as JSON in text instead of using the API.
        """
        import re
        import uuid
        
        tool_calls = []
        
        # Pattern to match {"name": "...", "arguments": {...}}
        pattern = r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\}|\[\]|\{\})\}'
        
        for match in re.finditer(pattern, text):
            name = match.group(1)
            arguments = match.group(2)
            
            # Verify this is a valid tool
            if name in self._tool_registry.tools:
                tool_calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=arguments,
                    ),
                ))
        
        return tool_calls if tool_calls else None
    
    def _clean_tool_call_text(self, text: str) -> str:
        """Remove tool call markers and JSON from text response."""
        import re
        
        # Remove all <|...|> special tokens (model control tokens)
        text = re.sub(r'<\|[^|]+\|>', '', text)
        
        # Remove common model directives like "commentary to=functions" etc
        text = re.sub(r'\b(commentary|constrain|message|channel)\s*(to=)?\w*', '', text, flags=re.IGNORECASE)
        
        # Remove tool call JSON patterns
        text = re.sub(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\}', '', text)
        
        # Clean up leftover artifacts
        text = re.sub(r'\bjson\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text)  # Multiple spaces to single
        
        return text.strip()

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
    ) -> list[str]:
        """Execute a list of tool calls.
        
        Handles parallelization and file write serialization.
        
        Args:
            tool_calls: List of tool calls to execute.
        
        Returns:
            List of tool results in the same order.
        """
        # Group tool calls by whether they need serialization
        write_tools = {"write_file", "edit_file"}
        
        # Extract file paths for serialization
        def get_file_path(tc: ToolCall) -> str | None:
            if tc.function.name in write_tools:
                try:
                    args = json.loads(tc.function.arguments)
                    return args.get("path")
                except json.JSONDecodeError:
                    return None
            return None
        
        # Execute tool calls
        async def execute_one(tc: ToolCall) -> str:
            name = tc.function.name
            
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            
            # Log to transcript
            transcript_tool_call(name, args)
            
            # Notify output handler
            await self._output.on_tool_start(name, args)
            
            # Get file lock if needed
            file_path = get_file_path(tc)
            lock = None
            if file_path and self._config.serialize_file_writes:
                if file_path not in self._file_locks:
                    self._file_locks[file_path] = asyncio.Lock()
                lock = self._file_locks[file_path]
            
            # Execute with optional lock
            try:
                if lock:
                    async with lock:
                        result = await self._tool_registry.execute(name, args)
                else:
                    result = await self._tool_registry.execute(name, args)
                
                success = not result.startswith("Error:")
                
                # Log result to transcript
                transcript_tool_result(name, result, success)
                
                await self._output.on_tool_result(name, result, success)
                return result
                
            except Exception as e:
                error = f"Error: {type(e).__name__}: {e}"
                transcript_tool_result(name, error, False)
                await self._output.on_tool_result(name, error, False)
                return error
        
        # Execute in parallel with limited concurrency
        semaphore = asyncio.Semaphore(self._config.max_parallel_tools)
        
        async def bounded_execute(tc: ToolCall) -> str:
            async with semaphore:
                return await execute_one(tc)
        
        results = await asyncio.gather(
            *[bounded_execute(tc) for tc in tool_calls],
            return_exceptions=False,
        )
        
        return results

    async def chat(self, user_input: str) -> str:
        """Simplified interface for testing - just returns the response text.
        
        Args:
            user_input: The user's message.
        
        Returns:
            The assistant's response text.
        """
        result = await self.run(user_input)
        if result.error:
            return f"Error: {result.error}"
        return result.response

    def get_context_status(self) -> dict[str, Any]:
        """Get current context status."""
        status = self._context.get_status()
        return status.to_display_dict()

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "turn_count": self._turn_count,
            "total_tokens": self._llm_client.session_usage.total_tokens,
            "llm_calls": self._llm_client.call_count,
            "context_status": self.get_context_status(),
        }

    async def reset(self) -> None:
        """Reset the orchestrator state for a new conversation."""
        self._context = ContextManager(
            settings=self._settings,
            system_prompt=self._config.system_prompt,
            llm_client=self._llm_client,
        )
        self._plan_tracker = PlanTracker()
        self._llm_client.reset_session_stats()
        self._turn_count = 0
        self._file_locks.clear()
        logger.info("Orchestrator reset")


__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "TurnResult",
    "OutputHandler",
    "ConsoleOutputHandler",
    "NullOutputHandler",
    "ORCHESTRATOR_SYSTEM_PROMPT",
]