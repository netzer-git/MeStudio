"""Sub-agent base class and spawner."""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from mestudio.context.manager import ContextManager
from mestudio.core.config import Settings, get_settings
from mestudio.core.models import Message, ToolCall, FunctionCall
from mestudio.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""

    name: str
    system_prompt: str
    available_tools: list[str]  # Tool names from registry
    max_turns: int = 10  # Prevent infinite loops
    max_context_tokens: int = 32000  # Sub-agent context limit


class SubAgentError(Exception):
    """Error during sub-agent execution."""


class SubAgent(ABC):
    """Base class for specialized sub-agents.
    
    Sub-agents run in isolated contexts with limited tool access.
    They complete focused tasks and return compact results to the orchestrator.
    """

    def __init__(
        self,
        config: SubAgentConfig,
        llm_client: LMStudioClient,
        global_registry: ToolRegistry,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize sub-agent.
        
        Args:
            config: Sub-agent configuration.
            llm_client: LLM client for completions.
            global_registry: Global tool registry to filter tools from.
            llm_semaphore: Optional semaphore for LLM call serialization.
        """
        self.config = config
        self.llm_client = llm_client
        self.global_registry = global_registry
        self._llm_semaphore = llm_semaphore or asyncio.Semaphore(1)
        
        # Build filtered tool list for this sub-agent
        self._tools = self._filter_tools()

    def _filter_tools(self) -> list[dict[str, Any]]:
        """Filter global tools to only those available to this sub-agent."""
        tools = []
        for tool_name in self.config.available_tools:
            tool_def = self.global_registry.get(tool_name)
            if tool_def:
                tools.append(tool_def.to_openai_schema())
            else:
                logger.warning(
                    f"SubAgent '{self.config.name}': Tool '{tool_name}' not found in registry"
                )
        return tools

    async def execute(self, task: str) -> str:
        """Execute a task with this sub-agent.
        
        Args:
            task: Task description to execute.
        
        Returns:
            Result string from the sub-agent.
        
        Raises:
            SubAgentError: If execution fails or max turns exceeded.
        """
        # Create isolated context for this sub-agent
        settings = get_settings()
        context = ContextManager(
            settings=settings,
            system_prompt=self.config.system_prompt,
            llm_client=self.llm_client,
        )
        
        # Add task as user message
        context.add_message(Message.user(task))
        
        logger.info(f"SubAgent '{self.config.name}' starting task: {task[:100]}...")
        
        turn = 0
        while turn < self.config.max_turns:
            turn += 1
            logger.debug(f"SubAgent '{self.config.name}' turn {turn}/{self.config.max_turns}")
            
            # Get messages for LLM
            messages = context.get_prompt_messages()
            
            # Call LLM with semaphore to prevent concurrent requests
            async with self._llm_semaphore:
                response = await self.llm_client.chat(
                    messages=messages,
                    tools=self._tools if self._tools else None,
                    stream=False,
                )
            
            # Handle response
            if response.tool_calls:
                # Execute tool calls
                assistant_msg = Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                context.add_message(assistant_msg)
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    
                    # Safety: sub-agents cannot delegate
                    if tool_name == "delegate_task":
                        result = "Error: Sub-agents cannot delegate tasks."
                        logger.warning(f"SubAgent '{self.config.name}' attempted to delegate")
                    else:
                        # Execute tool
                        result = await self.global_registry.execute(
                            tool_name,
                            tool_call.function.arguments,
                        )
                    
                    # Add tool result to context
                    context.add_message(Message.tool_result(tool_call.id, result))
                    logger.debug(f"SubAgent '{self.config.name}' tool '{tool_name}': {len(result)} chars")
            
            else:
                # No tool calls - agent is done, return the response
                final_response = response.content or ""
                logger.info(
                    f"SubAgent '{self.config.name}' completed in {turn} turns: "
                    f"{len(final_response)} chars"
                )
                return final_response
        
        # Max turns exceeded
        error = f"SubAgent '{self.config.name}' exceeded max turns ({self.config.max_turns})"
        logger.error(error)
        raise SubAgentError(error)

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this sub-agent's capabilities."""


class SubAgentSpawner:
    """Factory for creating and executing sub-agents.
    
    Manages pre-configured sub-agent types and handles execution.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        tool_registry: ToolRegistry,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize spawner.
        
        Args:
            llm_client: LLM client for sub-agent completions.
            tool_registry: Global tool registry.
            llm_semaphore: Semaphore for LLM call serialization.
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self._llm_semaphore = llm_semaphore or asyncio.Semaphore(1)
        
        # Import specialized agents here to avoid circular imports
        from mestudio.agents.file_agent import FileAgent
        from mestudio.agents.search_agent import SearchAgent
        from mestudio.agents.summary_agent import SummaryAgent
        
        self._agent_classes: dict[str, type[SubAgent]] = {
            "file": FileAgent,
            "search": SearchAgent,
            "summary": SummaryAgent,
        }

    def get_agent_types(self) -> list[str]:
        """Get available agent types."""
        return list(self._agent_classes.keys())

    def create_agent(self, agent_type: str) -> SubAgent:
        """Create a sub-agent instance.
        
        Args:
            agent_type: Type of agent to create.
        
        Returns:
            SubAgent instance.
        
        Raises:
            ValueError: If agent type is unknown.
        """
        agent_class = self._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(
                f"Unknown agent type: '{agent_type}'. "
                f"Available: {', '.join(self._agent_classes.keys())}"
            )
        
        return agent_class(
            llm_client=self.llm_client,
            global_registry=self.tool_registry,
            llm_semaphore=self._llm_semaphore,
        )

    async def spawn(self, agent_type: str, task: str) -> str:
        """Spawn a sub-agent and execute a task.
        
        Args:
            agent_type: Type of agent ('file', 'search', 'summary').
            task: Task description for the agent.
        
        Returns:
            Result string from the sub-agent.
        """
        try:
            agent = self.create_agent(agent_type)
            return await agent.execute(task)
        except ValueError as e:
            return f"Error: {e}"
        except SubAgentError as e:
            return f"Error: Sub-agent failed: {e}"
        except Exception as e:
            logger.exception(f"Sub-agent '{agent_type}' failed unexpectedly")
            return f"Error: Sub-agent '{agent_type}' failed: {type(e).__name__}: {e}"


__all__ = [
    "SubAgent",
    "SubAgentConfig",
    "SubAgentError",
    "SubAgentSpawner",
]