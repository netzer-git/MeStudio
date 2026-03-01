"""Delegate task tool for sub-agent invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from mestudio.tools.registry import tool

if TYPE_CHECKING:
    pass


# Type for agent handler functions
AgentHandler = Callable[[str], Coroutine[Any, Any, str]]

# Registry of agent types and their handlers
_agent_handlers: dict[str, AgentHandler] = {}


def register_agent_handler(agent_type: str, handler: AgentHandler) -> None:
    """Register a handler for an agent type.
    
    Called by the main agent during initialization to register
    sub-agent handlers.
    
    Args:
        agent_type: Type identifier (e.g., 'file', 'search', 'summary').
        handler: Async function that takes a task string and returns result.
    """
    _agent_handlers[agent_type] = handler
    logger.debug(f"Registered agent handler: {agent_type}")


def get_agent_handler(agent_type: str) -> AgentHandler | None:
    """Get the handler for an agent type."""
    return _agent_handlers.get(agent_type)


def list_agent_types() -> list[str]:
    """List all registered agent types."""
    return list(_agent_handlers.keys())


@tool(
    name="delegate_task",
    description=(
        "Delegate a focused task to a specialized sub-agent. "
        "Available agent types: 'file' (read/write/search files), "
        "'search' (web research), 'summary' (condense large text)."
    ),
    timeout=120.0,  # Sub-agents may need more time
)
async def delegate_task(agent_type: str, task: str) -> str:
    """Delegate a task to a sub-agent.
    
    Args:
        agent_type: Type of agent - 'file', 'search', or 'summary'.
        task: The task description for the sub-agent.
    """
    valid_types = ["file", "search", "summary"]
    
    if agent_type not in valid_types:
        return (
            f"Error: Invalid agent type '{agent_type}'. "
            f"Available types: {', '.join(valid_types)}"
        )
    
    handler = _agent_handlers.get(agent_type)
    
    if handler is None:
        return (
            f"Error: Agent '{agent_type}' is not currently available. "
            "The agent system may not be fully initialized."
        )
    
    logger.info(f"Delegating to {agent_type} agent: {task[:100]}...")
    
    try:
        result = await handler(task)
        logger.info(f"Sub-agent {agent_type} completed")
        return result
    
    except Exception as e:
        error = f"Error: Sub-agent '{agent_type}' failed: {e}"
        logger.error(error)
        return error


# Placeholder handlers for development/testing
async def _placeholder_file_handler(task: str) -> str:
    """Placeholder file agent handler."""
    return f"[File Agent] Task received: {task[:200]}... (not implemented)"


async def _placeholder_search_handler(task: str) -> str:
    """Placeholder search agent handler."""
    return f"[Search Agent] Task received: {task[:200]}... (not implemented)"


async def _placeholder_summary_handler(task: str) -> str:
    """Placeholder summary agent handler."""
    return f"[Summary Agent] Task received: {task[:200]}... (not implemented)"


def register_placeholder_handlers() -> None:
    """Register placeholder handlers for testing."""
    register_agent_handler("file", _placeholder_file_handler)
    register_agent_handler("search", _placeholder_search_handler)
    register_agent_handler("summary", _placeholder_summary_handler)


__all__ = [
    "delegate_task",
    "register_agent_handler",
    "get_agent_handler",
    "list_agent_types",
    "register_placeholder_handlers",
]
