"""Tool system components for MeStudio Agent."""

from mestudio.tools.registry import (
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
    get_registry,
    tool,
)

# Import tools to register them
from mestudio.tools import file_tools
from mestudio.tools import web_tools
from mestudio.tools import context_tools
from mestudio.tools import plan_tools
from mestudio.tools import agent_tools

# Expose key classes and utilities
from mestudio.tools.file_tools import is_binary
from mestudio.tools.web_tools import (
    SearchResult,
    SearchProvider,
    DDGSProvider,
    BraveSearchProvider,
    WebToolManager,
    get_web_manager,
    cleanup_web_tools,
)
from mestudio.tools.context_tools import set_context_manager, get_context_manager
from mestudio.tools.plan_tools import (
    PlanStep,
    TaskPlan,
    get_current_plan,
    set_current_plan,
)
from mestudio.tools.agent_tools import (
    register_agent_handler,
    get_agent_handler,
    list_agent_types,
    register_placeholder_handlers,
)


def register_all_tools() -> ToolRegistry:
    """Ensure all tools are registered and return the registry.
    
    This function is idempotent - calling it multiple times is safe.
    The @tool decorators register tools on import, so this just
    ensures the modules are loaded.
    
    Returns:
        The global tool registry with all tools registered.
    """
    # All tools are registered via decorators on module import
    # The imports above ensure they're loaded
    return get_registry()


__all__ = [
    # Registry
    "ToolRegistry",
    "ToolDefinition",
    "ToolParameter",
    "get_registry",
    "tool",
    "register_all_tools",
    # File tools
    "is_binary",
    # Web tools
    "SearchResult",
    "SearchProvider", 
    "DuckDuckGoProvider",
    "BraveSearchProvider",
    "WebToolManager",
    "get_web_manager",
    "cleanup_web_tools",
    # Context tools
    "set_context_manager",
    "get_context_manager",
    # Plan tools
    "PlanStep",
    "TaskPlan",
    "get_current_plan",
    "set_current_plan",
    # Agent tools
    "register_agent_handler",
    "get_agent_handler",
    "list_agent_types",
    "register_placeholder_handlers",
]
