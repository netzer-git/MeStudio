"""CLI interface components."""

from mestudio.cli.interface import CLIInterface, CLIOutputHandler
from mestudio.cli.renderers import (
    ContextStatusRenderer,
    DiffRenderer,
    ErrorRenderer,
    HelpRenderer,
    PlanRenderer,
    StreamingMarkdownRenderer,
    ToolCallRenderer,
)
from mestudio.cli.theme import (
    ASSISTANT_BORDER,
    ASSISTANT_STYLE,
    ERROR_BORDER,
    SYSTEM_STYLE,
    TOOL_ICONS,
    TOOL_STYLE,
    USER_BORDER,
    USER_STYLE,
    get_context_color,
    get_tool_icon,
)

__all__ = [
    # Interface
    "CLIInterface",
    "CLIOutputHandler",
    # Renderers
    "ContextStatusRenderer",
    "DiffRenderer",
    "ErrorRenderer",
    "HelpRenderer",
    "PlanRenderer",
    "StreamingMarkdownRenderer",
    "ToolCallRenderer",
    # Theme
    "ASSISTANT_BORDER",
    "ASSISTANT_STYLE",
    "ERROR_BORDER",
    "SYSTEM_STYLE",
    "TOOL_ICONS",
    "TOOL_STYLE",
    "USER_BORDER",
    "USER_STYLE",
    "get_context_color",
    "get_tool_icon",
]
