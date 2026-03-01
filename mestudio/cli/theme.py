"""Colors, styles, and icons for the CLI."""

from rich.style import Style

# ============================================================================
# Text Styles
# ============================================================================

USER_STYLE = Style(color="cyan", bold=True)
ASSISTANT_STYLE = Style(color="white")
TOOL_STYLE = Style(color="blue", bold=True)
ERROR_STYLE = Style(color="red", bold=True)
SYSTEM_STYLE = Style(color="yellow", dim=True)
SUCCESS_STYLE = Style(color="green", bold=True)
WARNING_STYLE = Style(color="yellow", bold=True)
DIM_STYLE = Style(dim=True)
HIGHLIGHT_STYLE = Style(color="magenta", bold=True)

# Panel borders
USER_BORDER = "cyan"
ASSISTANT_BORDER = "white"
TOOL_BORDER = "blue"
ERROR_BORDER = "red"
SYSTEM_BORDER = "yellow"

# ============================================================================
# Icons
# ============================================================================

# General
ICON_USER = ">"
ICON_ASSISTANT = "Assistant"
ICON_THINKING = "..."
ICON_ERROR = "X"
ICON_SUCCESS = "OK"
ICON_WARNING = "!"
ICON_INFO = "i"

# Tools
ICON_TOOL = "[tool]"
ICON_FILE_READ = "[read]"
ICON_FILE_WRITE = "[write]"
ICON_FILE_EDIT = "[edit]"
ICON_FOLDER = "[dir]"
ICON_SEARCH = "[search]"
ICON_WEB = "[web]"
ICON_BROWSER = "[fetch]"
ICON_PLAN = "[plan]"
ICON_CONTEXT = "[ctx]"
ICON_DELEGATE = "[agent]"

# Plan status
ICON_STEP_PENDING = "[ ]"
ICON_STEP_ACTIVE = "[>]"
ICON_STEP_DONE = "[x]"
ICON_STEP_FAILED = "[!]"
ICON_STEP_SKIPPED = "[-]"

# Context status
ICON_CONTEXT_OK = "[OK]"
ICON_CONTEXT_SOFT = "[SOFT]"
ICON_CONTEXT_AGGRESSIVE = "[AGG]"
ICON_CONTEXT_EMERGENCY = "[EMERG]"

# ============================================================================
# Tool Icon Mapping
# ============================================================================

TOOL_ICONS = {
    "read_file": ICON_FILE_READ,
    "write_file": ICON_FILE_WRITE,
    "edit_file": ICON_FILE_EDIT,
    "list_directory": ICON_FOLDER,
    "search_files": ICON_SEARCH,
    "find_files": ICON_SEARCH,
    "web_search": ICON_WEB,
    "read_webpage": ICON_BROWSER,
    "create_plan": ICON_PLAN,
    "update_step": ICON_PLAN,
    "get_plan": ICON_PLAN,
    "add_steps": ICON_PLAN,
    "remove_step": ICON_PLAN,
    "cancel_plan": ICON_PLAN,
    "replace_plan": ICON_PLAN,
    "save_context": ICON_CONTEXT,
    "load_context": ICON_CONTEXT,
    "compact_now": ICON_CONTEXT,
    "context_status": ICON_CONTEXT,
    "list_sessions": ICON_CONTEXT,
    "delegate_task": ICON_DELEGATE,
}


def get_tool_icon(tool_name: str) -> str:
    """Get the icon for a tool, falling back to generic tool icon."""
    return TOOL_ICONS.get(tool_name, ICON_TOOL)


# ============================================================================
# Progress Bar Colors
# ============================================================================

PROGRESS_COMPLETE = "green"
PROGRESS_REMAINING = "grey50"
PROGRESS_WARNING = "yellow"
PROGRESS_DANGER = "red"


def get_context_color(percent: float) -> str:
    """Get color for context usage percentage."""
    if percent < 65:
        return PROGRESS_COMPLETE
    elif percent < 80:
        return PROGRESS_WARNING
    elif percent < 95:
        return "orange1"
    else:
        return PROGRESS_DANGER
