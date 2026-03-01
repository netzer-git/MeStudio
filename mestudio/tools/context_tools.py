"""Context save, load, compact tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mestudio.context import CompactionLevel, ContextManager
from mestudio.tools.registry import tool

if TYPE_CHECKING:
    pass


# Global reference to the context manager (set by the agent)
_context_manager: ContextManager | None = None


def set_context_manager(cm: ContextManager) -> None:
    """Set the global context manager reference.
    
    Called by the agent during initialization.
    
    Args:
        cm: The context manager instance.
    """
    global _context_manager
    _context_manager = cm


def get_context_manager() -> ContextManager | None:
    """Get the current context manager."""
    return _context_manager


@tool(
    name="save_context",
    description="Save the current session to disk. Returns a session ID that can be used to load it later.",
)
async def save_context(label: str = "") -> str:
    """Save the current session.
    
    Args:
        label: Optional label to identify this session (e.g., 'refactoring project X').
    """
    if _context_manager is None:
        return "Error: Context manager not initialized"
    
    try:
        session_id = _context_manager.save_session(label=label)
        status = _context_manager.get_status()
        
        result = f"Session saved successfully!\n\n"
        result += f"Session ID: {session_id}\n"
        if label:
            result += f"Label: {label}\n"
        result += f"Messages: {status.message_count}\n"
        result += f"Tokens: {status.used_tokens:,}\n"
        
        return result
    
    except Exception as e:
        return f"Error saving session: {e}"


@tool(
    name="load_context",
    description="Load a previously saved session. Replaces the current context.",
)
async def load_context(session_id: str) -> str:
    """Load a saved session.
    
    Args:
        session_id: The session ID returned by save_context.
    """
    if _context_manager is None:
        return "Error: Context manager not initialized"
    
    try:
        success = _context_manager.load_session(session_id)
        
        if not success:
            # Try loading as checkpoint
            success = _context_manager.load_checkpoint(session_id)
        
        if not success:
            return f"Error: Session '{session_id}' not found"
        
        status = _context_manager.get_status()
        
        result = f"Session loaded successfully!\n\n"
        result += f"Session ID: {session_id}\n"
        result += f"Messages: {status.message_count}\n"
        result += f"Tokens: {status.used_tokens:,}\n"
        result += f"Budget usage: {status.percent_used:.1%}\n"
        
        return result
    
    except Exception as e:
        return f"Error loading session: {e}"


@tool(
    name="list_sessions",
    description="List all saved sessions.",
)
async def list_sessions() -> str:
    """List all available saved sessions."""
    if _context_manager is None:
        return "Error: Context manager not initialized"
    
    try:
        sessions = _context_manager.list_sessions()
        
        if not sessions:
            return "No saved sessions found."
        
        result = f"Found {len(sessions)} saved session(s):\n\n"
        
        for s in sessions:
            label = f" - {s['label']}" if s.get('label') else ""
            result += f"• {s['session_id']}{label}\n"
            result += f"  Created: {s['created_at']}\n"
            result += f"  Messages: {s['message_count']}, Tokens: {s['total_tokens_used']:,}\n\n"
        
        return result.strip()
    
    except Exception as e:
        return f"Error listing sessions: {e}"


@tool(
    name="compact_now",
    description="Manually trigger context compaction to free up space. Use when approaching context limits.",
)
async def compact_now(level: str = "soft") -> str:
    """Trigger manual compaction.
    
    Args:
        level: Compaction level - 'soft', 'preemptive', 'aggressive', or 'emergency'.
    """
    if _context_manager is None:
        return "Error: Context manager not initialized"
    
    level_map = {
        "soft": CompactionLevel.SOFT,
        "preemptive": CompactionLevel.PREEMPTIVE,
        "aggressive": CompactionLevel.AGGRESSIVE,
        "emergency": CompactionLevel.EMERGENCY,
    }
    
    compact_level = level_map.get(level.lower())
    if compact_level is None:
        return f"Error: Invalid level '{level}'. Use: soft, preemptive, aggressive, or emergency"
    
    status_before = _context_manager.get_status()
    
    try:
        success = await _context_manager.trigger_compaction(compact_level)
        
        if not success:
            return f"Error: Compaction failed (level: {level})"
        
        status_after = _context_manager.get_status()
        
        freed = status_before.used_tokens - status_after.used_tokens
        
        result = f"Compaction complete ({level})!\n\n"
        result += f"Before: {status_before.used_tokens:,} tokens ({status_before.percent_used:.1%})\n"
        result += f"After: {status_after.used_tokens:,} tokens ({status_after.percent_used:.1%})\n"
        result += f"Freed: {freed:,} tokens\n"
        result += f"Messages: {status_before.message_count} → {status_after.message_count}\n"
        
        return result
    
    except Exception as e:
        return f"Error during compaction: {e}"


@tool(
    name="context_status",
    description="Get current context window status including token usage, budget, and compaction level.",
)
async def context_status() -> str:
    """Get context status."""
    if _context_manager is None:
        return "Error: Context manager not initialized"
    
    try:
        status = _context_manager.get_status()
        
        # Build visual progress bar
        bar_width = 30
        filled = int(bar_width * status.percent_used)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        result = "Context Window Status\n"
        result += "=" * 40 + "\n\n"
        
        result += f"Budget: [{bar}] {status.percent_used:.1%}\n\n"
        
        result += f"Total budget:    {status.total_budget:>10,} tokens\n"
        result += f"Usable budget:   {status.usable_budget:>10,} tokens\n"
        result += f"Used:            {status.used_tokens:>10,} tokens\n"
        result += f"Available:       {status.usable_budget - status.used_tokens:>10,} tokens\n\n"
        
        result += f"Compaction level: {status.compaction_level.name.lower()}\n"
        result += f"Message count:    {status.message_count}\n\n"
        
        result += "Token breakdown:\n"
        result += f"  Compressed history: {status.compressed_history_tokens:>8,}\n"
        result += f"  Recent messages:    {status.recent_messages_tokens:>8,}\n"
        result += f"  Tool results:       {status.tool_results_tokens:>8,}\n"
        
        return result
    
    except Exception as e:
        return f"Error getting status: {e}"


__all__ = [
    "save_context",
    "load_context",
    "list_sessions",
    "compact_now",
    "context_status",
    "set_context_manager",
    "get_context_manager",
]
