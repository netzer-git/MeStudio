"""Comprehensive logging setup for MeStudio Agent.

This module provides structured logging with session context, file rotation,
and dual output (console + file). All operations are logged with timing and
summaries—never full content—to enable debugging without bloating logs.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from mestudio.core.config import Settings


# Track session metrics for final summary
class SessionMetrics:
    """Track cumulative metrics for a session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.llm_calls = 0
        self.tool_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.compaction_count = 0
        self.errors = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        duration = datetime.now() - self.start_time
        return {
            "session_id": self.session_id,
            "duration_seconds": int(duration.total_seconds()),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "compaction_count": self.compaction_count,
            "errors": self.errors,
        }


# Global session state
_session_id: str | None = None
_session_metrics: SessionMetrics | None = None


def generate_session_id() -> str:
    """Generate a unique session ID (8 hex chars)."""
    return uuid.uuid4().hex[:8]


def setup_logging(settings: Settings, session_id: str | None = None) -> str:
    """Configure loguru with session context, file rotation, and dual output.
    
    Args:
        settings: Application settings containing log configuration.
        session_id: Optional session ID. Generated if not provided.
    
    Returns:
        The session ID being used.
    """
    global _session_id, _session_metrics
    
    _session_id = session_id or generate_session_id()
    _session_metrics = SessionMetrics(_session_id)
    
    # Remove default handler
    logger.remove()
    
    # Console: human-readable, INFO+ only, colorized
    console_format = (
        "<dim>{time:HH:mm:ss}</dim> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[session]}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",
        colorize=True,
        filter=lambda record: record["extra"].get("session") is not None,
    )
    
    # File: structured logging with rotation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if settings.log_json_format:
        # JSON format for programmatic analysis
        logger.add(
            str(log_path),
            format="{message}",
            level=settings.log_level,
            rotation=settings.log_max_size,
            retention=settings.log_rotation_count,
            serialize=True,  # JSON format
            enqueue=True,  # Thread-safe
        )
    else:
        # Human-readable format
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[session]} | {name}:{function}:{line} | {message}"
        )
        logger.add(
            str(log_path),
            format=file_format,
            level=settings.log_level,
            rotation=settings.log_max_size,
            retention=settings.log_rotation_count,
            enqueue=True,
        )
    
    # Bind session ID to all future log calls
    logger.configure(extra={"session": _session_id})
    
    return _session_id


def get_session_logger():
    """Get the logger bound to the current session.
    
    Returns a no-op logger if logging hasn't been initialized yet.
    """
    if _session_id is None:
        # Return the base logger without session binding
        return logger
    return logger.bind(session=_session_id)


def get_session_id() -> str | None:
    """Get the current session ID."""
    return _session_id


def get_session_metrics() -> SessionMetrics | None:
    """Get the current session metrics tracker."""
    return _session_metrics


# ============================================================================
# Structured logging helpers
# ============================================================================

def log_session_start(python_version: str, model: str, max_tokens: int) -> None:
    """Log session start with system information."""
    log = get_session_logger()
    log.info(
        "Session started",
        event="session_start",
        python=python_version,
        model=model,
        max_tokens=max_tokens,
    )


def log_session_end() -> None:
    """Log session end with cumulative metrics."""
    if _session_metrics is None:
        return
    log = get_session_logger()
    log.info(
        "Session ended",
        event="session_end",
        **_session_metrics.to_dict(),
    )


def log_user_message(content: str, max_preview: int = 50) -> None:
    """Log user message receipt (preview only, not full content)."""
    log = get_session_logger()
    preview = content[:max_preview] + "..." if len(content) > max_preview else content
    log.info(
        "User message received",
        event="user_message",
        length=len(content),
        preview=preview.replace("\n", " "),
    )


def log_assistant_message(content: str, tool_calls: int = 0) -> None:
    """Log assistant message (summary only)."""
    log = get_session_logger()
    log.info(
        "Assistant response",
        event="assistant_message",
        length=len(content),
        tool_calls=tool_calls,
    )


def log_llm_request(
    message_count: int,
    tool_count: int,
    estimated_tokens: int | None = None,
) -> None:
    """Log LLM request being sent."""
    if _session_metrics:
        _session_metrics.llm_calls += 1
    log = get_session_logger()
    log.info(
        "LLM request",
        event="llm_request",
        message_count=message_count,
        tool_count=tool_count,
        estimated_tokens=estimated_tokens,
    )


def log_llm_response(
    duration_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: int = 0,
) -> None:
    """Log LLM response received."""
    if _session_metrics:
        _session_metrics.total_prompt_tokens += prompt_tokens
        _session_metrics.total_completion_tokens += completion_tokens
    log = get_session_logger()
    log.info(
        "LLM response",
        event="llm_response",
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_calls=tool_calls,
    )


def log_llm_retry(attempt: int, max_attempts: int, error: str, backoff_sec: float) -> None:
    """Log LLM retry attempt."""
    log = get_session_logger()
    log.warning(
        "LLM retry",
        event="llm_retry",
        attempt=attempt,
        max_attempts=max_attempts,
        error=error[:100],  # Truncate error message
        backoff_sec=backoff_sec,
    )


def log_tool_call(
    tool_name: str,
    args_keys: list[str],
    duration_ms: int,
    success: bool,
    result_length: int,
    error: str | None = None,
) -> None:
    """Log tool execution."""
    if _session_metrics:
        _session_metrics.tool_calls += 1
        if not success:
            _session_metrics.errors += 1
    log = get_session_logger()
    log.info(
        f"Tool: {tool_name}",
        event="tool_call",
        tool=tool_name,
        args_keys=args_keys,
        duration_ms=duration_ms,
        success=success,
        result_length=result_length,
        error=error[:100] if error else None,
    )


def log_tool_registered(tool_name: str, description: str) -> None:
    """Log tool registration."""
    log = get_session_logger()
    log.debug(
        f"Tool registered: {tool_name}",
        event="tool_registered",
        tool=tool_name,
        description=description[:80],
    )


def log_compaction(
    level: str,
    before_tokens: int,
    after_tokens: int,
    duration_ms: int,
) -> None:
    """Log context compaction event."""
    if _session_metrics:
        _session_metrics.compaction_count += 1
    log = get_session_logger()
    log.info(
        f"Compaction: {level}",
        event="compaction",
        level=level,
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        tokens_freed=before_tokens - after_tokens,
        duration_ms=duration_ms,
    )


def log_context_snapshot(
    total_tokens: int,
    used_pct: float,
    message_count: int,
    compressed: bool = False,
) -> None:
    """Log context state snapshot (DEBUG level)."""
    log = get_session_logger()
    log.debug(
        "Context snapshot",
        event="context_snapshot",
        total_tokens=total_tokens,
        used_pct=round(used_pct * 100, 1),
        message_count=message_count,
        compressed=compressed,
    )


def log_session_saved(path: str, message_count: int, file_size: int) -> None:
    """Log session save operation."""
    log = get_session_logger()
    log.info(
        "Session saved",
        event="session_saved",
        path=path,
        message_count=message_count,
        file_size_bytes=file_size,
    )


def log_session_loaded(path: str, message_count: int) -> None:
    """Log session load operation."""
    log = get_session_logger()
    log.info(
        "Session loaded",
        event="session_loaded",
        path=path,
        message_count=message_count,
    )


def log_error(operation: str, error: str, **kwargs: Any) -> None:
    """Log an error with context."""
    if _session_metrics:
        _session_metrics.errors += 1
    log = get_session_logger()
    log.error(
        f"Error in {operation}",
        event="error",
        operation=operation,
        error=error[:200],
        **kwargs,
    )


def log_warning(message: str, **kwargs: Any) -> None:
    """Log a warning with context."""
    log = get_session_logger()
    log.warning(message, **kwargs)


def log_debug(message: str, **kwargs: Any) -> None:
    """Log a debug message with context."""
    log = get_session_logger()
    log.debug(message, **kwargs)
