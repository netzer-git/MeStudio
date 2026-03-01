"""Disk-based context persistence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from mestudio.core.config import get_settings
from mestudio.core.models import Message, SessionMetadata
from mestudio.utils.logging import log_session_saved, log_session_loaded


class SessionSummary(BaseModel):
    """Summary of a saved session for listing."""

    session_id: str
    label: str
    created_at: datetime
    message_count: int
    total_tokens_used: int


class SessionData(BaseModel):
    """Full data for a saved session."""

    version: str = "1.0"
    session_id: str
    label: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    summaries: dict[str, Any] = Field(default_factory=dict)
    plan_state: dict[str, Any] = Field(default_factory=dict)
    metadata: SessionMetadata | None = None


class CheckpointData(BaseModel):
    """Data for a mid-task checkpoint."""

    version: str = "1.0"
    session_id: str
    checkpoint_at: datetime = Field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    compressed_history: str = ""
    plan_state: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)


class MemoryStore:
    """Disk-based persistence for sessions and checkpoints.
    
    Sessions are saved as JSON files in the data/context_store/ directory.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize the memory store.
        
        Args:
            data_dir: Base data directory. Uses settings if not provided.
        """
        if data_dir is None:
            data_dir = get_settings().data_path
        
        self._context_dir = data_dir / "context_store"
        self._context_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID.
        
        Format: YYYY-MM-DD_HH-MM-SS_{uuid4_short}
        Example: 2026-03-01_14-30-00_a3f2b1
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}_{short_uuid}"

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self._context_dir / f"{session_id}.json"

    def _checkpoint_path(self, session_id: str) -> Path:
        """Get the file path for a checkpoint."""
        return self._context_dir / f"{session_id}.checkpoint.json"

    def save_session(
        self,
        session_id: str,
        messages: list[Message],
        summaries: dict[str, Any],
        plan_state: dict[str, Any],
        metadata: SessionMetadata,
        label: str = "",
    ) -> Path:
        """Save a session to disk.
        
        Args:
            session_id: Unique session identifier.
            messages: List of messages in the conversation.
            summaries: Dictionary of summaries at different levels.
            plan_state: Current plan state if any.
            metadata: Session metadata (tokens, etc.).
            label: Optional user-provided label.
        
        Returns:
            Path to the saved session file.
        """
        # Convert messages to dicts for JSON serialization
        message_dicts = [m.model_dump() for m in messages]
        
        session_data = SessionData(
            session_id=session_id,
            label=label or f"Session {session_id}",
            created_at=datetime.now(),
            messages=message_dicts,
            summaries=summaries,
            plan_state=plan_state,
            metadata=metadata,
        )
        
        path = self._session_path(session_id)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data.model_dump(mode="json"), f, indent=2, default=str)
        
        file_size = path.stat().st_size
        log_session_saved(str(path), len(messages), file_size)
        logger.info(f"Saved session to {path}")
        return path

    def load_session(self, session_id: str) -> SessionData | None:
        """Load a session from disk.
        
        Args:
            session_id: The session ID to load.
        
        Returns:
            SessionData if found, None otherwise.
        """
        path = self._session_path(session_id)
        
        if not path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session_data = SessionData.model_validate(data)
            log_session_loaded(str(path), len(session_data.messages))
            return session_data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self) -> list[SessionSummary]:
        """List all saved sessions.
        
        Returns:
            List of session summaries sorted by creation time (newest first).
        """
        sessions: list[SessionSummary] = []
        
        for path in self._context_dir.glob("*.json"):
            # Skip checkpoint files
            if path.name.endswith(".checkpoint.json"):
                continue
            
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                sessions.append(
                    SessionSummary(
                        session_id=data.get("session_id", path.stem),
                        label=data.get("label", ""),
                        created_at=datetime.fromisoformat(data.get("created_at", "")),
                        message_count=len(data.get("messages", [])),
                        total_tokens_used=data.get("metadata", {}).get(
                            "total_tokens_used", 0
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to read session {path}: {e}")
        
        # Sort by creation time, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its checkpoint.
        
        Args:
            session_id: The session ID to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        session_path = self._session_path(session_id)
        checkpoint_path = self._checkpoint_path(session_id)
        
        deleted = False
        
        if session_path.exists():
            session_path.unlink()
            deleted = True
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        return deleted

    def save_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        compressed_history: str,
        plan_state: dict[str, Any],
        token_usage: dict[str, int],
    ) -> Path:
        """Save a mid-task checkpoint.
        
        Args:
            session_id: Session identifier.
            messages: Current messages.
            compressed_history: Compressed history string.
            plan_state: Current plan state.
            token_usage: Current token usage breakdown.
        
        Returns:
            Path to the checkpoint file.
        """
        message_dicts = [m.model_dump() for m in messages]
        
        checkpoint = CheckpointData(
            session_id=session_id,
            checkpoint_at=datetime.now(),
            messages=message_dicts,
            compressed_history=compressed_history,
            plan_state=plan_state,
            token_usage=token_usage,
        )
        
        path = self._checkpoint_path(session_id)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.model_dump(mode="json"), f, indent=2, default=str)
        
        logger.debug(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, session_id: str) -> CheckpointData | None:
        """Load a checkpoint from disk.
        
        Args:
            session_id: The session ID to load checkpoint for.
        
        Returns:
            CheckpointData if found, None otherwise.
        """
        path = self._checkpoint_path(session_id)
        
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CheckpointData.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {session_id}: {e}")
            return None

    def has_checkpoint(self, session_id: str) -> bool:
        """Check if a checkpoint exists for a session."""
        return self._checkpoint_path(session_id).exists()
