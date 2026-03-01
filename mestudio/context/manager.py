"""Context window manager — the heart of the context management system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from mestudio.context.budget import CompactionLevel, ContextUsage, TokenBudget
from mestudio.context.compactor import ContextCompactor
from mestudio.context.memory_store import MemoryStore
from mestudio.context.token_counter import TokenCounter, get_token_counter
from mestudio.core.config import Settings, get_settings
from mestudio.core.models import Message, SessionMetadata
from mestudio.utils.logging import log_compaction, log_context_snapshot

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


@dataclass
class ContextStatus:
    """Current status of the context window."""

    total_budget: int
    usable_budget: int
    used_tokens: int
    percent_used: float
    compaction_level: CompactionLevel
    message_count: int
    compressed_history_tokens: int
    recent_messages_tokens: int
    tool_results_tokens: int

    def to_display_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "total_budget": f"{self.total_budget:,}",
            "usable_budget": f"{self.usable_budget:,}",
            "used_tokens": f"{self.used_tokens:,}",
            "percent_used": f"{self.percent_used:.1%}",
            "compaction_level": self.compaction_level.name.lower(),
            "message_count": self.message_count,
            "compressed_history_tokens": f"{self.compressed_history_tokens:,}",
            "recent_messages_tokens": f"{self.recent_messages_tokens:,}",
            "tool_results_tokens": f"{self.tool_results_tokens:,}",
        }


class ContextManager:
    """Manages the context window with three-tier memory and automatic compaction.
    
    The three tiers are:
    1. Working memory (current messages)
    2. Compressed summaries (rolling history)
    3. Disk persistence (saved sessions/checkpoints)
    
    The five degradation levels are:
    - NONE: Normal operation
    - SOFT: Summarize older messages
    - PREEMPTIVE: Safety net compaction
    - AGGRESSIVE: Heavy summarization
    - EMERGENCY: Extractive only, no LLM
    """

    # Default system prompt for the agent
    DEFAULT_SYSTEM_PROMPT = """You are MeStudio Agent, a local AI assistant with tool-calling capabilities.

You can:
- Read, write, search, and edit local files
- Search the web for information
- Create and track multi-step plans for complex tasks
- Delegate focused tasks to sub-agents via delegate_task

Guidelines:
- For complex multi-step tasks, create a plan first using create_plan
- For focused sub-tasks, use delegate_task(agent_type, task) to delegate:
  - "file" agent for file read/write/search/edit operations
  - "search" agent for web research
  - "summary" agent for condensing large text
- Be concise in your responses — context is precious
- When reading files, request only the lines you need
- When tool results are large, summarize the key findings before responding
- Always report your progress on the current plan step
- If a plan is wrong or outdated, use cancel_plan or replace_plan to fix it"""

    def __init__(
        self,
        settings: Settings | None = None,
        system_prompt: str | None = None,
        llm_client: LMStudioClient | None = None,
    ) -> None:
        """Initialize the context manager.
        
        Args:
            settings: Application settings.
            system_prompt: Custom system prompt (uses default if not provided).
            llm_client: LLM client for summarization (required for compaction).
        """
        self._settings = settings or get_settings()
        self._budget = TokenBudget.from_settings(self._settings)
        self._token_counter = get_token_counter()
        self._compactor = ContextCompactor()
        self._memory_store = MemoryStore()
        self._llm_client = llm_client
        
        # Current state
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._messages: list[Message] = []
        self._compressed_history: str = ""
        self._plan_state: str = ""
        self._session_id: str = ""
        
        # Tracking
        self._compaction_count: int = 0
        self._tools_called: set[str] = set()
        
        # Cache token counts
        self._usage = ContextUsage()
        self._update_usage()

    def set_llm_client(self, client: LMStudioClient) -> None:
        """Set the LLM client (can be set after initialization)."""
        self._llm_client = client

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set a new system prompt and update usage."""
        self._system_prompt = value
        self._update_usage()

    @property
    def messages(self) -> list[Message]:
        """Get all messages (read-only view)."""
        return self._messages.copy()

    @property
    def message_count(self) -> int:
        """Get the number of messages."""
        return len(self._messages)

    @property
    def compressed_history(self) -> str:
        """Get the compressed history summary."""
        return self._compressed_history

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        if not self._session_id:
            self._session_id = MemoryStore.generate_session_id()
        return self._session_id

    def add_message(self, message: Message) -> CompactionLevel:
        """Add a message to the context.
        
        Automatically checks budget and triggers compaction if needed.
        
        Args:
            message: The message to add.
        
        Returns:
            The compaction level that was triggered (NONE if no compaction).
        """
        self._messages.append(message)
        
        # Track tool usage
        if message.tool_calls:
            for tc in message.tool_calls:
                self._tools_called.add(tc.function.name)
        
        self._update_usage()
        
        # Check if compaction is needed
        level = self._budget.should_compact(self._usage.total)
        
        if level != CompactionLevel.NONE:
            logger.info(f"Compaction needed: {level.name} (usage: {self._usage.total:,} tokens)")
        
        return level

    def add_messages(self, messages: list[Message]) -> CompactionLevel:
        """Add multiple messages to the context.
        
        Args:
            messages: List of messages to add.
        
        Returns:
            The highest compaction level triggered.
        """
        highest_level = CompactionLevel.NONE
        for msg in messages:
            level = self.add_message(msg)
            if level.value > highest_level.value:
                highest_level = level
        return highest_level

    def get_prompt_messages(self) -> list[Message]:
        """Build the message list for an LLM call.
        
        Order:
        1. System prompt (always first)
        2. Compressed history as system message (if any)
        3. Plan state as system message (if any)
        4. Recent messages (last N that fit budget)
        
        Returns:
            List of messages ready for LLM call.
        """
        result: list[Message] = []
        
        # 1. System prompt
        result.append(Message.system(self._system_prompt))
        
        # 2. Compressed history
        if self._compressed_history:
            result.append(
                Message.system(f"[Conversation Summary]\n{self._compressed_history}")
            )
        
        # 3. Plan state
        if self._plan_state:
            result.append(
                Message.system(f"[Current Plan]\n{self._plan_state}")
            )
        
        # 4. Recent messages (already trimmed if needed)
        result.extend(self._messages)
        
        return result

    def set_plan_state(self, plan_state: str) -> None:
        """Update the current plan state.
        
        Args:
            plan_state: Formatted plan state string.
        """
        self._plan_state = plan_state
        self._update_usage()

    async def trigger_compaction(self, level: CompactionLevel) -> bool:
        """Trigger context compaction at the specified level.
        
        Args:
            level: The compaction level to apply.
        
        Returns:
            True if compaction succeeded, False otherwise.
        """
        if level == CompactionLevel.NONE:
            return True
        
        if level != CompactionLevel.EMERGENCY and not self._llm_client:
            logger.error("Cannot perform compaction: no LLM client set")
            return False
        
        logger.info(f"Triggering {level.name} compaction")
        before_tokens = self._usage.total
        start_time = time.perf_counter()
        
        try:
            if level == CompactionLevel.SOFT:
                self._compressed_history = await self._compactor.compact_soft(
                    self._messages, self._llm_client
                )
                # Keep last 8 exchanges
                if len(self._messages) > 16:
                    self._messages = self._messages[-16:]
            
            elif level == CompactionLevel.PREEMPTIVE:
                summary, messages = await self._compactor.compact_preemptive(
                    self._messages, self._compressed_history, self._llm_client
                )
                self._compressed_history = summary
                self._messages = messages
            
            elif level == CompactionLevel.AGGRESSIVE:
                summary, messages = await self._compactor.compact_aggressive(
                    self._messages, self._compressed_history, self._llm_client
                )
                self._compressed_history = summary
                self._messages = messages
            
            elif level == CompactionLevel.EMERGENCY:
                summary, messages = self._compactor.compact_emergency(
                    self._messages, self._compressed_history
                )
                self._compressed_history = summary
                self._messages = messages
            
            self._compaction_count += 1
            self._update_usage()
            
            # Log compaction with structured data
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_compaction(
                level=level.name.lower(),
                before_tokens=before_tokens,
                after_tokens=self._usage.total,
                duration_ms=duration_ms,
            )
            
            logger.info(
                f"Compaction complete: {self._usage.total:,} tokens "
                f"({self._budget.usage_percent(self._usage.total):.1%})"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            
            # If non-emergency compaction failed, try emergency
            if level != CompactionLevel.EMERGENCY:
                logger.warning("Falling back to emergency compaction")
                return await self.trigger_compaction(CompactionLevel.EMERGENCY)
            
            return False

    async def ensure_budget(self, required_tokens: int = 0) -> bool:
        """Ensure there's enough budget for the next operation.
        
        Triggers compaction if needed to free up space.
        
        Args:
            required_tokens: Additional tokens needed beyond current usage.
        
        Returns:
            True if budget is available, False if compaction failed.
        """
        target_usage = self._usage.total + required_tokens
        level = self._budget.should_compact(target_usage)
        
        if level != CompactionLevel.NONE:
            return await self.trigger_compaction(level)
        
        return True

    def get_status(self) -> ContextStatus:
        """Get current context status for display.
        
        Returns:
            ContextStatus with current usage information.
        """
        return ContextStatus(
            total_budget=self._budget.total,
            usable_budget=self._budget.usable_budget,
            used_tokens=self._usage.total,
            percent_used=self._budget.usage_percent(self._usage.total),
            compaction_level=self._budget.should_compact(self._usage.total),
            message_count=len(self._messages),
            compressed_history_tokens=self._usage.compressed_history,
            recent_messages_tokens=self._usage.recent_messages,
            tool_results_tokens=self._usage.tool_results,
        )

    def clear(self, keep_system_prompt: bool = True) -> None:
        """Clear the context.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt.
        """
        self._messages.clear()
        self._compressed_history = ""
        self._plan_state = ""
        if not keep_system_prompt:
            self._system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self._update_usage()
        logger.info("Context cleared")

    def save_checkpoint(self) -> str:
        """Save a checkpoint of current state.
        
        Returns:
            The session ID.
        """
        self._memory_store.save_checkpoint(
            session_id=self.session_id,
            messages=self._messages,
            compressed_history=self._compressed_history,
            plan_state={"plan_state": self._plan_state},
            token_usage=self._usage.to_dict(),
        )
        return self.session_id

    def load_checkpoint(self, session_id: str) -> bool:
        """Load a checkpoint.
        
        Args:
            session_id: The session ID to load.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        checkpoint = self._memory_store.load_checkpoint(session_id)
        if not checkpoint:
            return False
        
        self._session_id = session_id
        self._messages = [Message.model_validate(m) for m in checkpoint.messages]
        self._compressed_history = checkpoint.compressed_history
        self._plan_state = checkpoint.plan_state.get("plan_state", "")
        self._update_usage()
        
        logger.info(f"Loaded checkpoint: {session_id}")
        return True

    def save_session(self, label: str = "") -> str:
        """Save the current session.
        
        Args:
            label: Optional label for the session.
        
        Returns:
            The session ID.
        """
        metadata = SessionMetadata(
            session_id=self.session_id,
            label=label,
            total_tokens_used=self._usage.total,
            compaction_count=self._compaction_count,
            tools_called=list(self._tools_called),
        )
        
        self._memory_store.save_session(
            session_id=self.session_id,
            messages=self._messages,
            summaries={"compressed_history": self._compressed_history},
            plan_state={"plan_state": self._plan_state},
            metadata=metadata,
            label=label,
        )
        
        return self.session_id

    def load_session(self, session_id: str) -> bool:
        """Load a saved session.
        
        Args:
            session_id: The session ID to load.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        session = self._memory_store.load_session(session_id)
        if not session:
            return False
        
        self._session_id = session_id
        self._messages = [Message.model_validate(m) for m in session.messages]
        self._compressed_history = session.summaries.get("compressed_history", "")
        self._plan_state = session.plan_state.get("plan_state", "")
        
        if session.metadata:
            self._compaction_count = session.metadata.compaction_count
            self._tools_called = set(session.metadata.tools_called)
        
        self._update_usage()
        
        logger.info(f"Loaded session: {session_id}")
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        """List available sessions.
        
        Returns:
            List of session summaries.
        """
        sessions = self._memory_store.list_sessions()
        return [
            {
                "session_id": s.session_id,
                "label": s.label,
                "created_at": s.created_at.isoformat(),
                "message_count": s.message_count,
                "total_tokens_used": s.total_tokens_used,
            }
            for s in sessions
        ]

    def _update_usage(self) -> None:
        """Recalculate token usage for all sections."""
        # System prompt
        system_tokens = self._token_counter.count_tokens(self._system_prompt)
        
        # Compressed history
        history_tokens = self._token_counter.count_tokens(self._compressed_history)
        
        # Plan state
        plan_tokens = self._token_counter.count_tokens(self._plan_state)
        
        # Recent messages and tool results
        message_tokens = 0
        tool_tokens = 0
        
        for msg in self._messages:
            tokens = self._token_counter.count_message(msg)
            if msg.role == "tool":
                tool_tokens += tokens
            else:
                message_tokens += tokens
        
        self._usage = ContextUsage(
            system_prompt=system_tokens,
            compressed_history=history_tokens,
            plan_state=plan_tokens,
            recent_messages=message_tokens,
            tool_results=tool_tokens,
        )
