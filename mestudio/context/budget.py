"""Token budget allocation and compaction level determination."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mestudio.core.config import Settings


class CompactionLevel(Enum):
    """Levels of context compaction based on usage thresholds."""
    
    NONE = auto()        # No compaction needed
    SOFT = auto()        # Summarize older messages
    PREEMPTIVE = auto()  # Safety net before aggressive
    AGGRESSIVE = auto()  # Summarize most messages, truncate tool results
    EMERGENCY = auto()   # No LLM call, extractive only


@dataclass
class TokenBudget:
    """Token budget allocation for different context sections.
    
    The sum of sub-budgets equals the total budget exactly.
    Thresholds are calculated against usable_budget (total - response).
    """

    # Total budget after safety margin (131K - 11K = 120K)
    total: int = 120_000
    
    # Sub-budget allocations (must sum to total)
    system_prompt: int = 2_000
    compressed_history: int = 8_000
    recent_messages: int = 16_000
    tool_results: int = 78_000
    response: int = 16_000  # Reserved for LLM response
    
    # Compaction thresholds (percentages of usable_budget)
    soft_threshold: float = 0.65
    preemptive_threshold: float = 0.80
    aggressive_threshold: float = 0.90
    emergency_threshold: float = 0.97

    @property
    def usable_budget(self) -> int:
        """Usable budget for prompt content (total minus response reservation)."""
        return self.total - self.response

    @property
    def soft_tokens(self) -> int:
        """Token threshold for soft compaction."""
        return int(self.usable_budget * self.soft_threshold)

    @property
    def preemptive_tokens(self) -> int:
        """Token threshold for preemptive compaction."""
        return int(self.usable_budget * self.preemptive_threshold)

    @property
    def aggressive_tokens(self) -> int:
        """Token threshold for aggressive compaction."""
        return int(self.usable_budget * self.aggressive_threshold)

    @property
    def emergency_tokens(self) -> int:
        """Token threshold for emergency compaction."""
        return int(self.usable_budget * self.emergency_threshold)

    def available_for_tools(self, current_usage: int) -> int:
        """Calculate remaining tokens available for tool results.
        
        Args:
            current_usage: Current token usage (excluding tool results budget).
        
        Returns:
            Number of tokens available for tool results.
        """
        remaining = self.usable_budget - current_usage
        return max(0, min(remaining, self.tool_results))

    def should_compact(self, current_usage: int) -> CompactionLevel:
        """Determine if compaction is needed based on current token usage.
        
        Args:
            current_usage: Current total token usage.
        
        Returns:
            CompactionLevel indicating what action to take.
        """
        if current_usage >= self.emergency_tokens:
            return CompactionLevel.EMERGENCY
        elif current_usage >= self.aggressive_tokens:
            return CompactionLevel.AGGRESSIVE
        elif current_usage >= self.preemptive_tokens:
            return CompactionLevel.PREEMPTIVE
        elif current_usage >= self.soft_tokens:
            return CompactionLevel.SOFT
        else:
            return CompactionLevel.NONE

    def usage_percent(self, current_usage: int) -> float:
        """Calculate usage as a percentage of usable budget.
        
        Args:
            current_usage: Current token usage.
        
        Returns:
            Usage percentage (0.0 to 1.0+).
        """
        if self.usable_budget == 0:
            return 1.0
        return current_usage / self.usable_budget

    @classmethod
    def from_settings(cls, settings: Settings) -> TokenBudget:
        """Create a TokenBudget from application settings.
        
        Args:
            settings: Application settings instance.
        
        Returns:
            TokenBudget configured from settings.
        """
        return cls(
            total=settings.total_budget,
            system_prompt=settings.system_prompt_budget,
            compressed_history=settings.compressed_history_budget,
            recent_messages=settings.recent_messages_budget,
            tool_results=settings.tool_results_budget,
            response=settings.response_budget,
            soft_threshold=settings.compaction_soft_pct,
            preemptive_threshold=settings.compaction_preemptive_pct,
            aggressive_threshold=settings.compaction_aggressive_pct,
            emergency_threshold=settings.compaction_emergency_pct,
        )


@dataclass
class ContextUsage:
    """Current token usage breakdown by section."""

    system_prompt: int = 0
    compressed_history: int = 0
    plan_state: int = 0
    recent_messages: int = 0
    tool_results: int = 0

    @property
    def total(self) -> int:
        """Total token usage across all sections."""
        return (
            self.system_prompt
            + self.compressed_history
            + self.plan_state
            + self.recent_messages
            + self.tool_results
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for logging/display."""
        return {
            "system_prompt": self.system_prompt,
            "compressed_history": self.compressed_history,
            "plan_state": self.plan_state,
            "recent_messages": self.recent_messages,
            "tool_results": self.tool_results,
            "total": self.total,
        }
