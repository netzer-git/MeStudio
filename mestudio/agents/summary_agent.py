"""Summarization specialist agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mestudio.agents.sub_agent import SubAgent, SubAgentConfig
from mestudio.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


# System prompt for the summary agent
SUMMARY_AGENT_SYSTEM_PROMPT = """You are a summarization specialist agent. Your job is to condense large amounts of text into concise, accurate summaries.

Guidelines:
- Preserve critical information:
  - Key facts and decisions
  - File paths and code locations
  - Important code patterns and structures
  - Error messages and their causes
  - Action items and next steps
- Remove:
  - Repetition and redundancy
  - Filler text and verbose explanations
  - Unnecessary formatting
  - Outdated or superseded information
- Maintain accuracy — never hallucinate or invent details.
- Use bullet points and structure for clarity.
- If summarizing code, preserve essential logic and interfaces.
- Indicate when information has been omitted ("[additional details omitted]")

Available tools:
- read_file: Read content from files to summarize

When given text directly in the task, summarize it without using tools.
When asked to summarize a file, read it first, then provide the summary."""


# Tools available to the summary agent
SUMMARY_AGENT_TOOLS = [
    "read_file",
]


class SummaryAgent(SubAgent):
    """Specialized agent for text summarization.
    
    Condenses large text into concise summaries while preserving key information.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        global_registry: ToolRegistry,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize summary agent."""
        config = SubAgentConfig(
            name="SummaryAgent",
            system_prompt=SUMMARY_AGENT_SYSTEM_PROMPT,
            available_tools=SUMMARY_AGENT_TOOLS,
            max_turns=5,  # Summaries are usually quick
            max_context_tokens=32000,
        )
        super().__init__(
            config=config,
            llm_client=llm_client,
            global_registry=global_registry,
            llm_semaphore=llm_semaphore,
        )

    def get_description(self) -> str:
        """Get description of summary agent capabilities."""
        return (
            "Summarization specialist: condenses large text into concise, "
            "accurate summaries. Use for compacting verbose content."
        )


__all__ = ["SummaryAgent", "SUMMARY_AGENT_SYSTEM_PROMPT", "SUMMARY_AGENT_TOOLS"]
