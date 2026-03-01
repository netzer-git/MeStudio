"""Web search specialist agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mestudio.agents.sub_agent import SubAgent, SubAgentConfig
from mestudio.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


# System prompt for the search agent
SEARCH_AGENT_SYSTEM_PROMPT = """You are a web research specialist agent. Your job is to search the web and extract relevant information.

Guidelines:
- Search for specific, targeted queries. Refine queries based on results.
- Read webpages to extract detailed information when search snippets aren't enough.
- Synthesize information from multiple sources when appropriate.
- Return a structured summary with:
  - Key facts and findings
  - Source URLs for verification
  - Relevance to the original query
- Be concise — the orchestrator has limited context.
- If search results are poor, try alternative query formulations.
- Clearly distinguish between facts and inferences.

Available tools:
- web_search: Search DuckDuckGo for information
- read_webpage: Read and extract content from a webpage

After completing research, provide a clear, organized summary of your findings."""


# Tools available to the search agent
SEARCH_AGENT_TOOLS = [
    "web_search",
    "read_webpage",
]


class SearchAgent(SubAgent):
    """Specialized agent for web research.
    
    Searches the web and extracts relevant information.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        global_registry: ToolRegistry,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize search agent."""
        config = SubAgentConfig(
            name="SearchAgent",
            system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
            available_tools=SEARCH_AGENT_TOOLS,
            max_turns=8,  # Search usually resolves quickly
            max_context_tokens=32000,
        )
        super().__init__(
            config=config,
            llm_client=llm_client,
            global_registry=global_registry,
            llm_semaphore=llm_semaphore,
        )

    def get_description(self) -> str:
        """Get description of search agent capabilities."""
        return (
            "Web research specialist: searches the web and extracts relevant "
            "information. Use for tasks requiring online research."
        )


__all__ = ["SearchAgent", "SEARCH_AGENT_SYSTEM_PROMPT", "SEARCH_AGENT_TOOLS"]
