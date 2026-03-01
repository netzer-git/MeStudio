"""Agent components.

Sub-agents are specialized agents that handle focused tasks:
- FileAgent: File operations (read, write, search, edit)
- SearchAgent: Web research (search, read webpages)
- SummaryAgent: Text summarization
"""

from mestudio.agents.sub_agent import (
    SubAgent,
    SubAgentConfig,
    SubAgentError,
    SubAgentSpawner,
)
from mestudio.agents.file_agent import (
    FileAgent,
    FILE_AGENT_SYSTEM_PROMPT,
    FILE_AGENT_TOOLS,
)
from mestudio.agents.search_agent import (
    SearchAgent,
    SEARCH_AGENT_SYSTEM_PROMPT,
    SEARCH_AGENT_TOOLS,
)
from mestudio.agents.summary_agent import (
    SummaryAgent,
    SUMMARY_AGENT_SYSTEM_PROMPT,
    SUMMARY_AGENT_TOOLS,
)

__all__ = [
    # Base classes
    "SubAgent",
    "SubAgentConfig",
    "SubAgentError",
    "SubAgentSpawner",
    # Specialized agents
    "FileAgent",
    "FILE_AGENT_SYSTEM_PROMPT",
    "FILE_AGENT_TOOLS",
    "SearchAgent",
    "SEARCH_AGENT_SYSTEM_PROMPT",
    "SEARCH_AGENT_TOOLS",
    "SummaryAgent",
    "SUMMARY_AGENT_SYSTEM_PROMPT",
    "SUMMARY_AGENT_TOOLS",
]
