"""File operations specialist agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mestudio.agents.sub_agent import SubAgent, SubAgentConfig
from mestudio.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


# System prompt for the file agent
FILE_AGENT_SYSTEM_PROMPT = """You are a file operations specialist agent. Your job is to read, write, search, and edit local files efficiently.

Guidelines:
- Be precise with file paths. Use absolute paths when possible.
- When reading files, focus on the parts relevant to the task. Don't read entire large files unless necessary.
- When editing files, make minimal targeted changes. Preserve existing formatting and style.
- When searching, use specific patterns to minimize noise.
- Report what you find concisely — the orchestrator has limited context.
- If a file doesn't exist or an operation fails, report the error clearly.

Available tools:
- read_file: Read content from a file (supports line ranges)
- write_file: Write or create a file
- edit_file: Make targeted edits to an existing file
- list_directory: List contents of a directory
- search_files: Search for patterns in files
- find_files: Find files by name pattern

After completing the task, provide a concise summary of what you found or did."""


# Tools available to the file agent
FILE_AGENT_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_files",
    "find_files",
]


class FileAgent(SubAgent):
    """Specialized agent for file operations.
    
    Can read, write, search, and edit local files.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        global_registry: ToolRegistry,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Initialize file agent."""
        config = SubAgentConfig(
            name="FileAgent",
            system_prompt=FILE_AGENT_SYSTEM_PROMPT,
            available_tools=FILE_AGENT_TOOLS,
            max_turns=15,  # File ops may need more turns
            max_context_tokens=32000,
        )
        super().__init__(
            config=config,
            llm_client=llm_client,
            global_registry=global_registry,
            llm_semaphore=llm_semaphore,
        )

    def get_description(self) -> str:
        """Get description of file agent capabilities."""
        return (
            "File operations specialist: reads, writes, searches, and edits "
            "local files. Use for tasks involving file manipulation."
        )


__all__ = ["FileAgent", "FILE_AGENT_SYSTEM_PROMPT", "FILE_AGENT_TOOLS"]
