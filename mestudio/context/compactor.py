"""Summarization and compression logic for context management."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from mestudio.context.token_counter import get_token_counter
from mestudio.core.models import Message

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


# Prompts for different compaction levels
SOFT_COMPACTION_PROMPT = """Summarize the following conversation, preserving:
- Key decisions made
- File paths and code changes
- Errors encountered
- Current task state and progress
- Important context for continuing the work

Be concise but complete. Output only the summary, no preamble."""

AGGRESSIVE_COMPACTION_PROMPT = """Create a very concise summary of this conversation. 
Focus ONLY on:
- The main task/goal
- Critical decisions
- File paths modified
- Current state

Keep it under 500 tokens. Output only the summary."""


class ContextCompactor:
    """Handles context compaction at different levels.
    
    Compaction levels:
    - SOFT: Summarize older messages, keep recent ones verbatim
    - PREEMPTIVE: Like soft but also truncate older tool results
    - AGGRESSIVE: Summarize most messages, heavily truncate tool results
    - EMERGENCY: No LLM call, extractive only
    """

    def __init__(self) -> None:
        """Initialize the compactor."""
        self._token_counter = get_token_counter()

    async def compact_soft(
        self,
        messages: list[Message],
        llm_client: LMStudioClient,
        keep_recent: int = 8,
    ) -> str:
        """Perform soft compaction by summarizing older messages.
        
        Args:
            messages: All messages in the conversation.
            llm_client: LLM client for summarization.
            keep_recent: Number of recent exchanges to keep verbatim.
        
        Returns:
            Summary string of older messages.
        """
        if len(messages) <= keep_recent * 2:
            # Not enough messages to compact
            return ""
        
        # Split messages
        older_messages = messages[:-keep_recent * 2]
        
        if not older_messages:
            return ""
        
        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(older_messages)
        
        summary_request = [
            Message.system(SOFT_COMPACTION_PROMPT),
            Message.user(f"Conversation to summarize:\n\n{conversation_text}"),
        ]
        
        try:
            response = await llm_client.chat(summary_request, stream=False)
            summary = response.content or ""
            
            token_count = self._token_counter.count_tokens(summary)
            logger.info(f"Soft compaction: summarized {len(older_messages)} messages into {token_count} tokens")
            
            return summary
        except Exception as e:
            logger.error(f"Soft compaction failed: {e}")
            # Fall back to extractive summary
            return self._extract_fallback_summary(messages)

    async def compact_preemptive(
        self,
        messages: list[Message],
        existing_summary: str,
        llm_client: LMStudioClient,
        keep_recent: int = 4,
    ) -> tuple[str, list[Message]]:
        """Perform preemptive compaction before things get critical.
        
        Like soft compaction but also truncates tool results in older messages.
        
        Args:
            messages: All messages in the conversation.
            existing_summary: Existing compressed history if any.
            llm_client: LLM client for summarization.
            keep_recent: Number of recent exchanges to keep.
        
        Returns:
            Tuple of (summary string, modified messages list).
        """
        # First truncate tool results in older messages
        modified_messages = self._truncate_old_tool_results(
            messages, keep_recent=keep_recent * 2, max_tokens=500
        )
        
        # Prepare content for summarization
        older_messages = modified_messages[:-keep_recent * 2] if len(modified_messages) > keep_recent * 2 else []
        
        if not older_messages and not existing_summary:
            return "", modified_messages
        
        # Build content to summarize
        content_parts = []
        if existing_summary:
            content_parts.append(f"Previous summary:\n{existing_summary}")
        if older_messages:
            content_parts.append(
                f"Additional conversation:\n{self._format_messages_for_summary(older_messages)}"
            )
        
        conversation_text = "\n\n".join(content_parts)
        
        summary_request = [
            Message.system(SOFT_COMPACTION_PROMPT),
            Message.user(f"Content to summarize:\n\n{conversation_text}"),
        ]
        
        try:
            response = await llm_client.chat(summary_request, stream=False)
            summary = response.content or ""
            
            # Return only recent messages
            recent_messages = modified_messages[-keep_recent * 2:] if len(modified_messages) > keep_recent * 2 else modified_messages
            
            token_count = self._token_counter.count_tokens(summary)
            logger.info(f"Preemptive compaction: {token_count} token summary")
            
            return summary, recent_messages
        except Exception as e:
            logger.error(f"Preemptive compaction failed: {e}")
            return existing_summary or self._extract_fallback_summary(messages), modified_messages

    async def compact_aggressive(
        self,
        messages: list[Message],
        existing_summary: str,
        llm_client: LMStudioClient,
        keep_recent: int = 2,
    ) -> tuple[str, list[Message]]:
        """Perform aggressive compaction under pressure.
        
        Summarizes most messages and heavily truncates tool results.
        
        Args:
            messages: All messages in the conversation.
            existing_summary: Existing compressed history.
            llm_client: LLM client for summarization.
            keep_recent: Number of recent exchanges to keep (default 2).
        
        Returns:
            Tuple of (summary string, modified messages list).
        """
        # Heavily truncate all tool results except in most recent messages
        modified_messages = self._truncate_old_tool_results(
            messages, keep_recent=keep_recent * 2, max_tokens=200
        )
        
        # Everything except last few messages goes into summary
        older_messages = modified_messages[:-keep_recent * 2] if len(modified_messages) > keep_recent * 2 else []
        
        content_parts = []
        if existing_summary:
            content_parts.append(f"Existing summary:\n{existing_summary}")
        if older_messages:
            content_parts.append(
                f"Recent conversation:\n{self._format_messages_for_summary(older_messages)}"
            )
        
        if not content_parts:
            return "", modified_messages
        
        conversation_text = "\n\n".join(content_parts)
        
        summary_request = [
            Message.system(AGGRESSIVE_COMPACTION_PROMPT),
            Message.user(f"Content:\n\n{conversation_text}"),
        ]
        
        try:
            response = await llm_client.chat(summary_request, stream=False)
            summary = response.content or ""
            
            # Ensure summary fits budget
            summary = self._token_counter.truncate_to_tokens(summary, 1000)
            
            recent_messages = modified_messages[-keep_recent * 2:] if len(modified_messages) > keep_recent * 2 else modified_messages
            
            token_count = self._token_counter.count_tokens(summary)
            logger.info(f"Aggressive compaction: {token_count} token summary")
            
            return summary, recent_messages
        except Exception as e:
            logger.error(f"Aggressive compaction failed: {e}")
            return self._extract_fallback_summary(messages), modified_messages[-keep_recent * 2:]

    def compact_emergency(
        self,
        messages: list[Message],
        existing_summary: str,
    ) -> tuple[str, list[Message]]:
        """Perform emergency compaction without LLM call.
        
        This is a last resort when context is critically full.
        Uses extractive methods only.
        
        Args:
            messages: All messages in the conversation.
            existing_summary: Existing compressed history.
        
        Returns:
            Tuple of (summary string, minimal messages list).
        """
        logger.warning("Emergency compaction triggered - using extractive methods only")
        
        # If we have an existing summary, use it
        if existing_summary:
            summary = existing_summary
        else:
            # Build a minimal summary from preserved info
            preserved = self.extract_preservable_info(messages)
            summary = self._build_emergency_summary(preserved)
        
        # Keep only the last exchange (user + assistant)
        recent_messages: list[Message] = []
        if messages:
            # Find last user message and any following assistant messages
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].role == "user":
                    recent_messages = messages[i:]
                    break
        
        # Limit to last 2 messages if we have more
        if len(recent_messages) > 2:
            recent_messages = recent_messages[-2:]
        
        # Strip tool results from remaining messages
        cleaned_messages = []
        for msg in recent_messages:
            if msg.role == "tool":
                cleaned_messages.append(
                    Message.tool_result(
                        msg.tool_call_id or "",
                        "[Tool result removed during emergency compaction]"
                    )
                )
            else:
                cleaned_messages.append(msg)
        
        token_count = self._token_counter.count_tokens(summary)
        logger.warning(f"Emergency compaction complete: {token_count} token summary, {len(cleaned_messages)} messages")
        
        return summary, cleaned_messages

    def extract_preservable_info(self, messages: list[Message]) -> dict[str, Any]:
        """Extract key information that should be preserved during compaction.
        
        Args:
            messages: All messages to extract from.
        
        Returns:
            Dictionary with extracted information.
        """
        info: dict[str, Any] = {
            "file_paths": set(),
            "errors": [],
            "decisions": [],
            "task_goal": None,
            "plan_state": None,
        }
        
        # Regex for file paths (Unix and Windows style)
        path_pattern = re.compile(
            r'(?:[a-zA-Z]:)?(?:[/\\][\w\-. ]+)+(?:\.\w+)?'
        )
        
        # Keywords for decisions
        decision_keywords = ["decided", "chose", "will use", "using", "selected", "picked"]
        
        for i, msg in enumerate(messages):
            if not msg.content:
                continue
            
            content = msg.content
            
            # Extract file paths
            paths = path_pattern.findall(content)
            info["file_paths"].update(paths)
            
            # First user message is likely the task goal
            if info["task_goal"] is None and msg.role == "user":
                # Take first 200 chars as goal
                info["task_goal"] = content[:200].strip()
            
            # Look for error messages
            if "error" in content.lower() or "failed" in content.lower():
                # Extract a snippet around the error
                error_snippet = content[:300] if len(content) > 300 else content
                info["errors"].append(error_snippet)
            
            # Look for decisions
            content_lower = content.lower()
            for keyword in decision_keywords:
                if keyword in content_lower and msg.role == "assistant":
                    # Extract sentence containing the keyword
                    sentences = content.split(".")
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            info["decisions"].append(sentence.strip())
                            break
        
        # Convert set to list for JSON serialization
        info["file_paths"] = list(info["file_paths"])
        
        return info

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Format messages as text for summarization."""
        parts = []
        for msg in messages:
            role = msg.role.upper()
            content = msg.content or ""
            
            # Truncate very long content
            if len(content) > 1000:
                content = content[:500] + "\n...[truncated]...\n" + content[-300:]
            
            if msg.tool_calls:
                tool_info = ", ".join(tc.function.name for tc in msg.tool_calls)
                parts.append(f"{role}: [Called tools: {tool_info}]")
            elif msg.tool_call_id:
                parts.append(f"TOOL RESULT ({msg.tool_call_id[:8]}...): {content}")
            else:
                parts.append(f"{role}: {content}")
        
        return "\n\n".join(parts)

    def _truncate_old_tool_results(
        self,
        messages: list[Message],
        keep_recent: int,
        max_tokens: int,
    ) -> list[Message]:
        """Truncate tool results in older messages.
        
        Args:
            messages: Messages to process.
            keep_recent: Number of recent messages to leave unchanged.
            max_tokens: Max tokens for truncated tool results.
        
        Returns:
            New list with truncated tool results.
        """
        if len(messages) <= keep_recent:
            return messages
        
        result = []
        cutoff = len(messages) - keep_recent
        
        for i, msg in enumerate(messages):
            if i >= cutoff:
                # Keep recent messages unchanged
                result.append(msg)
            elif msg.role == "tool" and msg.content:
                # Truncate old tool results
                truncated = self._token_counter.truncate_middle(
                    msg.content, max_tokens
                )
                result.append(
                    Message.tool_result(msg.tool_call_id or "", truncated)
                )
            else:
                result.append(msg)
        
        return result

    def _extract_fallback_summary(self, messages: list[Message]) -> str:
        """Create a fallback summary without LLM when compaction fails."""
        preserved = self.extract_preservable_info(messages)
        return self._build_emergency_summary(preserved)

    def _build_emergency_summary(self, preserved: dict[str, Any]) -> str:
        """Build an emergency summary from preserved info."""
        parts = []
        
        if preserved.get("task_goal"):
            parts.append(f"Task: {preserved['task_goal']}")
        
        if preserved.get("file_paths"):
            paths = preserved["file_paths"][:10]  # Limit to 10 paths
            parts.append(f"Files involved: {', '.join(paths)}")
        
        if preserved.get("decisions"):
            decisions = preserved["decisions"][:3]  # Limit to 3 decisions
            parts.append("Key decisions: " + "; ".join(decisions))
        
        if preserved.get("errors"):
            parts.append(f"Errors encountered: {len(preserved['errors'])} error(s)")
        
        return "\n".join(parts) if parts else "No context summary available."
