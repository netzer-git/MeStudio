"""Token counting utilities using tiktoken."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken
from loguru import logger

if TYPE_CHECKING:
    from mestudio.core.models import Message


class TokenCounter:
    """Count tokens using tiktoken with cl100k_base encoding.
    
    cl100k_base is used as a reasonable proxy for gpt-oss-20b tokenization.
    Actual token counts may vary by ±10%, which is acceptable for budget management.
    """

    # Overhead tokens per message for role/delimiter formatting
    MESSAGE_OVERHEAD = 4  # ~4 tokens for role, separators

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """Initialize the token counter.
        
        Args:
            encoding_name: The tiktoken encoding to use.
        """
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for.
        
        Returns:
            Number of tokens in the text.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message(self, message: Message) -> int:
        """Count tokens in a single message including overhead.
        
        Args:
            message: The message to count tokens for.
        
        Returns:
            Number of tokens including message formatting overhead.
        """
        tokens = self.MESSAGE_OVERHEAD  # role and delimiters
        
        if message.content:
            tokens += self.count_tokens(message.content)
        
        if message.name:
            tokens += self.count_tokens(message.name) + 1  # +1 for name separator
        
        if message.tool_calls:
            for tc in message.tool_calls:
                tokens += self.count_tokens(tc.function.name)
                tokens += self.count_tokens(tc.function.arguments)
                tokens += 4  # overhead for tool call structure
        
        if message.tool_call_id:
            tokens += self.count_tokens(message.tool_call_id)
        
        return tokens

    def count_messages(self, messages: list[Message]) -> int:
        """Count total tokens for a list of messages.
        
        Args:
            messages: List of messages to count tokens for.
        
        Returns:
            Total number of tokens including all message overhead.
        """
        total = 3  # Base overhead for message array structure
        for message in messages:
            total += self.count_message(message)
        return total

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "\n... [truncated]",
    ) -> str:
        """Truncate text to fit within a token budget.
        
        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens allowed.
            suffix: Suffix to append when truncation occurs.
        
        Returns:
            Truncated text with suffix if truncation was needed,
            or original text if it fits within budget.
        """
        if not text:
            return text
        
        tokens = self._encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Reserve space for suffix
        suffix_tokens = self._encoding.encode(suffix)
        available = max_tokens - len(suffix_tokens)
        
        if available <= 0:
            # Can't even fit the suffix, just return truncated text
            return self._encoding.decode(tokens[:max_tokens])
        
        truncated_tokens = tokens[:available]
        truncated_text = self._encoding.decode(truncated_tokens)
        
        return truncated_text + suffix

    def truncate_middle(
        self,
        text: str,
        max_tokens: int,
        separator: str = "\n\n... [middle content truncated] ...\n\n",
    ) -> str:
        """Truncate text from the middle, keeping start and end.
        
        Useful for preserving context from both beginning and end of
        long content like file contents or tool results.
        
        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens allowed.
            separator: Text to insert in place of removed middle.
        
        Returns:
            Text with middle truncated, or original if fits.
        """
        if not text:
            return text
        
        tokens = self._encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        separator_tokens = self._encoding.encode(separator)
        available = max_tokens - len(separator_tokens)
        
        if available <= 0:
            return self.truncate_to_tokens(text, max_tokens)
        
        # Split available tokens between start and end
        start_size = available // 2
        end_size = available - start_size
        
        start_tokens = tokens[:start_size]
        end_tokens = tokens[-end_size:]
        
        start_text = self._encoding.decode(start_tokens)
        end_text = self._encoding.decode(end_tokens)
        
        return start_text + separator + end_text


# Global singleton instance
_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Get the global TokenCounter instance."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens in text."""
    return get_token_counter().count_tokens(text)


def count_messages(messages: list[Message]) -> int:
    """Convenience function to count tokens in messages."""
    return get_token_counter().count_messages(messages)
