"""Test context management system."""

import sys
sys.path.insert(0, "c:/clones/MeStudio")

from mestudio.context import (
    ContextManager, ContextStatus,
    TokenBudget, CompactionLevel, ContextUsage,
    TokenCounter, get_token_counter,
    ContextCompactor,
    MemoryStore, SessionData, CheckpointData, SessionSummary,
)
from mestudio.core.models import Message
from mestudio.core.config import get_settings


def test_imports():
    """Test all imports work."""
    print("=== Import Test: PASSED ===")
    print()


def test_token_counter():
    """Test token counter functionality."""
    tc = get_token_counter()
    
    # Basic counting
    tokens = tc.count_tokens("Hello world")
    print(f'Token count for "Hello world": {tokens}')
    assert tokens > 0
    
    # Message counting
    msg = Message.user("What is the weather today?")
    msg_tokens = tc.count_message(msg)
    print(f"Message tokens: {msg_tokens}")
    assert msg_tokens > tokens  # Should include overhead
    
    # Truncation
    long_text = "word " * 1000
    truncated = tc.truncate_to_tokens(long_text, 50)
    print(f"Truncated to ~50 tokens: {len(truncated.split())} words")
    
    # Middle truncation
    middle = tc.truncate_middle(long_text, 100)
    assert "..." in middle
    print("Middle truncation: works")
    
    print("Token counter: PASSED\n")


def test_budget():
    """Test budget calculations."""
    settings = get_settings()
    budget = TokenBudget.from_settings(settings)
    
    print(f"Budget: total={budget.total:,}, usable={budget.usable_budget:,}")
    print(f"Thresholds: soft={budget.soft_threshold:,}, preemptive={budget.preemptive_threshold:,}")
    print(f"            aggressive={budget.aggressive_threshold:,}, emergency={budget.emergency_threshold:,}")
    
    # Test compaction level detection
    assert budget.should_compact(int(budget.usable_budget * 0.6)) == CompactionLevel.NONE
    assert budget.should_compact(int(budget.usable_budget * 0.72)) == CompactionLevel.SOFT
    assert budget.should_compact(int(budget.usable_budget * 0.82)) == CompactionLevel.PREEMPTIVE
    assert budget.should_compact(int(budget.usable_budget * 0.92)) == CompactionLevel.AGGRESSIVE
    assert budget.should_compact(int(budget.usable_budget * 0.98)) == CompactionLevel.EMERGENCY
    
    print("\nCompaction at 60%:", budget.should_compact(int(budget.usable_budget * 0.6)).name)
    print("Compaction at 72%:", budget.should_compact(int(budget.usable_budget * 0.72)).name)
    print("Compaction at 82%:", budget.should_compact(int(budget.usable_budget * 0.82)).name)
    print("Compaction at 92%:", budget.should_compact(int(budget.usable_budget * 0.92)).name)
    
    print("Budget: PASSED\n")


def test_context_usage():
    """Test context usage tracking."""
    usage = ContextUsage(
        system_prompt=500,
        compressed_history=1000,
        plan_state=200,
        recent_messages=3000,
        tool_results=800,
    )
    
    assert usage.total == 5500
    print(f"Context usage total: {usage.total}")
    
    d = usage.to_dict()
    assert d["total"] == 5500
    print("Context usage: PASSED\n")


def test_context_manager():
    """Test the main context manager."""
    print("=== Context Manager Test ===")
    
    cm = ContextManager()
    print(f"Default system prompt length: {len(cm.system_prompt)} chars")
    print(f"Session ID: {cm.session_id}")
    
    # Add messages
    level = cm.add_message(Message.user("Hello!"))
    assert level == CompactionLevel.NONE
    
    cm.add_message(Message.assistant("Hi there! How can I help you?"))
    cm.add_message(Message.user("What can you do?"))
    
    status = cm.get_status()
    print(f"\nStatus: {status.message_count} messages, {status.used_tokens:,} tokens")
    print(f"        {status.percent_used:.2%} used, level: {status.compaction_level.name}")
    
    assert status.message_count == 3
    assert status.used_tokens > 0
    assert status.compaction_level == CompactionLevel.NONE
    
    # Get prompt messages
    prompt_msgs = cm.get_prompt_messages()
    print(f"\nPrompt messages: {len(prompt_msgs)} (incl. system)")
    assert len(prompt_msgs) == 4  # system + 3 messages
    assert prompt_msgs[0].role == "system"
    
    # Test checkpointing
    session_id = cm.save_checkpoint()
    print(f"Checkpoint saved: {session_id}")
    
    # Clear and reload
    cm.clear()
    print(f"After clear: {cm.message_count} messages")
    assert cm.message_count == 0
    
    loaded = cm.load_checkpoint(session_id)
    print(f"Checkpoint loaded: {loaded}, messages: {cm.message_count}")
    assert loaded
    assert cm.message_count == 3
    
    print("Context Manager: PASSED\n")


def test_memory_store():
    """Test session persistence."""
    store = MemoryStore()
    
    # Generate unique session ID
    sid1 = MemoryStore.generate_session_id()
    sid2 = MemoryStore.generate_session_id()
    assert sid1 != sid2  # UUIDs should be unique
    print(f"Session IDs: {sid1[:20]}... (unique: {sid1 != sid2})")
    
    print("Memory store: PASSED\n")


def test_compactor():
    """Test compactor (without LLM - just structure)."""
    compactor = ContextCompactor()
    
    # Test preservable info extraction
    messages = [
        Message.user("Please search for files containing 'test'"),
        Message.assistant("I found 5 files."),
        Message.user("Now edit the main.py file"),
        Message.assistant("Done! I edited line 42."),
    ]
    
    info = compactor.extract_preservable_info(messages)
    print(f"Extracted preservable info: {len(info)} items")
    
    # Test emergency compaction (no LLM needed)
    summary, remaining = compactor.compact_emergency(messages, "")
    print(f"Emergency compaction: kept {len(remaining)} messages")
    print(f"Summary length: {len(summary)} chars")
    
    print("Compactor: PASSED\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("MeStudio Context System Tests")
    print("=" * 50 + "\n")
    
    test_imports()
    test_token_counter()
    test_budget()
    test_context_usage()
    test_context_manager()
    test_memory_store()
    test_compactor()
    
    print("=" * 50)
    print("=== ALL CONTEXT TESTS PASSED ===")
    print("=" * 50)


if __name__ == "__main__":
    main()
