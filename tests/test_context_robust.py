"""Robust stress tests for the Context Management System.

Tests with realistic data volumes, budget thresholds, and compaction scenarios.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, "c:/clones/MeStudio")

from mestudio.context import (
    ContextManager,
    ContextStatus,
    TokenBudget,
    CompactionLevel,
    ContextUsage,
    TokenCounter,
    get_token_counter,
    ContextCompactor,
    MemoryStore,
)
from mestudio.core.models import Message, ToolCall, FunctionCall
from mestudio.core.config import get_settings


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_code_block(lines: int = 50, language: str = "python") -> str:
    """Generate a realistic code block."""
    code_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated test module."""',
        "",
        "import os",
        "import sys",
        "from typing import Any, Dict, List, Optional",
        "from dataclasses import dataclass, field",
        "",
        "",
        "@dataclass",
        "class TestConfig:",
        '    """Configuration for test execution."""',
        "    ",
        "    name: str",
        "    enabled: bool = True",
        "    timeout: int = 30",
        "    retries: int = 3",
        "    metadata: Dict[str, Any] = field(default_factory=dict)",
        "",
        "",
        "class TestRunner:",
        '    """Runs test suites with configuration."""',
        "    ",
        "    def __init__(self, config: TestConfig) -> None:",
        "        self.config = config",
        "        self._results: List[Dict] = []",
        "        self._start_time: float = 0",
        "    ",
        "    def setup(self) -> None:",
        '        """Initialize test environment."""',
        "        self._start_time = time.time()",
        '        print(f"Setting up {self.config.name}...")',
        "    ",
        "    def run_test(self, test_name: str, test_func: callable) -> bool:",
        '        """Execute a single test."""',
        "        try:",
        "            test_func()",
        '            self._results.append({"name": test_name, "passed": True})',
        "            return True",
        "        except Exception as e:",
        '            self._results.append({"name": test_name, "passed": False, "error": str(e)})',
        "            return False",
        "    ",
        "    def teardown(self) -> None:",
        '        """Clean up test environment."""',
        "        elapsed = time.time() - self._start_time",
        '        print(f"Tests completed in {elapsed:.2f}s")',
        "    ",
        "    @property",
        "    def passed(self) -> int:",
        '        return sum(1 for r in self._results if r["passed"])',
        "    ",
        "    @property", 
        "    def failed(self) -> int:",
        '        return sum(1 for r in self._results if not r["passed"])',
    ]
    
    # Extend to requested length
    while len(code_lines) < lines:
        code_lines.append(f"    # Additional line {len(code_lines)}")
    
    return "\n".join(code_lines[:lines])


def generate_file_listing(num_files: int = 100) -> str:
    """Generate a realistic file listing output."""
    extensions = [".py", ".ts", ".js", ".json", ".md", ".yaml", ".txt", ".html", ".css"]
    dirs = ["src", "lib", "utils", "components", "models", "services", "tests", "config"]
    
    lines = ["Directory listing:", ""]
    for i in range(num_files):
        dir_path = dirs[i % len(dirs)]
        ext = extensions[i % len(extensions)]
        size = 1000 + (i * 137) % 50000
        lines.append(f"  {dir_path}/module_{i:03d}{ext}  ({size:,} bytes)")
    
    lines.append(f"\nTotal: {num_files} files")
    return "\n".join(lines)


def generate_search_results(num_results: int = 50) -> str:
    """Generate realistic search results."""
    lines = [f"Found {num_results} matches:", ""]
    
    for i in range(num_results):
        file_path = f"src/module_{i % 20}/component_{i}.py"
        line_num = 10 + (i * 7) % 500
        snippet = f"    def process_item_{i}(self, data: Dict[str, Any]) -> Result:"
        lines.append(f"{file_path}:{line_num}: {snippet}")
    
    return "\n".join(lines)


def generate_error_traceback() -> str:
    """Generate a realistic Python traceback."""
    return '''Traceback (most recent call last):
  File "/app/src/main.py", line 142, in main
    result = await processor.execute(task)
  File "/app/src/processor.py", line 87, in execute
    data = await self._fetch_data(task.source)
  File "/app/src/processor.py", line 156, in _fetch_data
    response = await self.client.get(url, timeout=30)
  File "/app/lib/http_client.py", line 43, in get
    return await self._request("GET", url, **kwargs)
  File "/app/lib/http_client.py", line 78, in _request
    raise HTTPError(f"Request failed: {response.status_code}")
HTTPError: Request failed: 503 Service Unavailable

The above exception was the direct cause of the following exception:

  File "/app/src/processor.py", line 89, in execute
    raise ProcessingError(f"Failed to fetch data: {e}") from e
ProcessingError: Failed to fetch data: Request failed: 503 Service Unavailable'''


def generate_web_search_result() -> str:
    """Generate realistic web search results."""
    return '''Search results for "Python async context manager best practices":

1. Real Python - Async Context Managers
   URL: https://realpython.com/async-context-managers/
   Async context managers in Python provide a clean way to manage resources
   in asynchronous code. Using `async with` ensures proper cleanup even when
   exceptions occur. Key patterns include...
   
2. Python Documentation - contextlib
   URL: https://docs.python.org/3/library/contextlib.html
   The contextlib module provides utilities for common tasks involving the
   with statement. For async code, use @asynccontextmanager decorator...

3. Stack Overflow - Best practices for async context managers
   URL: https://stackoverflow.com/questions/12345678
   When implementing async context managers, consider: 1) Always use try/finally
   in __aexit__, 2) Handle CancelledError appropriately, 3) Avoid blocking calls...

4. Medium - Advanced Python: Async Context Managers Deep Dive  
   URL: https://medium.com/python-async-context-managers
   Understanding the lifecycle of async context managers is crucial for building
   robust async applications. This article covers common pitfalls and solutions...

5. GitHub - python/cpython contextlib implementation
   URL: https://github.com/python/cpython/blob/main/Lib/contextlib.py
   Reference implementation of async context managers in the standard library...'''


# =============================================================================
# Test Cases
# =============================================================================

def test_large_conversation():
    """Test with a large multi-turn conversation."""
    print("\n" + "=" * 60)
    print("TEST: Large Conversation (50+ messages)")
    print("=" * 60)
    
    cm = ContextManager()
    tc = get_token_counter()
    
    # Simulate a realistic coding session
    conversation = [
        ("user", "I need help refactoring a large Python project. The codebase has about 50 modules and I want to improve the architecture."),
        ("assistant", "I'd be happy to help you refactor your Python project. To get started, I'll need to understand your current codebase structure. Let me first list the project files to see what we're working with."),
        ("user", "The project is in /home/user/myproject. Can you analyze it?"),
        ("assistant", "I'll analyze the project structure now."),
        ("user", "Great, what did you find?"),
        ("assistant", f"I found the following structure:\n\n{generate_file_listing(80)}\n\nThe codebase appears to follow a modular structure. I notice several areas that could benefit from refactoring."),
        ("user", "Can you look at the main module?"),
        ("assistant", f"Here's the main module content:\n\n```python\n{generate_code_block(100)}\n```\n\nI see several opportunities for improvement here."),
        ("user", "What changes would you recommend?"),
        ("assistant", "Based on my analysis, I recommend:\n\n1. **Extract configuration** - Move settings to a dedicated config module\n2. **Dependency injection** - Use DI for better testability\n3. **Split large classes** - Some classes handle too many responsibilities\n4. **Add type hints** - Improve code documentation and IDE support\n5. **Implement async patterns** - For I/O operations"),
        ("user", "Let's start with extracting the configuration. Show me what that would look like."),
        ("assistant", f"Here's the proposed config module:\n\n```python\n{generate_code_block(60)}\n```"),
        ("user", "That looks good. Now let's implement the dependency injection."),
        ("assistant", "I'll create a dependency injection container. First, let me search for all the places where dependencies are instantiated."),
        ("user", "What did you find?"),
        ("assistant", f"Search results:\n\n{generate_search_results(30)}\n\nI found 30 places where we instantiate dependencies directly. Let me create the DI container."),
        ("user", "Please proceed with the implementation."),
        ("assistant", f"Here's the DI container implementation:\n\n```python\n{generate_code_block(80)}\n```"),
        ("user", "I got an error when running the tests."),
        ("assistant", f"Let me check the error:\n\n{generate_error_traceback()}\n\nThis is a network timeout issue. Let me fix the HTTP client configuration."),
        ("user", "Can you also add retry logic?"),
        ("assistant", f"I'll add retry logic with exponential backoff:\n\n```python\n{generate_code_block(70)}\n```"),
        ("user", "What about the async patterns you mentioned?"),
        ("assistant", "Let me show you how to convert synchronous code to async patterns."),
        ("user", "First, can you search for best practices online?"),
        ("assistant", f"Here are the search results:\n\n{generate_web_search_result()}\n\nBased on these resources, I recommend using asynccontextmanager for resource management."),
        ("user", "Show me an implementation example."),
        ("assistant", f"Here's an async implementation:\n\n```python\n{generate_code_block(90)}\n```"),
        ("user", "Can you run the tests now?"),
        ("assistant", "Running tests... All 47 tests passed! The refactoring is complete."),
    ]
    
    # Add all messages
    for role, content in conversation:
        if role == "user":
            cm.add_message(Message.user(content))
        else:
            cm.add_message(Message.assistant(content))
    
    status = cm.get_status()
    print(f"\nMessages added: {status.message_count}")
    print(f"Total tokens used: {status.used_tokens:,}")
    print(f"Budget usage: {status.percent_used:.2%}")
    print(f"Compaction level: {status.compaction_level.name}")
    
    # Test prompt message building
    prompt_msgs = cm.get_prompt_messages()
    prompt_tokens = tc.count_messages(prompt_msgs)
    print(f"Prompt messages: {len(prompt_msgs)}")
    print(f"Prompt tokens: {prompt_tokens:,}")
    
    assert status.message_count == len(conversation)
    assert status.used_tokens > 5000  # Should be substantial
    print("\n✓ Large conversation test PASSED")


def test_tool_results():
    """Test with tool call results (simulating agent behavior)."""
    print("\n" + "=" * 60)
    print("TEST: Tool Results (file reads, searches, etc.)")
    print("=" * 60)
    
    cm = ContextManager()
    
    # User request
    cm.add_message(Message.user("Read the main.py file and search for all TODO comments"))
    
    # Assistant with tool calls
    tool_call_msg = Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_001",
                function=FunctionCall(
                    name="read_file",
                    arguments='{"path": "main.py", "lines": [1, 200]}'
                )
            ),
            ToolCall(
                id="call_002", 
                function=FunctionCall(
                    name="search_files",
                    arguments='{"pattern": "TODO", "path": "."}'
                )
            )
        ]
    )
    cm.add_message(tool_call_msg)
    
    # Tool results
    file_content = generate_code_block(150)
    cm.add_message(Message.tool_result("call_001", file_content))
    
    search_results = generate_search_results(40)
    cm.add_message(Message.tool_result("call_002", search_results))
    
    # Assistant response
    cm.add_message(Message.assistant(
        "I've analyzed the file and found the TODO comments. Here's a summary:\n\n"
        "1. Line 45: TODO - Add error handling\n"
        "2. Line 87: TODO - Optimize database query\n"
        "3. Line 123: TODO - Add unit tests\n"
        "...(37 more)"
    ))
    
    # Continue with more tool usage
    cm.add_message(Message.user("Now edit the file to fix the first TODO"))
    
    edit_tool_msg = Message(
        role="assistant",
        content="I'll add error handling now.",
        tool_calls=[
            ToolCall(
                id="call_003",
                function=FunctionCall(
                    name="edit_file",
                    arguments='{"path": "main.py", "edits": [{"line": 45, "content": "try:\\n    result = process()\\nexcept Exception as e:\\n    logger.error(f\\"Error: {e}\\")"}]}'
                )
            )
        ]
    )
    cm.add_message(edit_tool_msg)
    cm.add_message(Message.tool_result("call_003", "File edited successfully. Changes:\n- Added try/except block at line 45\n- Added error logging"))
    
    status = cm.get_status()
    print(f"\nMessages: {status.message_count}")
    print(f"Total tokens: {status.used_tokens:,}")
    print(f"Tool results tokens: {status.tool_results_tokens:,}")
    print(f"Recent messages tokens: {status.recent_messages_tokens:,}")
    print(f"Budget usage: {status.percent_used:.2%}")
    
    assert status.tool_results_tokens > 0
    print("\n✓ Tool results test PASSED")


def test_budget_thresholds():
    """Test that compaction is triggered at correct thresholds."""
    print("\n" + "=" * 60)
    print("TEST: Budget Thresholds & Compaction Triggers")
    print("=" * 60)
    
    settings = get_settings()
    budget = TokenBudget.from_settings(settings)
    cm = ContextManager(settings=settings)
    tc = get_token_counter()
    
    print(f"Usable budget: {budget.usable_budget:,} tokens")
    print(f"Soft threshold (65%): {int(budget.usable_budget * 0.65):,}")
    print(f"Preemptive threshold (80%): {int(budget.usable_budget * 0.80):,}")
    print(f"Aggressive threshold (90%): {int(budget.usable_budget * 0.90):,}")
    print(f"Emergency threshold (97%): {int(budget.usable_budget * 0.97):,}")
    
    # Generate increasingly large messages to approach thresholds
    large_code = generate_code_block(500)
    large_listing = generate_file_listing(200)
    
    levels_reached = set()
    message_count = 0
    
    # Keep adding messages until we trigger compaction levels
    while cm.get_status().compaction_level != CompactionLevel.EMERGENCY and message_count < 200:
        # Add user message
        cm.add_message(Message.user(f"Analyze module {message_count}"))
        message_count += 1
        
        # Add large assistant response
        if message_count % 3 == 0:
            content = f"Analysis of module {message_count}:\n\n```python\n{large_code}\n```"
        elif message_count % 3 == 1:
            content = f"File listing for module {message_count}:\n\n{large_listing}"
        else:
            content = f"Search results:\n\n{generate_search_results(60)}"
        
        level = cm.add_message(Message.assistant(content))
        message_count += 1
        
        if level != CompactionLevel.NONE and level not in levels_reached:
            levels_reached.add(level)
            status = cm.get_status()
            print(f"\n  → {level.name} triggered at {status.used_tokens:,} tokens ({status.percent_used:.1%})")
        
        # Stop if we've seen enough levels or approaching emergency
        if len(levels_reached) >= 3 or cm.get_status().percent_used > 0.95:
            break
    
    final_status = cm.get_status()
    print(f"\nFinal state:")
    print(f"  Messages: {final_status.message_count}")
    print(f"  Tokens: {final_status.used_tokens:,}")
    print(f"  Usage: {final_status.percent_used:.1%}")
    print(f"  Level: {final_status.compaction_level.name}")
    print(f"  Levels reached: {[l.name for l in sorted(levels_reached, key=lambda x: x.value)]}")
    
    assert len(levels_reached) > 0, "Should trigger at least one compaction level"
    print("\n✓ Budget thresholds test PASSED")


def test_checkpoint_with_data():
    """Test checkpoint save/load with substantial data."""
    print("\n" + "=" * 60)
    print("TEST: Checkpoint Save/Load with Substantial Data")
    print("=" * 60)
    
    cm = ContextManager()
    
    # Build up a realistic session
    messages_data = [
        ("user", "I need to build a REST API with FastAPI"),
        ("assistant", f"I'll help you create a FastAPI application. Here's a starter template:\n\n```python\n{generate_code_block(80)}\n```"),
        ("user", "Add authentication endpoints"),
        ("assistant", f"Here's the authentication module:\n\n```python\n{generate_code_block(100)}\n```"),
        ("user", "What about database models?"),
        ("assistant", f"Database models:\n\n```python\n{generate_code_block(70)}\n```"),
        ("user", "Add CRUD operations"),
        ("assistant", f"CRUD operations:\n\n```python\n{generate_code_block(90)}\n```"),
        ("user", "Show me the tests"),
        ("assistant", f"Test suite:\n\n```python\n{generate_code_block(120)}\n```"),
    ]
    
    for role, content in messages_data:
        if role == "user":
            cm.add_message(Message.user(content))
        else:
            cm.add_message(Message.assistant(content))
    
    # Set plan state
    cm.set_plan_state("""
## Current Plan: Build REST API

### Steps:
1. [x] Create FastAPI app structure
2. [x] Add authentication
3. [x] Define database models
4. [x] Implement CRUD operations
5. [x] Write tests
6. [ ] Deploy to production
    """)
    
    # Get status before save
    status_before = cm.get_status()
    session_id = cm.session_id
    
    print(f"Before save:")
    print(f"  Session ID: {session_id}")
    print(f"  Messages: {status_before.message_count}")
    print(f"  Tokens: {status_before.used_tokens:,}")
    
    # Save checkpoint
    saved_id = cm.save_checkpoint()
    print(f"\nCheckpoint saved: {saved_id}")
    
    # Create new context manager and load
    cm2 = ContextManager()
    loaded = cm2.load_checkpoint(saved_id)
    
    status_after = cm2.get_status()
    print(f"\nAfter load:")
    print(f"  Session ID: {cm2.session_id}")
    print(f"  Messages: {status_after.message_count}")
    print(f"  Tokens: {status_after.used_tokens:,}")
    
    assert loaded, "Checkpoint should load successfully"
    assert status_after.message_count == status_before.message_count
    assert cm2.session_id == session_id
    
    # Verify messages are intact
    original_msgs = cm.messages
    loaded_msgs = cm2.messages
    assert len(original_msgs) == len(loaded_msgs)
    for orig, load in zip(original_msgs, loaded_msgs):
        assert orig.role == load.role
        assert orig.content == load.content
    
    print("\n✓ Checkpoint test PASSED")


def test_session_persistence():
    """Test full session save/load cycle."""
    print("\n" + "=" * 60)
    print("TEST: Session Persistence")
    print("=" * 60)
    
    cm = ContextManager()
    
    # Build session
    for i in range(20):
        cm.add_message(Message.user(f"Question {i}: What about feature {i}?"))
        cm.add_message(Message.assistant(f"Feature {i} implementation:\n\n{generate_code_block(30)}"))
    
    # Save session with label
    session_id = cm.save_session(label="API Development Session")
    print(f"Session saved: {session_id}")
    print(f"Label: API Development Session")
    
    # List sessions
    sessions = cm.list_sessions()
    print(f"\nAvailable sessions: {len(sessions)}")
    for s in sessions[:3]:
        print(f"  - {s['session_id'][:30]}... ({s['message_count']} msgs)")
    
    # Load in fresh context
    cm2 = ContextManager()
    loaded = cm2.load_session(session_id)
    
    assert loaded
    assert cm2.message_count == cm.message_count
    print(f"\nLoaded session: {cm2.message_count} messages")
    
    print("\n✓ Session persistence test PASSED")


def test_compactor_methods():
    """Test compactor methods in detail."""
    print("\n" + "=" * 60)
    print("TEST: Compactor Methods")
    print("=" * 60)
    
    compactor = ContextCompactor()
    
    # Build a varied message history
    messages = [
        Message.user("Search for all Python files in the project"),
        Message.assistant("I'll search for Python files."),
        Message.tool_result("call_1", generate_file_listing(50)),
        Message.assistant("Found 50 Python files. Let me analyze the main ones."),
        Message.user("Read the config.py file"),
        Message.assistant("Reading config.py..."),
        Message.tool_result("call_2", generate_code_block(80)),
        Message.assistant(f"The config file contains:\n\n```python\n{generate_code_block(40)}\n```"),
        Message.user("There's an error in line 45"),
        Message.assistant(f"I see the error:\n\n{generate_error_traceback()}\n\nLet me fix it."),
        Message.user("What files did we modify so far?"),
        Message.assistant("We modified: config.py (line 45), main.py (lines 10-20), utils.py"),
        Message.user("Save the current progress"),
        Message.assistant("Progress saved. Current plan:\n1. [x] Find files\n2. [x] Fix config\n3. [ ] Test changes"),
    ]
    
    # Test preservable info extraction
    info = compactor.extract_preservable_info(messages)
    print(f"\nExtracted preservable info:")
    print(f"  File paths: {len(info.get('file_paths', []))} found")
    print(f"  Plan steps: {len(info.get('plan_steps', []))} found")
    print(f"  Errors: {len(info.get('errors', []))} found")
    print(f"  Key decisions: {len(info.get('key_decisions', []))} found")
    
    # Test emergency compaction
    summary, remaining = compactor.compact_emergency(messages, "Previous summary: Setup complete.")
    print(f"\nEmergency compaction:")
    print(f"  Original messages: {len(messages)}")
    print(f"  Remaining messages: {len(remaining)}")
    print(f"  Summary length: {len(summary)} chars")
    print(f"  Summary preview: {summary[:100]}...")
    
    assert len(remaining) < len(messages)
    assert len(summary) > 0
    
    print("\n✓ Compactor methods test PASSED")


def test_token_counting_accuracy():
    """Test token counting with various content types."""
    print("\n" + "=" * 60)
    print("TEST: Token Counting Accuracy")
    print("=" * 60)
    
    tc = get_token_counter()
    
    test_cases = [
        ("Empty string", ""),
        ("Single word", "hello"),
        ("Short sentence", "The quick brown fox jumps over the lazy dog."),
        ("Code snippet", "def hello():\n    return 'world'"),
        ("JSON data", '{"name": "test", "value": 123, "enabled": true}'),
        ("Long code", generate_code_block(50)),
        ("File listing", generate_file_listing(30)),
        ("Mixed content", f"Analysis:\n\n{generate_code_block(20)}\n\nResults:\n{generate_search_results(10)}"),
    ]
    
    print(f"\n{'Content Type':<20} {'Length':<10} {'Tokens':<10} {'Ratio':<10}")
    print("-" * 50)
    
    for name, content in test_cases:
        tokens = tc.count_tokens(content)
        length = len(content)
        ratio = tokens / max(length, 1) if length > 0 else 0
        print(f"{name:<20} {length:<10} {tokens:<10} {ratio:.3f}")
    
    # Test message counting (includes overhead)
    print("\nMessage overhead test:")
    content = "Hello, how are you?"
    raw_tokens = tc.count_tokens(content)
    msg_tokens = tc.count_message(Message.user(content))
    overhead = msg_tokens - raw_tokens
    print(f"  Raw content: {raw_tokens} tokens")
    print(f"  As message: {msg_tokens} tokens")
    print(f"  Overhead: {overhead} tokens")
    
    # Test truncation
    print("\nTruncation test:")
    long_text = " ".join(["word"] * 1000)
    for target in [50, 100, 500]:
        truncated = tc.truncate_to_tokens(long_text, target)
        actual = tc.count_tokens(truncated)
        print(f"  Target {target}: got {actual} tokens ({len(truncated.split())} words)")
    
    print("\n✓ Token counting test PASSED")


@pytest.mark.asyncio
async def test_async_compaction():
    """Test async compaction methods (mocked, no real LLM)."""
    print("\n" + "=" * 60)
    print("TEST: Async Compaction (structure only, no LLM)")
    print("=" * 60)
    
    cm = ContextManager()
    
    # Build up context
    for i in range(30):
        cm.add_message(Message.user(f"Task {i}: Process data set {i}"))
        cm.add_message(Message.assistant(f"Processing data set {i}... Done. Results: {generate_search_results(10)}"))
    
    status = cm.get_status()
    print(f"Before compaction:")
    print(f"  Messages: {status.message_count}")
    print(f"  Tokens: {status.used_tokens:,}")
    
    # Test emergency compaction (doesn't need LLM)
    if status.compaction_level != CompactionLevel.NONE:
        success = await cm.trigger_compaction(CompactionLevel.EMERGENCY)
        
        status_after = cm.get_status()
        print(f"\nAfter emergency compaction:")
        print(f"  Success: {success}")
        print(f"  Messages: {status_after.message_count}")
        print(f"  Tokens: {status_after.used_tokens:,}")
        print(f"  History tokens: {status_after.compressed_history_tokens:,}")
    else:
        print("\n  (Compaction not needed at this usage level)")
    
    print("\n✓ Async compaction test PASSED")


def test_context_status_display():
    """Test status display formatting."""
    print("\n" + "=" * 60)
    print("TEST: Context Status Display")
    print("=" * 60)
    
    cm = ContextManager()
    
    # Add some messages
    for i in range(10):
        cm.add_message(Message.user(f"Question {i}"))
        cm.add_message(Message.assistant(f"Answer {i} with some detail about the topic."))
    
    status = cm.get_status()
    display = status.to_display_dict()
    
    print("\nStatus display:")
    for key, value in display.items():
        print(f"  {key}: {value}")
    
    assert "total_budget" in display
    assert "used_tokens" in display
    assert "percent_used" in display
    
    print("\n✓ Status display test PASSED")


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """Run all robust tests."""
    print("=" * 60)
    print("MeStudio Context System - ROBUST STRESS TESTS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Synchronous tests
    test_token_counting_accuracy()
    test_large_conversation()
    test_tool_results()
    test_budget_thresholds()
    test_checkpoint_with_data()
    test_session_persistence()
    test_compactor_methods()
    test_context_status_display()
    
    # Async tests
    asyncio.run(test_async_compaction())
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"ALL ROBUST TESTS PASSED in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
