"""Intensive live tests for the Orchestrator using real LLM.

Run with: python tests/test_orchestrator_live.py

Requires LM Studio running on localhost:1234.

These tests exercise the full orchestrator with various task types:
- Simple Q&A
- File operations (read, write, edit, search)
- Web search and research
- Multi-step planning
- Sub-agent delegation
- Error handling
- Context management
"""

import asyncio
import os
import sys
import shutil
import uuid
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mestudio.core import Orchestrator, OrchestratorConfig, ConsoleOutputHandler
from mestudio.core.llm_client import LMStudioClient
from mestudio.core.config import get_settings


# Test temp directory inside workspace (accessible by file tools)
WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
TEST_TEMP_BASE = WORKSPACE_DIR / "test_temp"


def create_test_dir(prefix: str = "test") -> Path:
    """Create a unique test directory inside the workspace.
    
    This ensures paths are accessible by file tools which restrict
    access to the working directory.
    """
    TEST_TEMP_BASE.mkdir(exist_ok=True)
    test_dir = TEST_TEMP_BASE / f"{prefix}_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def cleanup_test_temp():
    """Clean up all test temp directories."""
    if TEST_TEMP_BASE.exists():
        shutil.rmtree(TEST_TEMP_BASE, ignore_errors=True)


class TestTracker:
    """Track test results."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []
        self.start_time = datetime.now()

    def add(self, name: str, passed: bool, details: str = "") -> None:
        self.results.append((name, passed, details))
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n{name}: {status}")
        if details:
            lines = details.split("\n")
            for line in lines[:5]:  # Show first 5 lines
                print(f"  {line}")
            if len(lines) > 5:
                print(f"  ... ({len(lines) - 5} more lines)")

    def summary(self) -> None:
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        
        for name, p, _ in self.results:
            status = "✓" if p else "✗"
            print(f"  {status} {name}")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n{passed}/{total} tests passed in {duration:.1f}s")
        
        if passed == total:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️ Some tests failed")


async def test_simple_qa(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test simple question-answering without tools."""
    print("\n" + "=" * 70)
    print("TEST: Simple Q&A (no tools)")
    print("=" * 70)
    
    # Reset for clean state
    await orchestrator.reset()
    
    response = await orchestrator.chat("What is 2 + 2? Just give me the number.")
    print(f"\nResponse: {response[:200]}")
    
    # Check response contains the answer
    passed = "4" in response
    tracker.add("Simple Q&A", passed, f"Response: {response[:100]}")


async def test_file_read(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test file reading capability."""
    print("\n" + "=" * 70)
    print("TEST: File Read")
    print("=" * 70)
    
    await orchestrator.reset()
    
    # Create test file inside workspace (accessible by file tools)
    test_dir = create_test_dir("file_read")
    test_file = test_dir / "sample.txt"
    test_file.write_text("Line 1: Hello World\nLine 2: Testing\nLine 3: File Read")
    
    # Use relative path for better compatibility
    rel_path = test_file.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Read the file {rel_path} and tell me what's on line 2."
    )
    print(f"\nResponse: {response[:300]}")
    
    passed = "testing" in response.lower() or "line 2" in response.lower()
    tracker.add("File Read", passed, f"Response: {response[:150]}")


async def test_file_write(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test file writing capability."""
    print("\n" + "=" * 70)
    print("TEST: File Write")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("file_write")
    test_file = test_dir / "output.txt"
    rel_path = test_file.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Write a file to {rel_path} with the content: 'Hello from MeStudio Agent!'"
    )
    print(f"\nResponse: {response[:300]}")
    
    # Verify file was created
    file_exists = test_file.exists()
    content_correct = False
    if file_exists:
        content = test_file.read_text()
        content_correct = "hello" in content.lower() or "mestudio" in content.lower()
    
    passed = file_exists and content_correct
    tracker.add("File Write", passed, 
        f"File exists: {file_exists}, Content correct: {content_correct}")


async def test_directory_listing(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test directory listing."""
    print("\n" + "=" * 70)
    print("TEST: Directory Listing")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("dir_list")
    (test_dir / "file1.py").write_text("# Python file")
    (test_dir / "file2.txt").write_text("Text file")
    (test_dir / "subdir").mkdir()
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"List the contents of the directory {rel_path}"
    )
    print(f"\nResponse: {response[:400]}")
    
    # Check response mentions the files
    passed = (
        "file1" in response.lower() or 
        "file2" in response.lower() or
        "subdir" in response.lower()
    )
    tracker.add("Directory Listing", passed, f"Response: {response[:200]}")


async def test_file_search(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test file search capability."""
    print("\n" + "=" * 70)
    print("TEST: File Search")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("file_search")
    (test_dir / "code.py").write_text("""
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b
""")
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Search for 'calculate' in the files under {rel_path} and tell me what functions you find."
    )
    print(f"\nResponse: {response[:400]}")
    
    passed = (
        "sum" in response.lower() or 
        "product" in response.lower() or
        "calculate" in response.lower()
    )
    tracker.add("File Search", passed, f"Response: {response[:200]}")


async def test_file_edit(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test file editing capability."""
    print("\n" + "=" * 70)
    print("TEST: File Edit")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("file_edit")
    test_file = test_dir / "config.txt"
    test_file.write_text("setting=old_value\nother=123")
    rel_path = test_file.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Edit the file {rel_path} to change 'old_value' to 'new_value'"
    )
    print(f"\nResponse: {response[:300]}")
    
    # Verify edit was made
    content = test_file.read_text()
    passed = "new_value" in content and "old_value" not in content
    tracker.add("File Edit", passed, f"File content: {content}")


async def test_web_search(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test web search capability."""
    print("\n" + "=" * 70)
    print("TEST: Web Search")
    print("=" * 70)
    
    await orchestrator.reset()
    
    response = await orchestrator.chat(
        "Search the web for 'Python programming language' and give me a brief summary."
    )
    print(f"\nResponse: {response[:500]}")
    
    # Check response mentions Python-related concepts
    passed = (
        "python" in response.lower() or
        "programming" in response.lower() or
        "language" in response.lower()
    )
    tracker.add("Web Search", passed, f"Response: {response[:200]}")


async def test_multi_step_task(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test a multi-step task requiring multiple tool calls."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Step Task")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("multi_step")
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"In the directory {rel_path}, create a file called 'hello.py' with a simple "
        f"Python function that prints 'Hello World', then read the file back and "
        f"confirm it was created correctly."
    )
    print(f"\nResponse: {response[:500]}")
    
    # Check if file was created
    hello_file = test_dir / "hello.py"
    file_exists = hello_file.exists()
    has_content = False
    if file_exists:
        content = hello_file.read_text()
        has_content = "hello" in content.lower() or "print" in content.lower()
    
    passed = file_exists and has_content
    tracker.add("Multi-Step Task", passed, 
        f"File exists: {file_exists}, Has content: {has_content}")


async def test_planning(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test task planning capability."""
    print("\n" + "=" * 70)
    print("TEST: Task Planning")
    print("=" * 70)
    
    await orchestrator.reset()
    
    response = await orchestrator.chat(
        "Create a plan for building a simple calculator application. "
        "The plan should have at least 3 steps."
    )
    print(f"\nResponse: {response[:600]}")
    
    # Check if response shows planning activity
    passed = (
        "step" in response.lower() or
        "plan" in response.lower() or
        "1." in response or
        "calculator" in response.lower()
    )
    tracker.add("Task Planning", passed, f"Response: {response[:250]}")


async def test_sub_agent_delegation(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test sub-agent delegation."""
    print("\n" + "=" * 70)
    print("TEST: Sub-Agent Delegation")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("sub_agent")
    test_file = test_dir / "analyze_me.py"
    test_file.write_text("""
'''This module contains math utilities.'''

def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b

def subtract(a: int, b: int) -> int:
    '''Subtract b from a.'''
    return a - b

class Calculator:
    '''A simple calculator class.'''
    
    def multiply(self, x: int, y: int) -> int:
        return x * y
""")
    rel_path = test_file.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Delegate to the file agent: analyze the Python file at {rel_path} "
        f"and describe all the functions and classes it contains."
    )
    print(f"\nResponse: {response[:500]}")
    
    passed = (
        "add" in response.lower() or
        "subtract" in response.lower() or
        "calculator" in response.lower() or
        "multiply" in response.lower()
    )
    tracker.add("Sub-Agent Delegation", passed, f"Response: {response[:200]}")


async def test_code_generation(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test code generation task."""
    print("\n" + "=" * 70)
    print("TEST: Code Generation")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("code_gen")
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Write a Python function to {rel_path}/factorial.py that calculates "
        f"the factorial of a number recursively. Include a docstring and type hints."
    )
    print(f"\nResponse: {response[:500]}")
    
    # Check if file was created with factorial function
    factorial_file = test_dir / "factorial.py"
    file_exists = factorial_file.exists()
    has_factorial = False
    if file_exists:
        content = factorial_file.read_text()
        has_factorial = "factorial" in content.lower() and "def " in content
    
    passed = file_exists and has_factorial
    tracker.add("Code Generation", passed, 
        f"File exists: {file_exists}, Has factorial: {has_factorial}")


async def test_error_handling(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test handling of errors gracefully."""
    print("\n" + "=" * 70)
    print("TEST: Error Handling")
    print("=" * 70)
    
    await orchestrator.reset()
    
    # Ask to read a non-existent file
    response = await orchestrator.chat(
        "Read the file /nonexistent/path/to/file/that/does/not/exist.txt"
    )
    print(f"\nResponse: {response[:400]}")
    
    # Agent should handle error gracefully (not crash)
    # Check for various ways errors might be expressed
    response_lower = response.lower()
    passed = (
        "error" in response_lower or
        "not found" in response_lower or
        "does not exist" in response_lower or
        "doesn't exist" in response_lower or
        "cannot" in response_lower or
        "can't" in response_lower or
        "unable" in response_lower or
        "outside" in response_lower  # outside working directory
    )
    tracker.add("Error Handling", passed, f"Response: {response[:200]}")


async def test_context_tracking(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test context status tracking."""
    print("\n" + "=" * 70)
    print("TEST: Context Tracking")
    print("=" * 70)
    
    await orchestrator.reset()
    
    # Have a conversation to build up context
    await orchestrator.chat("Hello! My name is Alice.")
    await orchestrator.chat("What's the capital of France?")
    
    # Get context status
    status = orchestrator.get_context_status()
    session_stats = orchestrator.get_session_stats()
    
    print(f"\nContext status: {status}")
    print(f"Session stats: {session_stats}")
    
    # Check context is being tracked
    passed = (
        session_stats["turn_count"] >= 2 and
        session_stats["total_tokens"] > 0
    )
    tracker.add("Context Tracking", passed, 
        f"Turns: {session_stats['turn_count']}, Tokens: {session_stats['total_tokens']}")


async def test_conversation_memory(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test that conversation context is maintained."""
    print("\n" + "=" * 70)
    print("TEST: Conversation Memory")
    print("=" * 70)
    
    await orchestrator.reset()
    
    # First message establishes context
    await orchestrator.chat("Remember this number: 42. It's important.")
    
    # Second message tests memory
    response = await orchestrator.chat("What was the important number I told you to remember?")
    print(f"\nResponse: {response[:300]}")
    
    passed = "42" in response
    tracker.add("Conversation Memory", passed, f"Response: {response[:150]}")


async def test_complex_file_task(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test a complex file manipulation task."""
    print("\n" + "=" * 70)
    print("TEST: Complex File Task")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("complex")
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    # Create initial files
    (test_dir / "module1.py").write_text("""
def greet():
    print("Hello")

def farewell():
    print("Goodbye")
""")
    (test_dir / "module2.py").write_text("""
def calculate():
    return 1 + 1
""")
    
    response = await orchestrator.chat(
        f"In the directory {rel_path}: "
        f"1. List all Python files "
        f"2. Read each file to find all function names "
        f"3. Create a new file called 'summary.txt' listing all functions found"
    )
    print(f"\nResponse: {response[:600]}")
    
    # Check if summary file was created
    summary_file = test_dir / "summary.txt"
    file_exists = summary_file.exists()
    has_functions = False
    if file_exists:
        content = summary_file.read_text().lower()
        has_functions = "greet" in content or "calculate" in content or "farewell" in content
    
    passed = file_exists and has_functions
    tracker.add("Complex File Task", passed, 
        f"Summary exists: {file_exists}, Has functions: {has_functions}")


async def test_research_task(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test a research task using web search."""
    print("\n" + "=" * 70)
    print("TEST: Research Task")
    print("=" * 70)
    
    await orchestrator.reset()
    
    response = await orchestrator.chat(
        "Research and summarize in 3-4 sentences: What is asyncio in Python? "
        "Include at least one specific feature or use case."
    )
    print(f"\nResponse: {response[:600]}")
    
    passed = (
        "async" in response.lower() or
        "concurrent" in response.lower() or
        "event loop" in response.lower() or
        "coroutine" in response.lower()
    )
    tracker.add("Research Task", passed, f"Response: {response[:250]}")


async def test_sequential_tools(orchestrator: Orchestrator, tracker: TestTracker) -> None:
    """Test sequential tool usage in a single turn."""
    print("\n" + "=" * 70)
    print("TEST: Sequential Tool Usage")
    print("=" * 70)
    
    await orchestrator.reset()
    
    test_dir = create_test_dir("sequential")
    rel_path = test_dir.relative_to(WORKSPACE_DIR)
    
    response = await orchestrator.chat(
        f"Do these steps in order: "
        f"1. Create a file {rel_path}/step1.txt with 'Step 1 complete' "
        f"2. Create a file {rel_path}/step2.txt with 'Step 2 complete' "
        f"3. List the directory to confirm both files exist"
    )
    print(f"\nResponse: {response[:500]}")
    
    # Check both files exist
    step1_exists = (test_dir / "step1.txt").exists()
    step2_exists = (test_dir / "step2.txt").exists()
    
    passed = step1_exists and step2_exists
    tracker.add("Sequential Tool Usage", passed, 
        f"Step1: {step1_exists}, Step2: {step2_exists}")


async def main():
    """Run all intensive tests."""
    print("\n" + "=" * 70)
    print("MeStudio Orchestrator - Intensive Live Tests")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Test temp: {TEST_TEMP_BASE}")
    
    # Set working directory in settings so file tools can access test_temp
    settings = get_settings()
    settings.working_directory = str(WORKSPACE_DIR)
    print(f"Working directory set to: {settings.working_directory}")
    
    # Clean up any previous test temp
    cleanup_test_temp()
    
    # Check LM Studio availability
    client = LMStudioClient()
    if not await client.is_available():
        print("\n❌ LM Studio is not running on localhost:1234")
        print("Please start LM Studio with a model before running these tests.")
        return
    
    print("✓ LM Studio is available")
    
    # Get available models
    models = await client.get_models()
    print(f"✓ Available models: {models}")
    
    # Create orchestrator with console output
    config = OrchestratorConfig(
        max_tool_iterations=15,
        max_parallel_tools=3,
    )
    orchestrator = Orchestrator(config=config, output_handler=ConsoleOutputHandler())
    
    # Initialize
    if not await orchestrator.initialize():
        print("❌ Failed to initialize orchestrator")
        return
    
    print("✓ Orchestrator initialized")
    
    # Track results
    tracker = TestTracker()
    
    # Run tests - ordered from simple to complex
    tests = [
        ("Simple Q&A", test_simple_qa),
        ("Conversation Memory", test_conversation_memory),
        ("Context Tracking", test_context_tracking),
        ("File Read", test_file_read),
        ("File Write", test_file_write),
        ("File Edit", test_file_edit),
        ("Directory Listing", test_directory_listing),
        ("File Search", test_file_search),
        ("Error Handling", test_error_handling),
        ("Multi-Step Task", test_multi_step_task),
        ("Sequential Tool Usage", test_sequential_tools),
        ("Code Generation", test_code_generation),
        ("Task Planning", test_planning),
        ("Web Search", test_web_search),
        ("Research Task", test_research_task),
        ("Sub-Agent Delegation", test_sub_agent_delegation),
        ("Complex File Task", test_complex_file_task),
    ]
    
    for name, test_func in tests:
        try:
            await test_func(orchestrator, tracker)
        except Exception as e:
            print(f"\n❌ Test {name} crashed: {e}")
            tracker.add(name, False, f"Exception: {type(e).__name__}: {e}")
    
    # Print summary
    tracker.summary()
    
    # Print final stats
    print("\n" + "=" * 70)
    print("FINAL SESSION STATS")
    print("=" * 70)
    stats = orchestrator.get_session_stats()
    print(f"  Total turns: {stats['turn_count']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  LLM calls: {stats['llm_calls']}")
    
    # Cleanup test temp directories
    print("\nCleaning up test temp directories...")
    cleanup_test_temp()
    print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
