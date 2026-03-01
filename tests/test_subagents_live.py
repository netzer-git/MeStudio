"""Live tests for sub-agent system using real LLM.

Run with: python tests/test_subagents_live.py

Requires LM Studio running on localhost:1234.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mestudio.agents import SubAgentSpawner, FileAgent, SearchAgent, SummaryAgent
from mestudio.core.llm_client import LMStudioClient
from mestudio.tools.registry import ToolRegistry


def print_test(name: str, passed: bool, details: str = "") -> None:
    """Print test result."""
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n{name}: {status}")
    if details:
        print(f"  {details}")


async def test_file_agent() -> bool:
    """Test FileAgent with a simple file task."""
    print("\n" + "=" * 60)
    print("TEST: FileAgent - Read and analyze a file")
    print("=" * 60)
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write('''"""Example module for testing."""

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """Simple calculator class."""
    
    def multiply(self, x: int, y: int) -> int:
        return x * y
''')
        test_file = f.name
    
    try:
        client = LMStudioClient()
        registry = ToolRegistry()
        spawner = SubAgentSpawner(client, registry)
        
        task = f"Read the file {test_file} and describe what functions and classes it contains."
        
        result = await spawner.spawn("file", task)
        print(f"\nTask: {task}")
        print(f"\nResult:\n{result[:500]}...")
        
        # Check for expected content
        passed = (
            "greet" in result.lower() or "add" in result.lower() or
            "calculator" in result.lower()
        )
        
        return passed
    finally:
        Path(test_file).unlink(missing_ok=True)


async def test_search_agent() -> bool:
    """Test SearchAgent with a simple web search."""
    print("\n" + "=" * 60)
    print("TEST: SearchAgent - Web search")
    print("=" * 60)
    
    client = LMStudioClient()
    registry = ToolRegistry()
    spawner = SubAgentSpawner(client, registry)
    
    task = "Search for information about Python asyncio and summarize what it is used for."
    
    result = await spawner.spawn("search", task)
    print(f"\nTask: {task}")
    print(f"\nResult:\n{result[:600]}...")
    
    # Check for expected content
    passed = (
        "async" in result.lower() or "await" in result.lower() or
        "concurrent" in result.lower() or "asynchronous" in result.lower()
    )
    
    return passed


async def test_summary_agent() -> bool:
    """Test SummaryAgent with text summarization."""
    print("\n" + "=" * 60)
    print("TEST: SummaryAgent - Text summarization")
    print("=" * 60)
    
    client = LMStudioClient()
    registry = ToolRegistry()
    spawner = SubAgentSpawner(client, registry)
    
    long_text = """
    The Python programming language has evolved significantly since its creation by Guido van Rossum 
    in the late 1980s. Originally conceived as a successor to the ABC language, Python was designed 
    with an emphasis on code readability and simplicity. The language uses significant whitespace 
    to delimit code blocks, which was quite unusual at the time but has since been recognized as 
    contributing to its clarity.
    
    Python supports multiple programming paradigms, including procedural, object-oriented, and 
    functional programming. This flexibility makes it suitable for a wide range of applications, 
    from simple scripts to complex enterprise software. The language includes a comprehensive 
    standard library that provides modules for handling files, regular expressions, networking, 
    databases, and much more.
    
    One of Python's key strengths is its extensive ecosystem of third-party packages. The Python 
    Package Index (PyPI) hosts hundreds of thousands of packages for tasks ranging from web 
    development to machine learning. Popular frameworks like Django for web applications, 
    NumPy for numerical computing, and TensorFlow for machine learning have made Python a 
    dominant force in many technology domains.
    
    The Python community follows the "Zen of Python" philosophy, which emphasizes simplicity, 
    readability, and explicit code. This philosophy is reflected in Python Enhancement Proposals 
    (PEPs), which guide the development of the language. Python 3, released in 2008, introduced 
    breaking changes to improve consistency and prepare the language for the future.
    """
    
    task = f"Summarize the following text in 3-4 bullet points:\n\n{long_text}"
    
    result = await spawner.spawn("summary", task)
    print(f"\nTask: Summarize Python history text")
    print(f"\nResult:\n{result}")
    
    # Check for expected content
    passed = (
        ("python" in result.lower() or "guido" in result.lower()) and
        len(result) < len(long_text)  # Should be shorter
    )
    
    return passed


async def test_spawner_error_handling() -> bool:
    """Test SubAgentSpawner error handling."""
    print("\n" + "=" * 60)
    print("TEST: SubAgentSpawner - Error handling")
    print("=" * 60)
    
    client = LMStudioClient()
    registry = ToolRegistry()
    spawner = SubAgentSpawner(client, registry)
    
    # Test invalid agent type
    result = await spawner.spawn("invalid_type", "test task")
    print(f"\nInvalid agent type result: {result}")
    
    error_handled = "error" in result.lower() and "invalid" in result.lower()
    
    # Test available types
    agent_types = spawner.get_agent_types()
    print(f"Available agent types: {agent_types}")
    
    types_correct = set(agent_types) == {"file", "search", "summary"}
    
    passed = error_handled and types_correct
    return passed


async def test_agent_creation() -> bool:
    """Test direct agent creation and configuration."""
    print("\n" + "=" * 60)
    print("TEST: Agent creation and configuration")
    print("=" * 60)
    
    client = LMStudioClient()
    registry = ToolRegistry()
    semaphore = asyncio.Semaphore(1)
    
    # Create each agent type
    file_agent = FileAgent(client, registry, semaphore)
    search_agent = SearchAgent(client, registry, semaphore)
    summary_agent = SummaryAgent(client, registry, semaphore)
    
    print(f"FileAgent: {file_agent.get_description()}")
    print(f"SearchAgent: {search_agent.get_description()}")
    print(f"SummaryAgent: {summary_agent.get_description()}")
    
    # Check configurations
    checks = [
        file_agent.config.name == "FileAgent",
        search_agent.config.name == "SearchAgent",
        summary_agent.config.name == "SummaryAgent",
        len(file_agent._tools) > 0,
        len(search_agent._tools) > 0,
        len(summary_agent._tools) > 0,
    ]
    
    print(f"\nFileAgent tools: {len(file_agent._tools)}")
    print(f"SearchAgent tools: {len(search_agent._tools)}")
    print(f"SummaryAgent tools: {len(summary_agent._tools)}")
    
    return all(checks)


async def main():
    """Run all live tests."""
    print("\n" + "=" * 60)
    print("MeStudio Sub-Agent Live Tests")
    print("=" * 60)
    
    # Check LM Studio availability
    client = LMStudioClient()
    if not await client.is_available():
        print("\n❌ LM Studio is not running on localhost:1234")
        print("Please start LM Studio before running these tests.")
        return
    
    print("✓ LM Studio is available")
    
    results = []
    
    # Run tests
    tests = [
        ("Agent Creation", test_agent_creation),
        ("Error Handling", test_spawner_error_handling),
        ("File Agent", test_file_agent),
        ("Search Agent", test_search_agent),
        ("Summary Agent", test_summary_agent),
    ]
    
    for name, test_func in tests:
        try:
            passed = await test_func()
            results.append((name, passed))
            print_test(name, passed)
        except Exception as e:
            results.append((name, False))
            print_test(name, False, f"Exception: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
