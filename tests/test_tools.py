"""Comprehensive tests for the Tool System (Step 4)."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "c:/clones/MeStudio")


def reload_tools():
    """Reload all tool modules to ensure tools are registered."""
    import importlib
    import mestudio.tools.registry
    import mestudio.tools.file_tools
    import mestudio.tools.web_tools
    import mestudio.tools.context_tools
    import mestudio.tools.plan_tools
    import mestudio.tools.agent_tools
    
    importlib.reload(mestudio.tools.registry)
    importlib.reload(mestudio.tools.file_tools)
    importlib.reload(mestudio.tools.web_tools)
    importlib.reload(mestudio.tools.context_tools)
    importlib.reload(mestudio.tools.plan_tools)
    importlib.reload(mestudio.tools.agent_tools)


from mestudio.tools import (
    ToolRegistry,
    ToolDefinition,
    ToolParameter,
    get_registry,
    tool,
    register_all_tools,
    is_binary,
    PlanStep,
    TaskPlan,
    get_current_plan,
    set_current_plan,
    set_context_manager,
    register_agent_handler,
    list_agent_types,
    register_placeholder_handlers,
)
from mestudio.tools.registry import _parse_docstring_params
from mestudio.context import ContextManager


def test_tool_registry():
    """Test the tool registry singleton."""
    print("\n" + "=" * 60)
    print("TEST: Tool Registry")
    print("=" * 60)
    
    # Don't reset - use existing registry
    registry = get_registry()
    assert isinstance(registry, ToolRegistry)
    
    # Singleton test
    registry2 = get_registry()
    assert registry is registry2
    print("✓ Singleton pattern works")
    
    # Manual registration
    async def dummy_tool(arg1: str, arg2: int = 10) -> str:
        return f"Result: {arg1}, {arg2}"
    
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters=[
            ToolParameter(name="arg1", type="string", description="First arg"),
            ToolParameter(name="arg2", type="integer", description="Second arg", required=False, default=10),
        ],
        handler=dummy_tool,
    )
    
    assert "test_tool" in registry.tools
    print("✓ Manual registration works")
    
    # Get OpenAI schema
    schemas = registry.get_openai_tools()
    test_schema = next(s for s in schemas if s["function"]["name"] == "test_tool")
    
    assert test_schema["type"] == "function"
    assert test_schema["function"]["name"] == "test_tool"
    assert "arg1" in test_schema["function"]["parameters"]["properties"]
    print("✓ OpenAI schema generation works")
    
    # Execute tool
    result = asyncio.run(registry.execute("test_tool", {"arg1": "hello", "arg2": 42}))
    assert result == "Result: hello, 42"
    print("✓ Tool execution works")
    
    # Execute with JSON string args
    result = asyncio.run(registry.execute("test_tool", '{"arg1": "world"}'))
    assert result == "Result: world, 10"
    print("✓ JSON argument parsing works")
    
    # Unknown tool
    result = asyncio.run(registry.execute("nonexistent", {}))
    assert "Error" in result
    print("✓ Unknown tool error handling works")
    
    print("\n✓ Tool Registry tests PASSED")


def test_tool_decorator():
    """Test the @tool decorator."""
    print("\n" + "=" * 60)
    print("TEST: @tool Decorator")
    print("=" * 60)
    
    registry = get_registry()
    
    @tool(name="decorated_tool_test", description="A decorated test tool")
    async def my_tool(name: str, count: int = 5) -> str:
        """
        Args:
            name: The name to use.
            count: How many times to repeat.
        """
        return f"Hello {name}! " * count
    
    assert "decorated_tool_test" in registry.tools
    print("✓ Decorator registers tool")
    
    # Check that parameter descriptions were extracted
    tool_def = registry.get("decorated_tool_test")
    name_param = next(p for p in tool_def.parameters if p.name == "name")
    assert "name to use" in name_param.description
    print("✓ Docstring parsing extracts parameter descriptions")
    
    # Execute
    result = asyncio.run(registry.execute("decorated_tool_test", {"name": "World", "count": 3}))
    assert result == "Hello World! Hello World! Hello World! "
    print("✓ Decorated tool executes correctly")
    
    print("\n✓ @tool Decorator tests PASSED")


def test_docstring_parsing():
    """Test docstring parameter extraction."""
    print("\n" + "=" * 60)
    print("TEST: Docstring Parsing")
    print("=" * 60)
    
    docstring = """
    Process some data.
    
    Args:
        input_file: Path to the input file.
        output_format: Output format (json, csv, xml).
        verbose: Enable verbose output.
    
    Returns:
        The processed result.
    """
    
    params = _parse_docstring_params(docstring)
    
    assert "input_file" in params
    assert "output_format" in params
    assert "verbose" in params
    assert "input" in params["input_file"].lower()
    print(f"✓ Parsed {len(params)} parameters from docstring")
    
    print("\n✓ Docstring parsing tests PASSED")


def test_file_tools():
    """Test file tools."""
    print("\n" + "=" * 60)
    print("TEST: File Tools")
    print("=" * 60)
    
    registry = get_registry()
    
    # Check file tools are registered
    file_tool_names = ["read_file", "write_file", "edit_file", "list_directory", "search_files", "find_files"]
    for name in file_tool_names:
        assert name in registry.tools, f"Missing tool: {name}"
    print(f"✓ All {len(file_tool_names)} file tools registered")
    
    # Test with a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override working directory
        from mestudio.core import config
        original_settings = config._settings
        
        try:
            config._settings = None
            os.environ["MESTUDIO_WORKING_DIRECTORY"] = tmpdir
            
            # Write a file
            test_file = Path(tmpdir) / "test.txt"
            result = asyncio.run(registry.execute("write_file", {
                "path": "test.txt",
                "content": "Line 1\nLine 2\nLine 3\n"
            }))
            assert "Written" in result
            assert test_file.exists()
            print("✓ write_file works")
            
            # Read the file
            result = asyncio.run(registry.execute("read_file", {"path": "test.txt"}))
            assert "Line 1" in result
            assert "Line 2" in result
            print("✓ read_file works")
            
            # Read with line range
            result = asyncio.run(registry.execute("read_file", {
                "path": "test.txt",
                "start_line": 2,
                "end_line": 2
            }))
            assert "Line 2" in result
            assert "Line 1" not in result
            print("✓ read_file with line range works")
            
            # Edit the file
            result = asyncio.run(registry.execute("edit_file", {
                "path": "test.txt",
                "edits": [{"old": "Line 2", "new": "Modified Line 2"}]
            }))
            assert "Edited" in result
            
            content = test_file.read_text()
            assert "Modified Line 2" in content
            print("✓ edit_file works")
            
            # List directory
            result = asyncio.run(registry.execute("list_directory", {"path": "."}))
            assert "test.txt" in result
            print("✓ list_directory works")
            
            # Search files
            result = asyncio.run(registry.execute("search_files", {
                "query": "Modified",
                "path": "."
            }))
            assert "Modified" in result
            print("✓ search_files works")
            
            # Find files
            result = asyncio.run(registry.execute("find_files", {
                "pattern": "*.txt",
                "path": "."
            }))
            assert "test.txt" in result
            print("✓ find_files works")
            
        finally:
            config._settings = original_settings
            if "MESTUDIO_WORKING_DIRECTORY" in os.environ:
                del os.environ["MESTUDIO_WORKING_DIRECTORY"]
    
    print("\n✓ File Tools tests PASSED")


def test_binary_detection():
    """Test binary file detection."""
    print("\n" + "=" * 60)
    print("TEST: Binary File Detection")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Text file
        text_file = Path(tmpdir) / "text.txt"
        text_file.write_text("Hello, world!")
        assert not is_binary(text_file)
        print("✓ Text file detected as non-binary")
        
        # Binary file
        binary_file = Path(tmpdir) / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03" * 100)
        assert is_binary(binary_file)
        print("✓ Binary file detected as binary")
    
    print("\n✓ Binary detection tests PASSED")


def test_plan_tools():
    """Test plan tools."""
    print("\n" + "=" * 60)
    print("TEST: Plan Tools")
    print("=" * 60)
    
    # Clear any existing plan
    set_current_plan(None)
    
    registry = get_registry()
    
    # Check plan tools are registered
    plan_tool_names = ["create_plan", "update_step", "get_plan", "add_steps", "remove_step", "cancel_plan", "replace_plan"]
    for name in plan_tool_names:
        assert name in registry.tools, f"Missing tool: {name}"
    print(f"✓ All {len(plan_tool_names)} plan tools registered")
    
    # Create a plan
    result = asyncio.run(registry.execute("create_plan", {
        "goal": "Build a web scraper",
        "steps": [
            "Set up project structure",
            "Implement URL fetching",
            "Parse HTML content",
            "Extract data",
            "Save results"
        ]
    }))
    assert "Plan created" in result
    assert get_current_plan() is not None
    print("✓ create_plan works")
    
    # Update a step
    result = asyncio.run(registry.execute("update_step", {
        "step_index": 1,
        "status": "done",
        "notes": "Created directories"
    }))
    assert "done" in result
    print("✓ update_step works")
    
    # Get plan
    result = asyncio.run(registry.execute("get_plan", {}))
    assert "Build a web scraper" in result
    assert "✓" in result  # Done icon
    print("✓ get_plan works")
    
    # Add steps
    result = asyncio.run(registry.execute("add_steps", {
        "steps": ["Add error handling", "Write tests"],
        "after_index": 3
    }))
    plan = get_current_plan()
    assert len(plan.steps) == 7
    print("✓ add_steps works")
    
    # Remove step
    result = asyncio.run(registry.execute("remove_step", {"step_index": 7}))
    plan = get_current_plan()
    assert len(plan.steps) == 6
    print("✓ remove_step works")
    
    # Replace plan
    result = asyncio.run(registry.execute("replace_plan", {
        "goal": "New goal",
        "steps": ["Step A", "Step B"]
    }))
    plan = get_current_plan()
    assert plan.goal == "New goal"
    assert len(plan.steps) == 2
    print("✓ replace_plan works")
    
    # Cancel plan
    result = asyncio.run(registry.execute("cancel_plan", {}))
    assert get_current_plan() is None
    print("✓ cancel_plan works")
    
    print("\n✓ Plan Tools tests PASSED")


def test_context_tools():
    """Test context tools."""
    print("\n" + "=" * 60)
    print("TEST: Context Tools")
    print("=" * 60)
    
    registry = get_registry()
    
    # Check context tools are registered
    context_tool_names = ["save_context", "load_context", "list_sessions", "compact_now", "context_status"]
    for name in context_tool_names:
        assert name in registry.tools, f"Missing tool: {name}"
    print(f"✓ All {len(context_tool_names)} context tools registered")
    
    # Set up context manager
    cm = ContextManager()
    set_context_manager(cm)
    
    # Add some messages
    from mestudio.core.models import Message
    cm.add_message(Message.user("Hello!"))
    cm.add_message(Message.assistant("Hi there!"))
    
    # Get status
    result = asyncio.run(registry.execute("context_status", {}))
    assert "Budget" in result
    assert "Message count" in result
    print("✓ context_status works")
    
    # Save session
    result = asyncio.run(registry.execute("save_context", {"label": "Test session"}))
    assert "saved" in result.lower()
    print("✓ save_context works")
    
    # List sessions
    result = asyncio.run(registry.execute("list_sessions", {}))
    assert "Test session" in result or "session" in result.lower()
    print("✓ list_sessions works")
    
    print("\n✓ Context Tools tests PASSED")


def test_agent_tools():
    """Test agent delegation tools."""
    print("\n" + "=" * 60)
    print("TEST: Agent Tools")
    print("=" * 60)
    
    registry = get_registry()
    
    assert "delegate_task" in registry.tools
    print("✓ delegate_task tool registered")
    
    # Register placeholder handlers
    register_placeholder_handlers()
    
    agents = list_agent_types()
    assert "file" in agents
    assert "search" in agents
    assert "summary" in agents
    print(f"✓ Agent types registered: {agents}")
    
    # Test delegation (placeholder)
    result = asyncio.run(registry.execute("delegate_task", {
        "agent_type": "file",
        "task": "Read the config file"
    }))
    assert "File Agent" in result
    print("✓ delegate_task executes placeholder handler")
    
    # Test invalid agent type
    result = asyncio.run(registry.execute("delegate_task", {
        "agent_type": "invalid",
        "task": "Some task"
    }))
    assert "Error" in result
    print("✓ Invalid agent type returns error")
    
    print("\n✓ Agent Tools tests PASSED")


def test_tool_timeout():
    """Test tool timeout handling."""
    print("\n" + "=" * 60)
    print("TEST: Tool Timeout")
    print("=" * 60)
    
    registry = get_registry()
    
    @tool(name="slow_tool_test", description="A slow tool", timeout=0.5)
    async def slow_tool() -> str:
        await asyncio.sleep(2)  # Will timeout
        return "Done"
    
    result = asyncio.run(registry.execute("slow_tool_test", {}))
    
    assert "timed out" in result.lower()
    print("✓ Timeout handling works")
    
    print("\n✓ Timeout tests PASSED")


def test_tool_error_handling():
    """Test tool error handling."""
    print("\n" + "=" * 60)
    print("TEST: Tool Error Handling")
    print("=" * 60)
    
    registry = get_registry()
    
    @tool(name="failing_tool_test", description="A tool that fails")
    async def failing_tool(value: int) -> str:
        if value < 0:
            raise ValueError("Value must be non-negative")
        return f"Value: {value}"
    
    # Successful execution
    result = asyncio.run(registry.execute("failing_tool_test", {"value": 10}))
    assert result == "Value: 10"
    print("✓ Successful execution works")
    
    # Failed execution
    result = asyncio.run(registry.execute("failing_tool_test", {"value": -5}))
    assert "Error" in result
    assert "ValueError" in result
    print("✓ Error is caught and reported")
    
    # Invalid arguments
    result = asyncio.run(registry.execute("failing_tool_test", {"wrong_arg": 10}))
    assert "Error" in result
    print("✓ Invalid arguments caught")
    
    # Invalid JSON
    result = asyncio.run(registry.execute("failing_tool_test", "{not valid json}"))
    assert "Error" in result
    print("✓ Invalid JSON caught")
    
    print("\n✓ Error handling tests PASSED")


def test_result_truncation():
    """Test result truncation."""
    print("\n" + "=" * 60)
    print("TEST: Result Truncation")
    print("=" * 60)
    
    registry = get_registry()
    
    @tool(name="verbose_tool_test", description="Returns lots of text", max_result_tokens=50)
    async def verbose_tool() -> str:
        return "word " * 1000  # Way more than 50 tokens
    
    result = asyncio.run(registry.execute("verbose_tool_test", {}))
    
    # Should be truncated
    from mestudio.context.token_counter import get_token_counter
    tc = get_token_counter()
    tokens = tc.count_tokens(result)
    
    assert tokens <= 55  # Allow small buffer
    assert "truncated" in result.lower()
    print(f"✓ Result truncated to ~{tokens} tokens")
    
    print("\n✓ Truncation tests PASSED")


def test_openai_schema_complete():
    """Test that the full OpenAI schema is correct."""
    print("\n" + "=" * 60)
    print("TEST: Complete OpenAI Schema")
    print("=" * 60)
    
    registry = get_registry()
    schemas = registry.get_openai_tools()
    
    print(f"Total tools registered: {len(schemas)}")
    
    # Check schema structure
    for schema in schemas:
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
    
    print("✓ All schemas have correct structure")
    
    # List all tools
    tool_names = [s["function"]["name"] for s in schemas]
    print(f"\nRegistered tools: {', '.join(sorted(tool_names))}")
    
    # Check expected tools exist
    expected = [
        "read_file", "write_file", "edit_file", "list_directory", "search_files", "find_files",
        "web_search", "read_webpage",
        "save_context", "load_context", "list_sessions", "compact_now", "context_status",
        "create_plan", "update_step", "get_plan", "add_steps", "remove_step", "cancel_plan", "replace_plan",
        "delegate_task",
    ]
    
    for name in expected:
        assert name in tool_names, f"Missing expected tool: {name}"
    
    print(f"\n✓ All {len(expected)} expected tools present")
    
    print("\n✓ OpenAI Schema tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MeStudio Tool System Tests (Step 4)")
    print("=" * 60)
    
    test_tool_registry()
    test_tool_decorator()
    test_docstring_parsing()
    test_binary_detection()
    test_file_tools()
    test_plan_tools()
    test_context_tools()
    test_agent_tools()
    test_tool_timeout()
    test_tool_error_handling()
    test_result_truncation()
    test_openai_schema_complete()
    
    print("\n" + "=" * 60)
    print("ALL TOOL SYSTEM TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
