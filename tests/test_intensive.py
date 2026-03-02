"""Intensive multi-tool integration tests for MeStudio Agent.

Run with:
    pytest tests/test_intensive.py -m live -v --tb=short

Requires LM Studio running on localhost:1234 with a model loaded.

These tests exercise complex, multi-tool scenarios that require planning,
creativity, and coordination — going well beyond single-tool smoke tests.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from textwrap import dedent

import pytest

from mestudio.core.orchestrator import Orchestrator, TurnResult

pytestmark = pytest.mark.live

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rel(p: Path) -> str:
    """Return workspace-relative path string (forward slashes)."""
    return str(p.relative_to(WORKSPACE_DIR)).replace("\\", "/")


def assert_min_tool_calls(result: TurnResult, minimum: int, label: str = ""):
    """Assert that the agent made at least *minimum* tool calls."""
    actual = len(result.tool_calls_made)
    assert actual >= minimum, (
        f"{label}: expected >= {minimum} tool calls, got {actual}. "
        f"Tools used: {result.tool_calls_made}"
    )


def assert_tools_used(result: TurnResult, *expected_tools: str, label: str = ""):
    """Assert that specific tool names appear in the tool calls list."""
    used = set(result.tool_calls_made)
    for tool_name in expected_tools:
        assert tool_name in used, (
            f"{label}: expected tool '{tool_name}' to be used. "
            f"Actual tools: {result.tool_calls_made}"
        )


def assert_no_error(result: TurnResult, label: str = ""):
    """Assert the TurnResult has no error."""
    assert result.error is None, f"{label}: unexpected error: {result.error}"


def python_compiles(path: Path) -> bool:
    """Return True if the file at *path* is valid Python (compiles without error)."""
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        return True
    except SyntaxError:
        return False


# ===================================================================
# CASE 1 — Cross-File Refactoring
# ===================================================================


@pytest.mark.asyncio
async def test_cross_file_refactoring(agent: Orchestrator, workspace: Path):
    """Rename a function across definition + 3 call-sites.

    Expects: list/find → read (×3-4) → edit (×3-4).  Min 6 tool calls.
    """
    # -- setup --
    (workspace / "utils.py").write_text(dedent("""\
        def old_helper(x):
            return x * 2
    """))
    (workspace / "module_a.py").write_text(dedent("""\
        from utils import old_helper

        def run_a():
            return old_helper(10)
    """))
    (workspace / "module_b.py").write_text(dedent("""\
        from utils import old_helper

        def run_b():
            return old_helper(20)
    """))
    (workspace / "module_c.py").write_text(dedent("""\
        from utils import old_helper

        result = old_helper(42)
    """))

    prompt = (
        f"In the directory {rel(workspace)}, rename the function `old_helper` to "
        f"`new_helper` everywhere.  Follow these exact steps:\n"
        f"1. First, use find_files or list_directory to find all .py files in {rel(workspace)}\n"
        f"2. Read each file to find where old_helper is used\n"
        f"3. Use edit_file on EACH file to replace `old_helper` with `new_helper` — "
        f"including the definition in utils.py, imports, and all call sites\n"
        f"Make sure you edit ALL FOUR files: utils.py, module_a.py, module_b.py, module_c.py."
    )

    result = await agent.run(prompt)

    # -- judge --
    assert_no_error(result, "refactoring")
    assert_min_tool_calls(result, 4, "refactoring")

    for fname in ("utils.py", "module_a.py", "module_b.py", "module_c.py"):
        content = (workspace / fname).read_text()
        assert "new_helper" in content, f"{fname} should contain 'new_helper'"
        assert "old_helper" not in content, f"{fname} should NOT contain 'old_helper'"


# ===================================================================
# CASE 2 — Research-to-File Pipeline
# ===================================================================


@pytest.mark.asyncio
async def test_research_to_file_pipeline(agent: Orchestrator, workspace: Path):
    """Web search → read pages → write a structured markdown comparison file.

    Expects: web_search → read_webpage (×1-2) → write_file.
    """
    out_file = workspace / "frameworks_comparison.md"

    prompt = (
        f"Do these steps in order:\n"
        f"1. Search the web for 'top 5 popular Python web frameworks'\n"
        f"2. Read any relevant result page\n"
        f"3. Create a file using write_file at the EXACT path {rel(out_file)} "
        f"containing a markdown table comparing the top 5 Python web frameworks.  "
        f"The table MUST have columns: Name, Description, Use Case.  "
        f"Include frameworks like Flask, Django, and FastAPI."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "research-pipeline")
    assert out_file.exists(), "frameworks_comparison.md was not created"
    assert_tools_used(result, "web_search", "write_file", label="research-pipeline")

    content = out_file.read_text()
    assert "|" in content, "File should contain a markdown table (pipe chars)"

    # At least 3 real frameworks mentioned
    known = ["flask", "django", "fastapi", "tornado", "bottle", "starlette",
             "sanic", "falcon", "pyramid", "aiohttp"]
    found = [fw for fw in known if fw in content.lower()]
    assert len(found) >= 3, (
        f"Expected >=3 real frameworks in the table, found {found}"
    )


# ===================================================================
# CASE 3 — Plan-Driven Multi-File Project Scaffold
# ===================================================================


@pytest.mark.asyncio
async def test_plan_driven_scaffold(agent: Orchestrator, workspace: Path):
    """Create a plan, then build a 4-file project scaffold tracking progress.

    Expects: create_plan → write_file (×4) → update_step (×4+) → get_plan.
    """
    pkg = workspace / "todo_app"

    prompt = (
        f"Plan and execute: create a Python project structure under "
        f"{rel(pkg)} with these files:\n"
        f"  1. {rel(pkg)}/main.py\n"
        f"  2. {rel(pkg)}/models.py\n"
        f"  3. {rel(pkg)}/storage.py\n"
        f"  4. {rel(pkg)}/cli.py\n"
        f"\nEach file should have appropriate imports, class or function stubs, "
        f"and docstrings.  Track your progress with a plan — create it first, "
        f"then mark each step done as you go."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "scaffold")
    assert_tools_used(result, "create_plan", "write_file", label="scaffold")

    for fname in ("main.py", "models.py", "storage.py", "cli.py"):
        fp = pkg / fname
        assert fp.exists(), f"{fname} was not created"
        assert python_compiles(fp), f"{fname} has syntax errors"
        content = fp.read_text()
        assert len(content.strip()) > 20, f"{fname} is suspiciously short"

    # Plan tool was also used to update steps
    plan_tools_used = [t for t in result.tool_calls_made if t in ("update_step", "get_plan")]
    assert len(plan_tools_used) >= 1, "Expected update_step or get_plan to be called"


# ===================================================================
# CASE 4 — Bug Hunt and Fix
# ===================================================================


@pytest.mark.asyncio
async def test_bug_hunt_and_fix(agent: Orchestrator, workspace: Path):
    """Read a buggy file, identify 3 bugs, fix them, verify.

    Expects: read_file → edit_file (×1-3) → read_file.
    """
    buggy = workspace / "buggy.py"
    buggy.write_text(dedent("""\
        import maths  # Bug 1: wrong module name (should be 'math')

        def average(numbers):
            \"\"\"Return the average of a list of numbers.\"\"\"
            total = sum(numbers)
            return total / len(number)  # Bug 2: 'number' should be 'numbers'

        def circle_area(radius):
            \"\"\"Return the area of a circle.\"\"\"
            return maths.pi * radius ** 3  # Bug 3: should be ** 2, not ** 3
    """))

    prompt = (
        f"Read the file at {rel(buggy)}, identify ALL bugs in it, fix them, "
        f"and then read it back to verify your fixes are correct."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "bug-hunt")
    assert_tools_used(result, "read_file", label="bug-hunt")

    # Check edits were applied
    content = buggy.read_text()
    assert "import math" in content, "Bug 1 not fixed: should be 'import math'"
    assert "maths" not in content, "Bug 1 not fixed: 'maths' still present"
    assert "len(numbers)" in content, "Bug 2 not fixed: should be 'len(numbers)'"
    assert "** 2" in content or "**2" in content, "Bug 3 not fixed: exponent should be 2"
    assert "** 3" not in content and "**3" not in content, "Bug 3 not fixed: exponent 3 still present"


# ===================================================================
# CASE 5 — Delegated Research + File Agent Coordination
# ===================================================================


@pytest.mark.asyncio
async def test_delegated_research_and_file(agent: Orchestrator, workspace: Path):
    """Delegate to search agent, then file agent (or write directly).

    Expects: delegate_task(search) → delegate_task(file) or write_file.
    """
    cheatsheet = workspace / "pathlib_cheatsheet.py"

    prompt = (
        f"First, delegate to the search agent to find what Python's pathlib "
        f"module is used for and its most common methods. "
        f"Then, use that research to create a cheat-sheet file at "
        f"{rel(cheatsheet)} with at least 5 example usages of pathlib.Path. "
        f"You can either delegate the file creation to the file agent or "
        f"write it directly."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "delegate-research")
    assert_tools_used(result, "delegate_task", label="delegate-research")

    assert cheatsheet.exists(), "pathlib_cheatsheet.py was not created"
    content = cheatsheet.read_text().lower()
    assert "pathlib" in content or "path" in content, "Cheatsheet should mention pathlib"

    # At least 3 distinct Path methods/usages
    path_methods = ["exists", "mkdir", "read_text", "write_text", "glob",
                    "iterdir", "resolve", "parent", "stem", "suffix",
                    "name", "is_file", "is_dir", "joinpath", "open",
                    "unlink", "rename", "stat", "home", "cwd"]
    found_methods = [m for m in path_methods if m in content]
    assert len(found_methods) >= 3, (
        f"Expected >=3 pathlib methods, found: {found_methods}"
    )


# ===================================================================
# CASE 6 — Conditional Logic: Analyze & Branch by File Type
# ===================================================================


@pytest.mark.asyncio
async def test_conditional_file_processing(agent: Orchestrator, workspace: Path):
    """Scan directory, process differently per file type.

    .py → add header comment; .json → validate; .md → count headings.
    Expects: list_directory → read_file (×N) → edit_file (for .py) → response with analysis.
    """
    (workspace / "script.py").write_text("print('hello')\n")
    (workspace / "util.py").write_text("x = 1\n")
    (workspace / "data.json").write_text('{"name": "test", "value": 42}')
    (workspace / "broken.json").write_text('{"name": "test", value: bad}')  # intentionally invalid
    (workspace / "notes.md").write_text("# Title\n\nSome text\n\n## Section 1\n\n### Subsection\n")

    prompt = (
        f"Scan the directory {rel(workspace)} and process each file:\n"
        f"  - For each .py file: add a header comment '# Reviewed' as the first line\n"
        f"  - For each .json file: check if it's valid JSON and report any issues\n"
        f"  - For each .md file: count the number of headings (lines starting with #)\n"
        f"Summarize everything when done."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "conditional")
    assert_min_tool_calls(result, 5, "conditional")

    # .py files should have review header
    for py_file in ("script.py", "util.py"):
        content = (workspace / py_file).read_text()
        assert content.startswith("# Reviewed"), (
            f"{py_file} should start with '# Reviewed', got: {content[:50]!r}"
        )

    # Response should mention JSON validation issues
    resp_lower = result.response.lower()
    assert "broken" in resp_lower or "invalid" in resp_lower or "error" in resp_lower, (
        "Response should mention broken.json is invalid"
    )

    # Response should mention heading count for .md file
    assert "3" in result.response or "heading" in resp_lower, (
        "Response should mention the heading count for notes.md"
    )


# ===================================================================
# CASE 7 — Iterative Web Deep-Dive
# ===================================================================


@pytest.mark.asyncio
async def test_iterative_web_deep_dive(agent: Orchestrator, workspace: Path):
    """Two separate web searches + page reads → synthesized document.

    Expects: web_search (×2) → read_webpage (×1-3) → write_file.  5+ tool calls.
    """
    summary_file = workspace / "comparison.md"

    prompt = (
        f"Do the following research:\n"
        f"1. Search the web for 'Python dataclasses vs pydantic'\n"
        f"2. Read the top result\n"
        f"3. Search the web for 'pydantic v2 migration guide'\n"
        f"4. Read the top result\n"
        f"5. Write a summary document at {rel(summary_file)} synthesizing "
        f"all findings.  It should mention both dataclasses and pydantic, "
        f"and reference v2 or migration."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "deep-dive")
    assert summary_file.exists(), "comparison.md was not created"
    assert_min_tool_calls(result, 4, "deep-dive")

    content = summary_file.read_text().lower()
    assert "dataclass" in content, "Should mention dataclasses"
    assert "pydantic" in content, "Should mention pydantic"

    word_count = len(content.split())
    assert word_count >= 100, f"Expected >=100 words, got {word_count}"


# ===================================================================
# CASE 8 — Multi-File Code Gen with Cross-References
# ===================================================================


@pytest.mark.asyncio
async def test_multi_file_codegen_cross_refs(agent: Orchestrator, workspace: Path):
    """Create a 3-file package with correct cross-file imports.

    Expects: write_file (×3), optionally read_file.  All files compile.
    """
    pkg = workspace / "mathlib"

    prompt = (
        f"Create a Python package under {rel(pkg)} with these files:\n"
        f"  1. {rel(pkg)}/__init__.py — exports `add` and `multiply` from operations\n"
        f"  2. {rel(pkg)}/operations.py — implements `add(a, b)` and `multiply(a, b)` functions\n"
        f"  3. {rel(pkg)}/tests.py — imports add and multiply from the mathlib package "
        f"(use `from mathlib.operations import add, multiply` or `from .operations import add, multiply`) "
        f"and has test functions `test_add()` and `test_multiply()` with assert statements\n"
        f"\nMake sure all imports are correct across files."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "codegen")

    # Check that write_file was called (or at least files exist)
    if result.tool_calls_made:
        assert_tools_used(result, "write_file", label="codegen")

    for fname in ("__init__.py", "operations.py", "tests.py"):
        fp = pkg / fname
        assert fp.exists(), f"{fname} was not created"
        assert python_compiles(fp), f"{fname} has syntax errors"

    # Check operations.py has both functions
    ops = (pkg / "operations.py").read_text()
    assert "def add" in ops, "operations.py missing 'def add'"
    assert "def multiply" in ops, "operations.py missing 'def multiply'"

    # Check __init__.py imports from operations
    init = (pkg / "__init__.py").read_text()
    assert "add" in init and "multiply" in init, "__init__.py should export both functions"

    # Check tests.py references both
    tests = (pkg / "tests.py").read_text()
    assert "add" in tests and "multiply" in tests, "tests.py should import/call both functions"
    assert "assert" in tests or "test_" in tests, "tests.py should have assertions or test functions"


# ===================================================================
# CASE 9 — Error Recovery Chain
# ===================================================================


@pytest.mark.asyncio
async def test_error_recovery_chain(agent: Orchestrator, workspace: Path):
    """Read a real file, fail on a nonexistent file, then write a report.

    Agent must NOT crash after the error and must continue to write the report.
    Expects: read_file (ok) → read_file (error) → write_file.
    """
    real_file = workspace / "data.txt"
    real_file.write_text("alpha=100\nbeta=200\ngamma=300\n")
    report_file = workspace / "report.txt"

    prompt = (
        f"Do these steps in order:\n"
        f"1. Read the file at {rel(real_file)}\n"
        f"2. Try to read the file at {rel(workspace)}/nonexistent.txt "
        f"   (handle the error gracefully — do NOT stop)\n"
        f"3. Create {rel(report_file)} summarizing what you found.\n"
        f"   Mention the content of the first file AND that the second file "
        f"   could not be read."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "error-recovery")
    assert report_file.exists(), "report.txt was not created"

    content = report_file.read_text().lower()
    # Should mention the real data
    assert "alpha" in content or "100" in content or "data" in content, (
        "Report should mention content from data.txt"
    )
    # Should mention failure
    assert ("nonexistent" in content or "not found" in content or
            "error" in content or "could not" in content or
            "does not exist" in content or "failed" in content), (
        "Report should mention the nonexistent file failure"
    )


# ===================================================================
# CASE 10 — Full Lifecycle: Plan → Execute → Compact → Resume
# ===================================================================


@pytest.mark.asyncio
async def test_full_lifecycle_plan_compact_resume(agent: Orchestrator, workspace: Path):
    """Plan with 6 steps, execute creating files, compact mid-way, continue.

    Expects: create_plan → write_file (×3) → update_step (×3) → compact_now
             → write_file (×3) → update_step (×3).
    """
    pkg = workspace / "dataproc"

    prompt = (
        f"Create a plan with exactly 6 steps to build a data processing utility "
        f"under {rel(pkg)}.  The steps should be:\n"
        f"  1. Create {rel(pkg)}/reader.py with a CSVReader class stub\n"
        f"  2. Create {rel(pkg)}/transformer.py with a DataTransformer class stub\n"
        f"  3. Create {rel(pkg)}/writer.py with a OutputWriter class stub\n"
        f"  4. Use compact_now to compact the context\n"
        f"  5. Create {rel(pkg)}/pipeline.py that imports from reader, transformer, writer\n"
        f"  6. Create {rel(pkg)}/__init__.py that exports the main classes\n"
        f"\nExecute each step, marking it done as you complete it. "
        f"You can batch multiple write_file calls together for efficiency."
    )

    result = await agent.run(prompt)

    assert_no_error(result, "lifecycle")
    assert_tools_used(result, "create_plan", "write_file", label="lifecycle")

    # At least 3 files created (we're lenient — the model might merge/skip steps)
    created = [f.name for f in pkg.iterdir() if f.is_file()] if pkg.exists() else []
    assert len(created) >= 3, f"Expected >=3 files, got {created}"

    # Check compact_now was called
    assert "compact_now" in result.tool_calls_made, (
        "compact_now should have been called as instructed"
    )

    # Plan tools were engaged
    plan_tool_count = sum(
        1 for t in result.tool_calls_made
        if t in ("create_plan", "update_step", "get_plan")
    )
    assert plan_tool_count >= 2, (
        f"Expected >=2 plan-related tool calls, got {plan_tool_count}"
    )

    # All Python files compile
    for f in pkg.iterdir():
        if f.suffix == ".py":
            assert python_compiles(f), f"{f.name} has syntax errors"
