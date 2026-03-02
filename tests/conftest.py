"""Shared fixtures for MeStudio integration tests."""

from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from mestudio.core.orchestrator import Orchestrator, OrchestratorConfig, NullOutputHandler
from mestudio.core.config import get_settings
from mestudio.tools.context_tools import set_context_manager


WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
TEST_TEMP_BASE = WORKSPACE_DIR / "test_temp"


# ---------- module-scoped orchestrator (one init for all tests) ----------

_orchestrator: Orchestrator | None = None


async def _get_orchestrator() -> Orchestrator:
    """Lazily create and initialize a single Orchestrator for the whole session."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    settings = get_settings()
    settings.working_directory = str(WORKSPACE_DIR)

    config = OrchestratorConfig(
        max_tool_iterations=20,
        max_parallel_tools=5,
    )
    orch = Orchestrator(
        settings=settings,
        config=config,
        output_handler=NullOutputHandler(),
    )
    ok = await orch.initialize()
    if not ok:
        pytest.fail(
            "LM Studio is NOT running on localhost:1234. "
            "Start LM Studio with a model loaded before running these tests."
        )
    set_context_manager(orch.context)
    _orchestrator = orch
    return orch


@pytest.fixture()
async def agent() -> Orchestrator:
    """Per-test fixture: resets orchestrator state and re-wires context tools."""
    orch = await _get_orchestrator()
    await orch.reset()
    set_context_manager(orch.context)
    return orch


@pytest.fixture()
def workspace() -> Path:
    """Per-test temp directory inside the workspace root so file tools can reach it."""
    TEST_TEMP_BASE.mkdir(exist_ok=True)
    test_dir = TEST_TEMP_BASE / f"t_{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp():
    """Session-level cleanup of test_temp after all tests finish."""
    yield
    if TEST_TEMP_BASE.exists():
        shutil.rmtree(TEST_TEMP_BASE, ignore_errors=True)
