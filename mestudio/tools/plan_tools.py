"""Plan create, update, cancel, replace tools."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from mestudio.core.config import get_settings
from mestudio.tools.registry import tool


class PlanStep(BaseModel):
    """A single step in a task plan."""

    index: int
    description: str
    status: Literal["pending", "active", "done", "failed", "skipped"] = "pending"
    notes: str = ""
    sub_steps: list["PlanStep"] = Field(default_factory=list)

    def format(self, indent: int = 0) -> str:
        """Format step for display."""
        icons = {
            "pending": "○",
            "active": "●",
            "done": "✓",
            "failed": "✗",
            "skipped": "⊘",
        }
        icon = icons.get(self.status, "?")
        prefix = "  " * indent
        
        line = f"{prefix}{icon} {self.index}. {self.description}"
        if self.notes:
            line += f" ({self.notes})"
        
        lines = [line]
        for sub in self.sub_steps:
            lines.append(sub.format(indent + 1))
        
        return "\n".join(lines)


class TaskPlan(BaseModel):
    """A task plan with ordered steps."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def format(self) -> str:
        """Format plan for display."""
        lines = [
            f"Goal: {self.goal}",
            f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "Steps:",
        ]
        
        for step in self.steps:
            lines.append(step.format())
        
        # Add progress summary
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == "done")
        failed = sum(1 for s in self.steps if s.status == "failed")
        active = sum(1 for s in self.steps if s.status == "active")
        
        lines.append("")
        lines.append(f"Progress: {done}/{total} done")
        if active:
            lines.append(f"Active: {active}")
        if failed:
            lines.append(f"Failed: {failed}")
        
        return "\n".join(lines)

    def to_context_string(self) -> str:
        """Format plan for context injection."""
        lines = [f"## Plan: {self.goal}", ""]
        
        for step in self.steps:
            icons = {
                "pending": "[ ]",
                "active": "[→]",
                "done": "[x]",
                "failed": "[!]",
                "skipped": "[-]",
            }
            icon = icons.get(step.status, "[ ]")
            lines.append(f"{icon} {step.index}. {step.description}")
            if step.notes:
                lines.append(f"    Note: {step.notes}")
        
        return "\n".join(lines)


# Global current plan
_current_plan: TaskPlan | None = None


def get_current_plan() -> TaskPlan | None:
    """Get the current plan."""
    return _current_plan


def set_current_plan(plan: TaskPlan | None) -> None:
    """Set the current plan."""
    global _current_plan
    _current_plan = plan


def _get_plans_dir() -> Path:
    """Get the plans directory."""
    settings = get_settings()
    plans_dir = Path(settings.data_directory) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


def _save_plan(plan: TaskPlan) -> Path:
    """Save plan to disk."""
    plans_dir = _get_plans_dir()
    filename = f"plan_{plan.created_at.strftime('%Y%m%d_%H%M%S')}.json"
    path = plans_dir / filename
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))
    
    logger.debug(f"Saved plan to {path}")
    return path


@tool(
    name="create_plan",
    description="Create a new task plan with ordered steps. Use for complex multi-step tasks.",
)
async def create_plan(goal: str, steps: list[str]) -> str:
    """Create a new task plan.
    
    Args:
        goal: The overall goal of the plan.
        steps: List of step descriptions in order.
    """
    global _current_plan
    
    if _current_plan is not None:
        return (
            f"Error: A plan already exists (goal: '{_current_plan.goal}'). "
            "Use replace_plan to replace it, or cancel_plan to discard it first."
        )
    
    if not goal:
        return "Error: Goal cannot be empty"
    
    if not steps:
        return "Error: Plan must have at least one step"
    
    plan_steps = [
        PlanStep(index=i + 1, description=desc)
        for i, desc in enumerate(steps)
    ]
    
    _current_plan = TaskPlan(goal=goal, steps=plan_steps)
    
    # Save to disk
    try:
        path = _save_plan(_current_plan)
    except Exception as e:
        logger.warning(f"Failed to save plan: {e}")
    
    logger.info(f"Created plan: {goal} ({len(steps)} steps)")
    
    return f"Plan created successfully!\n\n{_current_plan.format()}"


@tool(
    name="update_step",
    description="Update the status of a plan step. Use to mark steps as done, failed, etc.",
)
async def update_step(
    step_index: int,
    status: str,
    notes: str = "",
) -> str:
    """Update a plan step.
    
    Args:
        step_index: The step number to update (1-indexed).
        status: New status - 'pending', 'active', 'done', 'failed', or 'skipped'.
        notes: Optional notes about this step.
    """
    global _current_plan
    
    if _current_plan is None:
        return "Error: No active plan. Use create_plan to create one."
    
    valid_statuses = ["pending", "active", "done", "failed", "skipped"]
    if status not in valid_statuses:
        return f"Error: Invalid status '{status}'. Use: {', '.join(valid_statuses)}"
    
    # Find the step
    step = None
    for s in _current_plan.steps:
        if s.index == step_index:
            step = s
            break
    
    if step is None:
        return f"Error: Step {step_index} not found. Plan has {len(_current_plan.steps)} steps."
    
    old_status = step.status
    step.status = status
    if notes:
        step.notes = notes
    
    _current_plan.updated_at = datetime.now()
    
    # Save updated plan
    try:
        _save_plan(_current_plan)
    except Exception as e:
        logger.warning(f"Failed to save plan: {e}")
    
    logger.info(f"Step {step_index}: {old_status} → {status}")
    
    return f"Step {step_index} updated: {old_status} → {status}\n\n{_current_plan.format()}"


@tool(
    name="get_plan",
    description="Get the current plan with status of all steps.",
)
async def get_plan() -> str:
    """Get the current plan."""
    if _current_plan is None:
        return "No active plan. Use create_plan to create one."
    
    return _current_plan.format()


@tool(
    name="add_steps",
    description="Add new steps to the current plan.",
)
async def add_steps(
    steps: list[str],
    after_index: int | None = None,
) -> str:
    """Add steps to the plan.
    
    Args:
        steps: List of new step descriptions.
        after_index: Insert after this step index. If None, appends to end.
    """
    global _current_plan
    
    if _current_plan is None:
        return "Error: No active plan. Use create_plan to create one."
    
    if not steps:
        return "Error: No steps provided"
    
    # Find insertion point
    if after_index is None:
        insert_pos = len(_current_plan.steps)
    else:
        insert_pos = None
        for i, s in enumerate(_current_plan.steps):
            if s.index == after_index:
                insert_pos = i + 1
                break
        
        if insert_pos is None:
            return f"Error: Step {after_index} not found"
    
    # Create new steps (with temporary indices)
    new_steps = [PlanStep(index=0, description=desc) for desc in steps]
    
    # Insert
    _current_plan.steps[insert_pos:insert_pos] = new_steps
    
    # Re-index all steps
    for i, step in enumerate(_current_plan.steps):
        step.index = i + 1
    
    _current_plan.updated_at = datetime.now()
    
    try:
        _save_plan(_current_plan)
    except Exception as e:
        logger.warning(f"Failed to save plan: {e}")
    
    logger.info(f"Added {len(steps)} steps")
    
    return f"Added {len(steps)} step(s).\n\n{_current_plan.format()}"


@tool(
    name="remove_step",
    description="Remove a step from the plan.",
)
async def remove_step(step_index: int) -> str:
    """Remove a step from the plan.
    
    Args:
        step_index: The step number to remove (1-indexed).
    """
    global _current_plan
    
    if _current_plan is None:
        return "Error: No active plan."
    
    # Find and remove
    found = False
    for i, step in enumerate(_current_plan.steps):
        if step.index == step_index:
            _current_plan.steps.pop(i)
            found = True
            break
    
    if not found:
        return f"Error: Step {step_index} not found"
    
    # Re-index
    for i, step in enumerate(_current_plan.steps):
        step.index = i + 1
    
    _current_plan.updated_at = datetime.now()
    
    try:
        _save_plan(_current_plan)
    except Exception as e:
        logger.warning(f"Failed to save plan: {e}")
    
    logger.info(f"Removed step {step_index}")
    
    return f"Step removed.\n\n{_current_plan.format()}"


@tool(
    name="cancel_plan",
    description="Discard the current plan entirely.",
)
async def cancel_plan() -> str:
    """Cancel and discard the current plan."""
    global _current_plan
    
    if _current_plan is None:
        return "No active plan to cancel."
    
    goal = _current_plan.goal
    _current_plan = None
    
    logger.info(f"Cancelled plan: {goal}")
    
    return f"Plan cancelled: '{goal}'"


@tool(
    name="replace_plan",
    description="Replace the current plan with a new one. Preserves notes from completed steps in history.",
)
async def replace_plan(goal: str, steps: list[str]) -> str:
    """Replace the current plan.
    
    Args:
        goal: The new goal.
        steps: List of new step descriptions.
    """
    global _current_plan
    
    old_plan = _current_plan
    _current_plan = None
    
    # Create new plan
    result = await create_plan(goal, steps)
    
    if old_plan:
        # Log completed steps from old plan
        completed = [s for s in old_plan.steps if s.status in ("done", "failed")]
        if completed:
            notes = [f"  • Step {s.index} ({s.status}): {s.description}" for s in completed]
            logger.info(f"Replaced plan. Previous completed steps:\n" + "\n".join(notes))
    
    return result


__all__ = [
    "PlanStep",
    "TaskPlan",
    "create_plan",
    "update_step",
    "get_plan",
    "add_steps",
    "remove_step",
    "cancel_plan",
    "replace_plan",
    "get_current_plan",
    "set_current_plan",
]
