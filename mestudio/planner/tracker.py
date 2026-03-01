"""Plan progress tracking and checkpoints."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger

from mestudio.core.config import get_settings
from mestudio.tools.plan_tools import PlanStep, TaskPlan, get_current_plan, set_current_plan


class PlanTracker:
    """Tracks progress through a task plan.
    
    Provides utilities for advancing through steps, detecting stuck states,
    and generating compact summaries for context injection.
    """

    def __init__(self, plan: TaskPlan | None = None) -> None:
        """Initialize the tracker.
        
        Args:
            plan: Initial plan to track. Can also use set_plan() later.
        """
        self._plan = plan
        self._failure_counts: dict[int, int] = {}  # step index -> failure count
        self._max_failures = 3  # Before marking as "stuck"

    @property
    def plan(self) -> TaskPlan | None:
        """Get the current plan."""
        return self._plan

    def set_plan(self, plan: TaskPlan | None) -> None:
        """Set a new plan to track.
        
        Args:
            plan: The plan to track, or None to clear.
        """
        self._plan = plan
        self._failure_counts = {}
        
        # Also update the global plan state
        set_current_plan(plan)
        
        if plan:
            logger.info(f"Tracking plan: {plan.goal}")

    def next_step(self) -> PlanStep | None:
        """Get the next step to work on.
        
        Returns the first pending or active step. If a step is already active,
        returns that one (so we continue where we left off).
        
        Returns:
            Next PlanStep to work on, or None if plan is complete/empty.
        """
        if not self._plan:
            return None

        # First, look for an active step (resume)
        for step in self._plan.steps:
            if step.status == "active":
                return step

        # Then, find the first pending step
        for step in self._plan.steps:
            if step.status == "pending":
                return step

        return None

    def start_step(self, index: int) -> bool:
        """Mark a step as active (in progress).
        
        Args:
            index: Step index (1-based).
        
        Returns:
            True if step was started, False if not found or invalid state.
        """
        step = self._get_step(index)
        if not step:
            return False

        if step.status not in ("pending", "failed"):
            logger.warning(f"Cannot start step {index}: status is {step.status}")
            return False

        step.status = "active"
        self._plan.updated_at = datetime.now()
        logger.info(f"Started step {index}: {step.description[:50]}...")
        return True

    def mark_done(self, index: int, notes: str = "") -> bool:
        """Mark a step as completed.
        
        Args:
            index: Step index (1-based).
            notes: Optional notes about completion.
        
        Returns:
            True if step was marked done, False if not found.
        """
        step = self._get_step(index)
        if not step:
            return False

        step.status = "done"
        if notes:
            step.notes = notes
        self._plan.updated_at = datetime.now()
        
        # Clear failure count
        self._failure_counts.pop(index, None)
        
        logger.info(f"Completed step {index}: {step.description[:50]}...")
        return True

    def mark_failed(self, index: int, notes: str = "") -> bool:
        """Mark a step as failed.
        
        Increments failure count. After max_failures, the step remains
        failed and is_stuck() will return True.
        
        Args:
            index: Step index (1-based).
            notes: Optional notes about the failure.
        
        Returns:
            True if step was marked failed, False if not found.
        """
        step = self._get_step(index)
        if not step:
            return False

        step.status = "failed"
        if notes:
            step.notes = notes
        self._plan.updated_at = datetime.now()
        
        # Track failures
        self._failure_counts[index] = self._failure_counts.get(index, 0) + 1
        
        logger.warning(
            f"Step {index} failed ({self._failure_counts[index]}/{self._max_failures}): "
            f"{step.description[:50]}..."
        )
        return True

    def skip_step(self, index: int, reason: str = "") -> bool:
        """Mark a step as skipped.
        
        Args:
            index: Step index (1-based).
            reason: Optional reason for skipping.
        
        Returns:
            True if step was skipped, False if not found.
        """
        step = self._get_step(index)
        if not step:
            return False

        step.status = "skipped"
        if reason:
            step.notes = reason
        self._plan.updated_at = datetime.now()
        
        logger.info(f"Skipped step {index}: {step.description[:50]}...")
        return True

    def is_complete(self) -> bool:
        """Check if the plan is complete.
        
        A plan is complete when all steps are done, failed, or skipped.
        
        Returns:
            True if plan is complete or no plan exists.
        """
        if not self._plan:
            return True

        return all(
            step.status in ("done", "failed", "skipped")
            for step in self._plan.steps
        )

    def is_stuck(self) -> bool:
        """Check if the plan is stuck.
        
        A plan is stuck when a step has failed max_failures times
        and there are still pending steps.
        
        Returns:
            True if stuck on a failing step.
        """
        if not self._plan:
            return False

        for step in self._plan.steps:
            if step.status == "failed":
                if self._failure_counts.get(step.index, 0) >= self._max_failures:
                    return True

        return False

    def get_progress(self) -> dict[str, int]:
        """Get progress statistics.
        
        Returns:
            Dict with counts: total, done, failed, pending, active, skipped.
        """
        if not self._plan:
            return {"total": 0, "done": 0, "failed": 0, "pending": 0, "active": 0, "skipped": 0}

        stats = {"total": len(self._plan.steps), "done": 0, "failed": 0, "pending": 0, "active": 0, "skipped": 0}
        for step in self._plan.steps:
            stats[step.status] = stats.get(step.status, 0) + 1
        return stats

    def get_summary(self) -> str:
        """Get a compact plan summary for context injection.
        
        This summary is always preserved during context compaction.
        
        Returns:
            Compact string representation of plan state.
        """
        if not self._plan:
            return ""

        return self._plan.to_context_string()

    def save(self, path: Path | str | None = None) -> Path:
        """Save the current plan to disk.
        
        Args:
            path: Path to save to. Uses default location if not provided.
        
        Returns:
            Path where plan was saved.
        """
        if not self._plan:
            raise ValueError("No plan to save")

        if path is None:
            settings = get_settings()
            plans_dir = Path(settings.data_directory) / "plans"
            plans_dir.mkdir(parents=True, exist_ok=True)
            filename = f"plan_{self._plan.created_at.strftime('%Y%m%d_%H%M%S')}.json"
            path = plans_dir / filename
        else:
            path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self._plan.model_dump_json(indent=2))

        logger.info(f"Saved plan to {path}")
        return path

    def load(self, path: Path | str) -> TaskPlan:
        """Load a plan from disk.
        
        Args:
            path: Path to load from.
        
        Returns:
            The loaded TaskPlan.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Plan file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = TaskPlan.model_validate(data)
        self.set_plan(plan)
        
        logger.info(f"Loaded plan from {path}")
        return plan

    def retry_step(self, index: int) -> bool:
        """Retry a failed step by resetting it to pending.
        
        Args:
            index: Step index (1-based).
        
        Returns:
            True if step was reset, False if not found or not failed.
        """
        step = self._get_step(index)
        if not step:
            return False

        if step.status != "failed":
            logger.warning(f"Cannot retry step {index}: status is {step.status}")
            return False

        step.status = "pending"
        step.notes = f"Retrying (previous attempts: {self._failure_counts.get(index, 0)})"
        self._plan.updated_at = datetime.now()
        
        logger.info(f"Retrying step {index}: {step.description[:50]}...")
        return True

    def _get_step(self, index: int) -> PlanStep | None:
        """Get a step by index.
        
        Args:
            index: Step index (1-based).
        
        Returns:
            PlanStep or None if not found.
        """
        if not self._plan:
            return None

        for step in self._plan.steps:
            if step.index == index:
                return step
        return None