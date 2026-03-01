"""Task decomposition into steps using LLM structured output."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from mestudio.core.models import Message
from mestudio.tools.plan_tools import PlanStep, TaskPlan

if TYPE_CHECKING:
    from mestudio.core.llm_client import LMStudioClient


# JSON Schema for structured plan output
PLAN_SCHEMA = {
    "name": "task_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "A concise statement of the overall goal",
            },
            "steps": {
                "type": "array",
                "description": "Ordered list of concrete, actionable steps",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "What this step accomplishes",
                        },
                        "tools_needed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tool names likely needed for this step",
                        },
                    },
                    "required": ["description", "tools_needed"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 10,
            },
        },
        "required": ["goal", "steps"],
        "additionalProperties": False,
    },
}

PLANNER_SYSTEM_PROMPT = """\
You are a task planning assistant. Your job is to break down complex tasks into concrete, actionable steps.

Guidelines:
- Create 3-10 steps (fewer for simple tasks, more for complex ones)
- Each step should be small enough to complete in one focused action
- Steps should be in logical order, with dependencies respected
- Be specific: "Write a Python function that validates email addresses" not "Handle validation"
- Consider what tools will be needed: file operations, web search, code editing, etc.

Available tools the agent can use:
- File tools: read_file, write_file, edit_file, list_directory, search_files, find_files
- Web tools: web_search, read_webpage
- Plan tools: update_step, add_steps, get_plan
- Context tools: save_context, compact_now, context_status

Think step by step about what needs to happen to accomplish the goal."""


class TaskPlanner:
    """Decomposes complex tasks into structured plans using LLM.
    
    The planner uses structured output to generate validated TaskPlan
    objects that can be tracked and executed step by step.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        """Initialize the planner.
        
        Args:
            system_prompt: Custom system prompt. Uses default if not provided.
        """
        self.system_prompt = system_prompt or PLANNER_SYSTEM_PROMPT

    async def decompose(
        self,
        task_description: str,
        llm_client: "LMStudioClient",
    ) -> TaskPlan:
        """Break a complex task into ordered steps.
        
        Uses LLM structured output to generate a validated plan.
        
        Args:
            task_description: The task to decompose.
            llm_client: LLM client for inference.
        
        Returns:
            A TaskPlan with ordered steps.
        """
        logger.info(f"Decomposing task: {task_description[:100]}...")

        messages = [
            Message.system(self.system_prompt),
            Message.user(f"Create a plan for: {task_description}"),
        ]

        try:
            result = await llm_client.structured_output(
                messages=messages,
                schema=PLAN_SCHEMA,
            )

            # Convert to TaskPlan
            steps = [
                PlanStep(
                    index=i + 1,
                    description=step["description"],
                    status="pending",
                    notes=", ".join(step.get("tools_needed", [])),
                )
                for i, step in enumerate(result["steps"])
            ]

            plan = TaskPlan(
                goal=result["goal"],
                steps=steps,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            logger.info(f"Created plan with {len(steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Failed to decompose task: {e}")
            raise

    async def refine_plan(
        self,
        plan: TaskPlan,
        feedback: str,
        llm_client: "LMStudioClient",
    ) -> TaskPlan:
        """Refine an existing plan based on feedback.
        
        Args:
            plan: The current plan.
            feedback: User feedback or context about what needs to change.
            llm_client: LLM client for inference.
        
        Returns:
            A refined TaskPlan.
        """
        logger.info(f"Refining plan based on feedback: {feedback[:100]}...")

        messages = [
            Message.system(self.system_prompt),
            Message.user(
                f"Current plan:\n{plan.format()}\n\n"
                f"Feedback: {feedback}\n\n"
                f"Please create an improved plan addressing the feedback."
            ),
        ]

        try:
            result = await llm_client.structured_output(
                messages=messages,
                schema=PLAN_SCHEMA,
            )

            # Convert to TaskPlan, preserving status of completed steps if descriptions match
            old_step_status = {s.description: s.status for s in plan.steps}
            
            steps = [
                PlanStep(
                    index=i + 1,
                    description=step["description"],
                    status=old_step_status.get(step["description"], "pending"),
                    notes=", ".join(step.get("tools_needed", [])),
                )
                for i, step in enumerate(result["steps"])
            ]

            refined_plan = TaskPlan(
                goal=result["goal"],
                steps=steps,
                created_at=plan.created_at,  # Preserve original creation time
                updated_at=datetime.now(),
            )

            logger.info(f"Refined plan: {len(steps)} steps")
            return refined_plan

        except Exception as e:
            logger.error(f"Failed to refine plan: {e}")
            raise

    async def estimate_complexity(
        self,
        task_description: str,
        llm_client: "LMStudioClient",
    ) -> dict[str, Any]:
        """Estimate task complexity without creating a full plan.
        
        Useful for deciding whether planning is needed.
        
        Args:
            task_description: The task to evaluate.
            llm_client: LLM client for inference.
        
        Returns:
            Dict with complexity assessment.
        """
        complexity_schema = {
            "name": "complexity_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "complexity": {
                        "type": "string",
                        "enum": ["simple", "moderate", "complex"],
                        "description": "Task complexity level",
                    },
                    "estimated_steps": {
                        "type": "integer",
                        "description": "Estimated number of steps needed",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "needs_planning": {
                        "type": "boolean",
                        "description": "Whether formal planning is recommended",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the assessment",
                    },
                },
                "required": ["complexity", "estimated_steps", "needs_planning", "reasoning"],
                "additionalProperties": False,
            },
        }

        messages = [
            Message.system(
                "You are a task complexity assessor. Evaluate tasks and determine "
                "if they need formal planning or can be handled directly."
            ),
            Message.user(f"Assess the complexity of this task: {task_description}"),
        ]

        try:
            return await llm_client.structured_output(
                messages=messages,
                schema=complexity_schema,
            )
        except Exception as e:
            logger.error(f"Failed to estimate complexity: {e}")
            # Default to suggesting planning for safety
            return {
                "complexity": "moderate",
                "estimated_steps": 5,
                "needs_planning": True,
                "reasoning": f"Could not assess: {e}",
            }
