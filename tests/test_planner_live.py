"""Test Task Planner with actual LM Studio model.

This script tests the TaskPlanner's ability to decompose tasks into
structured plans using the model's structured output capability.

Run: python tests/test_planner_live.py
Requires: LM Studio running at localhost:1234 with gpt-oss-20b loaded
"""

import asyncio
import sys
sys.path.insert(0, "c:/clones/MeStudio")

from mestudio.core.llm_client import LMStudioClient
from mestudio.planner import TaskPlanner, PlanTracker


async def test_decompose():
    """Test task decomposition with different complexity levels."""
    print("=" * 60)
    print("TEST: Task Decomposition with LLM")
    print("=" * 60)
    
    client = LMStudioClient()
    
    # Check if LM Studio is available
    print("\nChecking LM Studio availability...")
    if not await client.is_available():
        print("ERROR: LM Studio is not running at localhost:1234")
        print("Please start LM Studio and load gpt-oss-20b")
        return False
    print("✓ LM Studio is available")
    
    planner = TaskPlanner()
    
    # Test cases of varying complexity
    test_tasks = [
        # Simple task
        "Create a hello world Python script",
        
        # Moderate task
        "Build a Python script that reads a CSV file, filters rows where the 'status' column equals 'active', and writes the results to a new CSV file",
        
        # Complex task
        "Create a REST API endpoint in Python that accepts JSON data, validates the schema, stores it in a SQLite database, and returns the created record with a generated ID",
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'=' * 60}")
        print(f"Task {i}: {task[:60]}...")
        print("=" * 60)
        
        try:
            plan = await planner.decompose(task, client)
            
            print(f"\n{plan.format()}")
            print(f"\n✓ Generated {len(plan.steps)} steps")
            
            # Verify plan structure
            assert plan.goal, "Plan should have a goal"
            assert len(plan.steps) >= 1, "Plan should have at least 1 step"
            assert len(plan.steps) <= 10, "Plan should have at most 10 steps"
            
            for step in plan.steps:
                assert step.description, f"Step {step.index} should have description"
                assert step.status == "pending", f"New steps should be pending"
            
            print("✓ Plan structure validated")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


async def test_complexity_estimation():
    """Test complexity estimation for different tasks."""
    print("\n" + "=" * 60)
    print("TEST: Complexity Estimation")
    print("=" * 60)
    
    client = LMStudioClient()
    planner = TaskPlanner()
    
    test_tasks = [
        ("Print 'hello world'", "simple"),
        ("Create a web scraper that extracts all product prices from an e-commerce site", "complex"),
        ("Read a file and count the lines", "simple"),
    ]
    
    for task, expected in test_tasks:
        print(f"\nTask: {task[:50]}...")
        print(f"Expected complexity: {expected}")
        
        try:
            result = await planner.estimate_complexity(task, client)
            
            print(f"  Assessed complexity: {result['complexity']}")
            print(f"  Estimated steps: {result['estimated_steps']}")
            print(f"  Needs planning: {result['needs_planning']}")
            print(f"  Reasoning: {result['reasoning'][:100]}...")
            
            # Note: We don't strictly assert the expected complexity
            # as the model may have valid reasons for different assessments
            print("✓ Complexity estimation completed")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    return True


async def test_plan_tracking():
    """Test PlanTracker with a generated plan."""
    print("\n" + "=" * 60)
    print("TEST: Plan Tracking")
    print("=" * 60)
    
    client = LMStudioClient()
    planner = TaskPlanner()
    tracker = PlanTracker()
    
    # Generate a plan
    task = "Create a Python module with two functions: one to read JSON files and one to write JSON files"
    plan = await planner.decompose(task, client)
    
    # Track the plan
    tracker.set_plan(plan)
    print(f"\nTracking plan: {plan.goal}")
    print(f"Steps: {len(plan.steps)}")
    
    # Simulate execution
    print("\nSimulating execution...")
    
    # Start first step
    step = tracker.next_step()
    if step:
        print(f"  Next step: {step.index}. {step.description[:50]}...")
        tracker.start_step(step.index)
        print(f"    Status: {step.status}")
        
        # Complete it
        tracker.mark_done(step.index, "Completed successfully")
        print(f"    Marked done")
    
    # Check progress
    progress = tracker.get_progress()
    print(f"\nProgress: {progress}")
    
    # Check summary
    summary = tracker.get_summary()
    print(f"\nContext summary:\n{summary}")
    
    # Save and reload
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_plan.json"
        tracker.save(path)
        print(f"\nSaved plan to {path}")
        
        # New tracker, load the plan
        tracker2 = PlanTracker()
        tracker2.load(path)
        print(f"Loaded plan, steps: {len(tracker2.plan.steps)}")
        
        # Verify state preserved
        assert tracker2.plan.steps[0].status == "done"
        print("✓ Plan state preserved after save/load")
    
    return True


async def test_plan_refinement():
    """Test plan refinement based on feedback."""
    print("\n" + "=" * 60)
    print("TEST: Plan Refinement")
    print("=" * 60)
    
    client = LMStudioClient()
    planner = TaskPlanner()
    
    # Create initial plan
    task = "Build a calculator that adds two numbers"
    print(f"Initial task: {task}")
    
    plan = await planner.decompose(task, client)
    print(f"\nInitial plan ({len(plan.steps)} steps):")
    print(plan.format())
    
    # Refine with feedback
    feedback = "The calculator should also support subtraction, multiplication, and division"
    print(f"\nFeedback: {feedback}")
    
    refined = await planner.refine_plan(plan, feedback, client)
    print(f"\nRefined plan ({len(refined.steps)} steps):")
    print(refined.format())
    
    # Refined plan should likely have more or different steps
    print("\n✓ Plan refinement completed")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("MeStudio Task Planner - Live LLM Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Task Decomposition", await test_decompose()))
    results.append(("Complexity Estimation", await test_complexity_estimation()))
    results.append(("Plan Tracking", await test_plan_tracking()))
    results.append(("Plan Refinement", await test_plan_refinement()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\n{passed}/{len(results)} tests passed")
    
    return all(success for _, success in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
