"""Unit tests for TaskPlanner."""

from agents.agents.task_planner import (
    task_planner_node,
    _build_messages,
    _extract_plan_from_output,
    _create_synthetic_task,
)
from agents.task_states import create_initial_state


def test_task_planner_handles_no_active_milestone():
    """TaskPlanner returns abort when no active milestone."""
    state = create_initial_state(user_request="Add X", repo_root="/tmp")
    state["milestones_list"] = []
    state["active_milestone_index"] = -1
    result = task_planner_node(state)
    assert result["task_planner_action"] == "abort"
    assert "No active milestone" in result.get("escalation_context", "")


def test_extract_plan_from_output():
    """_extract_plan_from_output extracts PRP from markdown block."""
    content = "Some text\n```markdown\n# Implementation Plan: Test\n\n## Changes\n```"
    plan = _extract_plan_from_output(content)
    assert "# Implementation Plan" in plan
    assert "## Changes" in plan


def test_create_synthetic_task():
    """_create_synthetic_task produces valid Task."""
    task = _create_synthetic_task(
        "Add colony model",
        "# Plan\n\nCreate Colony.gd",
        "milestone_001",
        1,
    )
    assert task.id.startswith("task_tp_")
    assert task.description == "Add colony model"
    assert task.milestone_id == "milestone_001"
    assert task.status.value == "ready"
