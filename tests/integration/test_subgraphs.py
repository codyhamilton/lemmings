"""Integration tests for multi-node agent sequences.

These tests verify that agents work together correctly, with outputs
from one agent feeding correctly into the next.

Note: task_planner and scope agents use create_agent (LangGraph) which requires
models that support bind_tools. FakeMessagesListChatModel does not, so these
tests are skipped. Use e2e tests or real LLM for full integration testing.
"""

import pytest
from unittest.mock import patch
from agents.agents.task_planner import task_planner_node
from agents.agents.implementor import implementor_node
from agents.task_states import TaskStatus
from agents.testing.fixtures import create_test_state_with_task
from agents.testing.mock_llm import create_mock_llm


@pytest.mark.skip(reason="create_agent requires model with bind_tools; use e2e for integration")
def test_task_planner_to_implementor_flow_with_mocks(mock_llm_factory):
    """Test task_planner -> implementor flow with mock LLMs."""
    # Mock task_planner response (action=implement with PRP)
    task_planner_llm = mock_llm_factory([{
        "action": "implement",
        "task_description": "Add player health property",
        "implementation_plan": """
# Implementation Plan: Add player health

## Task
- Outcome: Player has health property

## Changes
### Modify: `scripts/Player.gd`
**Location**: Top of file
**Change**: Add health property
```gdscript
var health = 100
```

## Files Summary
- Modify: scripts/Player.gd
""",
        "carry_forward": [],
        "escalation_context": "",
    }])

    # Mock implementor response
    implementor_llm = mock_llm_factory([{
        "files_modified": ["scripts/Player.gd"],
        "result_summary": "Created Player.gd with health property",
        "issues_noticed": [],
        "success": True,
    }])

    state = create_test_state_with_task(
        description="Add player health",
        measurable_outcome="Player has health property",
    )
    state["milestones_list"] = [
        {
            "id": "milestone_001",
            "description": "Player has health",
            "sketch": "player scripts",
        },
    ]
    state["active_milestone_index"] = 0
    state["done_list"] = []
    state["carry_forward"] = []

    # Run task_planner
    with patch("agents.agents.task_planner.planning_llm", task_planner_llm):
        state.update(task_planner_node(state))

    assert state["task_planner_action"] == "implement"
    assert state["current_implementation_plan"] is not None
    assert state["current_task_id"] is not None

    # Run implementor
    with patch("agents.agents.implementor.coding_llm", implementor_llm):
        state.update(implementor_node(state))

    assert state["current_implementation_result"] is not None
    assert state["current_implementation_result"]["success"] is True


@pytest.mark.skip(reason="create_agent requires model with bind_tools; use e2e for integration")
def test_task_planner_skip_action_with_mocks(mock_llm_factory):
    """Test task_planner skip action (gap already closed)."""
    task_planner_llm = mock_llm_factory([{
        "action": "skip",
        "task_description": "",
        "implementation_plan": "",
        "carry_forward": ["Next task description"],
        "escalation_context": "",
    }])

    state = create_test_state_with_task()
    state["milestones_list"] = [
        {
            "id": "milestone_001",
            "description": "Player has health",
            "sketch": "player scripts",
        },
    ]
    state["active_milestone_index"] = 0
    state["done_list"] = []
    state["carry_forward"] = []

    with patch("agents.agents.task_planner.planning_llm", task_planner_llm):
        result = task_planner_node(state)

    assert result["task_planner_action"] == "skip"
    assert len(result["done_list"]) == 1
    assert "skip" in result["done_list"][0]["description"].lower() or "gap" in result["done_list"][0]["description"].lower()
