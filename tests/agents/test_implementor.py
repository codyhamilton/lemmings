"""Tests for implementor agent with mock tools."""

import pytest
from unittest.mock import patch
from agents.agents.implementor import implementor_node
from agents.task_states import TaskStatus
from agents.testing.fixtures import create_test_state_with_task


def test_implementor_executes_plan(mock_llm_factory, mock_tools_factory, test_state_with_task):
    """Test implementor executes implementation plan."""
    # Mock LLM that returns implementation result
    mock_llm = mock_llm_factory([{
        "files_modified": ["scripts/entities/Player.gd"],
        "result_summary": "Added health property and methods",
        "issues_noticed": [],
        "success": True,
    }])
    
    # Mock tools that track calls
    mock_responses = {}
    mock_tools = mock_tools_factory(mock_responses)
    
    state = test_state_with_task
    state["current_implementation_plan"] = """
# Implementation Plan

## Create File: scripts/entities/Player.gd
```gdscript
extends Node
var health = 100
var max_health = 100

func damage(amount: int):
    health = max(0, health - amount)

func heal(amount: int):
    health = min(max_health, health + amount)

func is_alive() -> bool:
    return health > 0
```
"""
    
    with patch("agents.agents.implementor.coding_llm", mock_llm):
        # Note: In a real test, we'd also need to mock the tools
        # For now, this tests the LLM response parsing
        result = implementor_node(state)
    
    assert result["current_implementation_result"] is not None
    impl_result = result["current_implementation_result"]
    assert impl_result["success"] is True
    assert "health" in impl_result["result_summary"].lower()


def test_implementor_handles_missing_plan(test_state_with_task):
    """Test implementor handles missing implementation plan."""
    state = test_state_with_task
    state["current_implementation_plan"] = None
    
    result = implementor_node(state)
    
    assert "error" in result
    assert "No implementation plan" in result["error"]


def test_implementor_handles_missing_task():
    """Test implementor handles missing current_task_id."""
    from agents.testing.fixtures import create_test_state
    
    state = create_test_state()
    state["current_task_id"] = None
    state["current_implementation_plan"] = "Test plan"
    
    result = implementor_node(state)
    
    assert "error" in result
    assert "No current task" in result["error"]


def test_implementor_stores_result_in_task(mock_llm_factory, test_state_with_task):
    """Test that implementation result is stored in the task."""
    mock_llm = mock_llm_factory([{
        "files_modified": ["scripts/Player.gd"],
        "result_summary": "Added health system",
        "issues_noticed": [],
        "success": True,
    }])
    
    state = test_state_with_task
    state["current_implementation_plan"] = "Test plan"
    
    with patch("agents.agents.implementor.coding_llm", mock_llm):
        result = implementor_node(state)
    
    from agents.task_states import TaskTree
    task_tree = TaskTree.from_dict(result["tasks"])
    task = task_tree.tasks["task_001"]
    assert task.result_summary is not None
    assert "health" in task.result_summary.lower()
