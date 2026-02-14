"""Tests for implementor agent with mock tools."""

import pytest
from unittest.mock import patch, MagicMock
from agents.agents.implementor import implementor_node
from agents.testing.fixtures import create_test_state_with_task


def _make_mock_agent(structured_response: dict):
    """Create a mock agent whose invoke returns the given structured_response."""
    mock = MagicMock()
    mock.invoke.return_value = {"structured_response": structured_response}
    return mock


def test_implementor_executes_plan(test_state_with_task):
    """Test implementor executes implementation plan and parses structured response."""
    state = test_state_with_task
    state["current_implementation_plan"] = """
# Implementation Plan

## Create File: scripts/entities/Player.gd
```gdscript
extends Node
var health = 100
```
"""
    mock_agent = _make_mock_agent({
        "files_modified": ["scripts/entities/Player.gd"],
        "result_summary": "Added health property and methods",
        "issues_noticed": [],
        "success": True,
    })

    with patch("agents.agents.implementor.create_implementor_agent", return_value=mock_agent):
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


def test_implementor_stores_result_in_task(test_state_with_task):
    """Test that implementation result is stored in the task."""
    state = test_state_with_task
    state["current_implementation_plan"] = "Test plan"
    mock_agent = _make_mock_agent({
        "files_modified": ["scripts/Player.gd"],
        "result_summary": "Added health system",
        "issues_noticed": [],
        "success": True,
    })

    with patch("agents.agents.implementor.create_implementor_agent", return_value=mock_agent):
        result = implementor_node(state)

    from agents.task_states import TaskTree
    task_tree = TaskTree.from_dict(result["tasks"])
    task = task_tree.tasks["task_001"]
    assert task.result_summary is not None
    assert "health" in task.result_summary.lower()
