"""Integration tests for multi-node agent sequences.

These tests verify that agents work together correctly, with outputs
from one agent feeding correctly into the next.
"""

import pytest
from unittest.mock import patch
from agents.agents.researcher import researcher_node
from agents.agents.planner import planner_node
from agents.agents.implementor import implementor_node
from agents.state import TaskStatus
from agents.testing.fixtures import create_test_state_with_task
from agents.testing.mock_llm import create_mock_llm


@pytest.mark.skip(reason="Requires real LLM - enable for integration testing")
def test_researcher_to_planner_flow(real_llm_if_available):
    """Test that researcher output feeds correctly into planner.
    
    This test requires a real LLM connection. Skip by default.
    """
    state = create_test_state_with_task(
        description="Add player health bar UI",
        measurable_outcome="Health bar visible in game",
    )
    
    # Run researcher
    state.update(researcher_node(state))
    assert state["current_gap_analysis"] is not None
    
    # Run planner with researcher output
    state.update(planner_node(state))
    assert state["current_implementation_plan"] is not None
    assert "health" in state["current_implementation_plan"].lower()


def test_researcher_to_planner_flow_with_mocks(mock_llm_factory):
    """Test researcher -> planner flow with mock LLMs."""
    # Mock researcher response
    researcher_llm = mock_llm_factory([{
        "gap_exists": True,
        "current_state_summary": "No health bar UI exists",
        "desired_state_summary": "Need health bar UI component",
        "gap_description": "Health bar UI missing",
        "relevant_files": ["scripts/ui/HealthBar.gd"],
        "keywords": ["health", "ui", "bar"],
    }])
    
    # Mock planner response
    planner_llm = mock_llm_factory(["""
# Implementation Plan

## Create File: scripts/ui/HealthBar.gd
```gdscript
extends Control
var max_health = 100
var current_health = 100

func _ready():
    update_display()

func update_display():
    # Update health bar visual
    pass
```
"""])
    
    state = create_test_state_with_task(
        description="Add player health bar UI",
        measurable_outcome="Health bar visible in game",
    )
    
    # Run researcher
    with patch("agents.agents.researcher.planning_llm", researcher_llm):
        state.update(researcher_node(state))
    
    assert state["current_gap_analysis"] is not None
    assert state["current_gap_analysis"]["gap_exists"] is True
    
    # Run planner with researcher output
    with patch("agents.agents.planner.planning_llm", planner_llm):
        state.update(planner_node(state))
    
    assert state["current_implementation_plan"] is not None
    assert "HealthBar" in state["current_implementation_plan"]


def test_planner_to_implementor_flow_with_mocks(mock_llm_factory):
    """Test planner -> implementor flow with mock LLMs."""
    # Mock planner response
    planner_llm = mock_llm_factory(["""
# Implementation Plan

## Create File: scripts/Player.gd
```gdscript
extends Node
var health = 100
```
"""])
    
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
    state["current_gap_analysis"] = {
        "gap_exists": True,
        "task_id": "task_001",
        "current_state_summary": "No health system",
        "desired_state_summary": "Need health property",
        "gap_description": "Missing",
        "relevant_files": ["scripts/Player.gd"],
        "keywords": ["health"],
    }
    
    # Run planner
    with patch("agents.agents.planner.planning_llm", planner_llm):
        state.update(planner_node(state))
    
    assert state["current_implementation_plan"] is not None
    
    # Run implementor
    with patch("agents.agents.implementor.coding_llm", implementor_llm):
        state.update(implementor_node(state))
    
    assert state["current_implementation_result"] is not None
    assert state["current_implementation_result"]["success"] is True


def test_full_execution_loop_with_mocks(mock_llm_factory):
    """Test researcher -> planner -> implementor sequence."""
    # Setup mock LLMs for each agent
    researcher_llm = mock_llm_factory([{
        "gap_exists": True,
        "current_state_summary": "No health system",
        "desired_state_summary": "Need health system",
        "gap_description": "Missing",
        "relevant_files": ["scripts/Player.gd"],
        "keywords": ["health"],
    }])
    
    planner_llm = mock_llm_factory(["""
# Implementation Plan

## Create File: scripts/Player.gd
```gdscript
extends Node
var health = 100
```
"""])
    
    implementor_llm = mock_llm_factory([{
        "files_modified": ["scripts/Player.gd"],
        "result_summary": "Created Player.gd",
        "issues_noticed": [],
        "success": True,
    }])
    
    state = create_test_state_with_task(
        description="Add player health",
        measurable_outcome="Player has health property",
    )
    
    # Run researcher
    with patch("agents.agents.researcher.planning_llm", researcher_llm):
        state.update(researcher_node(state))
    
    assert state["current_gap_analysis"]["gap_exists"] is True
    
    # Run planner
    with patch("agents.agents.planner.planning_llm", planner_llm):
        state.update(planner_node(state))
    
    assert state["current_implementation_plan"] is not None
    
    # Run implementor
    with patch("agents.agents.implementor.coding_llm", implementor_llm):
        state.update(implementor_node(state))
    
    assert state["current_implementation_result"]["success"] is True
    
    # Verify task has stored summaries
    from agents.state import TaskTree
    task_tree = TaskTree.from_dict(state["tasks"])
    task = task_tree.tasks["task_001"]
    assert task.gap_analysis is not None
    assert task.implementation_plan_summary is not None
    assert task.result_summary is not None
