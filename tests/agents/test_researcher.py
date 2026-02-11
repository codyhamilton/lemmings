"""Tests for researcher agent with mock LLM."""

import pytest
from unittest.mock import patch
from agents.agents.researcher import researcher_node
from agents.task_states import TaskStatus
from agents.testing.fixtures import create_test_state_with_task


def test_researcher_detects_gap(mock_llm_factory, test_state_with_task):
    """Test researcher detects gap when gap_exists is True."""
    mock_llm = mock_llm_factory([{
        "gap_exists": True,
        "current_state_summary": "No health system exists",
        "desired_state_summary": "Need player health with damage/heal",
        "gap_description": "Health system not implemented",
        "relevant_files": ["scripts/entities/Player.gd"],
        "keywords": ["health", "damage"],
    }])
    
    with patch("agents.agents.researcher.planning_llm", mock_llm):
        result = researcher_node(test_state_with_task)
    
    assert result["current_gap_analysis"] is not None
    assert result["current_gap_analysis"]["gap_exists"] is True
    assert "health" in result["current_gap_analysis"]["keywords"]
    assert "scripts/entities/Player.gd" in result["current_gap_analysis"]["relevant_files"]


def test_researcher_detects_no_gap(mock_llm_factory, test_state_with_task):
    """Test researcher detects no gap when gap_exists is False."""
    mock_llm = mock_llm_factory([{
        "gap_exists": False,
        "current_state_summary": "Health system already exists in Player.gd",
        "desired_state_summary": "Need player health with damage/heal",
        "gap_description": "Feature already implemented",
        "relevant_files": ["scripts/entities/Player.gd"],
        "keywords": ["health"],
    }])
    
    with patch("agents.agents.researcher.planning_llm", mock_llm):
        result = researcher_node(test_state_with_task)
    
    assert result["current_gap_analysis"] is not None
    assert result["current_gap_analysis"]["gap_exists"] is False


def test_researcher_with_fixture(mock_llm_factory, researcher_input_fixture, researcher_gap_exists_response):
    """Test researcher using fixture state and response."""
    mock_llm = mock_llm_factory([researcher_gap_exists_response])
    
    with patch("agents.agents.researcher.planning_llm", mock_llm):
        result = researcher_node(researcher_input_fixture)
    
    assert result["current_gap_analysis"] is not None
    gap_analysis = result["current_gap_analysis"]
    assert gap_analysis["gap_exists"] is True
    assert gap_analysis["task_id"] == "task_001"


def test_researcher_handles_missing_task():
    """Test researcher handles missing current_task_id."""
    from agents.testing.fixtures import create_test_state
    
    state = create_test_state()
    state["current_task_id"] = None
    
    result = researcher_node(state)
    
    assert "error" in result
    assert "No current task" in result["error"]


def test_researcher_stores_gap_analysis_in_task(mock_llm_factory, test_state_with_task):
    """Test that gap analysis is stored in the task."""
    mock_llm = mock_llm_factory([{
        "gap_exists": True,
        "current_state_summary": "No health system",
        "desired_state_summary": "Need health system",
        "gap_description": "Missing",
        "relevant_files": ["scripts/Player.gd"],
        "keywords": ["health"],
    }])
    
    with patch("agents.agents.researcher.planning_llm", mock_llm):
        result = researcher_node(test_state_with_task)
    
    from agents.task_states import TaskTree
    task_tree = TaskTree.from_dict(result["tasks"])
    task = task_tree.tasks["task_001"]
    assert task.gap_analysis is not None
    assert task.gap_analysis["gap_exists"] is True
