"""Unit tests for scope agents (InitialScopeAgent and ScopeReviewAgent)."""

from unittest.mock import MagicMock, patch

from agents.agents.scope_agent import (
    initial_scope_agent_node,
    scope_review_agent_node,
    _scope_output_to_state,
    _build_initial_messages,
    _build_review_messages,
    ScopeAgentOutput,
    MilestoneItem,
)
from agents.task_states import create_initial_state


def test_initial_scope_agent_handles_missing_request():
    """InitialScopeAgent returns error when user_request is missing."""
    state = create_initial_state(user_request="", repo_root="/tmp")
    result = initial_scope_agent_node(state)
    assert "error" in result
    assert "No user request" in result["error"]
    assert result.get("remit") == ""


def test_scope_review_agent_handles_missing_request():
    """ScopeReviewAgent returns error when user_request is missing."""
    state = create_initial_state(user_request="", repo_root="/tmp")
    result = scope_review_agent_node(state)
    assert "error" in result
    assert "No user request" in result["error"]


def test_scope_output_to_state():
    """_scope_output_to_state maps agent output directly to state format."""
    data = ScopeAgentOutput(
        remit="Add colony management",
        explicit_needs=["colony creation"],
        implied_needs=["data models"],
        milestones=[
            MilestoneItem(description="User can create colonies", sketch="data models, state"),
            MilestoneItem(description="User can view colony stats", sketch="UI, display"),
        ],
    )
    result = _scope_output_to_state(data)
    assert result["remit"] == "Add colony management"
    assert len(result["milestones_list"]) == 2
    assert result["active_milestone_index"] == 0
    assert result["milestones_list"][0]["id"] == "milestone_001"
    assert result["milestones_list"][0]["description"] == "User can create colonies"
    assert result["milestones_list"][0]["sketch"] == "data models, state"


def test_build_initial_messages():
    """_build_initial_messages includes only user request and repo root."""
    state = create_initial_state(user_request="Add health", repo_root="/repo")
    messages = _build_initial_messages(state)
    assert len(messages) == 1
    content = messages[0].content
    assert "Add health" in content
    assert "/repo" in content
    assert "RE-PLAN" not in content


def test_build_review_messages():
    """_build_review_messages includes done_list and last_assessment."""
    state = create_initial_state(user_request="Add health", repo_root="/repo")
    state["done_list"] = [{"description": "Done task", "result": "OK"}]
    state["last_assessment"] = {"assessment_notes": "Divergence"}
    messages = _build_review_messages(state)
    assert len(messages) == 1
    content = messages[0].content
    assert "RE-PLAN CONTEXT" in content
    assert "accomplished" in content
    assert "Divergence" in content


def test_initial_scope_agent_raises_when_no_milestones():
    """InitialScopeAgent returns failed state when agent produces no milestones."""
    empty_output = ScopeAgentOutput(
        remit="Done",
        explicit_needs=[],
        implied_needs=[],
        milestones=[],
    )
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"structured_response": empty_output}

    state = create_initial_state(user_request="Add X", repo_root="/tmp")
    with patch("agents.agents.scope_agent._create_initial_scope_agent", return_value=mock_agent):
        result = initial_scope_agent_node(state)

    assert result.get("status") == "failed"
    assert "no milestones" in result.get("error", "").lower()


def test_scope_review_agent_accepts_empty_milestones():
    """ScopeReviewAgent sets status complete when agent produces no remaining milestones."""
    empty_output = ScopeAgentOutput(
        remit="Scope complete",
        explicit_needs=[],
        implied_needs=[],
        milestones=[],
    )
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"structured_response": empty_output}

    state = create_initial_state(user_request="Add X", repo_root="/tmp")
    with patch("agents.agents.scope_agent._create_scope_review_agent", return_value=mock_agent):
        result = scope_review_agent_node(state)

    assert result.get("status") == "complete"
    assert result.get("milestones_list") == []
    assert "error" not in result or not result.get("error")
