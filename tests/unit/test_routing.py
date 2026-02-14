"""Unit tests for routing functions in graph.py."""

import pytest
from agents.graph import (
    after_qa,
    after_assessor,
    after_task_planner,
    after_scope_agent,
)
from agents.task_states import (
    Task,
    TaskStatus,
    TaskTree,
    QAResult,
    AssessmentResult,
)
from agents.testing.fixtures import create_test_state, create_test_state_with_task


class TestAfterQA:
    """Test routing after QA node."""

    def test_routes_to_mark_complete_when_passed(self):
        """Test routing to mark_complete when QA passes."""
        state = create_test_state_with_task()
        state["current_qa_result"] = {
            "task_id": "task_001",
            "passed": True,
            "feedback": "Looks good",
            "failure_type": None,
            "issues": [],
        }

        result = after_qa(state)

        assert result == "mark_complete"

    def test_routes_to_retry_task_planner_on_task_tp_failure(self):
        """Test routing to retry task_planner for TaskPlanner tasks."""
        state = create_test_state_with_task(
            task_id="task_tp_abc123",
            attempt_count=0,
            max_attempts=3,
        )
        state["current_qa_result"] = {
            "task_id": "task_tp_abc123",
            "passed": False,
            "feedback": "Wrong approach",
            "failure_type": "wrong_approach",
            "issues": ["Fundamental misunderstanding"],
        }

        result = after_qa(state)

        assert result == "increment_attempt_and_retry_task_planner"

    def test_routes_to_retry_implementor_on_incomplete(self):
        """Test routing to retry implementor on incomplete failure."""
        state = create_test_state_with_task(
            attempt_count=0,
            max_attempts=3,
        )
        state["current_qa_result"] = {
            "task_id": "task_001",
            "passed": False,
            "feedback": "Incomplete",
            "failure_type": "incomplete",
            "issues": ["Missing pieces"],
        }

        result = after_qa(state)

        assert result == "increment_attempt_and_retry_implementor"

    def test_routes_to_retry_task_planner_on_wrong_approach(self):
        """Test routing to retry task_planner on wrong_approach for non-task_tp."""
        state = create_test_state_with_task(
            attempt_count=0,
            max_attempts=3,
        )
        state["current_qa_result"] = {
            "task_id": "task_001",
            "passed": False,
            "feedback": "Wrong approach",
            "failure_type": "wrong_approach",
            "issues": ["Fundamental misunderstanding"],
        }

        result = after_qa(state)

        assert result == "increment_attempt_and_retry_task_planner"

    def test_routes_to_mark_failed_when_max_retries_reached(self):
        """Test routing to mark_failed when max retries reached."""
        state = create_test_state_with_task(
            attempt_count=3,
            max_attempts=3,
        )
        state["current_qa_result"] = {
            "task_id": "task_001",
            "passed": False,
            "feedback": "Failed",
            "failure_type": "incomplete",
            "issues": ["Still incomplete"],
        }

        result = after_qa(state)

        assert result == "mark_failed"


class TestAfterAssessor:
    """Test routing after assessor node."""

    def test_routes_to_end_when_complete(self):
        """Test routing to end when workflow is complete."""
        state = create_test_state()
        state["status"] = "complete"

        result = after_assessor(state)

        assert result == "end"

    def test_routes_to_advance_milestone_when_milestone_complete(self):
        """Test routing to advance_milestone when milestone is complete."""
        state = create_test_state()
        state["milestones_list"] = [
            {"id": "milestone_001", "description": "First"},
            {"id": "milestone_002", "description": "Second"},
        ]
        state["active_milestone_index"] = 0
        state["last_assessment"] = {
            "uncovered_gaps": [],
            "is_complete": False,
            "stability_check": True,
            "milestone_complete": True,
            "next_milestone_id": "milestone_002",
            "assessment_notes": "Milestone complete",
        }

        result = after_assessor(state)

        assert result == "advance_milestone"

    def test_routes_to_scope_review_agent_when_escalate_to_scope(self):
        """Test routing to scope_review_agent when major divergence."""
        state = create_test_state()
        state["milestones_list"] = [{"id": "milestone_001", "description": "First"}]
        state["active_milestone_index"] = 0
        state["last_assessment"] = {
            "uncovered_gaps": ["Wrong direction"],
            "is_complete": False,
            "stability_check": False,
            "milestone_complete": False,
            "next_milestone_id": None,
            "assessment_notes": "Major divergence",
            "escalate_to_scope": True,
        }

        result = after_assessor(state)

        assert result == "scope_review_agent"

    def test_routes_to_task_planner_when_gaps_uncovered(self):
        """Test routing to task_planner when gaps are uncovered."""
        state = create_test_state()
        state["milestones_list"] = [{"id": "milestone_001", "description": "First"}]
        state["active_milestone_index"] = 0
        state["last_assessment"] = {
            "uncovered_gaps": ["Gap 1", "Gap 2"],
            "is_complete": False,
            "stability_check": False,
            "milestone_complete": False,
            "next_milestone_id": None,
            "assessment_notes": "Gaps found",
        }

        result = after_assessor(state)

        assert result == "task_planner"

    def test_routes_to_task_planner_when_stable_with_tasks(self):
        """Test routing to task_planner when stable but tasks remain."""
        state = create_test_state()
        state["milestones_list"] = [{"id": "milestone_001", "description": "First"}]
        state["active_milestone_index"] = 0

        # Add a ready task
        task = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
            milestone_id="milestone_001",
        )
        task_tree = TaskTree()
        task_tree.add_task(task)
        state["tasks"] = task_tree.to_dict()

        state["last_assessment"] = {
            "uncovered_gaps": [],
            "is_complete": False,
            "stability_check": True,
            "milestone_complete": False,
            "next_milestone_id": None,
            "assessment_notes": "Stable",
        }

        result = after_assessor(state)

        assert result == "task_planner"


class TestAfterTaskPlanner:
    """Test routing after task_planner node."""

    def test_routes_to_implementor_when_implement(self):
        """Test routing to implementor when action is implement."""
        state = create_test_state()
        state["task_planner_action"] = "implement"

        result = after_task_planner(state)

        assert result == "implementor"

    def test_routes_to_task_planner_when_skip(self):
        """Test routing back to task_planner when action is skip."""
        state = create_test_state()
        state["task_planner_action"] = "skip"

        result = after_task_planner(state)

        assert result == "task_planner"

    def test_routes_to_assessor_when_abort(self):
        """Test routing to assessor when action is abort."""
        state = create_test_state()
        state["task_planner_action"] = "abort"

        result = after_task_planner(state)

        assert result == "assessor"

    def test_routes_to_assessor_when_milestone_done(self):
        """Test routing to assessor when action is milestone_done."""
        state = create_test_state()
        state["task_planner_action"] = "milestone_done"

        result = after_task_planner(state)

        assert result == "assessor"


class TestAfterScopeAgent:
    """Test routing after initial_scope_agent / scope_review_agent nodes."""

    def test_routes_to_set_active_milestone_when_success(self):
        """Test routing to set_active_milestone when scope definition succeeded."""
        state = create_test_state()
        state["remit"] = "Add health system"
        state["milestones_list"] = [{"id": "milestone_001", "description": "Health"}]
        state["status"] = "running"

        result = after_scope_agent(state)

        assert result == "set_active_milestone"

    def test_routes_to_end_when_failed(self):
        """Test routing to end when scope definition failed."""
        state = create_test_state()
        state["status"] = "failed"

        result = after_scope_agent(state)

        assert result == "end"

    def test_routes_to_end_when_no_milestones(self):
        """Test routing to end when no milestones produced."""
        state = create_test_state()
        state["remit"] = "Add health"
        state["milestones_list"] = []
        state["status"] = "running"

        result = after_scope_agent(state)

        assert result == "end"
