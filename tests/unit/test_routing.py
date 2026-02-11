"""Unit tests for routing functions in graph.py."""

import pytest
from agents.graph import (
    after_researcher,
    after_validator,
    after_qa,
    after_assessor,
    after_prioritizer,
)
from agents.task_states import (
    Task,
    TaskStatus,
    TaskTree,
    GapAnalysis,
    QAResult,
    ValidationResult,
    AssessmentResult,
)
from agents.testing.fixtures import create_test_state, create_test_state_with_task


class TestAfterResearcher:
    """Test routing after researcher node."""
    
    def test_routes_to_planner_when_gap_exists(self):
        """Test routing to planner when gap exists."""
        state = create_test_state_with_task()
        state["current_gap_analysis"] = {
            "gap_exists": True,
            "task_id": "task_001",
            "current_state_summary": "No health system",
            "desired_state_summary": "Need health system",
            "gap_description": "Missing",
            "relevant_files": [],
            "keywords": [],
        }
        
        result = after_researcher(state)
        
        assert result == "planner"
    
    def test_routes_to_mark_complete_when_no_gap(self):
        """Test routing to mark_complete when no gap exists."""
        state = create_test_state_with_task()
        state["current_gap_analysis"] = {
            "gap_exists": False,
            "task_id": "task_001",
            "current_state_summary": "Health system exists",
            "desired_state_summary": "Need health system",
            "gap_description": "Already satisfied",
            "relevant_files": [],
            "keywords": [],
        }
        
        result = after_researcher(state)
        
        assert result == "mark_complete"
    
    def test_routes_to_planner_when_no_gap_analysis(self):
        """Test routing to planner when gap analysis is missing (assume gap exists)."""
        state = create_test_state_with_task()
        state["current_gap_analysis"] = None
        
        result = after_researcher(state)
        
        assert result == "planner"


class TestAfterValidator:
    """Test routing after validator node."""
    
    def test_routes_to_qa_when_validation_passed(self):
        """Test routing to QA when validation passes."""
        state = create_test_state_with_task()
        state["current_validation_result"] = {
            "task_id": "task_001",
            "files_verified": ["scripts/Player.gd"],
            "files_missing": [],
            "validation_passed": True,
            "validation_issues": [],
        }
        
        result = after_validator(state)
        
        assert result == "qa"
    
    def test_routes_to_retry_implementor_when_validation_failed(self):
        """Test routing to retry implementor when validation fails."""
        state = create_test_state_with_task(
            attempt_count=0,
            max_attempts=3,
        )
        state["current_validation_result"] = {
            "task_id": "task_001",
            "files_verified": [],
            "files_missing": ["scripts/Player.gd"],
            "validation_passed": False,
            "validation_issues": ["File not found"],
        }
        
        result = after_validator(state)
        
        assert result == "increment_attempt_and_retry_implementor"
    
    def test_routes_to_mark_failed_when_max_retries_reached(self):
        """Test routing to mark_failed when max retries reached."""
        state = create_test_state_with_task(
            attempt_count=3,
            max_attempts=3,
        )
        state["current_validation_result"] = {
            "task_id": "task_001",
            "files_verified": [],
            "files_missing": ["scripts/Player.gd"],
            "validation_passed": False,
            "validation_issues": ["File not found"],
        }
        
        result = after_validator(state)
        
        assert result == "mark_failed"
    
    def test_routes_to_qa_when_no_validation_result(self):
        """Test routing to QA when validation result is missing."""
        state = create_test_state_with_task()
        state["current_validation_result"] = None
        
        result = after_validator(state)
        
        assert result == "qa"


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
    
    def test_routes_to_retry_researcher_on_wrong_approach(self):
        """Test routing to retry researcher on wrong_approach failure."""
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
        
        assert result == "increment_attempt_and_retry_researcher"
    
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
    
    def test_routes_to_retry_planner_on_plan_issue(self):
        """Test routing to retry planner on plan_issue failure."""
        state = create_test_state_with_task(
            attempt_count=0,
            max_attempts=3,
        )
        state["current_qa_result"] = {
            "task_id": "task_001",
            "passed": False,
            "feedback": "Plan issue",
            "failure_type": "plan_issue",
            "issues": ["Plan insufficient"],
        }
        
        result = after_qa(state)
        
        assert result == "increment_attempt_and_retry_planner"
    
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
        state["active_milestone_id"] = "milestone_001"
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
    
    def test_routes_to_expander_when_gaps_uncovered(self):
        """Test routing to expander when gaps are uncovered."""
        state = create_test_state()
        state["active_milestone_id"] = "milestone_001"
        state["last_assessment"] = {
            "uncovered_gaps": ["Gap 1", "Gap 2"],
            "is_complete": False,
            "stability_check": False,
            "milestone_complete": False,
            "next_milestone_id": None,
            "assessment_notes": "Gaps found",
        }
        
        result = after_assessor(state)
        
        assert result == "expander"
    
    def test_routes_to_prioritizer_when_stable_with_tasks(self):
        """Test routing to prioritizer when stable but tasks remain."""
        state = create_test_state()
        state["active_milestone_id"] = "milestone_001"
        
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
        
        assert result == "prioritizer"


class TestAfterPrioritizer:
    """Test routing after prioritizer node."""
    
    def test_routes_to_researcher_when_task_selected(self):
        """Test routing to researcher when task is selected."""
        state = create_test_state()
        state["current_task_id"] = "task_001"
        state["active_milestone_id"] = "milestone_001"
        
        result = after_prioritizer(state)
        
        assert result == "researcher"
    
    def test_routes_to_assessor_when_no_task(self):
        """Test routing to assessor when no task is selected."""
        state = create_test_state()
        state["current_task_id"] = None
        state["active_milestone_id"] = "milestone_001"
        
        result = after_prioritizer(state)
        
        assert result == "assessor"
    
    def test_routes_to_end_when_complete(self):
        """Test routing to end when workflow is complete."""
        state = create_test_state()
        state["status"] = "complete"
        
        result = after_prioritizer(state)
        
        assert result == "end"
