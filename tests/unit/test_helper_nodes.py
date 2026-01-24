"""Unit tests for helper nodes in graph.py."""

import pytest
from agents.graph import (
    mark_task_complete_node,
    mark_task_failed_node,
    increment_iteration_node,
)
from agents.state import Task, TaskStatus, TaskTree
from agents.testing.fixtures import create_test_state_with_task


class TestMarkTaskCompleteNode:
    """Test mark_task_complete_node."""
    
    def test_marks_task_complete(self):
        """Test that task is marked as complete."""
        state = create_test_state_with_task(
            task_id="task_001",
            status=TaskStatus.IN_PROGRESS,
        )
        
        result = mark_task_complete_node(state)
        
        task_tree = TaskTree.from_dict(result["tasks"])
        task = task_tree.tasks["task_001"]
        assert task.status == TaskStatus.COMPLETE
        assert result["current_task_id"] is None
    
    def test_adds_to_completed_task_ids(self):
        """Test that task ID is added to completed_task_ids."""
        state = create_test_state_with_task(
            task_id="task_001",
            status=TaskStatus.IN_PROGRESS,
        )
        state["completed_task_ids"] = []
        
        result = mark_task_complete_node(state)
        
        assert "task_001" in result["completed_task_ids"]
    
    def test_clears_ephemeral_state(self):
        """Test that ephemeral state is cleared."""
        state = create_test_state_with_task()
        state["current_gap_analysis"] = {"gap_exists": True}
        state["current_implementation_plan"] = "Plan content"
        state["current_implementation_result"] = {"success": True}
        
        result = mark_task_complete_node(state)
        
        assert result["current_gap_analysis"] is None
        assert result["current_implementation_plan"] is None
        assert result["current_implementation_result"] is None
    
    def test_updates_blocked_tasks_to_ready(self):
        """Test that blocked tasks become ready when dependencies complete."""
        state = create_test_state()
        
        # Create task tree with dependencies
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.IN_PROGRESS,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.PENDING,
            depends_on=["task_001"],
        )
        
        task_tree = TaskTree()
        task_tree.add_task(task1)
        task_tree.add_task(task2)
        state["tasks"] = task_tree.to_dict()
        state["current_task_id"] = "task_001"
        state["completed_task_ids"] = []
        
        result = mark_task_complete_node(state)
        
        task_tree = TaskTree.from_dict(result["tasks"])
        assert task_tree.tasks["task_002"].status == TaskStatus.READY
    
    def test_handles_missing_task(self):
        """Test handling when current_task_id doesn't exist."""
        state = create_test_state()
        state["current_task_id"] = "nonexistent_task"
        
        result = mark_task_complete_node(state)
        
        assert "error" in result
        assert "not found" in result["error"]


class TestMarkTaskFailedNode:
    """Test mark_task_failed_node."""
    
    def test_marks_task_failed(self):
        """Test that task is marked as failed."""
        state = create_test_state_with_task(
            task_id="task_001",
            status=TaskStatus.IN_PROGRESS,
        )
        state["error"] = "Test error"
        
        result = mark_task_failed_node(state)
        
        task_tree = TaskTree.from_dict(result["tasks"])
        task = task_tree.tasks["task_001"]
        assert task.status == TaskStatus.FAILED
        assert task.last_failure_reason == "Test error"
        assert result["current_task_id"] is None
    
    def test_adds_to_failed_task_ids(self):
        """Test that task ID is added to failed_task_ids."""
        state = create_test_state_with_task(
            task_id="task_001",
            status=TaskStatus.IN_PROGRESS,
        )
        state["error"] = "Test error"
        state["failed_task_ids"] = []
        
        result = mark_task_failed_node(state)
        
        assert "task_001" in result["failed_task_ids"]
    
    def test_blocks_dependent_tasks(self):
        """Test that dependent tasks are blocked."""
        state = create_test_state()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.IN_PROGRESS,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.PENDING,
            depends_on=["task_001"],
        )
        
        task_tree = TaskTree()
        task_tree.add_task(task1)
        task_tree.add_task(task2)
        state["tasks"] = task_tree.to_dict()
        state["current_task_id"] = "task_001"
        state["error"] = "Test error"
        state["failed_task_ids"] = []
        
        result = mark_task_failed_node(state)
        
        task_tree = TaskTree.from_dict(result["tasks"])
        assert task_tree.tasks["task_002"].status == TaskStatus.BLOCKED
    
    def test_determines_failure_stage(self):
        """Test that failure stage is determined from state."""
        state = create_test_state_with_task()
        state["current_qa_result"] = {"passed": False}
        state["error"] = "QA error"
        
        result = mark_task_failed_node(state)
        
        task_tree = TaskTree.from_dict(result["tasks"])
        task = task_tree.tasks[state["current_task_id"]]
        assert task.last_failure_stage == "qa"


class TestIncrementIterationNode:
    """Test increment_iteration_node."""
    
    def test_increments_iteration(self):
        """Test that iteration counter is incremented."""
        state = create_test_state()
        state["iteration"] = 5
        
        result = increment_iteration_node(state)
        
        assert result["iteration"] == 6
    
    def test_resets_tasks_created_this_iteration(self):
        """Test that tasks_created_this_iteration is reset."""
        state = create_test_state()
        state["iteration"] = 1
        state["tasks_created_this_iteration"] = 10
        
        result = increment_iteration_node(state)
        
        assert result["tasks_created_this_iteration"] == 0
