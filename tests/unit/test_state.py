"""Unit tests for task_states.py - TaskTree, Task, WorkflowState operations."""

import pytest
from agents.task_states import (
    Task,
    TaskStatus,
    TaskTree,
    Milestone,
    MilestoneStatus,
    GapAnalysis,
    NeedGap,
    QAResult,
    ValidationResult,
    AssessmentResult,
)


class TestTask:
    """Test Task dataclass operations."""
    
    def test_task_creation(self):
        """Test creating a basic task."""
        task = Task(
            id="task_001",
            description="Add health system",
            measurable_outcome="Player has health property",
        )
        
        assert task.id == "task_001"
        assert task.description == "Add health system"
        assert task.status == TaskStatus.PENDING
        assert task.attempt_count == 0
        assert task.max_attempts == 3
    
    def test_task_serialization(self):
        """Test Task to_dict and from_dict."""
        task = Task(
            id="task_001",
            description="Test task",
            measurable_outcome="Outcome",
            status=TaskStatus.READY,
            attempt_count=1,
        )
        
        task_dict = task.to_dict()
        assert task_dict["id"] == "task_001"
        assert task_dict["status"] == "ready"
        assert task_dict["attempt_count"] == 1
        
        restored = Task.from_dict(task_dict)
        assert restored.id == task.id
        assert restored.status == task.status
        assert restored.attempt_count == task.attempt_count


class TestTaskTree:
    """Test TaskTree DAG operations."""
    
    def test_add_task(self):
        """Test adding a task to the tree."""
        tree = TaskTree()
        task = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
        )
        
        tree.add_task(task)
        
        assert "task_001" in tree.tasks
        assert tree.tasks["task_001"].description == "Task 1"
    
    def test_add_task_with_dependencies(self):
        """Test adding tasks with dependencies."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            depends_on=["task_001"],
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        assert task1.id in tree.tasks
        assert task2.id in tree.tasks
        assert "task_001" in task2.depends_on
        assert "task_002" in task1.blocks
    
    def test_add_task_prevents_cycles(self):
        """Test that adding a task that would create a cycle raises ValueError."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            depends_on=["task_002"],
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            depends_on=["task_001"],
        )
        
        tree.add_task(task1)
        
        with pytest.raises(ValueError, match="would create a cycle"):
            tree.add_task(task2)
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.PENDING,
            depends_on=["task_001"],
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        ready = tree.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "task_001"
    
    def test_get_ready_tasks_with_milestone_filter(self):
        """Test getting ready tasks filtered by milestone."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
            milestone_id="milestone_001",
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.READY,
            milestone_id="milestone_002",
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        ready = tree.get_ready_tasks(milestone_id="milestone_001")
        assert len(ready) == 1
        assert ready[0].id == "task_001"
    
    def test_mark_complete(self):
        """Test marking a task as complete and updating dependencies."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.PENDING,
            depends_on=["task_001"],
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        tree.mark_complete("task_001")
        
        assert tree.tasks["task_001"].status == TaskStatus.COMPLETE
        assert tree.tasks["task_002"].status == TaskStatus.READY
    
    def test_mark_failed(self):
        """Test marking a task as failed and blocking dependencies."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.PENDING,
            depends_on=["task_001"],
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        tree.mark_failed("task_001", "Test error", "implementor")
        
        assert tree.tasks["task_001"].status == TaskStatus.FAILED
        assert tree.tasks["task_001"].last_failure_reason == "Test error"
        assert tree.tasks["task_002"].status == TaskStatus.BLOCKED
    
    def test_get_statistics(self):
        """Test getting task tree statistics."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.READY,
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.COMPLETE,
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        stats = tree.get_statistics()
        assert stats["total"] == 2
        assert stats["ready"] == 1
        assert stats["complete"] == 1
    
    def test_is_milestone_complete(self):
        """Test checking if a milestone is complete."""
        tree = TaskTree()
        
        task1 = Task(
            id="task_001",
            description="Task 1",
            measurable_outcome="Outcome 1",
            status=TaskStatus.COMPLETE,
            milestone_id="milestone_001",
        )
        task2 = Task(
            id="task_002",
            description="Task 2",
            measurable_outcome="Outcome 2",
            status=TaskStatus.COMPLETE,
            milestone_id="milestone_001",
        )
        
        tree.add_task(task1)
        tree.add_task(task2)
        
        assert tree.is_milestone_complete("milestone_001") is True
        
        task3 = Task(
            id="task_003",
            description="Task 3",
            measurable_outcome="Outcome 3",
            status=TaskStatus.READY,
            milestone_id="milestone_001",
        )
        tree.add_task(task3)
        
        assert tree.is_milestone_complete("milestone_001") is False


class TestResultTypes:
    """Test agent result dataclasses."""
    
    def test_need_gap_serialization(self):
        """Test NeedGap serialization."""
        need_gap = NeedGap(
            need="Add colony management",
            need_type="explicit",
            gap_exists=True,
            current_state_summary="No colony system",
            desired_state_summary="Full colony system with UI",
            gap_description="Colony data model and UI missing",
            relevant_areas=["scripts/domain/", "scripts/ui/"],
            keywords=["colony", "colony_ui"],
        )

        gap_dict = need_gap.to_dict()
        assert gap_dict["gap_exists"] is True
        assert gap_dict["need"] == "Add colony management"
        assert gap_dict["need_type"] == "explicit"
        assert gap_dict["relevant_areas"] == ["scripts/domain/", "scripts/ui/"]

        restored = NeedGap.from_dict(gap_dict)
        assert restored.gap_exists is True
        assert restored.need == "Add colony management"
        assert restored.relevant_areas == ["scripts/domain/", "scripts/ui/"]

    def test_gap_analysis_serialization(self):
        """Test GapAnalysis serialization."""
        gap = GapAnalysis(
            task_id="task_001",
            gap_exists=True,
            current_state_summary="No health system",
            desired_state_summary="Need health system",
            gap_description="Health system missing",
            relevant_files=["scripts/Player.gd"],
            keywords=["health"],
        )
        
        gap_dict = gap.to_dict()
        assert gap_dict["gap_exists"] is True
        assert gap_dict["task_id"] == "task_001"
        
        restored = GapAnalysis.from_dict(gap_dict)
        assert restored.gap_exists is True
        assert restored.relevant_files == ["scripts/Player.gd"]
    
    def test_qa_result_serialization(self):
        """Test QAResult serialization."""
        qa = QAResult(
            task_id="task_001",
            passed=True,
            feedback="Looks good",
            failure_type=None,
            issues=[],
        )
        
        qa_dict = qa.to_dict()
        assert qa_dict["passed"] is True
        
        restored = QAResult.from_dict(qa_dict)
        assert restored.passed is True
        assert restored.failure_type is None
    
    def test_assessment_result_serialization(self):
        """Test AssessmentResult serialization."""
        assessment = AssessmentResult(
            uncovered_gaps=["Gap 1", "Gap 2"],
            is_complete=False,
            stability_check=True,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes="Test notes",
        )
        
        assessment_dict = assessment.to_dict()
        assert len(assessment_dict["uncovered_gaps"]) == 2
        assert assessment_dict["is_complete"] is False
        
        restored = AssessmentResult.from_dict(assessment_dict)
        assert len(restored.uncovered_gaps) == 2
