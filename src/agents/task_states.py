"""State definitions for the iterative task tree workflow.

This module defines the core data structures for the new workflow:
- Task: The fundamental unit of work, forming a DAG
- WorkflowState: The top-level state object
- TaskTree: DAG operations for managing tasks
- Various result types for agent outputs
"""

from typing import TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# =============================================================================
# Task Status and Types
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"          # Not yet ready (dependencies incomplete)
    READY = "ready"              # All dependencies met, can be picked
    IN_PROGRESS = "in_progress"  # Currently being executed
    COMPLETE = "complete"        # Successfully finished
    FAILED = "failed"            # Failed after max retries
    BLOCKED = "blocked"          # Blocked by failed dependency
    DEFERRED = "deferred"        # Needs human input


# =============================================================================
# Milestone Model
# =============================================================================

class MilestoneStatus(str, Enum):
    """Status of a milestone in the workflow."""
    PENDING = "pending"      # Not yet active
    ACTIVE = "active"        # Currently being worked on
    COMPLETE = "complete"    # All tasks complete/failed/deferred


@dataclass
class Milestone:
    """A milestone representing a sequential phase in the workflow.
    
    Milestones are linear and sequential. Each milestone represents an interim
    state that will be achieved. Tasks are created within milestones and can
    only depend on other tasks within the same milestone.
    """
    id: str                      # Unique identifier (e.g., "milestone_001")
    description: str             # Short description of interim state (max 200 chars)
    order: int = 0              # Sequential order (0, 1, 2, ...)
    status: MilestoneStatus = MilestoneStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "order": self.order,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Milestone":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            status=MilestoneStatus(data.get("status", "pending")),
            order=data.get("order", 0),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


# =============================================================================
# Core Task Model
# =============================================================================

@dataclass
class Task:
    """A task in the workflow DAG.
    
    Tasks are the fundamental unit of work. They form a directed acyclic graph
    with explicit dependencies. Each task represents a discrete piece of work
    with a measurable outcome.
    """
    # Identity
    id: str                      # Unique identifier (e.g., "task_001")
    description: str             # What needs to be done
    measurable_outcome: str      # How we know it's complete
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Dependency graph
    depends_on: list[str] = field(default_factory=list)  # Task IDs this depends on
    blocks: list[str] = field(default_factory=list)      # Task IDs blocked by this
    
    # Hierarchy (optional - for task decomposition)
    parent_id: str | None = None
    
    # Milestone association
    milestone_id: str | None = None  # Which milestone this task belongs to
    
    # Provenance
    created_by: str = "unknown"           # Which agent created this ("intake", "expander", etc.)
    created_at_iteration: int = 0         # Which expansion iteration
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Compressed context from agents (set during execution)
    gap_analysis: dict | None = None      # From Researcher (GapAnalysis.to_dict())
    implementation_plan_summary: str | None = None  # Compressed summary (full plan in ephemeral state)
    result_summary: str | None = None     # From Implementor
    qa_feedback: str | None = None        # From QA
    
    # Retry tracking
    attempt_count: int = 0
    max_attempts: int = 3
    last_failure_reason: str | None = None
    last_failure_stage: str | None = None  # "researcher", "planner", "implementor", "validator", "qa"
    
    # Metadata
    tags: list[str] = field(default_factory=list)  # For grouping/filtering
    estimated_complexity: str | None = None  # "simple", "moderate", "complex"
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "measurable_outcome": self.measurable_outcome,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "parent_id": self.parent_id,
            "milestone_id": self.milestone_id,
            "created_by": self.created_by,
            "created_at_iteration": self.created_at_iteration,
            "created_at": self.created_at,
            "gap_analysis": self.gap_analysis,
            "implementation_plan_summary": self.implementation_plan_summary,
            "result_summary": self.result_summary,
            "qa_feedback": self.qa_feedback,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "last_failure_reason": self.last_failure_reason,
            "last_failure_stage": self.last_failure_stage,
            "tags": self.tags,
            "estimated_complexity": self.estimated_complexity,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            measurable_outcome=data["measurable_outcome"],
            status=TaskStatus(data.get("status", "pending")),
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            parent_id=data.get("parent_id"),
            milestone_id=data.get("milestone_id"),
            created_by=data.get("created_by", "unknown"),
            created_at_iteration=data.get("created_at_iteration", 0),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            gap_analysis=data.get("gap_analysis"),
            implementation_plan_summary=data.get("implementation_plan_summary"),
            result_summary=data.get("result_summary"),
            qa_feedback=data.get("qa_feedback"),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            last_failure_reason=data.get("last_failure_reason"),
            last_failure_stage=data.get("last_failure_stage"),
            tags=data.get("tags", []),
            estimated_complexity=data.get("estimated_complexity"),
        )


# =============================================================================
# Agent Result Types
# =============================================================================

@dataclass
class GapAnalysis:
    """Result from Researcher agent - gap between current and desired state.
    
    This is the critical gate: if gap_exists is False, the task is already
    satisfied and we skip implementation entirely.
    """
    task_id: str
    gap_exists: bool
    current_state_summary: str   # What exists now (compressed, max 1k chars)
    desired_state_summary: str   # What task requires (compressed, max 1k chars)
    gap_description: str         # The delta - what's missing (max 2k chars)
    relevant_files: list[str]    # Files that would need changes
    keywords: list[str]          # Terms for further searching
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "gap_exists": self.gap_exists,
            "current_state_summary": self.current_state_summary,
            "desired_state_summary": self.desired_state_summary,
            "gap_description": self.gap_description,
            "relevant_files": self.relevant_files,
            "keywords": self.keywords,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GapAnalysis":
        return cls(
            task_id=data["task_id"],
            gap_exists=data.get("gap_exists", True),
            current_state_summary=data.get("current_state_summary", ""),
            desired_state_summary=data.get("desired_state_summary", ""),
            gap_description=data.get("gap_description", ""),
            relevant_files=data.get("relevant_files", []),
            keywords=data.get("keywords", []),
        )


@dataclass
class ImplementationResult:
    """Result from Implementor agent."""
    task_id: str
    files_modified: list[str]    # Files actually modified
    result_summary: str          # What was done (compressed, max 1k chars)
    issues_noticed: list[str]    # Problems encountered
    success: bool                # Did tool calls succeed?
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "files_modified": self.files_modified,
            "result_summary": self.result_summary,
            "issues_noticed": self.issues_noticed,
            "success": self.success,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ImplementationResult":
        return cls(
            task_id=data["task_id"],
            files_modified=data.get("files_modified", []),
            result_summary=data.get("result_summary", ""),
            issues_noticed=data.get("issues_noticed", []),
            success=data.get("success", False),
        )


@dataclass
class ValidationResult:
    """Result from Validator agent - file verification."""
    task_id: str
    files_verified: list[str]    # Files that exist and were modified
    files_missing: list[str]     # Reported files that don't exist
    validation_passed: bool
    validation_issues: list[str]
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "files_verified": self.files_verified,
            "files_missing": self.files_missing,
            "validation_passed": self.validation_passed,
            "validation_issues": self.validation_issues,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ValidationResult":
        return cls(
            task_id=data["task_id"],
            files_verified=data.get("files_verified", []),
            files_missing=data.get("files_missing", []),
            validation_passed=data.get("validation_passed", False),
            validation_issues=data.get("validation_issues", []),
        )


@dataclass
class QAResult:
    """Result from QA agent - requirement satisfaction check."""
    task_id: str
    passed: bool
    feedback: str                # Detailed assessment (max 500 chars)
    failure_type: str | None     # "wrong_approach", "incomplete", "plan_issue", or None if passed
    issues: list[str]
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "feedback": self.feedback,
            "failure_type": self.failure_type,
            "issues": self.issues,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QAResult":
        return cls(
            task_id=data["task_id"],
            passed=data.get("passed", False),
            feedback=data.get("feedback", ""),
            failure_type=data.get("failure_type"),
            issues=data.get("issues", []),
        )


@dataclass
class AssessmentResult:
    """Result from Assessor agent - gap and completion check."""
    uncovered_gaps: list[str]    # Gap descriptions not covered by open tasks
    is_complete: bool            # Is the remit fully satisfied?
    stability_check: bool        # Did this iteration create new tasks?
    milestone_complete: bool     # Is the active milestone complete?
    next_milestone_id: str | None  # Next milestone to expand (if milestone_complete)
    assessment_notes: str        # Additional notes
    
    def to_dict(self) -> dict:
        return {
            "uncovered_gaps": self.uncovered_gaps,
            "is_complete": self.is_complete,
            "stability_check": self.stability_check,
            "milestone_complete": self.milestone_complete,
            "next_milestone_id": self.next_milestone_id,
            "assessment_notes": self.assessment_notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AssessmentResult":
        return cls(
            uncovered_gaps=data.get("uncovered_gaps", []),
            is_complete=data.get("is_complete", False),
            stability_check=data.get("stability_check", False),
            milestone_complete=data.get("milestone_complete", False),
            next_milestone_id=data.get("next_milestone_id"),
            assessment_notes=data.get("assessment_notes", ""),
        )


# =============================================================================
# Workflow State
# =============================================================================

def reducer_append(current: list, new: list | None) -> list:
    """Reducer that appends new items to current list."""
    if new is None:
        return current
    return current + new


class WorkflowState(TypedDict, total=False):
    """Main state object for the iterative task tree workflow.
    
    This state object is passed through the LangGraph workflow. It contains
    the task tree, current execution state, and ephemeral data for the
    currently executing task.
    """
    # Configuration
    verbose: bool
    max_iterations: int
    max_task_retries: int
    
    # Immutable inputs
    user_request: str
    repo_root: str
    
    # Scope boundary (from Intent and Milestone agents)
    remit: str
    explicit_needs: list[str]
    implied_needs: list[str]
    
    # Milestones (sequential phases)
    milestones: dict[str, dict]   # milestone_id -> Milestone.to_dict()
    active_milestone_id: str | None  # Current milestone being worked on
    milestone_order: list[str]    # Ordered list of milestone IDs
    
    # Task tree (core data structure)
    tasks: dict[str, dict]       # task_id -> Task.to_dict()
    
    # Execution tracking
    current_task_id: str | None
    completed_task_ids: list[str]
    failed_task_ids: list[str]
    deferred_task_ids: list[str]
    
    # Iteration control
    iteration: int               # Current expansion iteration
    is_stable: bool              # True when no new tasks created
    tasks_created_this_iteration: int
    
    # Ephemeral data (cleared between tasks, stored in full detail)
    current_gap_analysis: dict | None         # GapAnalysis.to_dict() for current task
    current_implementation_plan: str | None   # Full PRP markdown for current task
    current_implementation_result: dict | None  # ImplementationResult.to_dict()
    current_validation_result: dict | None    # ValidationResult.to_dict()
    current_qa_result: dict | None            # QAResult.to_dict()
    
    # Last assessment (from Assessor)
    last_assessment: dict | None  # AssessmentResult.to_dict()
    
    # Workflow status
    status: str                  # "running", "complete", "failed"
    error: str | None
    messages: Annotated[list[str], reducer_append]
    
    # Dashboard state
    dashboard_mode: bool         # Whether dashboard is active
    current_node: str | None     # Currently executing node
    node_statuses: dict[str, str]  # node_name -> status ("pending", "active", "complete", "failed")
    node_history: list[dict]      # Recent node executions with summaries
    suppress_output: bool        # Whether to suppress verbose output (when dashboard active)


# =============================================================================
# Task Tree Operations
# =============================================================================

class TaskTree:
    """Task tree with DAG operations.
    
    Provides operations for managing tasks as a directed acyclic graph:
    - Adding tasks with dependencies
    - Marking tasks as complete/failed
    - Finding ready tasks
    - Detecting cycles
    - Computing task summaries
    """
    
    def __init__(self, tasks: dict[str, Task] | None = None):
        """Initialize task tree.
        
        Args:
            tasks: Optional initial tasks (task_id -> Task)
        """
        self.tasks: dict[str, Task] = tasks or {}
    
    def add_task(self, task: Task) -> None:
        """Add a task to the tree.
        
        Args:
            task: Task to add
        
        Raises:
            ValueError: If task ID already exists or would create a cycle
        """
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already exists")
        
        # Check for cycles
        if self._would_create_cycle(task):
            raise ValueError(f"Adding task {task.id} would create a cycle")
        
        # Add task
        self.tasks[task.id] = task
        
        # Update reverse dependencies (blocks)
        for dep_id in task.depends_on:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if task.id not in dep_task.blocks:
                    dep_task.blocks.append(task.id)
    
    def _would_create_cycle(self, task: Task) -> bool:
        """Check if adding this task would create a cycle.
        
        Args:
            task: Task to check
        
        Returns:
            True if adding task would create a cycle
        """
        # For each dependency, check if we can reach the new task
        for dep_id in task.depends_on:
            if dep_id not in self.tasks:
                continue
            if self._can_reach(dep_id, task.id, visited=set()):
                return True
        return False
    
    def _can_reach(self, from_id: str, to_id: str, visited: set[str]) -> bool:
        """Check if we can reach to_id from from_id following depends_on edges.
        
        Args:
            from_id: Starting task ID
            to_id: Target task ID
            visited: Already visited task IDs
        
        Returns:
            True if to_id is reachable from from_id
        """
        if from_id == to_id:
            return True
        if from_id in visited:
            return False
        
        visited.add(from_id)
        
        # Follow the "blocks" edges (reverse of depends_on)
        task = self.tasks.get(from_id)
        if not task:
            return False
        
        for blocked_id in task.blocks:
            if self._can_reach(blocked_id, to_id, visited):
                return True
        
        return False
    
    def get_tasks_by_milestone(self, milestone_id: str) -> list[Task]:
        """Get all tasks belonging to a specific milestone.
        
        Args:
            milestone_id: ID of milestone to filter by
        
        Returns:
            List of tasks in the milestone
        """
        return [task for task in self.tasks.values() if task.milestone_id == milestone_id]
    
    def is_milestone_complete(self, milestone_id: str) -> bool:
        """Check if a milestone is complete.
        
        A milestone is complete when all its tasks are in final states:
        COMPLETE, FAILED, or DEFERRED.
        
        Args:
            milestone_id: ID of milestone to check
        
        Returns:
            True if all tasks in milestone are complete/failed/deferred
        """
        milestone_tasks = self.get_tasks_by_milestone(milestone_id)
        if not milestone_tasks:
            return False  # No tasks means not complete
        
        final_states = {TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.DEFERRED}
        return all(task.status in final_states for task in milestone_tasks)
    
    def get_ready_tasks(self, milestone_id: str | None = None) -> list[Task]:
        """Get all tasks that are ready to execute.
        
        A task is ready if:
        - Status is READY
        - All dependencies are COMPLETE
        - If milestone_id is provided, task must belong to that milestone
        
        Args:
            milestone_id: Optional milestone ID to filter by
        
        Returns:
            List of ready tasks, sorted by priority
        """
        ready = []
        for task in self.tasks.values():
            # Filter by milestone if specified
            if milestone_id is not None and task.milestone_id != milestone_id:
                continue
            
            if task.status != TaskStatus.READY:
                continue
            
            # Check all dependencies are complete
            all_deps_complete = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETE
                for dep_id in task.depends_on
                if dep_id in self.tasks
            )
            
            if all_deps_complete:
                ready.append(task)
        
        # Sort by priority: more blockers first, then earlier creation
        ready.sort(key=lambda t: (-len(t.blocks), t.created_at_iteration, t.created_at))
        return ready
    
    def mark_complete(self, task_id: str) -> None:
        """Mark a task as complete and update dependent tasks.
        
        Args:
            task_id: ID of task to mark complete
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.status = TaskStatus.COMPLETE
        
        # Update blocked tasks - check if they're now ready
        for blocked_id in task.blocks:
            blocked_task = self.tasks.get(blocked_id)
            if not blocked_task or blocked_task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are complete
            all_deps_complete = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETE
                for dep_id in blocked_task.depends_on
                if dep_id in self.tasks
            )
            
            if all_deps_complete:
                blocked_task.status = TaskStatus.READY
    
    def mark_failed(self, task_id: str, reason: str, stage: str) -> None:
        """Mark a task as failed and block dependent tasks.
        
        Args:
            task_id: ID of task to mark failed
            reason: Failure reason
            stage: Which stage failed (e.g., "implementor", "qa")
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.status = TaskStatus.FAILED
        task.last_failure_reason = reason
        task.last_failure_stage = stage
        
        # Mark blocked tasks as blocked
        for blocked_id in task.blocks:
            blocked_task = self.tasks.get(blocked_id)
            if blocked_task and blocked_task.status in (TaskStatus.PENDING, TaskStatus.READY):
                blocked_task.status = TaskStatus.BLOCKED
    
    def get_task_summary(self, task_id: str, max_chars: int = 300) -> str:
        """Get a compressed summary of a task for passing to agents.
        
        Args:
            task_id: ID of task to summarize
            max_chars: Maximum characters in summary
        
        Returns:
            Compressed summary string
        """
        task = self.tasks.get(task_id)
        if not task:
            return ""
        
        summary_parts = [
            f"{task.id}: {task.description}",
            f"Status: {task.status.value}",
        ]
        
        if task.measurable_outcome:
            summary_parts.append(f"Outcome: {task.measurable_outcome[:100]}")
        
        if task.result_summary:
            summary_parts.append(f"Result: {task.result_summary[:100]}")
        
        summary = " | ".join(summary_parts)
        
        if len(summary) > max_chars:
            summary = summary[:max_chars-3] + "..."
        
        return summary
    
    def get_statistics(self) -> dict:
        """Get task tree statistics.
        
        Returns:
            Dictionary with counts by status
        """
        stats = {
            "total": len(self.tasks),
            "pending": 0,
            "ready": 0,
            "in_progress": 0,
            "complete": 0,
            "failed": 0,
            "blocked": 0,
            "deferred": 0,
        }
        
        for task in self.tasks.values():
            stats[task.status.value] += 1
        
        return stats
    
    def to_dict(self) -> dict[str, dict]:
        """Serialize task tree to dictionary.
        
        Returns:
            Dictionary of task_id -> Task.to_dict()
        """
        return {task_id: task.to_dict() for task_id, task in self.tasks.items()}
    
    @classmethod
    def from_dict(cls, data: dict[str, dict]) -> "TaskTree":
        """Deserialize task tree from dictionary.
        
        Args:
            data: Dictionary of task_id -> Task dict
        
        Returns:
            TaskTree instance
        """
        tasks = {task_id: Task.from_dict(task_dict) for task_id, task_dict in data.items()}
        return cls(tasks)


# =============================================================================
# State Initialization
# =============================================================================

def create_initial_state(
    user_request: str,
    repo_root: str,
    verbose: bool = False,
    max_iterations: int = 10,
    max_task_retries: int = 3,
    dashboard_mode: bool = False,
) -> WorkflowState:
    """Create the initial workflow state.
    
    Args:
        user_request: The user's development request
        repo_root: Path to the repository root
        verbose: Enable verbose agent output for debugging
        max_iterations: Maximum expansion iterations
        max_task_retries: Maximum retries per task
    
    Returns:
        Initial WorkflowState
    """
    return WorkflowState(
        # Configuration
        verbose=verbose,
        max_iterations=max_iterations,
        max_task_retries=max_task_retries,
        
        # Immutable inputs
        user_request=user_request,
        repo_root=repo_root,
        
        # Scope
        remit="",
        explicit_needs=[],
        implied_needs=[],
        
        # Milestones
        milestones={},
        active_milestone_id=None,
        milestone_order=[],
        
        # Task tree
        tasks={},
        
        # Execution tracking
        current_task_id=None,
        completed_task_ids=[],
        failed_task_ids=[],
        deferred_task_ids=[],
        
        # Iteration control
        iteration=0,
        is_stable=False,
        tasks_created_this_iteration=0,
        
        # Ephemeral data
        current_gap_analysis=None,
        current_implementation_plan=None,
        current_implementation_result=None,
        current_validation_result=None,
        current_qa_result=None,
        
        # Last assessment
        last_assessment=None,
        
        # Workflow status
        status="running",
        error=None,
        messages=[],
        
        # Dashboard state
        dashboard_mode=dashboard_mode,
        current_node=None,
        node_statuses={},
        node_history=[],
        suppress_output=dashboard_mode,  # Suppress output when dashboard is active
    )
