"""State fixture builders for testing agents.

Provides utilities to create WorkflowState instances with various configurations
for testing different agent scenarios.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from ..task_states import (
    WorkflowState,
    Task,
    TaskStatus,
    TaskTree,
    create_initial_state,
)


def create_test_state(
    user_request: str = "Add health system",
    repo_root: str | Path | None = None,
    with_task: Task | None = None,
    current_task_id: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> WorkflowState:
    """Create a WorkflowState for testing.
    
    Args:
        user_request: The user's development request
        repo_root: Repository root path (defaults to temp directory)
        with_task: Optional Task to add to the state
        current_task_id: Optional current task ID to set
        verbose: Enable verbose output
        **kwargs: Additional state fields to set
    
    Returns:
        WorkflowState configured for testing
    
    Example:
        >>> task = Task(
        ...     id="task_001",
        ...     description="Add player health",
        ...     measurable_outcome="Player has health property"
        ... )
        >>> state = create_test_state(with_task=task, current_task_id="task_001")
    """
    if repo_root is None:
        # Use a temporary directory for testing
        repo_root = Path(tempfile.mkdtemp(prefix="agent_test_"))
    else:
        repo_root = Path(repo_root)
    
    state = create_initial_state(
        user_request=user_request,
        repo_root=str(repo_root),
        verbose=verbose,
    )
    
    if with_task:
        task_tree = TaskTree()
        task_tree.add_task(with_task)
        state["tasks"] = task_tree.to_dict()
        state["current_task_id"] = current_task_id or with_task.id
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        state[key] = value
    
    return state


def create_test_state_with_task(
    task_id: str = "task_001",
    description: str = "Add player health system",
    measurable_outcome: str = "Player has health property",
    status: TaskStatus = TaskStatus.IN_PROGRESS,
    milestone_id: str | None = "milestone_001",
    **task_kwargs: Any,
) -> WorkflowState:
    """Create a test state with a sample task ready for execution.
    
    Args:
        task_id: Task identifier
        description: Task description
        measurable_outcome: Measurable outcome
        status: Task status
        milestone_id: Optional milestone ID
        **task_kwargs: Additional Task fields
    
    Returns:
        WorkflowState with the task set as current
    
    Example:
        >>> state = create_test_state_with_task(
        ...     description="Add health bar UI",
        ...     measurable_outcome="Health bar visible in game"
        ... )
    """
    task = Task(
        id=task_id,
        description=description,
        measurable_outcome=measurable_outcome,
        status=status,
        milestone_id=milestone_id,
        **task_kwargs,
    )
    
    return create_test_state(
        user_request=description,
        with_task=task,
        current_task_id=task_id,
    )


def create_test_state_from_fixture(fixture_path: str) -> WorkflowState:
    """Create a WorkflowState from a JSON fixture file.
    
    Args:
        fixture_path: Path to JSON file containing state data
    
    Returns:
        WorkflowState loaded from fixture
    
    Example:
        >>> state = create_test_state_from_fixture("tests/fixtures/states/researcher_input.json")
    """
    fixture_file = Path(fixture_path)
    if not fixture_file.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    
    with open(fixture_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create initial state and update with fixture data
    state = create_initial_state(
        user_request=data.get("user_request", "Test request"),
        repo_root=data.get("repo_root", str(Path.cwd())),
        verbose=data.get("verbose", False),
    )
    
    # Update with all fixture fields
    for key, value in data.items():
        if key not in ("user_request", "repo_root", "verbose"):
            state[key] = value
    
    return state


def save_state_fixture(state: WorkflowState, fixture_path: str) -> None:
    """Save a WorkflowState to a JSON fixture file.
    
    Args:
        state: WorkflowState to save
        fixture_path: Path where to save the fixture
    
    Example:
        >>> state = create_test_state_with_task()
        >>> save_state_fixture(state, "tests/fixtures/states/researcher_input.json")
    """
    fixture_file = Path(fixture_path)
    fixture_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert state to JSON-serializable dict
    state_dict = dict(state)
    
    # Convert any Path objects to strings
    for key, value in state_dict.items():
        if isinstance(value, Path):
            state_dict[key] = str(value)
    
    with open(fixture_file, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2, default=str)
