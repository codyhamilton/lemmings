"""Pytest configuration and fixtures for agent testing."""

import pytest
import tempfile
from pathlib import Path

from agents.task_states import Task, TaskStatus
from agents.testing.mock_llm import create_mock_llm
from agents.testing.mock_tools import create_mock_tools
from agents.testing.fixtures import (
    create_test_state,
    create_test_state_with_task,
    create_test_state_from_fixture,
)


@pytest.fixture
def mock_llm_factory():
    """Factory for creating mock LLMs with scripted responses.
    
    Returns:
        Function that takes a list of responses and returns FakeMessagesListChatModel
    
    Example:
        >>> def test_agent(mock_llm_factory):
        ...     mock_llm = mock_llm_factory([{"gap_exists": True}])
        ...     # Use mock_llm in test
    """
    def _factory(responses):
        return create_mock_llm(responses)
    return _factory


@pytest.fixture
def mock_tools_factory():
    """Factory for creating mock tools with controlled responses.
    
    Returns:
        Function that takes a responses dict and returns list of mock tools
    
    Example:
        >>> def test_agent(mock_tools_factory):
        ...     tools = mock_tools_factory({
        ...         "read_file": {"path/to/file.gd": "content"}
        ...     })
        ...     # Use tools in test
    """
    def _factory(responses=None):
        return create_mock_tools(responses)
    return _factory


@pytest.fixture
def test_repo_root():
    """Create a temporary directory for testing repository operations.
    
    Returns:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="agent_test_repo_"))
    yield temp_dir
    # Cleanup handled by tempfile


@pytest.fixture
def test_state():
    """Basic empty test state.
    
    Returns:
        WorkflowState with minimal configuration
    """
    return create_test_state()


@pytest.fixture
def test_state_with_task():
    """Test state with a sample task ready for execution.
    
    Returns:
        WorkflowState with task_001 set as current task
    """
    return create_test_state_with_task(
        task_id="task_001",
        description="Add player health system",
        measurable_outcome="Player has health property",
        status=TaskStatus.IN_PROGRESS,
    )


@pytest.fixture
def researcher_input_fixture():
    """Load researcher input state from fixture.
    
    Returns:
        WorkflowState loaded from fixture
    """
    fixture_path = Path(__file__).parent / "fixtures" / "states" / "researcher_input.json"
    return create_test_state_from_fixture(str(fixture_path))


@pytest.fixture
def researcher_gap_exists_response():
    """Load expected researcher response when gap exists.
    
    Returns:
        Dict with gap analysis data
    """
    import json
    fixture_path = Path(__file__).parent / "fixtures" / "responses" / "researcher_gap_exists.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def researcher_no_gap_response():
    """Load expected researcher response when no gap exists.
    
    Returns:
        Dict with gap analysis data
    """
    import json
    fixture_path = Path(__file__).parent / "fixtures" / "responses" / "researcher_no_gap.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)
