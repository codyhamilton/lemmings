# Agent Testing Framework

This directory contains the testing infrastructure for the LangGraph agent workflow. The framework supports multiple testing layers from fast unit tests to full integration tests.

## Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── fixtures/
│   ├── states/              # Sample WorkflowState snapshots
│   └── responses/           # Expected LLM responses
├── unit/                    # Unit tests (no LLM required)
│   ├── test_state.py        # TaskTree, Task, WorkflowState
│   ├── test_routing.py       # Routing functions
│   └── test_helper_nodes.py # Helper nodes
├── agents/                  # Agent tests with mock LLM
│   ├── test_researcher.py
│   └── test_implementor.py
└── integration/             # Multi-node sequence tests
    └── test_subgraphs.py
```

## Running Tests

### Run all tests
```bash
cd agents
pytest tests/
```

### Run specific test categories
```bash
# Unit tests only (fast, no LLM)
pytest tests/unit/

# Agent tests with mocks
pytest tests/agents/

# Integration tests
pytest tests/integration/
```

### Run with verbose output
```bash
pytest tests/ -v
```

## Testing Layers

### Layer 1: Unit Tests (No LLM)

Fast, deterministic tests for state management and routing logic:

- `TaskTree` operations (add_task, mark_complete, get_ready_tasks)
- Routing functions (after_researcher, after_validator, after_qa)
- Helper nodes (mark_task_complete_node, mark_task_failed_node)

These run in milliseconds and don't require any external dependencies.

### Layer 2: Agent Tests with Mock LLM

Test individual agents with controlled LLM responses using `FakeMessagesListChatModel`:

```python
def test_researcher_detects_gap(mock_llm_factory, test_state_with_task):
    mock_llm = mock_llm_factory([{
        "gap_exists": True,
        "current_state_summary": "No health system",
        # ...
    }])
    
    with patch("agents.agents.researcher.planning_llm", mock_llm):
        result = researcher_node(test_state_with_task)
    
    assert result["current_gap_analysis"]["gap_exists"] is True
```

### Layer 3: Integration Tests

Test multi-node sequences to verify agents work together:

```python
def test_researcher_to_planner_flow_with_mocks(mock_llm_factory):
    # Setup mocks for each agent
    researcher_llm = mock_llm_factory([{...}])
    planner_llm = mock_llm_factory([{...}])
    
    # Run sequence
    state.update(researcher_node(state))
    state.update(planner_node(state))
    
    assert state["current_implementation_plan"] is not None
```

## Interactive Debugging

Use the CLI runner to test individual agents during development:

```bash
# Run researcher with fixture state
python -m agents.testing.runner researcher --state tests/fixtures/states/researcher_input.json

# Run with mock LLM
python -m agents.testing.runner researcher \
    --mock-llm \
    --response tests/fixtures/responses/researcher_gap_exists.json

# Run with real LLM (for prompt testing)
python -m agents.testing.runner researcher \
    --state tests/fixtures/states/researcher_input.json \
    --verbose

# Save output as fixture
python -m agents.testing.runner researcher \
    --state input.json \
    --save-output output.json
```

## Fixtures

### State Fixtures

Create test states using the fixture factory:

```python
from agents.testing.fixtures import create_test_state_with_task

state = create_test_state_with_task(
    description="Add player health",
    measurable_outcome="Player has health property",
    status=TaskStatus.IN_PROGRESS,
)
```

### Response Fixtures

Store expected LLM responses as JSON:

```json
{
  "gap_exists": true,
  "current_state_summary": "No health system",
  "desired_state_summary": "Need health system",
  "gap_description": "Missing",
  "relevant_files": ["scripts/Player.gd"],
  "keywords": ["health"]
}
```

## Best Practices

1. **Start with unit tests**: Test state logic and routing first (fastest feedback)
2. **Use mocks for agent tests**: Mock LLM responses for deterministic testing
3. **Test in isolation**: Each agent test should be independent
4. **Use fixtures**: Reuse state and response fixtures across tests
5. **Test edge cases**: Missing tasks, max retries, error conditions
6. **Integration tests sparingly**: Use for critical multi-node flows only

## Adding New Tests

### Unit Test Example

```python
# tests/unit/test_my_feature.py
from agents.state import Task, TaskStatus

def test_my_feature():
    task = Task(id="task_001", description="Test", measurable_outcome="Outcome")
    assert task.status == TaskStatus.PENDING
```

### Agent Test Example

```python
# tests/agents/test_my_agent.py
from unittest.mock import patch
from agents.agents.my_agent import my_agent_node

def test_my_agent(mock_llm_factory, test_state_with_task):
    mock_llm = mock_llm_factory([{"result": "expected"}])
    
    with patch("agents.agents.my_agent.planning_llm", mock_llm):
        result = my_agent_node(test_state_with_task)
    
    assert result["expected_field"] == "expected_value"
```

## CI/CD

Unit tests and mock-based tests can run in CI without API access. Integration tests that require real LLMs should be marked with `@pytest.mark.skip` or run only in specific environments.
