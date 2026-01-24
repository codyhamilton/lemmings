"""Mock LLM factory for testing agents without real API calls.

Uses LangChain's FakeMessagesListChatModel to provide scripted responses
for deterministic agent testing.
"""

import json
from typing import Any
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


def create_mock_llm(responses: list[str | dict | AIMessage]) -> FakeMessagesListChatModel:
    """Create a mock LLM with scripted responses.
    
    Args:
        responses: List of responses. Can be:
            - str: Raw text response
            - dict: Will be JSON-stringified
            - AIMessage: Direct message object
    
    Returns:
        FakeMessagesListChatModel configured with the responses
    
    Example:
        >>> mock_llm = create_mock_llm([
        ...     {"gap_exists": True, "current_state_summary": "No health system"},
        ...     "Additional response text"
        ... ])
    """
    messages = []
    for resp in responses:
        if isinstance(resp, AIMessage):
            messages.append(resp)
        elif isinstance(resp, dict):
            # Wrap JSON in code block for better parsing
            content = f"```json\n{json.dumps(resp, indent=2)}\n```"
            messages.append(AIMessage(content=content))
        else:
            messages.append(AIMessage(content=str(resp)))
    
    return FakeMessagesListChatModel(responses=messages)


def create_mock_llm_from_fixture(fixture_path: str) -> FakeMessagesListChatModel:
    """Create a mock LLM from a JSON fixture file.
    
    Args:
        fixture_path: Path to JSON file containing response(s)
    
    Returns:
        FakeMessagesListChatModel configured with fixture responses
    
    Example:
        >>> mock_llm = create_mock_llm_from_fixture("tests/fixtures/responses/researcher_gap_exists.json")
    """
    from pathlib import Path
    
    fixture_file = Path(fixture_path)
    if not fixture_file.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    
    with open(fixture_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # If it's a single response dict, wrap in list
    if isinstance(data, dict):
        responses = [data]
    elif isinstance(data, list):
        responses = data
    else:
        raise ValueError(f"Fixture must contain dict or list, got {type(data)}")
    
    return create_mock_llm(responses)
