"""Unit tests for subagent tools (explain_code, ask, web_search)."""

from unittest.mock import patch

from agents.subagents import explain_code, ask, web_search


def test_web_search_returns_formatted_results():
    """web_search returns non-empty formatted results for valid query."""
    result = web_search.invoke({"query": "Python", "max_results": 2})
    assert "Found" in result or "results" in result.lower()
    assert len(result) > 50


def test_web_search_handles_empty_query():
    """web_search handles edge case gracefully."""
    result = web_search.invoke({"query": "xyznonexistent12345", "max_results": 2})
    # Should return something (either results or "No search results")
    assert isinstance(result, str)
    assert len(result) > 0


def test_explain_code_handles_exception():
    """explain_code returns error message on exception."""
    import agents.subagents.explain_code as explain_module
    explain_module._explain_agent = None

    with patch("agents.subagents.explain_code._get_explain_agent") as mock_get:
        mock_get.side_effect = RuntimeError("LLM unavailable")
        result = explain_code.invoke({"query": "test"})

    assert "error" in result.lower()
    assert "unavailable" in result.lower() or "runtimeerror" in result.lower()


def test_ask_handles_exception():
    """ask returns error message on exception."""
    import agents.subagents.ask as ask_module
    ask_module._ask_agent = None

    with patch("agents.subagents.ask._get_ask_agent") as mock_get:
        mock_get.side_effect = RuntimeError("LLM unavailable")
        result = ask.invoke({"query": "test"})

    assert "error" in result.lower()
