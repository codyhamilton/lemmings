"""Mock tool implementations for testing agents without filesystem side effects.

These tools return controlled responses based on a response dictionary,
allowing deterministic testing of agent behavior.
"""

from typing import Any
from langchain_core.tools import tool


def create_mock_tools(responses: dict[str, Any] | None = None) -> list:
    """Create mock versions of all tools used by agents.
    
    Args:
        responses: Dictionary mapping tool names to response values.
            Can be:
            - Simple string responses: {"read_file": "content"}
            - Per-path responses: {"read_file": {"path/to/file.gd": "content"}}
            - Callable functions: {"search_files": lambda pattern: "results"}
    
    Returns:
        List of mock tool functions
    
    Example:
        >>> tools = create_mock_tools({
        ...     "read_file": {"scripts/Player.gd": "extends Node\nvar health = 100"},
        ...     "search_files": "Found 3 matches in scripts/entities/Player.gd:42"
        ... })
    """
    responses = responses or {}
    
    @tool
    def mock_read_file(path: str) -> str:
        """Mock read_file that returns predefined content."""
        tool_resp = responses.get("read_file", {})
        if isinstance(tool_resp, dict):
            return tool_resp.get(path, f"# Mock content for {path}\n# File not in mock responses")
        return tool_resp
    
    @tool
    def mock_read_file_lines(
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 100,
    ) -> str:
        """Mock read_file_lines that returns predefined content with line numbers."""
        tool_resp = responses.get("read_file_lines", {})
        if isinstance(tool_resp, dict):
            content = tool_resp.get(path, f"# Mock content for {path}")
        else:
            content = tool_resp
        
        # Format with line numbers
        lines = content.split("\n")
        start_idx = max(0, start_line - 1)
        end_idx = min(start_idx + max_lines, len(lines)) if end_line is None else min(end_line, len(lines))
        
        result_lines = []
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
            result_lines.append(f"{i:4d}| {line}")
        
        return f"File: {path} (lines {start_idx + 1}-{end_idx} of {len(lines)})\n" + "\n".join(result_lines)
    
    @tool
    def mock_search_files(
        query: str,
        file_pattern: str = "*",
        max_results: int = 10,
    ) -> str:
        """Mock search_files that returns predefined results."""
        tool_resp = responses.get("search_files", {})
        if callable(tool_resp):
            return tool_resp(query, file_pattern, max_results)
        elif isinstance(tool_resp, dict):
            return tool_resp.get(query, f"No matches found for '{query}'")
        return tool_resp or f"Found 0 matches for '{query}'"
    
    @tool
    def mock_list_directory(path: str = ".") -> str:
        """Mock list_directory that returns predefined directory structure."""
        tool_resp = responses.get("list_directory", {})
        if isinstance(tool_resp, dict):
            return tool_resp.get(path, f"Directory listing for {path}:\n  (empty)")
        return tool_resp or f"Directory listing for {path}:\n  (empty)"
    
    @tool
    def mock_find_files_by_name(pattern: str, path: str = ".") -> str:
        """Mock find_files_by_name that returns predefined file matches."""
        tool_resp = responses.get("find_files_by_name", {})
        if isinstance(tool_resp, dict):
            return tool_resp.get(pattern, f"No files found matching '{pattern}'")
        return tool_resp or f"No files found matching '{pattern}'"
    
    @tool
    def mock_rag_search(query: str, top_k: int = 5) -> str:
        """Mock rag_search that returns predefined semantic search results."""
        tool_resp = responses.get("rag_search", {})
        if isinstance(tool_resp, dict):
            return tool_resp.get(query, f"No semantic matches for '{query}'")
        return tool_resp or f"No semantic matches for '{query}'"
    
    @tool
    def mock_write_file(path: str, content: str) -> str:
        """Mock write_file that tracks writes without touching filesystem."""
        # Track writes in responses for verification
        if "write_file_calls" not in responses:
            responses["write_file_calls"] = []
        responses["write_file_calls"].append({"path": path, "content": content})
        return f"Successfully wrote {len(content)} bytes to {path} (mocked)"
    
    @tool
    def mock_apply_edit(path: str, old_text: str, new_text: str) -> str:
        """Mock apply_edit that tracks edits without touching filesystem."""
        if "apply_edit_calls" not in responses:
            responses["apply_edit_calls"] = []
        responses["apply_edit_calls"].append({
            "path": path,
            "old_text": old_text,
            "new_text": new_text
        })
        return f"Successfully applied edit to {path} (mocked)"
    
    @tool
    def mock_create_file(path: str, content: str) -> str:
        """Mock create_file that tracks creates without touching filesystem."""
        if "create_file_calls" not in responses:
            responses["create_file_calls"] = []
        responses["create_file_calls"].append({"path": path, "content": content})
        return f"Successfully created {path} (mocked)"
    
    return [
        mock_read_file,
        mock_read_file_lines,
        mock_search_files,
        mock_list_directory,
        mock_find_files_by_name,
        mock_rag_search,
        mock_write_file,
        mock_apply_edit,
        mock_create_file,
    ]


def get_mock_tool_calls(responses: dict[str, Any]) -> dict[str, list]:
    """Extract tool call history from mock responses.
    
    Args:
        responses: The responses dict passed to create_mock_tools
    
    Returns:
        Dictionary with tool call histories:
        {
            "write_file_calls": [...],
            "apply_edit_calls": [...],
            "create_file_calls": [...]
        }
    """
    return {
        "write_file_calls": responses.get("write_file_calls", []),
        "apply_edit_calls": responses.get("apply_edit_calls", []),
        "create_file_calls": responses.get("create_file_calls", []),
    }
