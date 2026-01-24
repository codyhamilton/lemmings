"""File reading tools with bounded results."""

from pathlib import Path
from langchain_core.tools import tool


def _clean_path(path: str) -> str:
    """Clean path by removing Godot res:// prefix if present."""
    if path.startswith("res://"):
        return path[6:]
    return path


@tool
def read_file(path: str) -> str:
    """Read the complete contents of a file.
    
    Path is relative to the current working directory.
    
    Args:
        path: Path relative to current directory
    
    Returns:
        File contents or error message
    """
    path = _clean_path(path)
    full_path = Path.cwd() / path
    
    if not full_path.exists():
        return f"File does not exist: {path}"
    
    if not full_path.is_file():
        return f"Not a file: {path}"
    
    try:
        content = full_path.read_text(encoding="utf-8")
        
        # Warn if file is very large
        lines = content.split("\n")
        if len(lines) > 50:
            return (
                f"Warning: Large file ({len(lines)} lines). "
                f"Consider using read_file_lines for specific sections.\n\n"
                f"{content}"
            )
        
        return content
        
    except UnicodeDecodeError:
        return f"Cannot read file (not text): {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def read_file_lines(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    max_lines: int = 100,
) -> str:
    """Read specific lines from a file.
    
    This is the PRIMARY way to read file content. Use this to read line ranges
    around matches found by search_files. Never read entire files.
    Path is relative to the current working directory.
    
    Typical usage:
    1. Use search_files to find line numbers where symbols/keywords appear
    2. Read surrounding context: read_file_lines(path, match_line-10, match_line+10)
    3. Use max_lines to limit the range (default 100 lines max)
    
    Args:
        path: Path relative to current directory
        start_line: First line to read (1-indexed)
        end_line: Last line to read (inclusive), or None for max_lines from start
        max_lines: Maximum number of lines to return (enforced limit)
    
    Returns:
        File contents with line numbers formatted as "line| content" or error message
    """
    path = _clean_path(path)
    full_path = Path.cwd() / path
    
    if not full_path.exists():
        return f"File does not exist: {path}"
    
    if not full_path.is_file():
        return f"Not a file: {path}"
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        total_lines = len(all_lines)
        
        # Adjust indices (1-indexed to 0-indexed)
        start_idx = max(0, start_line - 1)
        
        if end_line is None:
            end_idx = min(start_idx + max_lines, total_lines)
        else:
            end_idx = min(end_line, total_lines)
        
        # Enforce max_lines limit
        if end_idx - start_idx > max_lines:
            end_idx = start_idx + max_lines
        
        selected_lines = all_lines[start_idx:end_idx]
        
        # Format with line numbers, truncating long lines
        result_lines = []
        for i, line in enumerate(selected_lines, start=start_idx + 1):
            line_content = line.rstrip()
            if len(line_content) > 100:
                line_content = line_content[:100] + "..."
            result_lines.append(f"{i:4d}| {line_content}")
        
        header = f"File: {path} (lines {start_idx + 1}-{end_idx} of {total_lines})"
        
        return header + "\n" + "\n".join(result_lines)
        
    except UnicodeDecodeError:
        return f"Cannot read file (not text): {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def get_file_info(path: str) -> str:
    """Get information about a file without reading its contents.
    
    Path is relative to the current working directory.
    
    Args:
        path: Path relative to current directory
    
    Returns:
        File information (size, line count, type)
    """
    path = _clean_path(path)
    full_path = Path.cwd() / path
    
    if not full_path.exists():
        return f"File does not exist: {path}"
    
    if full_path.is_dir():
        return f"{path} is a directory"
    
    try:
        stat = full_path.stat()
        size_kb = stat.st_size / 1024
        
        # Count lines
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        except UnicodeDecodeError:
            line_count = "N/A (binary)"
        
        return (
            f"File: {path}\n"
            f"Size: {size_kb:.1f} KB\n"
            f"Lines: {line_count}\n"
            f"Extension: {full_path.suffix or 'none'}"
        )
        
    except Exception as e:
        return f"Error getting file info: {e}"
