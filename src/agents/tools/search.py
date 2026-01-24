"""File and code search tools with bounded results."""

import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool

from .gitignore import load_gitignore_patterns, should_ignore


def _get_line_count(file_path: Path) -> int:
    """Get the line count for a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, IOError):
        return -1  # Binary or unreadable


@tool
def search_files(
    query: str,
    file_pattern: str = "*",
    max_results: int = 10,
) -> str:
    """Search for files containing a pattern using ripgrep/regex.
    
    Returns results with file paths and LINE NUMBERS for each match.
    Format: "path/to/file.gd:42:  matching line content"
    
    All paths are relative to the current working directory.
    
    Use this to:
    - Find where symbols (function/class names) are defined or used
    - Search for keywords across the codebase
    - Get line numbers to then read specific ranges with read_file_lines
    
    Args:
        query: The search pattern (supports regex, e.g., "^func my_function" or "class.*Colony")
              Search is case-insensitive
        file_pattern: Glob pattern to filter files (e.g., "*.gd", "*.tscn")
        max_results: Maximum number of results to return
    
    Returns:
        Search results with format "file:line: content (N lines total)" - includes line counts
        Use line numbers to read ranges with read_file_lines
    """
    repo_root = Path.cwd()
    
    # Load .gitignore patterns
    gitignore_patterns = load_gitignore_patterns(repo_root)
    
    try:
        # Use ripgrep for fast searching (case-insensitive)
        # Note: ripgrep respects .gitignore by default, but we'll also filter results
        cmd = [
            "rg",
            "--max-count", str(max_results),
            "--line-number",
            "--no-heading",
            "--color", "never",
            "-i",  # Case-insensitive
            "-g", file_pattern,
            query,
            str(repo_root),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[:max_results]
            if not lines:
                return "No matches found."
            
            # Add line counts for each file mentioned
            file_line_counts = {}
            output_lines = []
            
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        file_path_str = parts[0]
                        line_num = parts[1] if len(parts) > 1 else ""
                        content = parts[2] if len(parts) > 2 else ""
                        
                        # Convert to Path and ensure it's relative to repo_root
                        file_path_obj = Path(file_path_str)
                        if file_path_obj.is_absolute():
                            try:
                                rel_path = file_path_obj.relative_to(repo_root)
                            except ValueError:
                                # Path is outside repo_root, skip
                                continue
                        else:
                            rel_path = file_path_obj
                        
                        # Skip if ignored by .gitignore
                        full_file_path = repo_root / rel_path
                        if should_ignore(full_file_path, repo_root, gitignore_patterns):
                            continue
                        
                        rel_path_str = str(rel_path)
                        
                        # Get line count if not already cached
                        if rel_path_str not in file_line_counts:
                            full_path = repo_root / rel_path_str
                            if full_path.exists():
                                line_count = _get_line_count(full_path)
                                if line_count >= 0:
                                    file_line_counts[rel_path_str] = line_count
                        
                        # Reconstruct line with relative path
                        if len(parts) >= 3:
                            new_line = f"{rel_path_str}:{line_num}:{content}"
                        else:
                            new_line = f"{rel_path_str}:{line_num}"
                        
                        # Add line count info if we have it
                        if rel_path_str in file_line_counts:
                            line_count = file_line_counts[rel_path_str]
                            output_lines.append(f"{new_line} ({line_count} lines total)")
                        else:
                            output_lines.append(new_line)
                    else:
                        output_lines.append(line)
                else:
                    output_lines.append(line)
            
            return "\n".join(output_lines)
        elif result.returncode == 1:
            return "No matches found."
        else:
            return f"Search error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return "Search timed out."
    except FileNotFoundError:
        return "Error: ripgrep (rg) not found. Please install ripgrep for search functionality."


@tool
def list_directory(
    path: str = ".",
    max_depth: int = 2,
    max_items: int = 50,
) -> str:
    """List contents of a directory with limited depth.
    
    Files show their line count in parentheses, e.g., "file.gd (150 lines)"
    All paths are relative to the current working directory.
    
    Args:
        path: Path relative to current directory (use "." for root)
        max_depth: Maximum depth to recurse
        max_items: Maximum number of items to return
    
    Returns:
        Directory listing as a tree-like string with line counts for files
    """
    repo_root = Path.cwd()
    full_path = repo_root / path
    
    if not full_path.exists():
        return f"Path does not exist: {path}"
    
    if not full_path.is_dir():
        return f"Not a directory: {path}"
    
    # Load .gitignore patterns once
    gitignore_patterns = load_gitignore_patterns(repo_root)
    
    items = []
    count = 0
    
    def walk_dir(dir_path: Path, depth: int, prefix: str = ""):
        nonlocal count
        
        if depth > max_depth or count >= max_items:
            return
        
        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        
        for entry in entries:
            if count >= max_items:
                items.append(f"{prefix}... (truncated)")
                return
            
            # Skip if ignored by .gitignore
            if should_ignore(entry, repo_root, gitignore_patterns):
                continue
            
            rel_path = entry.relative_to(full_path)
            
            if entry.is_dir():
                items.append(f"{prefix}{entry.name}/")
                count += 1
                walk_dir(entry, depth + 1, prefix + "  ")
            else:
                # Add line count for files
                line_count = _get_line_count(entry)
                if line_count >= 0:
                    items.append(f"{prefix}{entry.name} ({line_count} lines)")
                else:
                    items.append(f"{prefix}{entry.name}")
                count += 1
    
    walk_dir(full_path, 0)
    
    if not items:
        return "Directory is empty."
    
    return "\n".join(items)


def _expand_brace_pattern(pattern: str) -> list[str]:
    """Expand bash-style brace patterns into multiple glob patterns.
    
    E.g., "**/*.{gd,ts}" -> ["**/*.gd", "**/*.ts"]
    """
    import re
    
    # Match {option1,option2,...} pattern
    brace_match = re.search(r'\{([^}]+)\}', pattern)
    if not brace_match:
        return [pattern]
    
    options = brace_match.group(1).split(',')
    prefix = pattern[:brace_match.start()]
    suffix = pattern[brace_match.end():]
    
    # Recursively expand in case of multiple brace groups
    expanded = []
    for opt in options:
        expanded.extend(_expand_brace_pattern(prefix + opt.strip() + suffix))
    
    return expanded


@tool
def find_files_by_name(
    pattern: str,
    max_results: int = 20,
) -> str:
    """Find files matching a glob pattern.
    
    Use this to verify file hints from requirements - they may be partial or incorrect.
    Results include line counts for each file.
    All paths are relative to the current working directory.
    
    Args:
        pattern: Glob pattern (e.g., "*.gd", "**/test_*.py", "*colony*.gd", "**/*.{gd,tscn}")
                 Supports bash-style brace expansion: {a,b,c}
        max_results: Maximum number of results
    
    Returns:
        List of matching file paths with line counts, e.g., "file.gd (150 lines)"
    """
    root = Path.cwd()
    
    try:
        # Expand brace patterns (Python glob doesn't support them natively)
        patterns = _expand_brace_pattern(pattern)
        
        # Collect matches from all expanded patterns
        all_matches = []
        for p in patterns:
            all_matches.extend(root.glob(p))
        
        # Dedupe while preserving order
        seen = set()
        matches = []
        for m in all_matches:
            if m not in seen:
                seen.add(m)
                matches.append(m)
        
        matches = matches[:max_results]
        
        if not matches:
            return f"No files matching pattern: {pattern}"
        
        # Load .gitignore patterns
        gitignore_patterns = load_gitignore_patterns(root)
        
        # Return relative paths with line counts
        results = []
        for match in matches:
            # Skip if ignored by .gitignore
            if should_ignore(match, root, gitignore_patterns):
                continue
            
            rel_path = str(match.relative_to(root))
            line_count = _get_line_count(match)
            if line_count >= 0:
                results.append(f"{rel_path} ({line_count} lines)")
            else:
                results.append(rel_path)
        
        if not results:
            return f"No files matching pattern: {pattern}"
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Error finding files: {e}"


