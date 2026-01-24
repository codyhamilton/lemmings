"""Utilities for parsing and checking .gitignore patterns."""

import re
from pathlib import Path
from typing import Set


def _parse_gitignore_pattern(pattern: str) -> tuple[str | None, bool, bool]:
    """Parse a gitignore pattern into a regex pattern.
    
    Returns:
        (regex_pattern, is_directory_pattern, negated) or (None, False, False) if should skip
    """
    # Remove leading/trailing whitespace
    pattern = pattern.strip()
    
    # Skip empty lines and comments
    if not pattern or pattern.startswith("#"):
        return None, False, False
    
    # Negation pattern (starts with !)
    negated = pattern.startswith("!")
    if negated:
        pattern = pattern[1:]
    
    # Directory pattern (ends with /)
    is_dir = pattern.endswith("/")
    if is_dir:
        pattern = pattern[:-1]
    
    # Escape special regex chars except * and ?
    pattern = re.escape(pattern)
    
    # Convert gitignore wildcards to regex
    pattern = pattern.replace(r"\*\*", "GITIGNORE_DOUBLE_STAR")
    pattern = pattern.replace(r"\*", r"[^/]*")
    pattern = pattern.replace(r"\?", r"[^/]")
    pattern = pattern.replace("GITIGNORE_DOUBLE_STAR", r".*")
    
    # Anchor patterns appropriately
    if pattern.startswith("/"):
        # Absolute from repo root
        pattern = "^" + pattern[1:]
    else:
        # Can match anywhere
        pattern = "(^|/)" + pattern
    
    if is_dir:
        pattern = pattern + "/"
    
    pattern = pattern + "$"
    
    return pattern, is_dir, negated


def load_gitignore_patterns(repo_root: Path) -> list[tuple[re.Pattern, bool, bool]]:
    """Load and parse .gitignore file.
    
    Returns:
        List of (compiled_pattern, is_directory, negated) tuples
    """
    gitignore_path = repo_root / ".gitignore"
    patterns = []
    
    if not gitignore_path.exists():
        return patterns
    
    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                pattern_str, is_dir, negated = _parse_gitignore_pattern(line)
                if pattern_str:
                    try:
                        compiled = re.compile(pattern_str)
                        patterns.append((compiled, is_dir, negated))
                    except re.error:
                        # Skip invalid patterns
                        continue
    except Exception:
        # If we can't read .gitignore, just continue without it
        pass
    
    return patterns


def load_rag_ignore_patterns(repo_root: Path) -> list[tuple[re.Pattern, bool, bool]]:
    """Load and parse .rag-ignore file (optional additional ignore patterns for RAG).
    
    Supports globbing patterns like .gitignore.
    
    Returns:
        List of (compiled_pattern, is_directory, negated) tuples
    """
    ragignore_path = repo_root / ".rag-ignore"
    patterns = []
    
    if not ragignore_path.exists():
        return patterns
    
    try:
        with open(ragignore_path, "r", encoding="utf-8") as f:
            for line in f:
                pattern_str, is_dir, negated = _parse_gitignore_pattern(line)
                if pattern_str:
                    try:
                        compiled = re.compile(pattern_str)
                        patterns.append((compiled, is_dir, negated))
                    except re.error:
                        # Skip invalid patterns
                        continue
    except Exception:
        # If we can't read .rag-ignore, just continue without it
        pass
    
    return patterns


def load_ignore_patterns(repo_root: Path) -> list[tuple[re.Pattern, bool, bool]]:
    """Load and combine patterns from both .gitignore and .rag-ignore files.
    
    .rag-ignore patterns are appended after .gitignore patterns, so they can
    override .gitignore patterns using negation (!).
    
    Returns:
        List of (compiled_pattern, is_directory, negated) tuples
    """
    patterns = load_gitignore_patterns(repo_root)
    rag_patterns = load_rag_ignore_patterns(repo_root)
    patterns.extend(rag_patterns)
    return patterns


def should_ignore(path: Path, repo_root: Path, patterns: list[tuple[re.Pattern, bool, bool]] = None) -> bool:
    """Check if a path should be ignored based on .gitignore and .rag-ignore patterns.
    
    Args:
        path: Path to check (can be absolute or relative)
        repo_root: Root of the repository
        patterns: Pre-loaded patterns (will load from .gitignore and .rag-ignore if None)
    
    Returns:
        True if path should be ignored
    """
    if patterns is None:
        patterns = load_ignore_patterns(repo_root)
    
    # Convert to relative path
    try:
        rel_path = path.relative_to(repo_root)
    except ValueError:
        # Path is outside repo, don't ignore (let other checks handle it)
        return False
    
    rel_path_str = str(rel_path).replace("\\", "/")
    is_dir = path.is_dir()

    # Always ignore dot-prefixed paths and the git directory for tooling output.
    # This applies to *all* search/list tools regardless of .gitignore.
    # (We intentionally do not allow negation to "unignore" these.)
    parts = rel_path.parts
    if any(p.startswith(".") for p in parts):
        return True
    if ".git" in parts:
        return True
    
    # Check patterns (negated patterns override)
    ignored = False
    for pattern, pattern_is_dir, negated in patterns:
        if pattern_is_dir:
            # Directory pattern: check if the path or any parent directory matches
            # For "llm/", we want to match "llm/" and anything under "llm/"
            # The pattern regex is like "(^|/)llm/$" - we need to match paths starting with "llm/"
            
            # Create a pattern that matches the directory and anything under it
            # Remove the trailing "$" anchor to allow matching subpaths
            dir_pattern_str = pattern.pattern.rstrip("$")
            if dir_pattern_str.endswith("/"):
                # Pattern is like "(^|/)llm/" - this will match "llm/" and "llm/anything"
                dir_pattern = re.compile(dir_pattern_str)
            else:
                # Shouldn't happen, but handle it
                dir_pattern = pattern
            
            # Check if path matches (for directories, add trailing /)
            path_to_check = rel_path_str + ("/" if is_dir else "")
            if dir_pattern.search(path_to_check):
                if negated:
                    ignored = False
                else:
                    ignored = True
        else:
            # File pattern: check the path itself
            if pattern.search(rel_path_str):
                if negated:
                    ignored = False
                else:
                    ignored = True
    
    return ignored
