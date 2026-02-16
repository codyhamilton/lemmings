"""Git helpers for workspace detection and diff retrieval."""

import shutil
import subprocess
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


def is_git_workspace(repo_root: str | Path) -> bool:
    """Return True if repo_root is inside a git work tree and git is on PATH."""
    if not shutil.which("git"):
        return False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def get_diff(repo_root: str | Path, max_chars: int = 12_000) -> str:
    """Run git diff from repo_root; return capped output or empty string on error."""
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        diff = result.stdout
        if len(diff) > max_chars:
            diff = diff[:max_chars] + "\n... (truncated)"
        return diff
    except Exception as e:
        logger.debug("git diff failed: %s", e)
        return ""
