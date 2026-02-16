"""Workspace root for the current workflow run.

Set once at init (e.g. in create_initial_state). Path-aware tools resolve
paths relative to this root so agents do not need to chdir.
"""

from pathlib import Path

_workspace_root: Path | None = None


def set_workspace_root(path: str | Path) -> None:
    """Set the workspace root for this run. Called at workflow init."""
    global _workspace_root
    _workspace_root = Path(path).resolve()


def get_workspace_root() -> Path:
    """Return the workspace root, or cwd if not set (e.g. tests that never create state)."""
    if _workspace_root is not None:
        return _workspace_root
    return Path.cwd()
