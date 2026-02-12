"""File editing tools for making changes."""

from pathlib import Path
from langchain_core.tools import tool

from ..logging_config import get_logger
from .gitignore import should_ignore, load_gitignore_patterns

logger = get_logger(__name__)


def _find_git_root(start_path: Path) -> Path | None:
    """Find the git repository root by walking up from start_path."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def _clean_path(path: str) -> str:
    """Clean path by removing Godot res:// prefix if present."""
    if path.startswith("res://"):
        return path[6:]
    return path


def _validate_path_security(path: str, repo_root: Path | None = None) -> tuple[bool, str, Path]:
    """Validate that a path is safe to write to.
    
    The agent only knows relative paths. All paths are resolved relative to
    the current working directory, which should be set to the repository root
    before tools are called.
    
    Returns:
        (is_valid, error_message, resolved_path)
    """
    # Clean path - strip res:// prefix if present (Godot resource paths)
    path = _clean_path(path)
    
    # Use current working directory as repo root (set by coder agent before tool execution)
    if repo_root is None:
        repo_root = Path.cwd()
        # Verify we're actually in a git repo (safety check)
        if not (repo_root / ".git").exists():
            # Try to find git root by walking up
            found_root = _find_git_root(repo_root)
            if found_root:
                repo_root = found_root
    
    repo_root = repo_root.resolve()
    
    # Check 0: Prevent obvious path traversal attempts in the input
    # Normalize the path to detect ../ patterns
    normalized_path = Path(path)
    if normalized_path.is_absolute():
        return False, f"Security error: Path must be relative, not absolute", Path(path)
    
    # Check for path traversal components
    path_parts = normalized_path.parts
    if ".." in path_parts or path_parts and path_parts[0] == "/":
        return False, f"Security error: Path contains traversal components (../)", normalized_path
    
    # Resolve the full path relative to repo root
    # The agent only provides relative paths, so we construct the absolute path here
    # Use resolve() to normalize the path (handles . and .. components)
    try:
        full_path = (repo_root / path).resolve()
    except (OSError, RuntimeError) as e:
        return False, f"Security error: Invalid path - {e}", Path(path)
    
    # Check 1: CRITICAL - Must be within repository root after resolution
    # This prevents escaping via ../, symlinks, or other path manipulation
    # relative_to() will raise ValueError if full_path is not a subpath of repo_root
    try:
        rel_path = full_path.relative_to(repo_root)
    except ValueError:
        return False, f"Security error: Resolved path escapes repository root. Original: '{path}' resolves to '{full_path}' which is outside '{repo_root}'", full_path
    
    path_parts = rel_path.parts
    
    # Check 2: Protected folders - cannot write to agents, .cursor, .vscode
    protected_folders = {"agents", ".cursor", ".vscode"}
    if any(part in protected_folders for part in path_parts):
        return False, f"Security error: Cannot write to protected folder (agents, .cursor, or .vscode)", full_path
    
    # Check 3: Cannot write to .git directory
    if ".git" in path_parts:
        return False, f"Security error: Cannot write to .git directory", full_path
    
    # Check 4: Allow specific config files even if they're hidden
    allowed_hidden_files = {".gitignore", ".gitattributes", ".editorconfig"}
    is_allowed_config = len(path_parts) == 1 and path_parts[0] in allowed_hidden_files
    
    # Check 5: Prevent writing to other hidden files/directories
    if not is_allowed_config and any(p.startswith(".") for p in path_parts):
        return False, f"Security error: Cannot write to hidden files/directories", full_path
    
    # Check 6: Cannot write to gitignored files (but allow the config files we explicitly allow)
    if not is_allowed_config:
        gitignore_patterns = load_gitignore_patterns(repo_root)
        if should_ignore(full_path, repo_root, gitignore_patterns):
            return False, f"Security error: Cannot write to gitignored file/directory", full_path
    
    return True, "", full_path


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist.
    
    Path is relative to the current working directory.
    
    Args:
        path: Path relative to current directory
        content: Complete file contents to write
    
    Returns:
        Success or error message
    """
    # Security validation
    is_valid, error_msg, full_path = _validate_path_security(path)
    if not is_valid:
        logger.warning("write_file security validation failed: %s", error_msg)
        return error_msg

    try:
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        full_path.write_text(content, encoding="utf-8")

        return f"Successfully wrote {len(content)} bytes to {path}"

    except Exception as e:
        logger.error("write_file failed for %s: %s", path, e)
        return f"Error writing file: {e}"


@tool
def apply_edit(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Apply a targeted edit to a file by replacing old_text with new_text.
    
    Path is relative to the current working directory.
    
    Args:
        path: Path relative to current directory
        old_text: The exact text to find and replace
        new_text: The text to replace it with
    
    Returns:
        Success or error message
    """
    # Security validation
    is_valid, error_msg, full_path = _validate_path_security(path)
    if not is_valid:
        logger.warning("apply_edit security validation failed: %s", error_msg)
        return error_msg

    if not full_path.exists():
        return f"File does not exist: {path}"
    
    if not full_path.is_file():
        return f"Not a file: {path}"
    
    try:
        content = full_path.read_text(encoding="utf-8")
        
        # Check if old_text exists
        if old_text not in content:
            # Try to find similar text for better error message
            lines_with_partial = []
            old_lines = old_text.strip().split("\n")
            if old_lines:
                first_line = old_lines[0].strip()
                for i, line in enumerate(content.split("\n"), 1):
                    if first_line in line:
                        lines_with_partial.append(i)
            
            if lines_with_partial:
                return (
                    f"old_text not found exactly. "
                    f"Similar content may be on lines: {lines_with_partial[:5]}"
                )
            else:
                return "old_text not found in file"
        
        # Count occurrences
        occurrences = content.count(old_text)
        if occurrences > 1:
            return (
                f"old_text found {occurrences} times. "
                f"Please provide more context to make the match unique."
            )
        
        # Apply the edit
        new_content = content.replace(old_text, new_text, 1)
        full_path.write_text(new_content, encoding="utf-8")

        return f"Successfully applied edit to {path}"

    except UnicodeDecodeError:
        logger.warning("apply_edit: file not text: %s", path)
        return f"Cannot edit file (not text): {path}"
    except Exception as e:
        logger.error("apply_edit failed for %s: %s", path, e)
        return f"Error editing file: {e}"


@tool
def create_file(path: str, content: str) -> str:
    """Create a new file. Fails if file already exists.
    
    Path is relative to the current working directory.
    
    Args:
        path: Path relative to current directory
        content: File contents
    
    Returns:
        Success or error message
    """
    # Security validation
    is_valid, error_msg, full_path = _validate_path_security(path)
    if not is_valid:
        logger.warning("create_file security validation failed: %s", error_msg)
        return error_msg

    if full_path.exists():
        return f"File already exists: {path}"

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

        return f"Successfully created {path}"

    except Exception as e:
        logger.error("create_file failed for %s: %s", path, e)
        return f"Error creating file: {e}"
