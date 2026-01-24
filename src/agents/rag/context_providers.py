"""Pre-configured context providers for different agent personas."""

from pathlib import Path
from typing import Optional

from .retriever import retrieve, format_contexts_for_agent
from .indexer import get_default_persist_dir


def get_repo_overview(
    repo_root: Path | str | None = None,
    persist_dir: Path | str | None = None,
) -> str:
    """Get high-level repository structure overview.
    
    Retrieves documentation and key configuration files to give
    the planner agent a sense of the project structure.
    
    Args:
        repo_root: Repository root directory
        persist_dir: Directory where index is persisted
    
    Returns:
        Formatted overview string
    """
    repo_root = Path(repo_root).resolve()
    if persist_dir is None:
        persist_dir = get_default_persist_dir(repo_root)
    
    # Query for overview documentation
    overview_contexts = retrieve(
        query="project structure architecture overview components",
        n_results=5,
        chunk_types=["doc", "config"],
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    if not overview_contexts:
        return "No overview documentation found."
    
    return format_contexts_for_agent(overview_contexts, max_tokens=2000)


def get_relevant_docs(
    user_request: str,
    repo_root: Path | str | None = None,
    persist_dir: Path | str | None = None,
) -> list[str]:
    """Get relevant documentation snippets for a user request.
    
    Args:
        user_request: The user's request
        repo_root: Repository root directory
        persist_dir: Directory where index is persisted
    
    Returns:
        List of documentation snippets
    """
    if repo_root:
        repo_root = Path(repo_root).resolve()
    if persist_dir is None and repo_root:
        persist_dir = get_default_persist_dir(repo_root)
    
    # Query for relevant docs
    doc_contexts = retrieve(
        query=user_request,
        n_results=3,
        chunk_types=["doc"],
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    return [ctx.content for ctx in doc_contexts]


def get_coding_patterns(
    description: str,
    file_types: list[str] | None = None,
    repo_root: Path | str | None = None,
    persist_dir: Path | str | None = None,
) -> str:
    """Get similar coding patterns for the coder agent.
    
    Args:
        description: Description of what to code
        file_types: File extensions to search (e.g., [".gd"])
        persist_dir: Directory where index is persisted
    
    Returns:
        Formatted coding patterns
    """
    if repo_root:
        repo_root = Path(repo_root).resolve()
    if persist_dir is None and repo_root:
        persist_dir = get_default_persist_dir(repo_root)
    
    if file_types is None:
        file_types = [".gd"]  # Default to GDScript
    
    # Build file pattern
    file_pattern = None
    if len(file_types) == 1:
        file_pattern = f"*{file_types[0]}"
    
    # Query for similar patterns
    pattern_contexts = retrieve(
        query=description,
        n_results=5,
        file_pattern=file_pattern,
        chunk_types=["class", "function"],
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    if not pattern_contexts:
        return "No similar patterns found."
    
    return format_contexts_for_agent(pattern_contexts, max_tokens=4000)


def get_implementation_context(
    requirement: str,
    keywords: list[str],
    symbols: list[str] | None = None,
    repo_root: Path | str | None = None,
    persist_dir: Path | str | None = None,
) -> str:
    """Get comprehensive implementation context for a requirement.
    
    This is the main function used by the researcher agent.
    
    Args:
        requirement: The requirement description
        keywords: Keywords to search for
        symbols: Optional symbol names
        persist_dir: Directory where index is persisted
    
    Returns:
        Formatted context string
    """
    from .retriever import retrieve_for_requirement
    
    if repo_root:
        repo_root = Path(repo_root).resolve()
    if persist_dir is None and repo_root:
        persist_dir = get_default_persist_dir(repo_root)
    
    contexts = retrieve_for_requirement(
        requirement=requirement,
        keywords=keywords,
        symbols=symbols,
        n_results=15,
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    if not contexts:
        return "No relevant context found. You may need to create new files or components."
    
    return format_contexts_for_agent(contexts, max_tokens=8000)


def get_review_standards(
    code_type: str = "gdscript",
    repo_root: Path | str | None = None,
    persist_dir: Path | str | None = None,
) -> str:
    """Get coding standards and best practices for review.
    
    Args:
        code_type: Type of code being reviewed
        persist_dir: Directory where index is persisted
    
    Returns:
        Formatted standards and examples
    """
    if repo_root:
        repo_root = Path(repo_root).resolve()
    if persist_dir is None and repo_root:
        persist_dir = get_default_persist_dir(repo_root)
    
    # Query for well-structured examples
    standard_contexts = retrieve(
        query=f"{code_type} best practices clean code examples",
        n_results=5,
        chunk_types=["class", "function"],
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    if not standard_contexts:
        return "No coding standard examples found."
    
    return format_contexts_for_agent(standard_contexts, max_tokens=3000)
