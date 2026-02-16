"""RAG search tool for semantic codebase search.

This module provides both:
1. A library function `perform_rag_search()` that can be called directly for prefetching
2. A LangChain tool `rag_search` that wraps the library function for agent use
"""

from pathlib import Path
from langchain_core.tools import tool

from ..logging_config import get_logger
from ..workspace import get_workspace_root
from ..rag.retriever import retrieve, format_contexts_for_agent

logger = get_logger(__name__)


def perform_rag_search(
    query: str,
    n_results: int = 10,
    repo_root: Path | str | None = None,
    max_tokens: int = 4000,
) -> str:
    """Perform RAG search and return formatted results.
    
    This is the library function that can be called directly (not as a tool).
    Use this for prefetching context before agent invocation.
    
    Args:
        query: Natural language description of what to find
        n_results: Maximum number of results to return
        repo_root: Repository root path (defaults to workspace root from workflow init)
        max_tokens: Maximum tokens for formatted output
    
    Returns:
        Formatted string with file paths, symbols, and line numbers of relevant code
    """
    try:
        if repo_root is None:
            repo_root = get_workspace_root()

        contexts = retrieve(
            query=query,
            n_results=n_results,
            repo_root=repo_root,
        )
        
        if not contexts:
            return f"No relevant code found for query: {query}"
        
        return format_contexts_for_agent(contexts, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("RAG search failed: %s", e)
        return f"RAG search error: {e}"


@tool
def rag_search(query: str, n_results: int = 10) -> str:
    """Search the codebase using semantic search (RAG) to find relevant files, classes, and functions.
    
    Use this to understand existing patterns and architecture in the codebase.
    This helps you formulate appropriate tasks that fit the existing structure.
    
    Args:
        query: Natural language description of what to find (e.g., "data models", "UI patterns", "state management")
        n_results: Maximum number of results to return
    
    Returns:
        Formatted string with file paths, symbols, and line numbers of relevant code
    """
    return perform_rag_search(query=query, n_results=n_results)
