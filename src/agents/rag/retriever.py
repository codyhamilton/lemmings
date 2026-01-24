"""RAG retriever with filtering and query interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import fnmatch

from .vectorstore import get_vectorstore, query_store


@dataclass
class RetrievedContext:
    """A retrieved context chunk with relevance score."""
    
    path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str
    symbols: list[str]
    score: float  # Similarity score (lower is better for cosine distance)
    
    def __str__(self) -> str:
        """Format for display."""
        symbol_str = f" [{', '.join(self.symbols)}]" if self.symbols else ""
        return f"{self.path}:{self.start_line}-{self.end_line}{symbol_str} (score: {self.score:.3f})"


def retrieve(
    query: str,
    n_results: int = 10,
    file_pattern: str | None = None,
    chunk_types: list[str] | None = None,
    symbol_filter: str | None = None,
    persist_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> list[RetrievedContext]:
    """Retrieve relevant code chunks using semantic search.
    
    Args:
        query: Natural language query describing what to find
        n_results: Maximum number of results to return
        file_pattern: Glob pattern to filter files (e.g., "*.gd", "config/*.json")
        chunk_types: Filter by chunk types (e.g., ["class", "function"])
        symbol_filter: Filter by symbol name (partial match)
        persist_dir: Directory where index is persisted
        repo_root: Repository root (to find default persist dir)
    
    Returns:
        List of RetrievedContext objects, sorted by relevance
    """
    if persist_dir is None:
        if repo_root:
            # Lazy import to avoid circular dependency
            from .indexer import get_default_persist_dir
            persist_dir = get_default_persist_dir(Path(repo_root).resolve())
        else:
            persist_dir = Path.cwd() / ".rag_index"
    else:
        persist_dir = Path(persist_dir)
    
    # Get collection
    collection = get_vectorstore(persist_dir)
    
    # Build metadata filter
    where = {}
    if chunk_types:
        # ChromaDB uses $in operator for list matching
        where["chunk_type"] = {"$in": chunk_types}
    
    # Query vector store
    # Note: file_pattern and symbol_filter will be applied post-query
    # because they require more complex matching
    results = query_store(
        collection=collection,
        query_text=query,
        n_results=n_results * 2,  # Get extra to allow for post-filtering
        where=where if where else None,
    )
    
    # Process results
    contexts = []
    
    if not results["ids"] or not results["ids"][0]:
        return contexts
    
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        distance = results["distances"][0][i]
        
        path = metadata.get("path", "")
        
        # Apply file pattern filter
        if file_pattern and not fnmatch.fnmatch(path, file_pattern):
            continue
        
        # Extract symbols from metadata
        symbols_str = metadata.get("symbols", "")
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        
        # Apply symbol filter
        if symbol_filter:
            if not any(symbol_filter.lower() in s.lower() for s in symbols):
                continue
        
        contexts.append(RetrievedContext(
            path=path,
            start_line=metadata.get("start_line", 1),
            end_line=metadata.get("end_line", 1),
            content=document,
            chunk_type=metadata.get("chunk_type", "unknown"),
            symbols=symbols,
            score=distance,
        ))
        
        # Stop if we have enough results
        if len(contexts) >= n_results:
            break
    
    return contexts


def retrieve_for_requirement(
    requirement: str,
    keywords: list[str],
    symbols: list[str] | None = None,
    n_results: int = 15,
    persist_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> list[RetrievedContext]:
    """Retrieve context for a specific requirement.
    
    This is a convenience function that combines the requirement text
    with keywords for better retrieval.
    
    Args:
        requirement: The requirement description
        keywords: Keywords to search for
        symbols: Optional symbol names to prioritize
        n_results: Maximum number of results
        persist_dir: Directory where index is persisted
        repo_root: Repository root (to find default persist dir)
    
    Returns:
        List of RetrievedContext objects
    """
    # Build enriched query
    query_parts = [requirement]
    
    if keywords:
        query_parts.append("Keywords: " + ", ".join(keywords))
    
    query = "\n".join(query_parts)
    
    # Retrieve with symbol filter if provided
    symbol_filter = None
    if symbols and len(symbols) > 0:
        # Use first symbol as filter
        symbol_filter = symbols[0]
    
    return retrieve(
        query=query,
        n_results=n_results,
        symbol_filter=symbol_filter,
        persist_dir=persist_dir,
        repo_root=repo_root,
    )


def retrieve_similar_code(
    code_snippet: str,
    file_types: list[str] | None = None,
    n_results: int = 5,
    persist_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> list[RetrievedContext]:
    """Find similar code patterns.
    
    Useful for coder agent to find existing patterns to follow.
    
    Args:
        code_snippet: Code to find similar examples of
        file_types: File extensions to search (e.g., [".gd", ".json"])
        n_results: Maximum number of results
        persist_dir: Directory where index is persisted
        repo_root: Repository root (to find default persist dir)
    
    Returns:
        List of RetrievedContext objects
    """
    # Convert file types to pattern
    file_pattern = None
    if file_types:
        if len(file_types) == 1:
            file_pattern = f"*{file_types[0]}"
        else:
            # For multiple types, we'll filter post-query
            pass
    
    contexts = retrieve(
        query=code_snippet,
        n_results=n_results,
        file_pattern=file_pattern,
        persist_dir=persist_dir,
        repo_root=repo_root,
    )
    
    # Additional filtering for multiple file types
    if file_types and len(file_types) > 1:
        contexts = [
            c for c in contexts
            if any(c.path.endswith(ft) for ft in file_types)
        ]
    
    return contexts


def format_contexts_for_agent(
    contexts: list[RetrievedContext],
    max_tokens: int = 8000,
) -> str:
    """Format retrieved contexts for agent consumption.
    
    Args:
        contexts: List of retrieved contexts
        max_tokens: Approximate max tokens (using ~4 chars per token)
    
    Returns:
        Formatted string suitable for agent prompts
    """
    if not contexts:
        return "No relevant context found."
    
    lines = []
    lines.append(f"Found {len(contexts)} relevant code sections:\n")
    
    total_chars = 0
    max_chars = max_tokens * 4
    
    for i, ctx in enumerate(contexts, 1):
        # Header for this context
        symbol_str = f" [{', '.join(ctx.symbols)}]" if ctx.symbols else ""
        header = f"\n--- {i}. {ctx.path}:{ctx.start_line}-{ctx.end_line}{symbol_str} ({ctx.chunk_type}) ---"
        
        # Check if we have space
        if total_chars + len(header) + len(ctx.content) > max_chars:
            lines.append(f"\n... ({len(contexts) - i + 1} more contexts truncated)")
            break
        
        lines.append(header)
        lines.append(ctx.content)
        
        total_chars += len(header) + len(ctx.content)
    
    return "\n".join(lines)
