# RAG-Based Context Management

This module implements Retrieval-Augmented Generation (RAG) for fast, semantic code search and context retrieval.

## Architecture

```
User Request
     ↓
[Preload Context Node] → Loads repo overview + relevant docs
     ↓
[Planner] → Plans with RAG context
     ↓
[Researcher] → Uses RAG semantic search (replaces 3-pass shell tool loop)
     ↓
[Rest of pipeline...]
```

## Components

### 1. Vector Store (`vectorstore.py`)
- Uses **ChromaDB** for local, persistent vector storage
- Embeddings via **sentence-transformers/all-MiniLM-L6-v2**
- Fast CPU inference (~90MB model, 384-dim embeddings)
- Cosine similarity for semantic search

### 2. Smart Chunker (`chunker.py`)
Preserves semantic units instead of arbitrary splits:

- **GDScript (.gd)**: Chunks by class/function definitions
- **JSON configs**: Chunks by top-level keys (e.g., each ship definition)
- **TSCN scenes**: Extracts node hierarchy and script references
- **Markdown docs**: Chunks by headers

### 3. Indexer (`indexer.py`)
- Full index build on first run
- Incremental updates based on file modification times
- Metadata tracking for efficient updates
- Respects .gitignore patterns

### 4. Retriever (`retriever.py`)
Core query interface with filtering:
- Semantic similarity search
- File pattern filtering (glob patterns)
- Chunk type filtering (class, function, config, etc.)
- Symbol filtering (by name)

### 5. Context Providers (`context_providers.py`)
Pre-configured retrievers for agent personas:
- `get_repo_overview()` - High-level structure for Planner
- `get_relevant_docs()` - Documentation for Planner
- `get_implementation_context()` - Main function for Researcher
- `get_coding_patterns()` - Examples for Coder
- `get_review_standards()` - Standards for Reviewer

## Usage

### Building the Index

```bash
# One-time build (or after major changes)
rag build

# Build and watch for changes (10s debounce)
rag build -w

# Watch for changes only (if already built)
rag watch

# Incremental update (automatic when running agents)
rag update

# Check statistics
rag stats
```

**Note**: The agent system automatically updates the index before each run (only changed files), so manual updates are rarely needed.

### Querying from Code

```python
from agents.rag.retriever import retrieve

# Basic semantic search
contexts = retrieve(
    query="how to add a new resource type",
    n_results=10
)

# With filters
contexts = retrieve(
    query="economy resource management",
    file_pattern="*.gd",
    chunk_types=["class", "function"],
    symbol_filter="Resource"
)

# For requirements (used by Researcher)
from agents.rag.retriever import retrieve_for_requirement

contexts = retrieve_for_requirement(
    requirement="Add titanium resource",
    keywords=["resource", "economy", "materials"],
    symbols=["ResourceManager"]
)
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Index build (full) | 5-10s | ~500 files, ~2000 chunks |
| Index update | 1-5s | Only modified files |
| Single query | 50-200ms | CPU-based, very fast |
| Researcher (old) | 30-60s | 3-pass shell tool loop |
| Researcher (RAG) | 1-2s | **30-60x faster** |

## Storage

- **Location**: `<repo>/.rag_index/` (local to repository, git-ignored)
- **Size**: ~10-50MB depending on repo size
- **Embedding model**: `~/.cache/huggingface/` (~90MB, one-time download)

**Git Integration**: Add `.rag_index/` to your `.gitignore` (automatically ignored by common patterns)

## Integration Points

### 1. Graph Entry (`graph.py`)
- New `preload_context_node` runs before planner
- Pre-loads repo overview and relevant docs
- Refreshes on each refinement iteration

### 2. Researcher (`researcher.py`)
- RAG-based research as primary method
- Automatic fallback to shell tools if RAG fails
- Same output format (RequirementContext)

### 3. Planner (`planner.py`)
- Receives pre-loaded repo context
- Gets relevant documentation snippets
- Better understanding of project structure

### 4. State (`task_states.py`)
- New fields: `repo_context`, `relevant_docs`
- Pre-loaded at workflow start
- Available to all downstream agents

## Error Handling

The system gracefully handles RAG failures:
1. **Index not found**: Falls back to shell tools for Researcher
2. **Embedding model unavailable**: Downloads automatically on first run
3. **Query errors**: Catches and logs, returns empty results
4. **Corrupted index**: Rebuild with `rag build --force`

## Advantages over Shell Tools

1. **Speed**: 30-60x faster (1-2s vs 30-60s)
2. **Semantic**: Understands meaning, not just keywords
3. **Consistent**: Same performance regardless of query complexity
4. **No noise**: Clean results without tool call verbosity
5. **Proactive**: Context pre-loaded before agents need it

## Maintenance

```bash
# Rebuild after major refactoring
rag build --force

# Watch for changes (auto-update with 10s debounce)
rag build -w

# Manual update (usually not needed - auto-updates on agent run)
rag update

# Verify index health
rag stats
```

**Automatic Maintenance**: The index updates automatically when running agents, tracking only modified files for fast incremental updates.

## Future Enhancements

Possible improvements:
- [ ] Add code-specific embedding models (CodeBERT, GraphCodeBERT)
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add query result caching
- [ ] Support for more file types
- [ ] Real-time index updates (file watching)
- [ ] Query result re-ranking
