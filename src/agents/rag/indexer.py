"""Index manager for building and updating the vector store."""

import json
import time
from pathlib import Path
from typing import Optional

import chromadb

from ..logging_config import get_logger
from .chunker import chunk_file

logger = get_logger(__name__)
from .vectorstore import get_vectorstore, add_chunks_to_store, delete_chunks_by_path
from ..tools.gitignore import load_ignore_patterns, should_ignore


# Default persist directory - local to repo, git-ignored
def get_default_persist_dir(repo_root: Path) -> Path:
    """Get default persist directory for a repo (local .rag_index/)."""
    return repo_root / ".rag_index"


# Fallback if repo_root not available
DEFAULT_PERSIST_DIR = Path.cwd() / ".rag_index"

# Index metadata file
INDEX_META_FILE = "index_meta.json"


def _get_index_meta_path(persist_dir: Path) -> Path:
    """Get path to index metadata file."""
    return persist_dir / INDEX_META_FILE


def _get_indexed_paths_from_chromadb(collection: chromadb.Collection) -> set[str]:
    """Get all file paths that are currently indexed in ChromaDB.
    
    Returns:
        Set of normalized file paths (str)
    """
    indexed_paths = set()
    try:
        results = collection.get(include=['metadatas'])
        if results and results.get('metadatas'):
            for metadata in results['metadatas']:
                if metadata and 'path' in metadata:
                    path = metadata['path']
                    # Normalize path separators
                    normalized = str(path).replace('\\', '/')
                    indexed_paths.add(normalized)
    except Exception as e:
        logger.warning("Could not get indexed paths from ChromaDB: %s", e)
    
    return indexed_paths


def _load_index_meta_from_chromadb(collection: chromadb.Collection) -> dict:
    """Load file modification times from ChromaDB metadata.
    
    This is the source of truth - queries ChromaDB to get all indexed files
    and their modification times from chunk metadata.
    
    Returns:
        Dict mapping file paths (str) to modification times (float)
    """
    file_meta = {}
    try:
        # Get all chunks - ChromaDB's get() returns all items when no filters are provided
        # We only need metadata, not the full documents/embeddings
        results = collection.get(
            include=['metadatas']  # Only get metadata, not documents/embeddings (faster)
        )
        
        if results and results.get('metadatas'):
            for metadata in results['metadatas']:
                if metadata and 'path' in metadata:
                    path = metadata['path']
                    # Get mtime from metadata if available, otherwise skip
                    if 'file_mtime' in metadata:
                        try:
                            mtime = float(metadata['file_mtime'])
                            # Keep the latest mtime if we see the same file multiple times
                            if path not in file_meta or mtime > file_meta[path]:
                                file_meta[path] = mtime
                        except (ValueError, TypeError):
                            continue
    except Exception as e:
        logger.warning("Could not load metadata from ChromaDB: %s", e, exc_info=True)
    
    return file_meta


def _load_index_meta(persist_dir: Path, collection: chromadb.Collection = None) -> dict:
    """Load index metadata (file modification times).
    
    First tries to load from JSON cache file for speed.
    If that fails or is empty, falls back to loading from ChromaDB.
    
    Returns:
        Dict mapping file paths (str) to modification times (float).
    """
    # Try loading from JSON cache first (fast)
    meta_path = _get_index_meta_path(persist_dir)
    if meta_path.exists():
        try:
            raw_meta = json.loads(meta_path.read_text())
            # Validate and filter metadata: only keep entries with string keys and numeric values
            valid_meta = {}
            for key, value in raw_meta.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    # Filter out obvious test data (keys that don't look like file paths)
                    if (key and 
                        not key.startswith('test') and 
                        not key.startswith('timestamp') and
                        ('.' in key or '/' in key or len(key) > 3)):
                        valid_meta[key] = float(value)
            
            # If we got valid metadata, return it
            if valid_meta:
                return valid_meta
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted, fall through to ChromaDB
            pass
    
    # Fall back to loading from ChromaDB (slower but more reliable)
    if collection is not None:
        logger.info("Loading metadata from ChromaDB (cache file missing or invalid)...")
        return _load_index_meta_from_chromadb(collection)
    
    return {}


def _save_index_meta(persist_dir: Path, meta: dict) -> None:
    """Save index metadata."""
    # Ensure directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)
    meta_path = _get_index_meta_path(persist_dir)
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save index metadata: %s", e)
        raise


def _cleanup_deleted_files(
    collection: chromadb.Collection,
    repo_root: Path,
    gitignore_patterns: list,
    file_meta: dict | None = None,
    new_meta: dict | None = None,
) -> int:
    """Remove deleted files from the index.
    
    Args:
        collection: ChromaDB collection
        repo_root: Repository root directory
        gitignore_patterns: Pre-loaded gitignore patterns
        file_meta: Existing metadata dict (optional, will load from ChromaDB if not provided)
        new_meta: Metadata dict being built (will be updated to remove deleted files)
    
    Returns:
        Number of deleted files removed
    """
    # Get all currently indexed file paths
    if file_meta:
        indexed_paths = set(file_meta.keys())
    else:
        # Load from ChromaDB if no metadata available
        indexed_paths = _get_indexed_paths_from_chromadb(collection)
    
    if not indexed_paths:
        return 0
    
    # Get all current file paths in the repo
    current_paths = set()
    for file_path in repo_root.rglob('*'):
        if not file_path.is_file():
            continue
        if not _should_index_file(file_path, repo_root, gitignore_patterns):
            continue
        try:
            rel_path = str(file_path.relative_to(repo_root))
            # Normalize path separators for consistency
            rel_path_normalized = rel_path.replace('\\', '/')
            current_paths.add(rel_path_normalized)
        except ValueError:
            pass
    
    # Find deleted paths
    deleted_paths = indexed_paths - current_paths
    
    if not deleted_paths:
        return 0
    
    # Remove deleted files from metadata and ChromaDB
    for deleted_path in deleted_paths:
        # Try both normalized and original path format
        deleted_path_normalized = deleted_path.replace('\\', '/')
        for path_variant in [deleted_path, deleted_path_normalized]:
            # Remove from new_meta if provided
            if new_meta is not None and path_variant in new_meta:
                del new_meta[path_variant]
            # Delete chunks from ChromaDB
            delete_chunks_by_path(collection, path_variant)
    
    return len(deleted_paths)


def _should_index_file(file_path: Path, repo_root: Path, gitignore_patterns: list) -> bool:
    """Check if file should be indexed.
    
    Files are ignored based on .gitignore and .rag-ignore patterns.
    The .rag_index directory is automatically ignored (RAG's own output).
    """
    # Always ignore .rag_index directory (RAG's own output - don't index the index!)
    try:
        rel_path = file_path.relative_to(repo_root)
        rel_path_str = str(rel_path)
        if rel_path_str.startswith('.rag_index'):
            return False
    except ValueError:
        pass
    
    # Check ignore patterns from .gitignore and .rag-ignore
    if should_ignore(file_path, repo_root, gitignore_patterns):
        return False
    
    # Check file type
    suffix = file_path.suffix.lower()
    
    # Index code and config files (full content)
    if suffix in {'.gd', '.json', '.tscn', '.md', '.markdown'}:
        return True
    
    # Index asset files (name only, not content)
    # Common image formats
    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}:
        return True
    # Godot import files (store name for reference)
    if suffix == '.import':
        return True
    
    # Skip very large files
    try:
        if file_path.stat().st_size > 1_000_000:  # 1MB
            return False
    except OSError:
        return False
    
    return False


def build_index(
    repo_root: Path | str,
    persist_dir: Optional[Path | str] = None,
    force_rebuild: bool = False,
) -> dict:
    """Build or rebuild the complete index.
    
    Args:
        repo_root: Repository root directory
        persist_dir: Directory to persist the index (default: <repo>/.rag_index)
        force_rebuild: Force complete rebuild even if index exists
    
    Returns:
        Stats dictionary with files_indexed, chunks_created, time_taken
    """
    repo_root = Path(repo_root).resolve()
    if persist_dir is None:
        persist_dir = get_default_persist_dir(repo_root)
    else:
        persist_dir = Path(persist_dir)
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building index for %s", repo_root)
    logger.info("Persist directory: %s", persist_dir)
    
    start_time = time.time()
    
    # Load ignore patterns from .gitignore and .rag-ignore
    gitignore_patterns = load_ignore_patterns(repo_root)
    
    # Get vector store
    collection = get_vectorstore(persist_dir)
    
    # Check if collection has existing data
    try:
        existing_chunk_count = collection.count()
    except Exception:
        existing_chunk_count = 0
    
    # Clear existing index if force rebuild
    if force_rebuild:
        logger.info("Force rebuild: clearing existing index...")
        try:
            collection.delete(where={})  # Delete all
            existing_chunk_count = 0
        except Exception:
            pass
    
    # Load existing metadata (try JSON cache first, fall back to ChromaDB)
    file_meta = _load_index_meta(persist_dir, collection) if not force_rebuild else {}
    
    # If we have data in ChromaDB but no metadata, try to rebuild from ChromaDB
    if existing_chunk_count > 0 and not file_meta and not force_rebuild:
        logger.info("Found %s chunks in index but no metadata cache. Rebuilding metadata from ChromaDB...", existing_chunk_count)
        file_meta = _load_index_meta_from_chromadb(collection)
        if file_meta:
            logger.info("Recovered metadata for %s files from ChromaDB", len(file_meta))
    
    # Start with existing metadata - we'll update/add entries as we process files
    # Only copy valid metadata entries (file paths with numeric mtimes)
    new_meta = {}
    if file_meta:
        for key, value in file_meta.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                new_meta[key] = float(value)
    
    # Find all indexable files
    files_to_index = []
    files_skipped = 0
    for file_path in repo_root.rglob('*'):
        if not file_path.is_file():
            continue
        
        if not _should_index_file(file_path, repo_root, gitignore_patterns):
            continue
        
        # For incremental updates, check if file needs reindexing before adding to list
        if file_meta and len(file_meta) > 0:  # Only do this check if we have existing metadata (incremental mode)
            try:
                rel_path = str(file_path.relative_to(repo_root))
                # Normalize path separators (Windows vs Unix)
                rel_path_normalized = rel_path.replace('\\', '/')
                
                mtime = file_path.stat().st_mtime
                # Try both normalized and original path
                stored_mtime = file_meta.get(rel_path_normalized) or file_meta.get(rel_path)
                
                # Check if file hasn't changed (allow small floating point differences)
                # Use stored_mtime comparison: if stored >= current (within tolerance), file hasn't changed
                if stored_mtime is not None:
                    time_diff = abs(stored_mtime - mtime)
                    # File hasn't changed if mtime is same or older (within 0.1s tolerance)
                    # Also handle case where file was modified but mtime went backwards (rare but possible)
                    if time_diff < 0.1 or stored_mtime >= mtime:
                        # File hasn't changed, preserve its metadata in new_meta
                        # Use normalized path for consistency
                        new_meta[rel_path_normalized] = stored_mtime
                        files_skipped += 1
                        continue
            except (ValueError, OSError) as e:
                # If we can't check, include it to be safe
                pass
        
        files_to_index.append(file_path)
    
    if files_skipped > 0:
        logger.info("Found %s files to index (%s unchanged, skipping)", len(files_to_index), files_skipped)
    else:
        logger.info("Found %s files to index", len(files_to_index))
    
    # If no files to index, skip the expensive operations
    if not files_to_index:
        # Still check for deleted files
        deleted_count = _cleanup_deleted_files(
            collection=collection,
            repo_root=repo_root,
            gitignore_patterns=gitignore_patterns,
            file_meta=file_meta if not force_rebuild else None,
            new_meta=new_meta,
        )
        
        # Still need to save metadata (for unchanged files)
        try:
            clean_meta = {}
            for key, value in new_meta.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    clean_meta[key] = float(value)
            _save_index_meta(persist_dir, clean_meta)
            if deleted_count > 0:
                logger.info("Index is up to date (%s deleted file(s) removed)", deleted_count)
            else:
                logger.info("Index is up to date (no changes)")
        except Exception as e:
            logger.warning("Failed to save metadata: %s", e)
        
        elapsed = time.time() - start_time
        return {
            "files_indexed": 0,
            "chunks_created": 0,
            "time_taken": elapsed,
        }
    
    # Index files
    files_indexed = 0
    chunks_created = 0
    
    for file_path in files_to_index:
        try:
            rel_path = str(file_path.relative_to(repo_root))
            # Normalize path separators
            rel_path_normalized = rel_path.replace('\\', '/')
        except ValueError:
            continue
        
        # Get modification time
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            continue
        
        # Check if file needs reindexing (double-check, though we should have caught this earlier)
        stored_mtime = file_meta.get(rel_path_normalized) or file_meta.get(rel_path) if file_meta else None
        if stored_mtime is not None:
            time_diff = abs(stored_mtime - mtime)
            if time_diff < 0.1 or stored_mtime >= mtime:
                # File hasn't changed, preserve its metadata in new_meta
                new_meta[rel_path_normalized] = stored_mtime
                continue
        
        # Chunk the file
        chunks = chunk_file(file_path, repo_root)
        
        if not chunks:
            continue
        
        # Delete old chunks for this file
        delete_chunks_by_path(collection, rel_path)
        
        # Prepare chunks for storage
        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                "id": chunk.get_id(),
                "content": chunk.content,
                "metadata": {
                    "path": chunk.path,
                    "chunk_type": chunk.chunk_type,
                    "symbols": ",".join(chunk.symbols),  # Store as comma-separated string
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "file_mtime": str(mtime),  # Store file modification time in ChromaDB metadata
                }
            })
        
        # Add to vector store (this will load embedding model if needed)
        add_chunks_to_store(collection, chunk_dicts)
        
        files_indexed += 1
        chunks_created += len(chunks)
        # Store with normalized path
        new_meta[rel_path_normalized] = mtime
        
        # Show progress for every file when indexing small numbers, or every 10 for large batches
        if len(files_to_index) <= 10 or files_indexed % 10 == 0:
            logger.info("Indexed %s/%s files (%s chunks)...", files_indexed, len(files_to_index), chunks_created)
    
    # Always check for and remove deleted files from the index
    # This compares indexed paths (from ChromaDB or metadata) with current repo files
    deleted_count = _cleanup_deleted_files(
        collection=collection,
        repo_root=repo_root,
        gitignore_patterns=gitignore_patterns,
        file_meta=file_meta if not force_rebuild else None,
        new_meta=new_meta,
    )
    if deleted_count > 0:
        logger.info("Removed %s deleted file(s) from index", deleted_count)
    
    # Save metadata (even if empty - this marks that indexing was attempted)
    # Only save valid entries (file paths as strings, mtimes as numbers)
    clean_meta = {}
    for key, value in new_meta.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            clean_meta[key] = float(value)
    
    try:
        _save_index_meta(persist_dir, clean_meta)
        meta_path = _get_index_meta_path(persist_dir)
        if not meta_path.exists():
            logger.error("Metadata file was not created at %s (persist_dir exists=%s, is_dir=%s)", meta_path, persist_dir.exists(), persist_dir.is_dir())
        else:
            # Verify it's readable
            try:
                saved_meta = _load_index_meta(persist_dir)
                logger.info("Metadata saved: %s file entries", len(saved_meta))
            except Exception as e:
                logger.warning("Metadata file exists but couldn't be read: %s", e)
    except Exception as e:
        logger.error("Failed to save metadata: %s", e, exc_info=True)
    
    elapsed = time.time() - start_time
    
    stats = {
        "files_indexed": files_indexed,
        "chunks_created": chunks_created,
        "time_taken": elapsed,
    }
    
    logger.info("Index built: %s files, %s chunks in %.1fs", files_indexed, chunks_created, elapsed)
    
    return stats


def update_index(
    repo_root: Path | str,
    persist_dir: Optional[Path | str] = None,
) -> dict:
    """Update index incrementally based on file modifications.
    
    Args:
        repo_root: Repository root directory
        persist_dir: Directory to persist the index
    
    Returns:
        Stats dictionary with files_indexed, chunks_created, time_taken
    """
    repo_root = Path(repo_root).resolve()
    if persist_dir is None:
        persist_dir = get_default_persist_dir(repo_root)
    else:
        persist_dir = Path(persist_dir)
    
    # Check if this is a first-time build (no metadata file)
    meta_path = _get_index_meta_path(persist_dir)
    if not meta_path.exists():
        # First-time build - use build_index which will be more verbose
        logger.info("Building RAG index (first time)...")
        return build_index(repo_root, persist_dir, force_rebuild=False)
    
    # Incremental update - build_index already handles this efficiently
    # It checks mtimes and only reindexes changed files
    return build_index(repo_root, persist_dir, force_rebuild=False)


def get_index_stats(persist_dir: Optional[Path | str] = None, repo_root: Optional[Path | str] = None) -> dict:
    """Get statistics about the current index.
    
    Args:
        persist_dir: Directory where index is persisted
        repo_root: Repository root (to find default persist dir)
    
    Returns:
        Stats dictionary with total_files, total_chunks, last_update
    """
    if persist_dir is None:
        if repo_root:
            persist_dir = get_default_persist_dir(Path(repo_root).resolve())
        else:
            persist_dir = DEFAULT_PERSIST_DIR
    else:
        persist_dir = Path(persist_dir)
    
    if not persist_dir.exists():
        return {
            "total_files": 0,
            "total_chunks": 0,
            "last_update": None,
        }
    
    # Load metadata
    file_meta = _load_index_meta(persist_dir)
    
    # Get collection count
    try:
        collection = get_vectorstore(persist_dir)
        total_chunks = collection.count()
    except Exception:
        total_chunks = 0
    
    # Get last update time
    last_update = None
    if file_meta:
        # Filter to only numeric values (mtime should be float)
        numeric_values = [v for v in file_meta.values() if isinstance(v, (int, float))]
        if numeric_values:
            last_update = max(numeric_values)
    
    return {
        "total_files": len(file_meta),
        "total_chunks": total_chunks,
        "last_update": last_update,
    }
