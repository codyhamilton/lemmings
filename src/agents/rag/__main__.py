"""CLI tool for building and managing the RAG index.

Usage:
    python -m agents.rag build [--force] [-w]
    python -m agents.rag update
    python -m agents.rag stats
    python -m agents.rag watch
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

from .indexer import build_index, update_index, get_index_stats


def main():
    """Main entry point for RAG CLI."""
    parser = argparse.ArgumentParser(
        description="Build and manage the RAG index for agent context retrieval"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build the index")
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    build_parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Repository root directory (default: current directory)"
    )
    build_parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Watch for file changes and update index automatically (10s debounce)"
    )
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update the index incrementally")
    update_parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Repository root directory (default: current directory)"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Repository root directory (default: current directory)"
    )
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for file changes and update index")
    watch_parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Repository root directory (default: current directory)"
    )
    watch_parser.add_argument(
        "--debounce",
        type=int,
        default=10,
        help="Debounce time in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "build":
            repo_root = Path(args.repo).resolve()
            print(f"Building index for: {repo_root}")
            stats = build_index(repo_root, force_rebuild=args.force)
            print(f"\n✓ Index built successfully!")
            print(f"  Files indexed: {stats['files_indexed']}")
            print(f"  Chunks created: {stats['chunks_created']}")
            print(f"  Time taken: {stats['time_taken']:.2f}s")
            
            # Start watching if requested
            if args.watch:
                print(f"\n👀 Watching for file changes (10s debounce)...")
                print("   Press Ctrl+C to stop")
                return watch_files(repo_root, debounce_seconds=10)
            
            return 0
        
        elif args.command == "update":
            repo_root = Path(args.repo).resolve()
            print(f"Updating index for: {repo_root}")
            stats = update_index(repo_root)
            print(f"\n✓ Index updated!")
            print(f"  Files updated: {stats['files_indexed']}")
            print(f"  Chunks updated: {stats['chunks_created']}")
            print(f"  Time taken: {stats['time_taken']:.2f}s")
            return 0
        
        elif args.command == "stats":
            repo_root = Path(args.repo).resolve()
            stats = get_index_stats(repo_root=repo_root)
            print("\n📊 Index Statistics")
            print("=" * 50)
            print(f"  Repository: {repo_root}")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            if stats['last_update']:
                from datetime import datetime
                last_update = datetime.fromtimestamp(stats['last_update'])
                print(f"  Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"  Last update: Never")
            print("=" * 50)
            return 0
        
        elif args.command == "watch":
            repo_root = Path(args.repo).resolve()
            print(f"👀 Watching for file changes in: {repo_root}")
            print(f"   Debounce: {args.debounce}s")
            print("   Press Ctrl+C to stop")
            return watch_files(repo_root, debounce_seconds=args.debounce)
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def watch_files(repo_root: Path, debounce_seconds: int = 10) -> int:
    """Watch for file changes and update index with debouncing.
    
    Args:
        repo_root: Repository root directory
        debounce_seconds: Time to wait before updating after last change
    
    Returns:
        Exit code
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("\n❌ Error: watchdog package not installed")
        print("   Install with: pip install watchdog")
        return 1
    
    from .indexer import get_default_persist_dir
    from ..tools.gitignore import load_ignore_patterns, should_ignore
    
    # Load ignore patterns
    ignore_patterns = load_ignore_patterns(repo_root)
    
    # Track file changes with timestamps
    pending_changes = defaultdict(float)
    last_update = time.time()
    updating = False
    
    class IndexUpdateHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal last_update, updating
            
            # Skip directories and non-indexable files
            if event.is_directory:
                return
            
            path = Path(event.src_path)
            suffix = path.suffix.lower()
            
            # Only watch indexable file types
            if suffix not in {'.gd', '.json', '.tscn', '.md', '.markdown'}:
                return
            
            # Always ignore .rag_index directory (RAG's own output)
            try:
                rel_path = path.relative_to(repo_root)
                if str(rel_path).startswith('.rag_index'):
                    return
            except ValueError:
                pass
            
            # Check ignore patterns from .gitignore and .rag-ignore
            if should_ignore(path, repo_root, ignore_patterns):
                return
            
            # Record change
            pending_changes[str(path)] = time.time()
            last_update = time.time()
    
    # Set up observer
    observer = Observer()
    handler = IndexUpdateHandler()
    observer.schedule(handler, str(repo_root), recursive=True)
    observer.start()
    
    print(f"✓ Watching started")
    
    try:
        while True:
            time.sleep(1)
            
            # Check if we should update (debounce elapsed and have pending changes)
            if pending_changes and not updating:
                time_since_last_change = time.time() - last_update
                
                if time_since_last_change >= debounce_seconds:
                    updating = True
                    file_count = len(pending_changes)
                    
                    print(f"\n🔄 Updating index ({file_count} file(s) changed)...")
                    try:
                        stats = update_index(repo_root)
                        print(f"✓ Index updated: {stats['files_indexed']} files, {stats['chunks_created']} chunks ({stats['time_taken']:.2f}s)")
                    except Exception as e:
                        print(f"❌ Update failed: {e}")
                    
                    pending_changes.clear()
                    updating = False
    
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        return 0


if __name__ == "__main__":
    sys.exit(main())
