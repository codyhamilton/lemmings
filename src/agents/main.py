#!/usr/bin/env python3
"""CLI entry point for the multi-agent orchestration system."""

import argparse
import os
import sys
from pathlib import Path

from .task_states import create_initial_state
from .graph import graph
from .state import StatusHistory
from .stream.handler import StreamHandler
from .stream.messages import AIMessageStreamHandler
from .stream.status import StatusStreamHandler
from .ui.console import ConsoleUI


def run_workflow(
    user_request: str,
    repo_root: str,
    verbose: bool = False,
    max_retries: int = 3,
    dashboard_mode: bool = False,
) -> dict:
    """Run the multi-agent workflow.

    Args:
        user_request: The user's development request
        repo_root: Path to the repository root
        verbose: Whether to print extra debug output
        max_retries: Maximum retries for coder after review failure (0 = no retries)
        dashboard_mode: Ignored (dashboard removed); kept for CLI compatibility

    Returns:
        Result dict with status key
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)

        initial_state = create_initial_state(
            user_request,
            repo_root,
            verbose=verbose,
            max_iterations=10,
            max_task_retries=max_retries,
            dashboard_mode=False,
        )

        status_history = StatusHistory()
        message_handler = AIMessageStreamHandler()
        status_handler = StatusStreamHandler(status_history)
        stream_handler = StreamHandler(
            message_handler=message_handler,
            status_handler=status_handler,
        )
        console = ConsoleUI(message_handler, status_handler, show_thinking=verbose)
        console.print_workflow_start(user_request, repo_root)

        final_state = initial_state.copy()

        try:
            for chunk in graph.stream(
                initial_state,
                subgraphs=True,
                stream_mode=["messages", "updates"],
            ):
                stream_handler.handle(chunk)
        except KeyboardInterrupt:
            raise
        finally:
            stream_handler.finalize()

        console.render_final_summary(final_state)
        return {**final_state, "status": "completed"}

    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user (Ctrl-C)")
        return {"status": "interrupted", "error": "User cancelled"}
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        os.chdir(original_cwd)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-agent orchestration system for development tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agents "Add a new resource type 'titanium' to the economy"
  python -m agents "Fix the bug in colony view where population shows negative"
  python -m agents --repo /path/to/repo "Refactor the ship class to use composition"
        """,
    )

    parser.add_argument("request", help="The development task to accomplish")
    parser.add_argument(
        "--repo", "-r",
        default=None,
        help="Repository root path (defaults to current directory's git root)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with debug info",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for coder after review failure (0 = no retries, default: 3)",
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=3,
        help="Maximum refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Reserved for future use (no-op)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Update RAG index before running workflow",
    )

    args = parser.parse_args()

    if args.repo:
        repo_root = Path(args.repo).resolve()
    else:
        cwd = Path.cwd()
        repo_root = cwd
        for parent in [cwd] + list(cwd.parents):
            if (parent / ".git").exists():
                repo_root = parent
                break

    if not repo_root.exists():
        print(f"Error: Repository path does not exist: {repo_root}")
        sys.exit(1)

    if args.index:
        try:
            from .rag.indexer import get_index_stats, update_index
            stats = get_index_stats(repo_root=repo_root)
            persist_dir = Path(repo_root) / ".rag_index"
            meta_file = persist_dir / "index_meta.json"
            if meta_file.exists() or stats["total_chunks"] > 0:
                print("🔄 Checking RAG index for updates...")
                update_stats = update_index(repo_root)
                if update_stats["files_indexed"] > 0:
                    print(f"   Updated {update_stats['files_indexed']} file(s) in {update_stats['time_taken']:.1f}s")
                else:
                    print("   Index is up to date")
            else:
                print("\n⚠️  RAG index not found. Building initial index...")
                update_stats = update_index(repo_root)
                print(f"   Built index with {update_stats['files_indexed']} file(s) in {update_stats['time_taken']:.1f}s\n")
        except Exception as e:
            if args.verbose:
                print(f"\n⚠️  Could not update RAG index: {e}")
                import traceback
                traceback.print_exc()
            print("   Continuing with shell-based search...\n")

    result = run_workflow(
        user_request=args.request,
        repo_root=str(repo_root),
        verbose=args.verbose,
        max_retries=args.max_retries,
        dashboard_mode=args.dashboard,
    )

    status = result.get("status", "failed")
    if status == "completed":
        sys.exit(0)
    elif status in ("completed_with_issues", "interrupted"):
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
