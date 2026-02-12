#!/usr/bin/env python3
"""CLI entry point for the multi-agent orchestration system."""

import argparse
import os
import signal
import sys
from pathlib import Path

from .config import config
from .logging_config import get_logger, setup_logging
from .llm import initialise_llms

initialise_llms()  # Must run before graph/agents import planning_llm

from .task_states import create_initial_state
from .graph import graph
from .state import StatusHistory
from .stream.handler import StreamHandler, is_update_chunk
from .stream.messages import (
    AIMessageStreamHandler,
    StreamEvent,
    StreamEventType,
    BlockType,
)
from .stream.status import StatusStreamHandler
from .ui.console import ConsoleUI

logger = get_logger(__name__)

# Module-level so the signal handler can raise and we can restore in finally
_previous_signal_handlers: dict[int, object] = {}


def _termination_handler(signum: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM by raising KeyboardInterrupt for clean shutdown."""
    logger.info("Received signal %s; shutting down gracefully.", signum)
    raise KeyboardInterrupt()


def _install_signal_handlers() -> None:
    """Install SIGINT and SIGTERM handlers; save previous handlers for restore."""
    _previous_signal_handlers.clear()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old = signal.signal(sig, _termination_handler)
            _previous_signal_handlers[sig] = old
        except (ValueError, OSError):
            pass  # e.g. SIGTERM not available on this platform


def _restore_signal_handlers() -> None:
    """Restore previous signal handlers so we don't affect callers (e.g. pytest)."""
    for sig, old in _previous_signal_handlers.items():
        try:
            signal.signal(sig, old)
        except (ValueError, OSError):
            pass
    _previous_signal_handlers.clear()


def _make_stream_logging_subscriber():
    """Subscribe to message stream and log thinking/output at INFO."""
    buffer: list[str] = []
    in_think = False
    in_tool_call = False
    node_id: str | None = None

    def on_event(event: StreamEvent) -> None:
        nonlocal buffer, in_think, in_tool_call, node_id
        nid = event.node_id or "model"
        if event.type == StreamEventType.BLOCK_START and event.block == BlockType.THINK:
            in_think = True
            buffer = []
            node_id = nid
        elif event.type == StreamEventType.BLOCK_END and event.block == BlockType.THINK:
            if buffer:
                logger.info("Agent [%s] thinking: %s", node_id or nid, "".join(buffer))
            in_think = False
            buffer = []
        elif event.type == StreamEventType.BLOCK_START and event.block == BlockType.TOOL_CALL:
            in_tool_call = True
            buffer = []
            node_id = nid
        elif event.type == StreamEventType.BLOCK_END and event.block == BlockType.TOOL_CALL:
            if buffer:
                logger.info("Agent [%s] tool_call: %s", node_id or nid, "".join(buffer))
            in_tool_call = False
            buffer = []
        elif event.type == StreamEventType.TEXT_CHUNK and event.text:
            if in_think or in_tool_call:
                buffer.append(event.text)
            else:
                logger.info("Agent [%s] output: %s", nid, event.text)

    return on_event


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
        _install_signal_handlers()

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
        message_handler.subscribe(_make_stream_logging_subscriber())
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
                # With multiple stream modes, LangGraph yields (metadata, mode, data) tuples
                raw = chunk
                if isinstance(chunk, tuple) and len(chunk) >= 3:
                    mode, raw = chunk[1], chunk[2]
                    if mode == "updates" and isinstance(raw, dict):
                        for _node_name, state_update in raw.items():
                            if isinstance(state_update, dict):
                                final_state = {**final_state, **state_update}
                elif is_update_chunk(chunk):
                    for _node_name, state_update in chunk.items():
                        if isinstance(state_update, dict):
                            final_state = {**final_state, **state_update}
                stream_handler.handle(chunk)
        except KeyboardInterrupt:
            raise
        finally:
            stream_handler.finalize()

        console.render_final_summary(final_state)
        # Preserve failed status from nodes (intent, milestone, etc); otherwise completed
        result = {**final_state}
        if result.get("status") != "failed":
            result["status"] = "completed"
        return result

    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user (Ctrl-C)")
        return {"status": "interrupted", "error": "User cancelled"}
    except Exception as e:
        logger.error("Workflow failed with error: %s", e, exc_info=verbose)
        return {"status": "failed", "error": str(e)}
    finally:
        _restore_signal_handlers()
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

    # Setup logging as early as possible (LEMMINGS_LOG_LEVEL env, default INFO)
    log_level = config.get("log_level", "INFO")
    if args.verbose:
        log_level = "DEBUG"
    setup_logging(level=log_level, log_file=config.get("log_file"))

    verbose = args.verbose or config.get("verbose", False)

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
        logger.error("Repository path does not exist: %s", repo_root)
        sys.exit(1)

    if args.index:
        try:
            from .rag.indexer import get_index_stats, update_index
            stats = get_index_stats(repo_root=repo_root)
            persist_dir = Path(repo_root) / ".rag_index"
            meta_file = persist_dir / "index_meta.json"
            if meta_file.exists() or stats["total_chunks"] > 0:
                logger.info("Checking RAG index for updates...")
                update_stats = update_index(repo_root)
                if update_stats["files_indexed"] > 0:
                    logger.info("Updated %s file(s) in %.1fs", update_stats["files_indexed"], update_stats["time_taken"])
                else:
                    logger.info("Index is up to date")
            else:
                logger.info("RAG index not found. Building initial index...")
                update_stats = update_index(repo_root)
                logger.info("Built index with %s file(s) in %.1fs", update_stats["files_indexed"], update_stats["time_taken"])
        except Exception as e:
            logger.warning("Could not update RAG index: %s. Continuing with shell-based search.", e, exc_info=verbose)

    result = run_workflow(
        user_request=args.request,
        repo_root=str(repo_root),
        verbose=verbose,
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
