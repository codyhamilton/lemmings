#!/usr/bin/env python3
"""CLI entry point for the multi-agent orchestration system."""

import argparse
import os
import sys
from pathlib import Path

from .task_states import create_initial_state
from .graph import graph
from .ui.message_history import MessageHistory
from .ui.status_history import StatusHistory
from .ui.stream_handlers import AIMessageStreamHandler, StatusStreamHandler
from .ui.state_manager import UIStateManager, UIStateEvent
from .ui.base import UIBase
from .ui.verbose import VerboseUI
from .ui.dashboard import DashboardRenderer

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
        dashboard_mode: Whether to use dashboard UI
    
    Returns:
        Final state dictionary
    """
    # Change to repo root directory (tools work relative to current directory)
    original_cwd = os.getcwd()
    ui: UIBase = None
    message_history = MessageHistory()
    status_history = StatusHistory()
    ai_stream_handler = AIMessageStreamHandler()
    status_stream_handler = StatusStreamHandler(status_history)
    ui_state_manager: UIStateManager = None

    try:
        os.chdir(repo_root)
        
        # Create initial state
        initial_state = create_initial_state(
            user_request, 
            repo_root, 
            verbose=verbose,
            max_iterations=10,
            max_task_retries=max_retries,
            dashboard_mode=dashboard_mode,
        )
        
        # Initialize MessageHistory with initial state
        message_history.update_from_state(initial_state)
        
        # Create and start UI State Manager
        ui_state_manager = UIStateManager(initial_state=initial_state, max_refresh_rate=10.0)
        ui_state_manager.start()
        
        if dashboard_mode:
            ui = DashboardRenderer(initial_state, message_history, status_history, ui_state_manager)
        else:
            ui = VerboseUI(
                message_history, 
                status_history, 
                ui_state_manager, 
                show_thinking=verbose,
                stream_handler=ai_stream_handler
            )

        # Print workflow start header for verbose mode
        if not dashboard_mode and isinstance(ui, VerboseUI):
            ui.print_workflow_start(user_request, repo_root)
        
        # Run the graph with streaming
        try:
            final_state = initial_state.copy()
            previous_state = initial_state.copy()
            previous_node = previous_state.get("current_node")
            
            # Use stream() to get real-time updates and LLM token streaming
            # Stream both messages (for AI) and updates (for status)
            # This allows KeyboardInterrupt to propagate more reliably
            try:
                # Stream with both modes - LangGraph returns different chunk types
                for chunk in graph.stream(
                    initial_state,
                    subgraphs=True,
                    stream_mode=["messages", "updates"]
                ):
                    # Chunk structure depends on stream mode:
                    # - "messages": (node_info, (message_chunk, metadata)) or similar
                    # - "updates": {node_name: state_update}
                    # We need to detect which type we have
                    
                    # Check if it's an update chunk (dict with node names as keys)
                    if isinstance(chunk, dict) and all(isinstance(k, str) for k in chunk.keys()):
                        # This is an update chunk (state updates)
                        if status_stream_handler:
                            # Process each node update in the chunk
                            for node_name, state_update in chunk.items():
                                if isinstance(state_update, dict):
                                    status_stream_handler.process_state_update(
                                        node_name,
                                        state_update,
                                        full_state=previous_state
                                    )
                                    
                                    # Collect pending messages from stream handler and add to state
                                    if ai_stream_handler:
                                        pending_messages = ai_stream_handler.get_pending_messages()
                                        if pending_messages:
                                            # Add pending messages to state update
                                            current_messages = previous_state.get("messages", [])
                                            state_update["messages"] = current_messages + pending_messages
                                    
                                    # Update our tracked state
                                    previous_state = {**previous_state, **state_update}
                                    final_state = previous_state.copy()
                                    
                                    # Emit events to UI State Manager
                                    if ui_state_manager:
                                        # Detect node transitions
                                        current_node = state_update.get("current_node") or previous_state.get("current_node")
                                        
                                        # Check for node start (new node different from previous)
                                        if current_node and current_node != previous_node:
                                            # Emit node_complete for previous node if it existed
                                            if previous_node:
                                                complete_event = UIStateEvent(
                                                    event_type="node_complete",
                                                    node_name=previous_node,
                                                    state_update={"current_node": current_node},
                                                    full_state=final_state,
                                                    metadata={"node_name": previous_node}
                                                )
                                                ui_state_manager.emit_event(complete_event)
                                            
                                            # Emit node_start for new node
                                            start_event = UIStateEvent(
                                                event_type="node_start",
                                                node_name=current_node,
                                                state_update=state_update,
                                                full_state=final_state,
                                                metadata={"node_name": current_node}
                                            )
                                            ui_state_manager.emit_event(start_event)
                                        
                                        # Check for errors (node failed)
                                        if state_update.get("error"):
                                            error_node = current_node or node_name
                                            failed_event = UIStateEvent(
                                                event_type="node_failed",
                                                node_name=error_node,
                                                state_update=state_update,
                                                full_state=final_state,
                                                metadata={"node_name": error_node, "error": state_update.get("error")}
                                            )
                                            ui_state_manager.emit_event(failed_event)
                                        
                                        # Always emit state_update event
                                        update_event = UIStateEvent(
                                            event_type="state_update",
                                            node_name=node_name,
                                            state_update=state_update,
                                            full_state=final_state,
                                            metadata={"node_name": node_name}
                                        )
                                        ui_state_manager.emit_event(update_event)
                                        
                                        # Update previous_node for next iteration
                                        if current_node:
                                            previous_node = current_node
                                    
                                    # Update MessageHistory from new state
                                    if message_history:
                                        message_history.update_from_state(final_state)
                    else:
                        # This is likely a message chunk (AI messages)
                        # Message chunks come as tuples: (node_info, (message_chunk, metadata))
                        if ai_stream_handler:
                            ai_stream_handler.process_message_chunk(chunk)
            except KeyboardInterrupt:
                # Re-raise to be caught by outer handler
                raise
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
            # Finalize stream handlers
            if ai_stream_handler:
                ai_stream_handler.finalize()
                # Collect any remaining pending messages and add to final state
                pending_messages = ai_stream_handler.get_pending_messages()
                if pending_messages:
                    current_messages = final_state.get("messages", [])
                    final_state["messages"] = current_messages + pending_messages
                    # Update MessageHistory with final messages
                    if message_history:
                        message_history.update_from_state(final_state)
            if status_stream_handler:
                status_stream_handler.finalize()
            
            # Emit final state update to UI State Manager
            if ui_state_manager:
                final_event = UIStateEvent(
                    event_type="state_update",
                    state_update={},
                    full_state=final_state,
                    metadata={"final": True}
                )
                ui_state_manager.emit_event(final_event)
        
        # Render final summary
        if ui:
            if dashboard_mode:
                # Dashboard handles its own final state rendering
                if hasattr(ui, 'update_state'):
                    ui.update_state(final_state)
                    ui.render()
                import time
                time.sleep(0.5)
                print("\n" + "=" * 70)
                print("✅ Workflow Complete")
                print("=" * 70)
            else:
                # Verbose UI renders final summary
                ui.render_final_summary(final_state)
        
        return final_state
    finally:
        # Cleanup
        if ui:
            ui.cleanup()
        
        # Stop UI State Manager
        if ui_state_manager:
            ui_state_manager.stop()
        
        # Restore original working directory
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

Pipeline stages:
  1. UNDERSTAND  - Break request into requirements with keywords
  2. RESEARCH    - Find relevant context for each requirement  
  3. PLAN        - Create change summary for each requirement
  4. CONSOLIDATE - Reconcile into unified change list
  5. CODE        - Implement each change
  6. REVIEW      - Validate each change against requirements
  7. VALIDATE    - Final check that all requirements are met
        """,
    )
    
    parser.add_argument(
        "request",
        help="The development task to accomplish",
    )
    
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
        help="Enable dashboard view (press ESC to toggle detailed logs)",
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Update RAG index before running workflow",
    )
    
    args = parser.parse_args()
    
    # Determine repo root
    if args.repo:
        repo_root = Path(args.repo).resolve()
    else:
        # Try to find git root
        cwd = Path.cwd()
        repo_root = cwd
        
        # Walk up to find .git directory
        for parent in [cwd] + list(cwd.parents):
            if (parent / ".git").exists():
                repo_root = parent
                break
    
    if not repo_root.exists():
        print(f"Error: Repository path does not exist: {repo_root}")
        sys.exit(1)
    
    # Check and update RAG index if --index flag is provided
    if args.index:
        try:
            from .rag.indexer import get_index_stats, update_index
            
            stats = get_index_stats(repo_root=repo_root)
            
            # Check if index metadata exists (indicates index was built before)
            persist_dir = Path(repo_root) / ".rag_index"
            meta_file = persist_dir / "index_meta.json"
            
            if meta_file.exists() or stats['total_chunks'] > 0:
                # Index exists or was built before - do incremental update
                print("🔄 Checking RAG index for updates...")
                update_stats = update_index(repo_root)
                if update_stats['files_indexed'] > 0:
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
    
    # Run the workflow
    result = run_workflow(
        user_request=args.request,
        repo_root=str(repo_root),
        verbose=args.verbose,
        max_retries=args.max_retries,
        dashboard_mode=args.dashboard,
    )
    
    # Exit with appropriate code
    status = result.get("status", "failed")
    if status == "completed":
        sys.exit(0)
    elif status in ("completed_with_issues", "interrupted"):
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
