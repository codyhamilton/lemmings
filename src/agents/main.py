#!/usr/bin/env python3
"""CLI entry point for the multi-agent orchestration system."""

import argparse
import sys
from pathlib import Path

from .state import create_initial_state, TaskTree, TaskStatus
from .graph import graph


def print_milestone_summary(milestones: dict, milestone_order: list) -> None:
    """Print a summary of milestones."""
    if not milestones:
        return
    
    print("\n" + "─" * 70)
    print("🎯 MILESTONES")
    print("─" * 70)
    
    for milestone_id in milestone_order:
        milestone_dict = milestones.get(milestone_id, {})
        status = milestone_dict.get("status", "pending")
        description = milestone_dict.get("description", milestone_id)
        
        status_icon = {
            "pending": "⏸️",
            "active": "🔄",
            "complete": "✅",
        }.get(status, "❓")
        
        print(f"{status_icon} {milestone_id}: {description}")


def print_task_summary(tasks: dict) -> None:
    """Print a summary of tasks."""
    if not tasks:
        return
    
    task_tree = TaskTree.from_dict(tasks)
    stats = task_tree.get_statistics()
    
    print("\n" + "─" * 70)
    print("📋 TASKS")
    print("─" * 70)
    print(f"Total: {stats['total']} | "
          f"✅ Complete: {stats['complete']} | "
          f"🔄 Ready: {stats['ready']} | "
          f"⏸️  Pending: {stats['pending']} | "
          f"❌ Failed: {stats['failed']} | "
          f"🚫 Blocked: {stats['blocked']}")
    
    # Show some example tasks
    all_tasks = list(task_tree.tasks.values())
    for task in all_tasks[:10]:
        status_icon = {
            TaskStatus.READY: "🟢",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETE: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.PENDING: "⏸️",
        }.get(task.status, "❓")
        
        print(f"\n{status_icon} {task.id}: {task.description[:60]}...")
        if task.measurable_outcome:
            print(f"   Outcome: {task.measurable_outcome[:50]}...")
    
    if len(all_tasks) > 10:
        print(f"\n... and {len(all_tasks) - 10} more tasks")


def print_final_summary(state: dict) -> None:
    """Print the final workflow summary."""
    status = state.get("status", "unknown")
    iteration = state.get("iteration", 0)
    remit = state.get("remit", "")
    
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    
    # Status
    if status == "complete":
        print("✅ Status: COMPLETED SUCCESSFULLY")
    elif status == "failed":
        print("❌ Status: FAILED")
        if state.get("error"):
            print(f"   Error: {state['error']}")
    else:
        print(f"📌 Status: {status}")
    
    if remit:
        print(f"\nRemit: {remit[:200]}...")
    
    if iteration > 0:
        print(f"   Expansion iterations: {iteration}")
    
    # Milestones summary
    milestones = state.get("milestones", {})
    milestone_order = state.get("milestone_order", [])
    if milestones:
        print_milestone_summary(milestones, milestone_order)
    
    # Tasks summary
    tasks = state.get("tasks", {})
    if tasks:
        print_task_summary(tasks)
    
    # Assessment summary
    last_assessment = state.get("last_assessment")
    if last_assessment:
        from .state import AssessmentResult
        assessment = AssessmentResult.from_dict(last_assessment)
        print("\n" + "─" * 70)
        print("📊 ASSESSMENT")
        print("─" * 70)
        print(f"Overall Complete: {'✅' if assessment.is_complete else '❌'}")
        if assessment.uncovered_gaps:
            print(f"\n⚠️  {len(assessment.uncovered_gaps)} uncovered gaps:")
            for gap in assessment.uncovered_gaps[:5]:
                print(f"   - {gap[:80]}...")
        if assessment.assessment_notes:
            print(f"\nNotes: {assessment.assessment_notes[:200]}...")
    
    print("=" * 70)


def print_pipeline_overview() -> None:
    """Print an overview of the pipeline stages."""
    print("\nWorkflow stages:")
    print("  1. 📥 Intake - Understand request and create milestones")
    print("  2. 🌱 Expander - Discover tasks via IF X THEN Y reasoning")
    print("  3. 🎯 Prioritizer - Select next ready task")
    print("  4. 🔍 Researcher - Analyze gap between current and desired state")
    print("  5. 📝 Planner - Create implementation plan")
    print("  6. 💻 Implementor - Execute changes")
    print("  7. ✅ Validator - Verify files exist")
    print("  8. 🔍 QA - Verify requirements satisfied")
    print("  9. 📊 Assessor - Check completion and gaps")
    print()


def run_workflow(
    user_request: str, 
    repo_root: str, 
    verbose: bool = False,
    max_retries: int = 3,
    include_game_docs: bool = True,
    dashboard_mode: bool = False,
) -> dict:
    """Run the multi-agent workflow.
    
    Args:
        user_request: The user's development request
        repo_root: Path to the repository root
        verbose: Whether to print extra debug output
        max_retries: Maximum retries for coder after review failure (0 = no retries)
        include_game_docs: Include game design documentation in context
    
    Returns:
        Final state dictionary
    """
    import os
    
    print("\n" + "=" * 70)
    print("🤖 ITERATIVE TASK TREE WORKFLOW")
    print("=" * 70)
    print(f"Repository: {repo_root}")
    print()
    print("Request:")
    print(f"  {user_request}")
    
    print_pipeline_overview()
    
    print("=" * 70)
    
    # Change to repo root directory (tools work relative to current directory)
    original_cwd = os.getcwd()
    dashboard_renderer = None
    try:
        os.chdir(repo_root)
        
        # Initialize dashboard if enabled
        if dashboard_mode:
            from .dashboard import DashboardRenderer
            from .graph import set_dashboard_renderer, graph
            from .workflow_status import set_output_suppression
            
            initial_state_for_dashboard = create_initial_state(
                user_request,
                repo_root,
                verbose=verbose,
                max_iterations=10,
                max_task_retries=max_retries,
                dashboard_mode=True,
            )
            dashboard_renderer = DashboardRenderer(initial_state_for_dashboard)
            set_dashboard_renderer(dashboard_renderer)
            # Enable output suppression
            set_output_suppression(suppress=True, show_detailed=False)
        else:
            from .graph import graph
        
        # Create initial state
        initial_state = create_initial_state(
            user_request, 
            repo_root, 
            verbose=verbose,
            max_iterations=10,
            max_task_retries=max_retries,
            dashboard_mode=dashboard_mode,
        )
        
        # Run the graph
        try:
            # Use astream_events for better interrupt handling, but fall back to invoke
            # This allows KeyboardInterrupt to propagate more reliably
            try:
                final_state = graph.invoke(initial_state)
            except KeyboardInterrupt:
                # Re-raise to be caught by outer handler
                raise
        except KeyboardInterrupt:
            print("\n\n⚠️  Workflow interrupted by user (Ctrl-C)")
            # Clean up dashboard if active
            if dashboard_renderer:
                dashboard_renderer.cleanup()
            return {"status": "interrupted", "error": "User cancelled"}
        except Exception as e:
            print(f"\n❌ Workflow failed with error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return {"status": "failed", "error": str(e)}
        
        # Print final summary (unless in dashboard mode)
        if not dashboard_mode:
            print_final_summary(final_state)
        elif dashboard_renderer:
            # Show final state in dashboard
            dashboard_renderer.update_state(final_state)
            dashboard_renderer.render()
            # Give Textual time to render the final state
            import time
            time.sleep(0.5)
            print("\n" + "=" * 70)
            print("✅ Workflow Complete")
            print("=" * 70)
        
        return final_state
    finally:
        # Cleanup dashboard
        if dashboard_renderer:
            dashboard_renderer.cleanup()
            from .graph import set_dashboard_renderer
            set_dashboard_renderer(None)
        
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
        "--no-game-docs",
        action="store_true",
        help="Disable game design documentation context (reduces prompt size)",
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
        include_game_docs=not args.no_game_docs,  # Invert the flag
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
