"""Prioritizer - selects next task from ready tasks in the DAG.

This is a simple, deterministic selector that picks the next task to execute
based on priority rules. No LLM needed - just graph traversal logic.

Priority rules (in order):
1. Tasks with more blockers (higher impact)
2. Tasks created earlier in the workflow (FIFO within same blocker count)
3. Tasks with simpler estimated complexity (tie-breaker)
"""

from ..task_states import WorkflowState, Task, TaskStatus, TaskTree, MilestoneStatus


def prioritizer_node(state: WorkflowState) -> dict:
    """Select the next task to execute from ready tasks.
    
    This is a deterministic function that applies priority rules to select
    the best next task. No LLM needed.
    
    Priority logic:
    1. Get all READY tasks (status=READY, all dependencies complete)
    2. Sort by:
       - Number of tasks this blocks (DESC) - higher impact first
       - Creation iteration (ASC) - older tasks first
       - Creation timestamp (ASC) - FIFO within same iteration
       - Complexity (ASC) - simpler first as tie-breaker
    3. Select the top task
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_task_id set to next task, or status=complete if no tasks
    """
    tasks_dict = state["tasks"]
    active_milestone_id = state.get("active_milestone_id")
    milestones = state.get("milestones", {})
    
    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    
    # Check if we have an active milestone
    if not active_milestone_id:
        # Check if there are any tasks at all
        stats = task_tree.get_statistics()
        if stats["total"] == 0:
            return {
                "current_task_id": None,
                "status": "complete",  # Mark as complete to exit
                "messages": ["Prioritizer: No active milestone and no tasks - exiting"],
            }
        
        # No active milestone but tasks exist - this is an unrecoverable state
        return {
            "current_task_id": None,
            "status": "complete",  # Mark as complete to exit
            "messages": ["Prioritizer: No active milestone - exiting to prevent infinite loop"],
        }
    
    # Get milestone info
    milestone_dict = milestones.get(active_milestone_id)
    milestone_desc = milestone_dict.get("description", active_milestone_id) if milestone_dict else active_milestone_id
    
    # Get statistics for active milestone
    milestone_tasks = task_tree.get_tasks_by_milestone(active_milestone_id)
    milestone_stats = {
        "total": len(milestone_tasks),
        "ready": sum(1 for t in milestone_tasks if t.status == TaskStatus.READY),
        "pending": sum(1 for t in milestone_tasks if t.status == TaskStatus.PENDING),
        "in_progress": sum(1 for t in milestone_tasks if t.status == TaskStatus.IN_PROGRESS),
        "complete": sum(1 for t in milestone_tasks if t.status == TaskStatus.COMPLETE),
        "failed": sum(1 for t in milestone_tasks if t.status == TaskStatus.FAILED),
        "blocked": sum(1 for t in milestone_tasks if t.status == TaskStatus.BLOCKED),
    }
    
    # Get overall statistics for context
    stats = task_tree.get_statistics()
    
    # Get ready tasks filtered by active milestone
    ready_tasks = task_tree.get_ready_tasks(milestone_id=active_milestone_id)
    
    if not ready_tasks:
        # Check if milestone is complete
        is_milestone_complete = task_tree.is_milestone_complete(active_milestone_id)
        
        if is_milestone_complete:
            return {
                "current_task_id": None,
                "status": "running",  # Continue to assessor to advance milestone
                "messages": [f"Prioritizer: Milestone {active_milestone_id} complete"],
            }
        
        # Check if there are any pending tasks in milestone (waiting on dependencies)
        if milestone_stats['pending'] > 0 or milestone_stats['in_progress'] > 0:
            return {
                "current_task_id": None,
                "status": "running",  # Keep running, might unblock later
                "messages": ["Prioritizer: No ready tasks in milestone, may be blocked"],
            }
        
        # No tasks in milestone at all - should not happen but handle gracefully
        return {
            "current_task_id": None,
            "status": "running",
            "messages": [f"Prioritizer: No tasks in milestone {active_milestone_id}"],
        }
    
    # Select highest priority task (already sorted by TaskTree.get_ready_tasks)
    selected_task = ready_tasks[0]
    
    # Mark task as in progress
    selected_task.status = TaskStatus.IN_PROGRESS
    
    return {
        "current_task_id": selected_task.id,
        "tasks": task_tree.to_dict(),  # Updated with IN_PROGRESS status
        "messages": [f"Prioritizer: Selected {selected_task.id} for execution"],
    }
