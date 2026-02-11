"""LangGraph workflow definition for the iterative task tree workflow.

This workflow implements a nested loop architecture:
- **Phase 1 (Intent Understanding)**: Intent → Milestone (once at start)
- **Phase 2 (Expansion)**: Expander → Prioritizer → Execution Loop → Assessor → repeat
- **Phase 3 (Execution)**: Researcher → Planner → Implementor → Validator → QA → route/retry

The workflow iterates until:
1. All tasks are complete/failed/deferred
2. No uncovered gaps remain
3. The remit is fully satisfied
"""

from langgraph.graph import StateGraph, END
from typing import Callable

from .task_states import WorkflowState, Task, TaskStatus, TaskTree, GapAnalysis, QAResult, MilestoneStatus
from .agents import (
    intent_node,
    milestone_node,
    expander_node,
    prioritizer_node,
    researcher_node,
    planner_node,
    implementor_node,
    validator_node,
    qa_node,
)
from .agents.assessor import assessor_node


# =============================================================================
# Helper nodes (state transitions)
# =============================================================================

def mark_task_complete_node(state: WorkflowState) -> dict:
    """Mark current task as complete and clear ephemeral state.
    
    This node:
    1. Marks the current task as COMPLETE
    2. Updates dependent tasks (may become READY)
    3. Adds task to completed_task_ids
    4. Clears ephemeral state (current_* fields)
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    completed_task_ids = state.get("completed_task_ids", [])
    
    if not current_task_id:
        return {
            "messages": ["mark_complete: No current task"],
        }
    
    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"mark_complete: Task {current_task_id} not found"],
        }
    
    # Mark complete
    task_tree.mark_complete(current_task_id)
    
    # Add to completed list
    if current_task_id not in completed_task_ids:
        completed_task_ids = completed_task_ids + [current_task_id]
    
    # Check if any blocked tasks are now ready
    newly_ready = []
    for blocked_id in task.blocks:
        blocked_task = task_tree.tasks.get(blocked_id)
        if blocked_task and blocked_task.status == TaskStatus.READY:
            newly_ready.append(blocked_id)
    
    return {
        "tasks": task_tree.to_dict(),
        "completed_task_ids": completed_task_ids,
        "current_task_id": None,
        # Clear ephemeral state
        "current_gap_analysis": None,
        "current_implementation_plan": None,
        "current_implementation_result": None,
        "current_validation_result": None,
        "current_qa_result": None,
        "messages": [f"Task {current_task_id} marked complete"],
    }


def mark_task_failed_node(state: WorkflowState) -> dict:
    """Mark current task as failed and clear ephemeral state.
    
    This node:
    1. Marks the current task as FAILED
    2. Marks dependent tasks as BLOCKED
    3. Adds task to failed_task_ids
    4. Clears ephemeral state
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    failed_task_ids = state.get("failed_task_ids", [])
    error = state.get("error", "Unknown error")
    
    if not current_task_id:
        return {
            "messages": ["mark_failed: No current task"],
        }
    
    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"mark_failed: Task {current_task_id} not found"],
        }
    
    # Determine failure stage from last agent
    last_stage = "unknown"
    if state.get("current_qa_result"):
        last_stage = "qa"
    elif state.get("current_validation_result"):
        last_stage = "validator"
    elif state.get("current_implementation_result"):
        last_stage = "implementor"
    elif state.get("current_implementation_plan"):
        last_stage = "planner"
    elif state.get("current_gap_analysis"):
        last_stage = "researcher"
    
    # Mark failed
    task_tree.mark_failed(current_task_id, error, last_stage)
    
    # Add to failed list
    if current_task_id not in failed_task_ids:
        failed_task_ids = failed_task_ids + [current_task_id]
    
    # Check blocked tasks
    blocked_count = sum(
        1 for t in task_tree.tasks.values()
        if t.status == TaskStatus.BLOCKED
    )
    
    return {
        "tasks": task_tree.to_dict(),
        "failed_task_ids": failed_task_ids,
        "current_task_id": None,
        # Clear ephemeral state
        "current_gap_analysis": None,
        "current_implementation_plan": None,
        "current_implementation_result": None,
        "current_validation_result": None,
        "current_qa_result": None,
        "error": None,
        "messages": [f"Task {current_task_id} marked failed: {error}"],
    }


def increment_iteration_node(state: WorkflowState) -> dict:
    """Increment the expansion iteration counter."""
    iteration = state.get("iteration", 0)
    return {
        "iteration": iteration + 1,
        "tasks_created_this_iteration": 0,  # Reset counter
        "messages": [f"Iteration {iteration + 1} starting"],
    }


def increment_attempt_and_retry_implementor_node(state: WorkflowState) -> dict:
    """Increment attempt count and route to implementor for retry.
    
    This node ensures attempt_count is properly saved to state before retrying.
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    
    if not current_task_id:
        return {
            "error": "No current task for attempt increment",
            "messages": ["increment_attempt: No current task"],
        }
    
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"increment_attempt: Task {current_task_id} not found"],
        }
    
    # Safety check: don't increment if already at max
    if task.attempt_count >= task.max_attempts:
        return {
            "tasks": task_tree.to_dict(),
            "error": f"Max retries ({task.max_attempts}) already reached for {current_task_id}",
            "messages": [f"Max retries reached for {current_task_id}"],
        }
    
    # Increment attempt count
    task.attempt_count += 1
    
    return {
        "tasks": task_tree.to_dict(),
        "messages": [f"Incremented attempt count for {current_task_id} to {task.attempt_count}"],
    }


def increment_attempt_and_retry_researcher_node(state: WorkflowState) -> dict:
    """Increment attempt count and route to researcher for retry."""
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    
    if not current_task_id:
        return {
            "error": "No current task for attempt increment",
            "messages": ["increment_attempt: No current task"],
        }
    
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"increment_attempt: Task {current_task_id} not found"],
        }
    
    # Safety check: don't increment if already at max
    if task.attempt_count >= task.max_attempts:
        return {
            "tasks": task_tree.to_dict(),
            "error": f"Max retries ({task.max_attempts}) already reached for {current_task_id}",
            "messages": [f"Max retries reached for {current_task_id}"],
        }
    
    # Increment attempt count
    task.attempt_count += 1
    
    return {
        "tasks": task_tree.to_dict(),
        "messages": [f"Incremented attempt count for {current_task_id} to {task.attempt_count}"],
    }


def increment_attempt_and_retry_planner_node(state: WorkflowState) -> dict:
    """Increment attempt count and route to planner for retry."""
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    
    if not current_task_id:
        return {
            "error": "No current task for attempt increment",
            "messages": ["increment_attempt: No current task"],
        }
    
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"increment_attempt: Task {current_task_id} not found"],
        }
    
    # Safety check: don't increment if already at max
    if task.attempt_count >= task.max_attempts:
        return {
            "tasks": task_tree.to_dict(),
            "error": f"Max retries ({task.max_attempts}) already reached for {current_task_id}",
            "messages": [f"Max retries reached for {current_task_id}"],
        }
    
    # Increment attempt count
    task.attempt_count += 1
    
    return {
        "tasks": task_tree.to_dict(),
        "messages": [f"Incremented attempt count for {current_task_id} to {task.attempt_count}"],
    }


def advance_milestone_node(state: WorkflowState) -> dict:
    """Advance to the next milestone when current milestone is complete.
    
    This node:
    1. Marks current milestone as complete
    2. Sets active_milestone_id to next milestone
    3. Updates milestone statuses
    """
    active_milestone_id = state.get("active_milestone_id")
    milestones = state.get("milestones", {})
    milestone_order = state.get("milestone_order", [])
    last_assessment = state.get("last_assessment")
    
    if not active_milestone_id:
        return {"messages": ["advance_milestone: No active milestone"]}
    
    if not last_assessment:
        return {"messages": ["advance_milestone: No assessment"]}
    
    from .task_states import AssessmentResult
    assessment = AssessmentResult.from_dict(last_assessment)
    next_milestone_id = assessment.next_milestone_id
    
    if not next_milestone_id:
        return {"messages": ["advance_milestone: All milestones complete"]}
    
    # Mark current milestone as complete
    updated_milestones = milestones.copy()
    if active_milestone_id in updated_milestones:
        current_milestone = updated_milestones[active_milestone_id].copy()
        current_milestone["status"] = MilestoneStatus.COMPLETE.value
        updated_milestones[active_milestone_id] = current_milestone
    
    # Set next milestone as active
    if next_milestone_id in updated_milestones:
        next_milestone = updated_milestones[next_milestone_id].copy()
        next_milestone["status"] = MilestoneStatus.ACTIVE.value
        updated_milestones[next_milestone_id] = next_milestone
    
    return {
        "milestones": updated_milestones,
        "active_milestone_id": next_milestone_id,
        "messages": [f"Advanced from {active_milestone_id} to {next_milestone_id}"],
    }


# =============================================================================
# Routing functions
# =============================================================================

def after_researcher(state: WorkflowState) -> str:
    """Route after Researcher: check if gap exists.
    
    Returns:
        "mark_complete" if no gap (task already satisfied)
        "planner" if gap exists
    """
    current_gap_analysis = state.get("current_gap_analysis")
    
    if not current_gap_analysis:
        # No gap analysis - assume gap exists and proceed to planner
        return "planner"
    
    gap_analysis = GapAnalysis.from_dict(current_gap_analysis)
    
    if not gap_analysis.gap_exists:
        return "mark_complete"
    
    return "planner"


def after_validator(state: WorkflowState) -> str:
    """Route after Validator: check if validation passed.
    
    Returns:
        "qa" if validation passed
        "increment_attempt_and_retry_implementor" if validation failed (retry)
        "mark_failed" if max retries reached or no task
    """
    current_validation_result = state.get("current_validation_result")
    current_task_id = state.get("current_task_id")
    
    if not current_task_id:
        return "mark_failed"
    
    if not current_validation_result:
        # No validation result - proceed to QA (let it handle the error)
        return "qa"
    
    from .task_states import ValidationResult
    validation_result = ValidationResult.from_dict(current_validation_result)
    
    if validation_result.validation_passed:
        return "qa"
    
    # Validation failed - check retry count
    tasks_dict = state["tasks"]
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return "mark_failed"
    
    if task.attempt_count >= task.max_attempts:
        return "mark_failed"
    
    # Retry implementor - use intermediate node to increment attempt_count
    return "increment_attempt_and_retry_implementor"


def after_qa(state: WorkflowState) -> str:
    """Route after QA: check result and failure type.
    
    Returns:
        "mark_complete" if QA passed
        "increment_attempt_and_retry_researcher" if wrong_approach
        "increment_attempt_and_retry_implementor" if incomplete
        "increment_attempt_and_retry_planner" if plan_issue
        "mark_failed" if max retries or no task
    """
    current_qa_result = state.get("current_qa_result")
    current_task_id = state.get("current_task_id")
    
    if not current_task_id:
        return "mark_failed"
    
    if not current_qa_result:
        # No QA result - mark as failed
        return "mark_failed"
    
    qa_result = QAResult.from_dict(current_qa_result)
    
    if qa_result.passed:
        return "mark_complete"
    
    # QA failed - check retry count
    tasks_dict = state["tasks"]
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return "mark_failed"
    
    if task.attempt_count >= task.max_attempts:
        return "mark_failed"
    
    # Route based on failure type - use intermediate nodes to increment attempt_count
    failure_type = qa_result.failure_type
    
    if failure_type == "wrong_approach":
        # Fundamental misunderstanding - need new gap analysis
        return "increment_attempt_and_retry_researcher"
    
    elif failure_type == "incomplete":
        # Right approach, missing pieces - retry implementor
        return "increment_attempt_and_retry_implementor"
    
    elif failure_type == "plan_issue":
        # Plan was insufficient - need new plan
        return "increment_attempt_and_retry_planner"
    
    else:
        # Unknown failure type - default to retry implementor
        return "increment_attempt_and_retry_implementor"


def after_assessor(state: WorkflowState) -> str:
    """Route after Assessor: check completion, milestone completion, and gaps.
    
    Returns:
        "end" if complete
        "advance_milestone" if milestone complete (then expand next)
        "expander" if gaps uncovered (within milestone)
        "prioritizer" if stable but tasks remain
        "end" if unrecoverable state (no active milestone)
    """
    last_assessment = state.get("last_assessment")
    status = state.get("status", "running")
    active_milestone_id = state.get("active_milestone_id")
    
    # Check if workflow is already marked complete
    if status == "complete":
        return "end"
    
    # If no active milestone, check if we should exit
    if not active_milestone_id:
        tasks_dict = state["tasks"]
        task_tree = TaskTree.from_dict(tasks_dict)
        stats = task_tree.get_statistics()
        
        # No tasks at all - exit
        if stats["total"] == 0:
            return "end"
        
        # No active milestone but tasks exist - unrecoverable, exit to prevent loop
        return "end"
    
    if not last_assessment:
        # No assessment - continue to prioritizer
        return "prioritizer"
    
    from .task_states import AssessmentResult
    assessment = AssessmentResult.from_dict(last_assessment)
    
    # If complete, end workflow
    if assessment.is_complete:
        return "end"
    
    # If milestone complete, advance to next milestone
    if assessment.milestone_complete and assessment.next_milestone_id:
        return "advance_milestone"
    
    # If gaps uncovered, expand (within current milestone)
    if assessment.uncovered_gaps:
        return "expander"
    
    # If stable but not complete, check if tasks remain
    if assessment.stability_check:
        # Check if there are any ready/pending tasks in active milestone
        tasks_dict = state["tasks"]
        task_tree = TaskTree.from_dict(tasks_dict)
        
        ready_tasks = task_tree.get_ready_tasks(milestone_id=active_milestone_id)
        milestone_tasks = task_tree.get_tasks_by_milestone(active_milestone_id)
        pending_count = sum(1 for t in milestone_tasks if t.status == TaskStatus.PENDING)
        
        if len(ready_tasks) > 0 or pending_count > 0:
            return "prioritizer"
        
        # Stable but no tasks - should be complete
        return "end"
    
    # Not stable - continue to prioritizer (may have new tasks from expander)
    return "prioritizer"


def after_intent(state: WorkflowState) -> str:
    """Route after Intent: check if intent succeeded.
    
    Returns:
        "milestone" if intent succeeded (has remit)
        "end" if intent failed (no remit, status failed)
    """
    remit = state.get("remit", "")
    status = state.get("status", "running")
    
    if status == "failed" or not remit:
        return "end"
    
    return "milestone"


def after_milestone(state: WorkflowState) -> str:
    """Route after Milestone: check if milestone succeeded.
    
    Returns:
        "increment_iteration" if milestone succeeded (has milestones)
        "end" if milestone failed (no milestones, status failed)
    """
    milestones = state.get("milestones", {})
    status = state.get("status", "running")
    
    if status == "failed" or not milestones:
        return "end"
    
    return "increment_iteration"


def after_prioritizer(state: WorkflowState) -> str:
    """Route after Prioritizer: check if task selected or workflow complete.
    
    Returns:
        "researcher" if task selected
        "assessor" if no task (workflow may be complete)
        "end" if unrecoverable state (no active milestone, no tasks)
    """
    current_task_id = state.get("current_task_id")
    status = state.get("status", "running")
    active_milestone_id = state.get("active_milestone_id")
    
    # Check if workflow is already marked complete
    if status == "complete":
        return "end"
    
    # If no active milestone, check if we should exit
    if not active_milestone_id:
        tasks_dict = state["tasks"]
        task_tree = TaskTree.from_dict(tasks_dict)
        stats = task_tree.get_statistics()
        
        # No tasks at all - exit
        if stats["total"] == 0:
            return "end"
        
        # No active milestone but tasks exist - unrecoverable, exit to prevent loop
        return "end"
    
    if current_task_id:
        return "researcher"
    
    # No task selected - assess completion
    return "assessor"


# =============================================================================
# Build the graph
# =============================================================================

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow.
    
    The workflow follows this structure:
    
    **Phase 1 (Intent Understanding - once at start)**:
    1. Intent → interprets user request, produces remit + needs
    2. Milestone → breaks remit into sequential interim states
    
    **Phase 2 (Expansion Loop)**:
    1. Expander → discovers new tasks via IF X THEN Y for active milestone
    2. Prioritizer → selects next ready task
    3. Execution Loop (phase 3) → executes single task
    4. Assessor → checks for gaps and completion
    5. Route back to Expander (if gaps) or Prioritizer (if tasks remain) or END
    
    **Phase 3 (Execution Loop - per task)**:
    1. Researcher → gap analysis
    2. Route: no gap → mark_complete, gap → Planner
    3. Planner → implementation plan
    4. Implementor → make changes
    5. Validator → verify files exist
    6. Route: validation failed → Implementor (retry), passed → QA
    7. QA → requirement satisfaction check
    8. Route: passed → mark_complete, failed → route by failure_type
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes (no wrapping - UI handled by StatusStreamHandler)
    workflow.add_node("intent", intent_node)
    workflow.add_node("milestone", milestone_node)
    workflow.add_node("expander", expander_node)
    workflow.add_node("prioritizer", prioritizer_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("implementor", implementor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("assessor", assessor_node)
    workflow.add_node("mark_complete", mark_task_complete_node)
    workflow.add_node("mark_failed", mark_task_failed_node)
    workflow.add_node("increment_iteration", increment_iteration_node)
    workflow.add_node("advance_milestone", advance_milestone_node)
    workflow.add_node("increment_attempt_and_retry_implementor", increment_attempt_and_retry_implementor_node)
    workflow.add_node("increment_attempt_and_retry_researcher", increment_attempt_and_retry_researcher_node)
    workflow.add_node("increment_attempt_and_retry_planner", increment_attempt_and_retry_planner_node)
    
    # Set entry point
    workflow.set_entry_point("intent")
    
    # === Phase 1: Intent and Milestone Planning (once at start) ===
    # Intent understands request → Milestone creates milestones → Expander expands first milestone
    workflow.add_conditional_edges(
        "intent",
        after_intent,
        {
            "milestone": "milestone",
            "end": END,
        }
    )
    workflow.add_conditional_edges(
        "milestone",
        after_milestone,
        {
            "increment_iteration": "increment_iteration",
            "end": END,
        }
    )
    workflow.add_edge("increment_iteration", "expander")
    
    # === Phase 2: Outer Loop (Expansion) ===
    
    # Expander → Prioritizer (expander sets active milestone if first expansion)
    workflow.add_edge("expander", "prioritizer")
    
    # Prioritizer → Researcher (if task) or Assessor (if no task)
    workflow.add_conditional_edges(
        "prioritizer",
        after_prioritizer,
        {
            "researcher": "researcher",
            "assessor": "assessor",
        }
    )
    
    # === Phase 3: Inner Loop (Execution) ===
    
    # Researcher → Planner (if gap) or mark_complete (if no gap)
    workflow.add_conditional_edges(
        "researcher",
        after_researcher,
        {
            "planner": "planner",
            "mark_complete": "mark_complete",
        }
    )
    
    # Planner → Implementor
    workflow.add_edge("planner", "implementor")
    
    # Implementor → Validator
    workflow.add_edge("implementor", "validator")
    
    # Validator → QA (if passed) or increment_attempt_and_retry_implementor (if failed, retry)
    workflow.add_conditional_edges(
        "validator",
        after_validator,
        {
            "qa": "qa",
            "increment_attempt_and_retry_implementor": "increment_attempt_and_retry_implementor",
            "mark_failed": "mark_failed",
        }
    )
    
    # Increment attempt nodes → route to appropriate agent
    workflow.add_edge("increment_attempt_and_retry_implementor", "implementor")
    workflow.add_edge("increment_attempt_and_retry_researcher", "researcher")
    workflow.add_edge("increment_attempt_and_retry_planner", "planner")
    
    # QA → route based on result
    workflow.add_conditional_edges(
        "qa",
        after_qa,
        {
            "mark_complete": "mark_complete",
            "increment_attempt_and_retry_researcher": "increment_attempt_and_retry_researcher",
            "increment_attempt_and_retry_implementor": "increment_attempt_and_retry_implementor",
            "increment_attempt_and_retry_planner": "increment_attempt_and_retry_planner",
            "mark_failed": "mark_failed",
        }
    )
    
    # === Phase 4: Task Completion ===
    
    # mark_complete → Assessor
    workflow.add_edge("mark_complete", "assessor")
    
    # mark_failed → Assessor
    workflow.add_edge("mark_failed", "assessor")
    
    # === Phase 5: Assessment and Routing ===
    
    # Assessor → route based on completion/gaps/milestone completion
    workflow.add_conditional_edges(
        "assessor",
        after_assessor,
        {
            "end": END,
            "advance_milestone": "advance_milestone",
            "expander": "expander",
            "prioritizer": "prioritizer",
        }
    )
    
    # Advance milestone → expander (to expand the next milestone)
    workflow.add_edge("advance_milestone", "expander")
    
    return workflow.compile()


# Compiled graph ready to use
graph = build_graph()
