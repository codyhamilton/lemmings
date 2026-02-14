"""LangGraph workflow definition for the iterative task tree workflow.

This workflow implements the subagent redesign (see WORKFLOW_ARCHITECTURE.md):
- **Phase 1 (Scope)**: ScopeAgent → set_active_milestone (once at start)
- **Phase 2 (Task Loop)**: TaskPlanner → Implementor → QA → mark_complete/mark_failed
- **Phase 3 (Assessment)**: Assessor (periodic) → task_planner | advance_milestone | scope_review_agent | end

Components: InitialScopeAgent (entry), ScopeReviewAgent (re-plan), TaskPlanner, Implementor, QA, Assessor.
"""

import json
from langgraph.graph import StateGraph, END
from typing import Callable

from .logging_config import get_logger
from .task_states import (
    WorkflowState,
    Task,
    TaskStatus,
    TaskTree,
    QAResult,
    get_milestones_list,
    get_active_milestone_id,
    get_active_milestone_index,
)

logger = get_logger(__name__)
from .agents import (
    initial_scope_agent_node,
    scope_review_agent_node,
    task_planner_node,
    implementor_node,
    qa_node,
    report_node,
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
        logger.debug("mark_complete: No current task")
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
    logger.debug("Task %s marked complete", current_task_id)

    # Add to completed list
    if current_task_id not in completed_task_ids:
        completed_task_ids = completed_task_ids + [current_task_id]

    # Add to done_list for TaskPlanner sliding window
    done_list = list(state.get("done_list", []))
    impl_result = state.get("current_implementation_result") or {}
    result_summary = impl_result.get("result_summary", "") if isinstance(impl_result, dict) else ""
    done_list.append({
        "description": task.description,
        "result": result_summary or task.result_summary or "Done",
    })

    # Check if any blocked tasks are now ready
    newly_ready = []
    for blocked_id in task.blocks:
        blocked_task = task_tree.tasks.get(blocked_id)
        if blocked_task and blocked_task.status == TaskStatus.READY:
            newly_ready.append(blocked_id)

    tasks_since_last_review = state.get("tasks_since_last_review", 0) + 1

    return {
        "tasks": task_tree.to_dict(),
        "completed_task_ids": completed_task_ids,
        "done_list": done_list,
        "tasks_since_last_review": tasks_since_last_review,
        "current_task_id": None,
        # Clear ephemeral state
        "current_gap_analysis": None,
        "current_implementation_plan": None,
        "current_implementation_result": None,
        "current_qa_result": None,
        "current_task_description": None,
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
        logger.debug("mark_failed: No current task")
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
    
    # Determine failure stage from last agent (validator absorbed into QA)
    last_stage = "unknown"
    if state.get("current_qa_result"):
        last_stage = "qa"
    elif state.get("current_implementation_result"):
        last_stage = "implementor"
    elif state.get("current_implementation_plan"):
        last_stage = "planner"
    elif state.get("current_gap_analysis"):
        last_stage = "researcher"
    
    # Mark failed
    task_tree.mark_failed(current_task_id, error, last_stage)
    logger.debug("Task %s marked failed at stage %s: %s", current_task_id, last_stage, error)
    
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
        "current_qa_result": None,
        "error": None,
        "messages": [f"Task {current_task_id} marked failed: {error}"],
    }


def set_active_milestone_node(state: WorkflowState) -> dict:
    """Set the first milestone as active after initial scope creation.
    
    ScopeAgent already sets active_milestone_index=0. This node is a no-op
    when index is already set; it exists for graph flow consistency.
    """
    milestones_list = get_milestones_list(state)
    active_milestone_index = get_active_milestone_index(state)
    
    if not milestones_list:
        return {"messages": ["set_active_milestone: No milestones"]}
    
    if active_milestone_index >= 0:
        active_id = get_active_milestone_id(state)
        logger.debug("set_active_milestone: Already set %s (index %d)", active_id, active_milestone_index)
        return {"messages": [f"Set active milestone: {active_id}"]}
    
    logger.debug("set_active_milestone: Set first milestone as active")
    return {
        "active_milestone_index": 0,
        "messages": [f"Set active milestone: {milestones_list[0].get('id', 'milestone_001')}"],
    }


def increment_iteration_node(state: WorkflowState) -> dict:
    """Increment the expansion iteration counter."""
    iteration = state.get("iteration", 0)
    logger.debug("Incrementing iteration to %s", iteration + 1)
    return {
        "iteration": iteration + 1,
        "tasks_created_this_iteration": 0,  # Reset counter
        "messages": [f"Iteration {iteration + 1} starting"],
    }


def make_increment_attempt_node(target_agent: str) -> Callable[[WorkflowState], dict]:
    """Create a retry node that increments attempt_count and routes to the target agent.
    
    The routing (which agent to retry) is handled by conditional edges; this node
    only ensures attempt_count is properly saved to state before retrying.
    """

    def node(state: WorkflowState) -> dict:
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
            "messages": [f"Incremented attempt count for {current_task_id} to {task.attempt_count} (retry {target_agent})"],
        }

    return node


def advance_milestone_node(state: WorkflowState) -> dict:
    """Advance to the next milestone when current milestone is complete."""
    active_milestone_id = get_active_milestone_id(state)
    milestones_list = get_milestones_list(state)
    current_index = get_active_milestone_index(state)
    last_assessment = state.get("last_assessment")
    
    logger.debug("advance_milestone: active_milestone_id=%s index=%d", active_milestone_id, current_index)
    
    if not active_milestone_id or current_index < 0:
        return {"messages": ["advance_milestone: No active milestone"]}
    
    if not last_assessment:
        return {"messages": ["advance_milestone: No assessment"]}
    
    from .task_states import AssessmentResult
    assessment = AssessmentResult.from_dict(last_assessment)
    next_milestone_id = assessment.next_milestone_id
    
    if not next_milestone_id:
        return {"messages": ["advance_milestone: All milestones complete"]}
    
    # Find next index
    next_index = current_index + 1
    if next_index >= len(milestones_list):
        return {"messages": ["advance_milestone: No next milestone"]}
    
    return {
        "active_milestone_index": next_index,
        "done_list": [],
        "carry_forward": [],
        "tasks_since_last_review": 0,
        "messages": [f"Advanced from {active_milestone_id} to {next_milestone_id}"],
    }


# =============================================================================
# Routing functions
# =============================================================================

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
        logger.debug("after_qa: passed, routing to mark_complete")
        return "mark_complete"
    
    # QA failed - check retry count
    tasks_dict = state["tasks"]
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return "mark_failed"
    
    if task.attempt_count >= task.max_attempts:
        return "mark_failed"

    # TaskPlanner tasks: retry task_planner (plan/approach adjustment)
    if current_task_id.startswith("task_tp_"):
        logger.debug("after_qa: TaskPlanner task failed, routing to retry task_planner")
        return "increment_attempt_and_retry_task_planner"

    # Other tasks: retry implementor (incomplete) or task_planner (wrong_approach/plan_issue)
    failure_type = qa_result.failure_type
    if failure_type == "wrong_approach" or failure_type == "plan_issue":
        return "increment_attempt_and_retry_task_planner"
    return "increment_attempt_and_retry_implementor"


def after_assessor(state: WorkflowState) -> str:
    """Route after Assessor: check completion, milestone completion, and gaps.
    
    Returns:
        "end" if complete
        "advance_milestone" if milestone complete (then task_planner for next)
        "scope_review_agent" if major divergence (re-plan)
        "task_planner" if gaps uncovered or stable but tasks remain
    """
    last_assessment = state.get("last_assessment")
    status = state.get("status", "running")
    active_milestone_id = get_active_milestone_id(state)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 100)
    
    # Check if workflow is already marked complete
    if status == "complete":
        return "end"
    
    # Max iteration guard: prevent infinite loops
    if iteration >= max_iterations:
        logger.info("after_assessor: max_iterations (%s) reached, routing to report", max_iterations)
        return "end"
    
    # If no active milestone, check if we should exit
    if not active_milestone_id:
        tasks_dict = state.get("tasks", {})
        task_tree = TaskTree.from_dict(tasks_dict)
        stats = task_tree.get_statistics()
        
        # No tasks at all - exit
        if stats["total"] == 0:
            return "end"
        
        # No active milestone but tasks exist - unrecoverable, exit to prevent loop
        return "end"
    
    if not last_assessment:
        return "task_planner"
    
    from .task_states import AssessmentResult
    assessment = AssessmentResult.from_dict(last_assessment)
    
    # If complete, end workflow
    if assessment.is_complete:
        return "end"
    
    # Major divergence: escalate to ScopeReview for re-plan
    if getattr(assessment, "escalate_to_scope", False):
        logger.debug("after_assessor: major divergence, routing to scope_review_agent")
        return "scope_review_agent"
    
    # If milestone complete, advance to next milestone
    if assessment.milestone_complete and assessment.next_milestone_id:
        logger.debug("after_assessor: milestone complete, routing to advance_milestone")
        return "advance_milestone"
    
    # Gaps uncovered or stable but tasks remain: continue with task_planner
    if assessment.uncovered_gaps:
        logger.debug("after_assessor: gaps uncovered, routing to task_planner")
        return "task_planner"
    
    # If stable but not complete, check if tasks remain
    if assessment.stability_check:
        tasks_dict = state.get("tasks", {})
        task_tree = TaskTree.from_dict(tasks_dict)
        ready_tasks = task_tree.get_ready_tasks(milestone_id=active_milestone_id)
        milestone_tasks = task_tree.get_tasks_by_milestone(active_milestone_id)
        pending_count = sum(1 for t in milestone_tasks if t.status == TaskStatus.PENDING)
        
        if len(ready_tasks) > 0 or pending_count > 0:
            return "task_planner"
        
        return "end"
    
    return "task_planner"


def after_task_planner(state: WorkflowState) -> str:
    """Route after TaskPlanner based on action.

    Returns:
        "implementor" if action=implement
        "task_planner" if action=skip (next round)
        "assessor" if action=abort or milestone_done
    """
    action = state.get("task_planner_action", "")
    if action == "implement":
        return "implementor"
    if action == "skip":
        return "task_planner"
    if action in ("abort", "milestone_done"):
        return "assessor"
    # Default: treat as abort
    return "assessor"


def after_scope_agent(state: WorkflowState) -> str:
    """Route after ScopeAgent: check if scope definition succeeded.
    
    Returns:
        "set_active_milestone" if succeeded (has remit and milestones)
        "end" if failed (no remit, status failed)
    """
    remit = state.get("remit", "")
    milestones_list = get_milestones_list(state)
    status = state.get("status", "running")

    if status == "failed" or not remit or not milestones_list:
        return "end"

    return "set_active_milestone"


# =============================================================================
# Node wrapper for INFO-level logging (transitions, agent input/output)
# =============================================================================

# Nodes that invoke the LLM; we log their input/output as formatted JSON
_AGENT_NODES = frozenset({
    "initial_scope_agent", "scope_review_agent", "task_planner", "implementor", "qa", "assessor", "report",
})


def _format_state_for_log(state: dict) -> str:
    """Format state dict as JSON for logging; avoid huge payloads."""
    try:
        s = json.dumps(state, indent=2, default=str)
        return s if len(s) <= 8000 else s[:8000] + "\n... (truncated)"
    except Exception:
        return str(state)[:2000]


def _wrap_node_for_logging(name: str, node_func: Callable) -> Callable:
    """Wrap a graph node to log transitions and agent input/output at INFO."""
    def wrapped(state: WorkflowState) -> dict:
        logger.info("Node transition: entering %s", name)
        if name in _AGENT_NODES:
            logger.info("Agent %s input:\n%s", name, _format_state_for_log(state))
        try:
            result = node_func(state)
            if name in _AGENT_NODES and isinstance(result, dict):
                logger.info("Agent %s output:\n%s", name, _format_state_for_log(result))
            logger.info("Node transition: exiting %s", name)
            return result
        except Exception:
            logger.info("Node transition: exiting %s (failed)", name)
            raise
    return wrapped


# =============================================================================
# Build the graph
# =============================================================================

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow.
    
    Five components: ScopeAgent, TaskPlanner, Implementor, QA, Assessor.
    See WORKFLOW_ARCHITECTURE.md for full design.
    
    **Phase 1 (Scope - once at start)**: InitialScopeAgent → set_active_milestone
    **Phase 2 (Task Loop)**: TaskPlanner → Implementor → QA → mark_complete/mark_failed
    **Phase 3 (Assessment)**: Assessor → task_planner | advance_milestone | scope_review_agent | end
    """
    workflow = StateGraph(WorkflowState)

    # Add all nodes with INFO logging wrapper (transitions + agent input/output)
    workflow.add_node("initial_scope_agent", _wrap_node_for_logging("initial_scope_agent", initial_scope_agent_node))
    workflow.add_node("scope_review_agent", _wrap_node_for_logging("scope_review_agent", scope_review_agent_node))
    workflow.add_node("task_planner", _wrap_node_for_logging("task_planner", task_planner_node))
    workflow.add_node("implementor", _wrap_node_for_logging("implementor", implementor_node))
    workflow.add_node("qa", _wrap_node_for_logging("qa", qa_node))
    workflow.add_node("assessor", _wrap_node_for_logging("assessor", assessor_node))
    workflow.add_node("mark_complete", _wrap_node_for_logging("mark_complete", mark_task_complete_node))
    workflow.add_node("mark_failed", _wrap_node_for_logging("mark_failed", mark_task_failed_node))
    workflow.add_node("set_active_milestone", _wrap_node_for_logging("set_active_milestone", set_active_milestone_node))
    workflow.add_node("increment_iteration", _wrap_node_for_logging("increment_iteration", increment_iteration_node))
    workflow.add_node("advance_milestone", _wrap_node_for_logging("advance_milestone", advance_milestone_node))
    workflow.add_node("increment_attempt_and_retry_implementor", _wrap_node_for_logging("increment_attempt_and_retry_implementor", make_increment_attempt_node("implementor")))
    workflow.add_node("increment_attempt_and_retry_task_planner", _wrap_node_for_logging("increment_attempt_and_retry_task_planner", make_increment_attempt_node("task_planner")))
    workflow.add_node("report", _wrap_node_for_logging("report", report_node))

    # Set entry point
    workflow.set_entry_point("initial_scope_agent")

    # === Phase 1: Scope Definition (once at start) ===
    workflow.add_conditional_edges(
        "initial_scope_agent",
        after_scope_agent,
        {
            "set_active_milestone": "set_active_milestone",
            "end": "report",
        }
    )
    # Scope review (re-plan from Assessor) uses same routing
    workflow.add_conditional_edges(
        "scope_review_agent",
        after_scope_agent,
        {
            "set_active_milestone": "set_active_milestone",
            "end": "report",
        }
    )
    workflow.add_edge("set_active_milestone", "increment_iteration")
    workflow.add_edge("increment_iteration", "task_planner")

    # === Phase 2: Task Planning Loop (TaskPlanner replaces Expander+Prioritizer+Researcher+Planner) ===

    workflow.add_conditional_edges(
        "task_planner",
        after_task_planner,
        {
            "implementor": "implementor",
            "task_planner": "task_planner",
            "assessor": "assessor",
        }
    )

    # === Phase 3: Execution Loop (TaskPlanner → Implementor → QA) ===

    # Implementor → QA (validator absorbed as pre-step; QA runs validation then LLM check)
    workflow.add_edge("implementor", "qa")

    # QA → route based on result (validation failures surface as QA failure_type=incomplete)
    workflow.add_conditional_edges(
        "qa",
        after_qa,
        {
            "mark_complete": "mark_complete",
            "increment_attempt_and_retry_task_planner": "increment_attempt_and_retry_task_planner",
            "increment_attempt_and_retry_implementor": "increment_attempt_and_retry_implementor",
            "mark_failed": "mark_failed",
        }
    )

    # Increment attempt nodes → route to appropriate agent
    workflow.add_edge("increment_attempt_and_retry_task_planner", "task_planner")
    workflow.add_edge("increment_attempt_and_retry_implementor", "implementor")

    # === Phase 4: Task Completion ===
    
    # mark_complete → task_planner (next round) or assessor (periodic)
    def after_mark_complete(state: WorkflowState) -> str:
        """Route after mark_complete: next round or periodic assessment."""
        tasks_since_last_review = state.get("tasks_since_last_review", 0)
        review_interval = state.get("review_interval", 5)
        # TaskPlanner mode: go to task_planner for next round
        if tasks_since_last_review >= review_interval:
            return "assessor"
        return "task_planner"

    workflow.add_conditional_edges(
        "mark_complete",
        after_mark_complete,
        {"task_planner": "task_planner", "assessor": "assessor"},
    )
    
    # mark_failed → Assessor
    workflow.add_edge("mark_failed", "assessor")
    
    # === Phase 5: Assessment and Routing ===
    
    # Assessor → route based on completion/gaps/milestone completion
    workflow.add_conditional_edges(
        "assessor",
        after_assessor,
        {
            "end": "report",
            "advance_milestone": "advance_milestone",
            "scope_review_agent": "scope_review_agent",
            "task_planner": "task_planner",
        }
    )

    # Report → END (runs before every exit)
    workflow.add_edge("report", END)

    # Advance milestone → task_planner (to plan next milestone's tasks)
    workflow.add_edge("advance_milestone", "task_planner")

    compiled = workflow.compile()
    logger.info("Workflow graph built and compiled")
    return compiled


# Compiled graph ready to use
graph = build_graph()
