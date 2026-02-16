"""Assessor agent - determines if workflow is complete or has uncovered gaps.

This agent reviews completed tasks, QA feedback, and the task tree to determine:
1. Are there uncovered gaps that need new tasks?
2. Is the remit fully satisfied?
3. Is the workflow stable (no new tasks this iteration)?

The Assessor acts as the gate between the execution loop and the expansion loop.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import (
    WorkflowState,
    TaskStatus,
    TaskTree,
    AssessmentResult,
    get_milestones_list,
    get_active_milestone_id,
    get_active_milestone_index,
)
from ..llm import planning_llm

logger = get_logger(__name__)


def _create_assessor_agent():
    """Create the Assessor agent with create_agent (no tools, structured output)."""
    return create_agent(
        model=planning_llm,
        tools=[],
        system_prompt=ASSESSOR_SYSTEM_PROMPT,
        response_format=AssessorOutput,
    )


class AssessorOutput(BaseModel):
    """Structured output for assessor agent."""

    uncovered_gaps: list[str] = Field(default_factory=list, description="Gaps not covered by open tasks")
    is_complete: bool = Field(description="Whether the remit is fully satisfied")
    stability_check: bool = Field(description="Whether the workflow is stable")
    milestone_complete: bool = Field(description="Whether the active milestone is complete")
    assessment_notes: str = Field(default="", description="Summary of assessment (max 500 chars)")
    correction_hint: str | None = Field(
        default=None,
        description="Minor drift hint for TaskPlanner when work is slightly off but not a full gap (max 200 chars)",
    )
    escalate_to_scope: bool = Field(
        default=False,
        description="Major divergence - wrong direction, re-plan via ScopeAgent",
    )


ASSESSOR_SYSTEM_PROMPT = """
## ROLE
You are an assessment agent for a software development project.

## PRIMARY OBJECTIVE
Review completed tasks within the active milestone and determine: gaps, milestone completion, overall completion, and stability.

## PROCESS
1. Review QA feedback from completed tasks - any loose ends or missing pieces?
2. Check if identified gaps are already covered by open tasks
3. Determine if active milestone is complete (all tasks in final states)
4. Assess stability (no new tasks created this iteration)
5. Determine if overall remit is fully satisfied

## GAP IDENTIFICATION
Look for:
- QA feedback mentioning missing pieces ("incomplete", "needs X", "missing Y")
- Tasks marked complete but QA noted gaps
- Logical dependencies not created as tasks
- Missing integration points
- Unhandled edge cases

## COVERAGE CHECK
Before reporting a gap as "uncovered":
- Is there an open task that addresses this gap?
- Check task descriptions, tags, and measurable outcomes
- Only report truly uncovered gaps

## COMPLETION CRITERIA

### Milestone Complete
- All tasks in milestone are COMPLETE, FAILED, or DEFERRED
- No uncovered gaps remain within milestone scope

### Overall Complete
- All milestones are complete
- All tasks across all milestones are COMPLETE, FAILED, or DEFERRED
- No uncovered gaps remain
- Remit scope is satisfied

### Stability
- No new tasks were created this iteration
- All tasks are in final states
- No pending expansion needed

## MINOR DRIFT (correction_hint)
When work is slightly off-track but not a full uncovered gap (e.g. implementation style drift,
minor missing polish, small refinement needed), set correction_hint to a brief actionable hint
for the TaskPlanner. Leave null if no minor drift.

## MAJOR DIVERGENCE (escalate_to_scope)
When the work is fundamentally going the wrong direction - wrong interpretation of remit,
wrong milestone breakdown, strategic misalignment - set escalate_to_scope=true. This triggers
a full re-plan via ScopeAgent. Use sparingly.

## OUTPUT FORMAT
```json
{
    "uncovered_gaps": ["Gap 1", "Gap 2"],
    "is_complete": true|false,
    "stability_check": true|false,
    "milestone_complete": true|false,
    "assessment_notes": "Summary (max 500 chars)",
    "correction_hint": null|"Brief hint for TaskPlanner (max 200 chars)"
}
```

## CONSTRAINTS
- Only report gaps NOT covered by open tasks in the active milestone
- Be specific about what's missing within milestone scope
- If everything is covered, return empty uncovered_gaps array
- milestone_complete=true only if all milestone tasks final AND no gaps
- is_complete=true only if remit satisfied across ALL milestones AND no gaps
- stability_check=true only if no new tasks this iteration AND all tasks final
"""


def assessor_node(state: WorkflowState) -> dict:
    """Assessor agent - determine if workflow is complete or has gaps.
    
    This agent reviews completed tasks, QA feedback, and the task tree to
    identify uncovered gaps and assess completion status.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with last_assessment set
    """
    logger.info("Assessor agent starting (periodic review)")
    remit = state.get("remit", "")
    tasks_dict = state.get("tasks", {})
    active_milestone_id = get_active_milestone_id(state)
    milestones_list = get_milestones_list(state)
    iteration = state.get("iteration", 0)
    completed_task_ids = state.get("completed_task_ids", [])
    tasks_created_this_iteration = state.get("tasks_created_this_iteration", 0)
    done_list = state.get("done_list", [])
    tasks_since_last_review = state.get("tasks_since_last_review", 0)

    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    
    # Check if we have an active milestone
    if not active_milestone_id:
        assessment_result = AssessmentResult(
            uncovered_gaps=[],
            is_complete=False,
            stability_check=False,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes="No active milestone",
            correction_hint=None,
        )
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": False,
            "tasks_since_last_review": 0,
            "messages": ["Assessor: No active milestone"],
        }
    
    # Get milestone info
    milestone_desc = active_milestone_id
    for m in milestones_list:
        if isinstance(m, dict) and m.get("id") == active_milestone_id:
            milestone_desc = m.get("description", active_milestone_id)
            break
    
    # Get tasks in active milestone
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
    
    # Check milestone completion (deterministic check)
    is_milestone_complete = task_tree.is_milestone_complete(active_milestone_id)
    
    # Get completed tasks with QA feedback (from active milestone)
    completed_tasks = [
        task_tree.tasks[tid] 
        for tid in completed_task_ids 
        if tid in task_tree.tasks and task_tree.tasks[tid].milestone_id == active_milestone_id
    ]
    
    # Get open tasks (from active milestone only)
    open_tasks = [
        t for t in milestone_tasks
        if t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS)
    ]
    
    # Deterministic fast-path: if READY tasks exist, skip LLM - route to prioritizer
    ready_tasks = task_tree.get_ready_tasks(milestone_id=active_milestone_id)
    if len(ready_tasks) > 0:
        logger.info("Assessor fast-path: %s READY tasks in %s, skipping LLM", len(ready_tasks), active_milestone_id)
        assessment_result = AssessmentResult(
            uncovered_gaps=[],
            is_complete=False,
            stability_check=True,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes="Fast-path: READY tasks remain, continue to prioritizer",
            correction_hint=None,
        )
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": False,
            "tasks_since_last_review": 0,
            "messages": [f"Assessor fast-path: {len(ready_tasks)} READY tasks, skipping LLM"],
        }
    
    # Determine next milestone ID if current is complete
    next_milestone_id = None
    if is_milestone_complete:
        current_index = get_active_milestone_index(state)
        if current_index >= 0 and current_index + 1 < len(milestones_list):
            next_m = milestones_list[current_index + 1]
            next_milestone_id = next_m.get("id") if isinstance(next_m, dict) else None
    
    # Build the assessment prompt
    prompt_parts = [
        f"Remit (scope boundary): {remit}",
        "",
        f"Active Milestone: {active_milestone_id}",
        f"Milestone Description: {milestone_desc}",
        f"Tasks since last review: {tasks_since_last_review}",
        "",
        "## Recently completed tasks (with QA feedback)",
    ]

    # TaskPlanner mode: include done_list for sliding-window context
    if done_list:
        prompt_parts.append("(Recent completions from done_list)")
        for i, item in enumerate(done_list[-10:], 1):
            if isinstance(item, dict):
                desc = item.get("description", item.get("task_description", str(item)))
                result = item.get("result", item.get("result_summary", ""))
                prompt_parts.append(f"  {i}. {str(desc)[:80]} → {str(result)[:60]}")
            else:
                prompt_parts.append(f"  {i}. {str(item)[:120]}")
        prompt_parts.append("")

    if completed_tasks:
        for task in completed_tasks[-10:]:  # Last 10 completed
            prompt_parts.append(f"\n{task.id}: {task.description}")
            prompt_parts.append(f"  Outcome: {task.measurable_outcome}")
            if task.qa_feedback:
                prompt_parts.append(f"  QA Feedback: {task.qa_feedback}")
            if task.result_summary:
                prompt_parts.append(f"  Result: {task.result_summary[:150]}")
    else:
        prompt_parts.append("No tasks completed yet")
    
    prompt_parts.extend([
        "",
        "## Open tasks in active milestone (what is planned)",
    ])
    
    if open_tasks:
        for task in open_tasks[:20]:  # Limit to 20
            summary = task_tree.get_task_summary(task.id, max_chars=200)
            prompt_parts.append(f"{summary}")
    else:
        prompt_parts.append("No open tasks")
    
    prompt_parts.extend([
        "",
        "## Milestone statistics",
        f"Milestone tasks: {milestone_stats['total']} total",
        f"Complete: {milestone_stats['complete']}, Ready: {milestone_stats['ready']}, Pending: {milestone_stats['pending']}, Failed: {milestone_stats['failed']}, Blocked: {milestone_stats['blocked']}",
        f"Milestone complete (deterministic check): {is_milestone_complete}",
        "",
        "## Overall statistics",
        f"Total tasks: {stats['total']}, Complete: {stats['complete']}, Ready: {stats['ready']}, Pending: {stats['pending']}, Failed: {stats['failed']}, Blocked: {stats['blocked']}",
        f"Tasks created this iteration: {tasks_created_this_iteration}",
        "",
        "## Instructions",
        "1. Review QA feedback for any gaps or missing pieces WITHIN THE ACTIVE MILESTONE",
        "2. Check if gaps are covered by open tasks in the milestone",
        "3. Determine if active milestone is complete (all tasks in final states, no gaps)",
        "4. Determine if overall remit is fully satisfied (all milestones complete)",
        "5. Assess stability (no new tasks this iteration + all tasks final = stable)",
        "6. If minor drift (work slightly off but not a gap), set correction_hint for TaskPlanner",
        "7. Output JSON with uncovered_gaps, is_complete, stability_check, milestone_complete, assessment_notes, correction_hint",
        "",
        "Remember:",
        "- Only report gaps NOT covered by open tasks in the active milestone",
        "- milestone_complete=true if all milestone tasks are final AND no gaps in milestone",
        "- is_complete=true only if remit satisfied across ALL milestones AND no gaps",
        "- stability_check=true if no new tasks this iteration AND all tasks final",
    ])
    
    try:
        agent = _create_assessor_agent()
        messages = [HumanMessage(content="\n".join(prompt_parts))]
        result = agent.invoke({"messages": messages})
        data = result.get("structured_response") if isinstance(result, dict) else None

        if not data:
            raise ValueError("No structured output received from assessor")
        
        # Use deterministic milestone completion check (override LLM if needed)
        milestone_complete = is_milestone_complete and not data.uncovered_gaps
        
        # Create AssessmentResult
        correction_hint = (data.correction_hint or "").strip() or None
        if correction_hint and len(correction_hint) > 200:
            correction_hint = correction_hint[:200]

        assessment_result = AssessmentResult(
            uncovered_gaps=data.uncovered_gaps,
            is_complete=data.is_complete,
            stability_check=data.stability_check,
            milestone_complete=milestone_complete,
            next_milestone_id=next_milestone_id if milestone_complete else None,
            assessment_notes=(data.assessment_notes or "")[:500],
            correction_hint=correction_hint,
            escalate_to_scope=getattr(data, "escalate_to_scope", False),
        )

        logger.info(
            "Assessor agent completed: %s gaps, complete=%s, stable=%s, correction_hint=%s",
            len(assessment_result.uncovered_gaps),
            assessment_result.is_complete,
            assessment_result.stability_check,
            bool(assessment_result.correction_hint),
        )
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": assessment_result.stability_check,
            "correction_hint": assessment_result.correction_hint,
            "tasks_since_last_review": 0,  # Reset for next periodic review
            "messages": [
                f"Assessor: {len(assessment_result.uncovered_gaps)} gaps, "
                f"complete={assessment_result.is_complete}, "
                f"stable={assessment_result.stability_check}"
            ],
        }
        
    except Exception as e:
        logger.error("Assessor agent exception: %s", e, exc_info=True)
        error_msg = f"Assessor failed during assessment: {e}"
        
        # Create failed assessment (assume not complete, not stable)
        assessment_result = AssessmentResult(
            uncovered_gaps=[],
            is_complete=False,
            stability_check=False,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes=error_msg,
            correction_hint=None,
        )

        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": False,
            "tasks_since_last_review": 0,
            "error": error_msg,
            "messages": [f"Assessor exception: {e}"],
        }
