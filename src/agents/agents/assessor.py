"""Assessor agent - determines if workflow is complete or has uncovered gaps.

This agent reviews completed tasks, QA feedback, and the task tree to determine:
1. Are there uncovered gaps that need new tasks?
2. Is the remit fully satisfied?
3. Is the workflow stable (no new tasks this iteration)?

The Assessor acts as the gate between the execution loop and the expansion loop.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from ..state import WorkflowState, Task, TaskStatus, TaskTree, AssessmentResult, MilestoneStatus
from ..llm import planning_llm
from ..normaliser import normalize_agent_output


ASSESSOR_SYSTEM_PROMPT = """
## ROLE
You are an assessment agent for a game development project using Godot in GDScript.

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

## OUTPUT FORMAT
```json
{
    "uncovered_gaps": [
        "Gap description 1 - what's missing and not covered in active milestone",
        "Gap description 2 - another missing piece"
    ],
    "is_complete": true|false,
    "stability_check": true|false,
    "milestone_complete": true|false,
    "assessment_notes": "Summary of assessment (max 500 chars)"
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
    remit = state["remit"]
    tasks_dict = state["tasks"]
    active_milestone_id = state.get("active_milestone_id")
    milestones = state.get("milestones", {})
    milestone_order = state.get("milestone_order", [])
    iteration = state.get("iteration", 0)
    completed_task_ids = state.get("completed_task_ids", [])
    tasks_created_this_iteration = state.get("tasks_created_this_iteration", 0)
    
    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    
    print("\n" + "="*70)
    print("📊 ASSESSOR AGENT")
    print("="*70)
    print(f"Iteration: {iteration}")
    print(f"Remit: {remit[:100]}...")
    print()
    
    # Check if we have an active milestone
    if not active_milestone_id:
        print("⚠️  No active milestone - cannot assess")
        assessment_result = AssessmentResult(
            uncovered_gaps=[],
            is_complete=False,
            stability_check=False,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes="No active milestone",
        )
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": False,
            "messages": ["Assessor: No active milestone"],
        }
    
    # Get milestone info
    milestone_dict = milestones.get(active_milestone_id)
    milestone_desc = milestone_dict.get("description", active_milestone_id) if milestone_dict else active_milestone_id
    print(f"Active Milestone: {active_milestone_id}")
    print(f"  {milestone_desc}")
    print()
    
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
    print(f"Milestone tasks: {milestone_stats['ready']} ready, {milestone_stats['pending']} pending, "
          f"{milestone_stats['complete']} complete, {milestone_stats['failed']} failed")
    print(f"Overall: {stats['complete']} complete, {stats['ready']} ready, "
          f"{stats['pending']} pending, {stats['failed']} failed, {stats['blocked']} blocked")
    print(f"Tasks created this iteration: {tasks_created_this_iteration}")
    print()
    
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
    
    # Determine next milestone ID if current is complete
    next_milestone_id = None
    if is_milestone_complete:
        current_index = milestone_order.index(active_milestone_id) if active_milestone_id in milestone_order else -1
        if current_index >= 0 and current_index + 1 < len(milestone_order):
            next_milestone_id = milestone_order[current_index + 1]
    
    # Build the assessment prompt
    prompt_parts = [
        f"Remit (scope boundary): {remit}",
        "",
        f"Active Milestone: {active_milestone_id}",
        f"Milestone Description: {milestone_desc}",
        "",
        "="*70,
        "RECENTLY COMPLETED TASKS IN ACTIVE MILESTONE (with QA feedback):",
        "="*70,
    ]
    
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
        "="*70,
        "OPEN TASKS IN ACTIVE MILESTONE (what is planned):",
        "="*70,
    ])
    
    if open_tasks:
        for task in open_tasks[:20]:  # Limit to 20
            summary = task_tree.get_task_summary(task.id, max_chars=200)
            prompt_parts.append(f"{summary}")
    else:
        prompt_parts.append("No open tasks")
    
    prompt_parts.extend([
        "",
        "="*70,
        "MILESTONE STATISTICS:",
        "="*70,
        f"Milestone tasks: {milestone_stats['total']} total",
        f"Complete: {milestone_stats['complete']}",
        f"Ready: {milestone_stats['ready']}",
        f"Pending: {milestone_stats['pending']}",
        f"Failed: {milestone_stats['failed']}",
        f"Blocked: {milestone_stats['blocked']}",
        f"Milestone complete (deterministic check): {is_milestone_complete}",
        "",
        "="*70,
        "OVERALL STATISTICS:",
        "="*70,
        f"Total tasks: {stats['total']}",
        f"Complete: {stats['complete']}",
        f"Ready: {stats['ready']}",
        f"Pending: {stats['pending']}",
        f"Failed: {stats['failed']}",
        f"Blocked: {stats['blocked']}",
        f"Tasks created this iteration: {tasks_created_this_iteration}",
        "",
        "="*70,
        "",
        "INSTRUCTIONS:",
        "1. Review QA feedback for any gaps or missing pieces WITHIN THE ACTIVE MILESTONE",
        "2. Check if gaps are covered by open tasks in the milestone",
        "3. Determine if active milestone is complete (all tasks in final states, no gaps)",
        "4. Determine if overall remit is fully satisfied (all milestones complete)",
        "5. Assess stability (no new tasks this iteration + all tasks final = stable)",
        "6. Output JSON with uncovered_gaps, is_complete, stability_check, milestone_complete, assessment_notes",
        "",
        "Remember:",
        "- Only report gaps NOT covered by open tasks in the active milestone",
        "- milestone_complete=true if all milestone tasks are final AND no gaps in milestone",
        "- is_complete=true only if remit satisfied across ALL milestones AND no gaps",
        "- stability_check=true if no new tasks this iteration AND all tasks final",
    ])
    
    try:
        print("💭 Starting assessment...\n")
        
        # Create a simple LLM call (no agent needed for assessment)
        messages = [
            SystemMessage(content=ASSESSOR_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(prompt_parts)),
        ]
        
        # Stream response
        content = ""
        for chunk in planning_llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                content += chunk.content
        print()
        
        if not content or not content.strip():
            raise ValueError("No content received from assessor")
        
        # Define expected schema for assessor output
        assessor_schema = {
            "uncovered_gaps": {
                "type": list,
                "required": False,
                "default": [],
            },
            "is_complete": {
                "type": bool,
                "required": True,
            },
            "stability_check": {
                "type": bool,
                "required": True,
            },
            "milestone_complete": {
                "type": bool,
                "required": True,
            },
            "assessment_notes": {
                "type": str,
                "required": False,
                "max_length": 500,
                "default": "",
            },
        }
        
        # Use normaliser for JSON extraction and parsing
        norm_result = normalize_agent_output(
            content,
            assessor_schema,
            use_llm_summarization=False  # Simple truncation
        )
        
        if not norm_result.success:
            raise ValueError(f"Normalisation failed: {norm_result.error}")
        
        data = norm_result.data
        
        # Use deterministic milestone completion check (override LLM if needed)
        milestone_complete = is_milestone_complete and not data.get("uncovered_gaps", [])
        
        # Create AssessmentResult
        assessment_result = AssessmentResult(
            uncovered_gaps=data.get("uncovered_gaps", []),
            is_complete=data.get("is_complete", False),
            stability_check=data.get("stability_check", False),
            milestone_complete=milestone_complete,
            next_milestone_id=next_milestone_id if milestone_complete else None,
            assessment_notes=data.get("assessment_notes", ""),
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print("ASSESSMENT RESULT:")
        print(f"{'='*70}")
        
        if assessment_result.uncovered_gaps:
            print(f"⚠️  {len(assessment_result.uncovered_gaps)} uncovered gaps:")
            for gap in assessment_result.uncovered_gaps[:10]:
                print(f"  - {gap}")
            if len(assessment_result.uncovered_gaps) > 10:
                print(f"  ... and {len(assessment_result.uncovered_gaps) - 10} more")
        else:
            print("✓ No uncovered gaps")
        
        print(f"\nMilestone Completion: {'✅ Complete' if assessment_result.milestone_complete else '⏸️  Not complete'}")
        if assessment_result.milestone_complete and assessment_result.next_milestone_id:
            print(f"Next Milestone: {assessment_result.next_milestone_id}")
        print(f"Overall Completion: {'✅ Complete' if assessment_result.is_complete else '⏸️  Not complete'}")
        print(f"Stability: {'✅ Stable' if assessment_result.stability_check else '🔄 Not stable'}")
        
        if assessment_result.assessment_notes:
            print(f"\nNotes:\n  {assessment_result.assessment_notes}")
        
        print("="*70)
        
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": assessment_result.stability_check,
            "messages": [
                f"Assessor: {len(assessment_result.uncovered_gaps)} gaps, "
                f"complete={assessment_result.is_complete}, "
                f"stable={assessment_result.stability_check}"
            ],
        }
        
    except Exception as e:
        error_msg = f"Assessor error: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Create failed assessment (assume not complete, not stable)
        assessment_result = AssessmentResult(
            uncovered_gaps=[],
            is_complete=False,
            stability_check=False,
            milestone_complete=False,
            next_milestone_id=None,
            assessment_notes=f"Assessment failed: {e}",
        )
        
        return {
            "last_assessment": assessment_result.to_dict(),
            "is_stable": False,
            "error": error_msg,
            "messages": [f"Assessor exception: {e}"],
        }
