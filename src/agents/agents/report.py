"""Report agent - produces a narrative summary of work done before exit.

Runs at the end of the graph flow, before every exit. Produces a concise
narrative summary regardless of how the workflow ended (complete, failed, early exit).
"""

from langchain_core.messages import HumanMessage, SystemMessage

from ..logging_config import get_logger
from ..task_states import WorkflowState, TaskTree, AssessmentResult
from ..llm import planning_llm

logger = get_logger(__name__)


REPORT_SYSTEM_PROMPT = """You are a report agent. Produce a concise narrative summary of the workflow execution.

## INPUT
You will receive: remit (original request), milestones, tasks, completed/failed task IDs, status, last assessment.

## OUTPUT
A single paragraph (max 300 words) covering:
1. What was requested (remit)
2. What was accomplished (milestones reached, tasks completed)
3. What failed or was deferred (if any)
4. Overall outcome (success / partial / failed)

Be factual and concise. No speculation.
"""


def report_node(state: WorkflowState) -> dict:
    """Produce a narrative summary of work done. Runs before every exit.

    Args:
        state: Current workflow state

    Returns:
        State update with work_report set
    """
    logger.info("Report agent starting")
    remit = state.get("remit", "")
    milestones = state.get("milestones", {})
    milestone_order = state.get("milestone_order", [])
    tasks_dict = state.get("tasks", {})
    completed_task_ids = state.get("completed_task_ids", [])
    failed_task_ids = state.get("failed_task_ids", [])
    status = state.get("status", "running")
    error = state.get("error")
    last_assessment = state.get("last_assessment")

    # Build context for the LLM
    task_tree = TaskTree.from_dict(tasks_dict) if tasks_dict else TaskTree()
    stats = task_tree.get_statistics()

    completed_descriptions = []
    for tid in completed_task_ids[-10:]:  # Last 10
        task = task_tree.tasks.get(tid)
        if task:
            completed_descriptions.append(f"- {tid}: {task.description[:80]}")

    failed_descriptions = []
    for tid in failed_task_ids[-5:]:
        task = task_tree.tasks.get(tid)
        if task:
            failed_descriptions.append(f"- {tid}: {task.description[:80]}")

    milestone_summary = []
    for mid in (milestone_order or list(milestones.keys())[:5]):
        m = milestones.get(mid, {})
        st = m.get("status", "pending")
        milestone_summary.append(f"- {mid} ({st}): {m.get('description', '')[:60]}")

    assessment_notes = ""
    if last_assessment:
        try:
            assessment = AssessmentResult.from_dict(last_assessment)
            assessment_notes = assessment.assessment_notes or ""
        except Exception:
            pass

    prompt = f"""Summarize this workflow execution.

REMIT (request): {remit[:500]}

MILESTONES: {chr(10).join(milestone_summary or ['None'])}

STATS: {stats['complete']} completed, {stats['failed']} failed, {stats['total']} total tasks

COMPLETED (last few): {chr(10).join(completed_descriptions or ['None'])}

FAILED (if any): {chr(10).join(failed_descriptions or ['None'])}

STATUS: {status}
ERROR: {error or 'None'}

ASSESSMENT NOTES: {assessment_notes[:200] if assessment_notes else 'None'}

Produce a single paragraph summary (max 300 words)."""

    try:
        response = planning_llm.invoke(
            [SystemMessage(content=REPORT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        content = response.content if hasattr(response, "content") else str(response)
        report_text = (content or "").strip()[:2000]  # Cap length

        if not report_text:
            report_text = f"Workflow ended with status: {status}. {stats['complete']} tasks completed, {stats['failed']} failed."
    except Exception as e:
        logger.error("Report agent exception: %s", e, exc_info=True)
        report_text = f"Workflow ended with status: {status}. Report generation failed: {e}"

    logger.info("Report agent completed")
    return {
        "work_report": report_text,
        "messages": ["Report: generated"],
    }
