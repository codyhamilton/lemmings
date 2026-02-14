"""TaskPlanner - sliding window task planning within milestone scope.

Consolidates Expander, Prioritizer, Researcher, and Planner. Uses subagent tools
(explain_code, ask) and direct search tools. Outputs implement/skip/abort/milestone_done
with PRP plan and updated carry_forward. Retry-aware with QA feedback.
"""

import re
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import (
    WorkflowState,
    Task,
    TaskStatus,
    TaskTree,
    get_milestones_list,
    get_active_milestone_id,
)
from ..llm import planning_llm
from ..subagents import explain_code, ask
from ..tools.rag import rag_search
from ..tools.search import search_files, find_files_by_name
from ..tools.read import read_file_lines

logger = get_logger(__name__)


class TaskPlannerOutput(BaseModel):
    """Structured output from TaskPlanner."""

    action: str = Field(
        description="One of: implement, skip, abort, milestone_done",
    )
    task_description: str = Field(
        default="",
        description="Description of task being implemented (when action=implement)",
    )
    implementation_plan: str = Field(
        default="",
        description="Full PRP markdown (when action=implement)",
    )
    carry_forward: list[str] = Field(
        default_factory=list,
        description="Updated lookahead task descriptions for next round (~100 chars each)",
    )
    escalation_context: str = Field(
        default="",
        description="Why abort (when action=abort)",
    )


TASK_PLANNER_SYSTEM_PROMPT = """
## ROLE
You are a task planning agent for a software development project. You work within a milestone scope using a sliding window: think a few steps ahead, pick a bite-sized chunk, produce a detailed plan.

## PRIMARY OBJECTIVE
Each round: review done list and carry-forward, research if needed, pick the next task, produce a PRP (Pull Request Plan) or signal skip/abort/milestone_done.

## SUBAGENT TOOLS
- **explain_code(query)**: Deep codebase research. Use for "How does X work?", "What exists for Y?"
- **ask(query)**: Quick factual lookups. Use for "Does X exist?", "Where is Y defined?"

## DIRECT TOOLS
- **rag_search**: Semantic code search
- **search_files**: Regex search for symbols
- **find_files_by_name**: Find files by glob
- **read_file_lines**: Read specific line ranges

## ACTIONS
- **implement**: You have a plan. Output task_description and implementation_plan (full PRP).
- **skip**: Gap already closed, no work needed. Update carry_forward for next round.
- **abort**: Fundamental conflict (task impossible, wrong approach). Output escalation_context.
- **milestone_done**: All work for this milestone is complete.

## PROCESS PER ROUND
1. Review milestone scope, done list, carry-forward
2. Research if needed (explain_code or direct tools)
3. Update carry-forward: refine, add, or remove based on understanding
4. Pick next bite-sized chunk from carry-forward (or think ahead if empty)
5. For implement: produce detailed PRP with file paths, code snippets, line refs
6. For skip: explain why no work needed, update carry_forward
7. For abort: explain the conflict in escalation_context
8. For milestone_done: confirm all user outcomes for this milestone are achieved

## RETRY BEHAVIOUR (when last_qa_feedback is provided)
- Read the QA feedback carefully
- Decide: re-research, adjust plan, or abort
- If adjusting: produce a revised implementation_plan that addresses the feedback

## PRP FORMAT (when action=implement)
```markdown
# Implementation Plan: [Task Description]

## Task
- Outcome: [measurable outcome]

## Changes
### Create: `path/to/file.gd`
**Purpose**: [Why]
\\`\\`\\`gdscript
[code]
\\`\\`\\`

### Modify: `path/to/file.gd`
**Location**: [Where]
**Change**: [What]
\\`\\`\\`gdscript
[code]
\\`\\`\\`

## Files Summary
- Create: file1.gd, file2.gd
- Modify: file3.gd
```

## CONSTRAINTS
- Use tools to verify - don't guess
- Keep carry_forward items ~100 chars each
- PRP must include actual code snippets
- Be specific: file paths, line numbers, function names
"""


def _create_task_planner_agent():
    """Create the TaskPlanner agent with subagent and direct tools."""
    tools = [explain_code, ask, rag_search, search_files, find_files_by_name, read_file_lines]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=TASK_PLANNER_SYSTEM_PROMPT,
        response_format=TaskPlannerOutput,
    )


def _build_messages(state: WorkflowState) -> list:
    """Build input messages for TaskPlanner."""
    milestones_list = get_milestones_list(state)
    active_milestone_id = get_active_milestone_id(state)
    done_list = state.get("done_list", [])
    carry_forward = state.get("carry_forward", [])
    current_qa_result = state.get("current_qa_result")
    correction_hint = state.get("correction_hint")
    repo_root = state.get("repo_root", "")

    parts = ["## MILESTONE SCOPE", ""]

    if active_milestone_id and milestones_list:
        m = next((x for x in milestones_list if isinstance(x, dict) and x.get("id") == active_milestone_id), None)
        if m:
            parts.append(f"**Milestone**: {m.get('description', active_milestone_id)}")
            if m.get("sketch"):
                parts.append(f"**Areas**: {m['sketch']}")
        else:
            parts.append(f"**Milestone**: {active_milestone_id}")
    elif milestones_list:
        m = milestones_list[0] if isinstance(milestones_list[0], dict) else {}
        parts.append(f"**Milestone**: {m.get('description', 'Unknown')}")
        if m.get("sketch"):
            parts.append(f"**Areas**: {m['sketch']}")
    else:
        parts.append("(No active milestone - check state)")
    parts.append("")

    if done_list:
        parts.append("## DONE (completed this milestone)")
        for i, item in enumerate(done_list[-7:], 1):  # Last 7
            if isinstance(item, dict):
                desc = item.get("description", item.get("task_description", str(item)))
                result = item.get("result", item.get("result_summary", ""))
                parts.append(f"  {i}. {str(desc)[:80]} → {str(result)[:60]}")
            else:
                parts.append(f"  {i}. {str(item)[:120]}")
        parts.append("")

    if carry_forward:
        parts.append("## CARRY-FORWARD (lookahead)")
        for i, cf in enumerate(carry_forward, 1):
            parts.append(f"  {i}. {str(cf)[:100]}")
        parts.append("")

    if current_qa_result and isinstance(current_qa_result, dict):
        feedback = current_qa_result.get("feedback", "")
        passed = current_qa_result.get("passed", False)
        if not passed and feedback:
            parts.append("## RETRY CONTEXT (QA failed - address this)")
            parts.append(feedback[:500])
            parts.append("")
    if correction_hint:
        parts.append("## CORRECTION HINT (from Assessor)")
        parts.append(correction_hint[:300])
        parts.append("")

    parts.append("## INSTRUCTIONS")
    parts.append(f"Repository root: {repo_root}")
    parts.append("")
    parts.append(
        "Review the scope, done list, and carry-forward. Research if needed. "
        "Pick the next task and output: action, and (when implement) task_description + implementation_plan. "
        "Update carry_forward for the next round."
    )

    return [HumanMessage(content="\n".join(parts))]


def _extract_plan_from_output(content: str) -> str:
    """Extract PRP markdown from agent output."""
    md_match = re.search(r"```markdown\s*(.*?)\s*```", content, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()
    plan_match = re.search(r"(# Implementation Plan:.*)", content, re.DOTALL)
    if plan_match:
        return plan_match.group(1).strip()
    return content.strip()


def _create_synthetic_task(
    task_description: str,
    implementation_plan: str,
    active_milestone_id: str | None,
    iteration: int,
) -> Task:
    """Create a synthetic task for the implementor (sliding window mode)."""
    import uuid
    task_id = f"task_tp_{uuid.uuid4().hex[:8]}"
    return Task(
        id=task_id,
        description=task_description[:500],
        measurable_outcome=task_description[:200],
        status=TaskStatus.READY,
        milestone_id=active_milestone_id,
        created_by="task_planner",
        created_at_iteration=iteration,
    )


def task_planner_node(state: WorkflowState) -> dict:
    """TaskPlanner - plan next task using sliding window.

    Replaces Expander, Prioritizer, Researcher, Planner. Outputs action and
    (when implement) creates synthetic task + implementation plan.

    Args:
        state: Current workflow state

    Returns:
        State update with task_planner_action, and when implement: task, plan, etc.
    """
    logger.info("TaskPlanner starting")
    active_milestone_id = get_active_milestone_id(state)
    milestones_list = get_milestones_list(state)
    tasks_dict = state.get("tasks", {})
    iteration = state.get("iteration", 0)

    if not active_milestone_id or not milestones_list:
        return {
            "task_planner_action": "abort",
            "escalation_context": "No active milestone for TaskPlanner",
            "error": "No active milestone",
            "messages": ["TaskPlanner: No active milestone"],
        }

    messages = _build_messages(state)

    try:
        agent = _create_task_planner_agent()
        result = agent.invoke({"messages": messages})

        data = result.get("structured_response") if isinstance(result, dict) else None
        if not data:
            raise ValueError("No structured output from TaskPlanner")

        if isinstance(data, TaskPlannerOutput):
            out = data
        else:
            action = str(data.get("action", "implement")).lower()
            if action not in ("implement", "skip", "abort", "milestone_done"):
                action = "implement"
            out = TaskPlannerOutput(
                action=action,
                task_description=str(data.get("task_description", ""))[:500],
                implementation_plan=str(data.get("implementation_plan", "")),
                carry_forward=data.get("carry_forward", []) or [],
                escalation_context=str(data.get("escalation_context", ""))[:500],
            )

        # Extract plan from markdown if needed
        impl_plan = out.implementation_plan
        if impl_plan and "```" in impl_plan:
            impl_plan = _extract_plan_from_output(impl_plan)

        update = {
            "task_planner_action": out.action,
            "carry_forward": out.carry_forward[:10],
            "correction_hint": None,  # Consumed
            "current_qa_result": None,  # Consumed
        }

        if out.action == "implement":
            if not impl_plan or len(impl_plan) < 100:
                logger.warning("TaskPlanner implement but plan too short, treating as skip")
                update["task_planner_action"] = "skip"
                update["carry_forward"] = out.carry_forward
                return {**update, "messages": ["TaskPlanner: Plan too short, skipping"]}

            task = _create_synthetic_task(
                out.task_description or "Implementation task",
                impl_plan,
                active_milestone_id,
                iteration,
            )
            task_tree = TaskTree.from_dict(tasks_dict)
            task_tree.tasks[task.id] = task

            update.update({
                "tasks": task_tree.to_dict(),
                "current_task_id": task.id,
                "current_implementation_plan": impl_plan,
                "current_task_description": out.task_description,
            })

        elif out.action == "abort":
            update["escalation_context"] = out.escalation_context or "Task aborted by TaskPlanner"
            update["error"] = update["escalation_context"]

        elif out.action == "skip":
            # Add to done_list so we don't re-pick
            done_list = state.get("done_list", [])
            done_list = list(done_list)
            done_list.append({
                "description": "Skipped (gap closed)",
                "result": "No work needed",
            })
            update["done_list"] = done_list

        logger.info("TaskPlanner completed: action=%s", out.action)
        update["messages"] = [f"TaskPlanner: {out.action}"]

        return update

    except Exception as e:
        logger.error("TaskPlanner exception: %s", e, exc_info=True)
        return {
            "task_planner_action": "abort",
            "escalation_context": str(e),
            "error": f"TaskPlanner failed: {e}",
            "messages": [f"TaskPlanner exception: {e}"],
        }
