"""QA agent - verifies that task requirements are satisfied.

Answers one question: does the implementation satisfy the task's measurable outcome?
Uses implementor summary, git diff (when in git workspace), and read/search tools as evidence.
Output: QAResult with passed, feedback, failure_type, issues.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import (
    WorkflowState,
    Task,
    TaskTree,
    QAResult,
    ImplementationResult,
)
from ..llm import planning_llm
from ..tools.git import get_diff
from ..tools.read import read_file, read_file_lines
from ..tools.search import search_files, find_files_by_name

logger = get_logger(__name__)


class QAOutput(BaseModel):
    """Structured output for QA agent."""

    passed: bool = Field(description="Whether the task requirements are met")
    feedback: str = Field(description="Detailed assessment (max 500 chars)")
    failure_type: str | None = Field(
        default=None,
        description="incomplete, wrong_approach, or plan_issue",
    )
    issues: list[str] = Field(default_factory=list, description="Specific issues found")


QA_SYSTEM_PROMPT = """
## ROLE
You are a QA agent for a software development project.

## PRIMARY OBJECTIVE
Verify that implemented changes satisfy the task's measurable outcome.

## PROCESS
1. Review the evidence (implementor summary, diff if present)
2. If needed, use read_file or search to verify specific files meet the measurable outcome
3. Decide if the outcome is satisfied or not
4. If failed, classify as: incomplete, wrong_approach, or plan_issue

## FAILURE TYPES (when passed=false)
- **incomplete**: Implementation started but missing key pieces
- **wrong_approach**: Implementation exists but doesn't solve problem correctly
- **plan_issue**: Implementation follows plan but plan was inadequate

Set passed=true only when the measurable outcome is satisfied.
"""


def _create_qa_agent():
    """Create the QA agent with read/search tools and structured output."""
    tools = [read_file, read_file_lines, search_files, find_files_by_name]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=QA_SYSTEM_PROMPT,
        response_format=QAOutput,
    )


def qa_node(state: WorkflowState) -> dict:
    """QA agent - verify task requirements are satisfied.

    Uses implementor summary and git diff (when in_git_workspace) as evidence.
    Returns state update with current_qa_result set.
    """
    logger.info("QA agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state.get("tasks", {})
    repo_root = state.get("repo_root", "")
    current_implementation_result = state.get("current_implementation_result")

    if not current_task_id:
        return {
            "error": "No current task set for QA",
            "messages": ["QA: No current task"],
        }

    if not current_implementation_result:
        task_tree = TaskTree.from_dict(tasks_dict)
        task = task_tree.tasks.get(current_task_id)
        qa_result = QAResult(
            task_id=current_task_id,
            passed=False,
            feedback="No implementation result available - implementor did not report changes",
            failure_type="incomplete",
            issues=["No implementation result available"],
        )
        if task:
            task.qa_feedback = qa_result.feedback[:500]
        return {
            "tasks": task_tree.to_dict() if task_tree else tasks_dict,
            "current_qa_result": qa_result.to_dict(),
            "messages": ["QA: Failed - no implementation result"],
        }

    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"QA: Task {current_task_id} not found"],
        }

    impl_result = ImplementationResult.from_dict(current_implementation_result)

    # Build evidence: implementor summary (always) + git diff (when in git workspace)
    evidence_parts = [
        "## Implementor report",
        f"Summary: {impl_result.result_summary}",
        f"Files modified: {', '.join(impl_result.files_modified) or '(none)'}",
    ]
    if impl_result.issues_noticed:
        evidence_parts.append(f"Issues noticed: {'; '.join(impl_result.issues_noticed)}")

    if state.get("in_git_workspace") and repo_root:
        diff_text = get_diff(repo_root, max_chars=12_000)
        if diff_text:
            evidence_parts.append("")
            evidence_parts.append("## Git diff (changes in repo)")
            evidence_parts.append(diff_text)
        else:
            evidence_parts.append("")
            evidence_parts.append("## Git diff: (no changes or not available)")

    task_desc = state.get("current_task_description") or task.description
    prompt_parts = [
        "## Task to verify",
        f"- ID: {task.id}",
        f"- Description: {task_desc}",
        f"- Measurable outcome: {task.measurable_outcome}",
        "",
        "## Evidence (what was done)",
        "\n".join(evidence_parts),
        "",
        "Use read_file or search_files if you need to verify file contents. "
        "Then decide: does the implementation satisfy the measurable outcome? "
        "If not, set passed=false and classify as incomplete, wrong_approach, or plan_issue.",
    ]

    try:
        agent = _create_qa_agent()
        messages = [HumanMessage(content="\n".join(prompt_parts))]
        result = agent.invoke({"messages": messages})
        data = result.get("structured_response") if isinstance(result, dict) else None

        if not data:
            raise ValueError("No structured output received from QA agent")

        valid_failure_types = ["incomplete", "wrong_approach", "plan_issue", None]
        failure_type = data.failure_type
        if failure_type not in valid_failure_types:
            failure_type = "incomplete"

        qa_result = QAResult(
            task_id=task.id,
            passed=data.passed,
            feedback=(data.feedback or "")[:500],
            failure_type=failure_type,
            issues=data.issues or [],
        )
        task.qa_feedback = qa_result.feedback[:500]

        logger.info(
            "QA agent completed: passed=%s, failure_type=%s",
            qa_result.passed,
            qa_result.failure_type,
        )
        return {
            "tasks": task_tree.to_dict(),
            "current_qa_result": qa_result.to_dict(),
            "messages": [
                f"QA: {'Passed' if qa_result.passed else 'Failed'} - {qa_result.failure_type or 'requirements met'}"
            ],
        }

    except Exception as e:
        logger.error("QA agent exception: %s", e, exc_info=True)
        error_msg = f"QA assessment failed for {current_task_id}: {e}"
        qa_result = QAResult(
            task_id=task.id,
            passed=False,
            feedback=error_msg,
            failure_type="incomplete",
            issues=[str(e)],
        )
        task.qa_feedback = qa_result.feedback[:500]
        return {
            "tasks": task_tree.to_dict(),
            "current_qa_result": qa_result.to_dict(),
            "error": error_msg,
            "messages": [f"QA exception: {e}"],
        }
