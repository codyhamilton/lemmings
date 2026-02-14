"""Implementor agent - executes implementation plans using file tools.

This agent takes the detailed implementation plan from Planner and actually
makes the file changes using tools (write_file, apply_edit, create_file).

Output: ImplementationResult with files_modified, result_summary, issues_noticed, success
"""

import os
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.subagents import ask

from ..logging_config import get_logger
from ..task_states import WorkflowState, Task, TaskTree, ImplementationResult
from ..llm import coding_llm
from ..tools.read import read_file, read_file_lines
from ..tools.edit import write_file, apply_edit, create_file
from ..tools.rag import rag_search, perform_rag_search
from ..tools.search import search_files, find_files_by_name

logger = get_logger(__name__)


class ImplementorOutput(BaseModel):
    """Structured output for implementor agent."""

    files_modified: list[str] = Field(default_factory=list, description="Files that were modified")
    result_summary: str = Field(description="Brief description of what was implemented (max 500 chars)")
    issues_noticed: list[str] = Field(default_factory=list, description="Any issues encountered")
    success: bool = Field(description="Whether most changes succeeded")


IMPLEMENTOR_SYSTEM_PROMPT = """## ROLE
You are a code implementation agent for a software development project.

## PRIMARY OBJECTIVE
Execute the implementation plan (PRP) you receive: create or modify files so that the task's outcome is achieved.

## PROCESS
1. Read existing files as needed to understand current code
2. Make changes to the files using the tools provided
3. Report on your changes

## FILE PATHS
- Use RELATIVE paths from project root
"""


def _build_implementor_messages(state: WorkflowState) -> list:
    """Build input messages for Implementor from graph state."""
    repo_root = state.get("repo_root", "")
    current_implementation_plan = state.get("current_implementation_plan", "")
    current_task_description = state.get("current_task_description", "")
    current_task_id = state.get("current_task_id")
    done_list = state.get("done_list", [])
    carry_forward = state.get("carry_forward", [])

    task = None
    tasks_dict = state.get("tasks", {})
    if current_task_id and tasks_dict:
        task_tree = TaskTree.from_dict(tasks_dict)
        task = task_tree.tasks.get(current_task_id)

    parts = ["## WORKING ENVIRONMENT", ""]
    parts.append(f"Repository root: {repo_root}")
    parts.append("All file paths are relative to this root.")
    parts.append("")

    # Project context from context.md at repo root (not agents/context.md)
    context_path = Path(repo_root) / "context.md"
    if context_path.exists():
        try:
            project_context = context_path.read_text(encoding="utf-8")
            if project_context.strip():
                parts.append("## PROJECT CONTEXT")
                parts.append(project_context[:3000])
                parts.append("")
        except Exception:
            pass

    # RAG prefetch based on task description or plan summary
    rag_query = current_task_description or current_implementation_plan[:300] or "implementation"
    if rag_query.strip():
        try:
            rag_context = perform_rag_search(
                query=rag_query.strip(),
                n_results=8,
                repo_root=Path(repo_root) if repo_root else None,
                max_tokens=3000,
            )
            if rag_context and "No relevant code found" not in rag_context and "RAG search error" not in rag_context:
                parts.append("## RELEVANT CODE (from codebase search)")
                parts.append(rag_context)
                parts.append("")
        except Exception as e:
            logger.debug("RAG prefetch for implementor failed: %s", e)

    if done_list:
        parts.append("## DONE (recently completed)")
        for i, item in enumerate(done_list[-5:], 1):
            if isinstance(item, dict):
                desc = item.get("description", item.get("task_description", str(item)))
                parts.append(f"  {i}. {str(desc)[:80]}")
            else:
                parts.append(f"  {i}. {str(item)[:100]}")
        parts.append("")

    if carry_forward:
        parts.append("## CARRY-FORWARD (lookahead)")
        for i, cf in enumerate(carry_forward[:5], 1):
            parts.append(f"  {i}. {str(cf)[:100]}")
        parts.append("")

    if task and task.attempt_count > 0:
        parts.append("## RETRY CONTEXT (address these issues)")
        if task.qa_feedback:
            parts.append(f"QA Feedback: {task.qa_feedback}")
        if task.last_failure_reason:
            parts.append(f"Failure: {task.last_failure_reason}")
        parts.append("")

    parts.append("## IMPLEMENTATION PLAN (execute this)")
    parts.append("")
    parts.append(current_implementation_plan)
    parts.append("")
    parts.append("Execute the plan using the file tools. Report the result via the structured response.")

    return [HumanMessage(content="\n".join(parts))]


def create_implementor_agent(repo_root: Path):
    """Create the implementor agent with file and search tools (no middleware).

    Matches TaskPlanner pattern: no middleware, response_format for structured output.
    """
    tools = [
        read_file,
        read_file_lines,
        write_file,
        apply_edit,
        create_file,
        rag_search,
        search_files,
        find_files_by_name,
        ask
    ]
    return create_agent(
        model=coding_llm,
        tools=tools,
        system_prompt=IMPLEMENTOR_SYSTEM_PROMPT,
        response_format=ImplementorOutput,
    )


def implementor_node(state: WorkflowState) -> dict:
    """Implementor agent - execute implementation plan for current task.

    Takes the plan from Planner and makes actual file changes using tools.

    Args:
        state: Current workflow state

    Returns:
        State update with current_implementation_result set
    """
    logger.info("Implementor agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    repo_root = state["repo_root"]
    current_implementation_plan = state.get("current_implementation_plan")

    if not current_task_id:
        return {
            "error": "No current task set for implementor",
            "messages": ["Implementor: No current task"],
        }

    if not current_implementation_plan:
        return {
            "error": "No implementation plan available for implementor",
            "messages": ["Implementor: No implementation plan found"],
        }

    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)

    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"Implementor: Task {current_task_id} not found"],
        }

    messages = _build_implementor_messages(state)

    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)

        agent = create_implementor_agent(repo_root=Path(repo_root))
        result = agent.invoke({"messages": messages})

        data = result.get("structured_response") if isinstance(result, dict) else None

        if data and isinstance(data, ImplementorOutput):
            impl_result = ImplementationResult(
                task_id=task.id,
                files_modified=data.files_modified,
                result_summary=data.result_summary[:500],
                issues_noticed=data.issues_noticed,
                success=data.success,
            )
        elif data and isinstance(data, dict):
            impl_result = ImplementationResult(
                task_id=task.id,
                files_modified=data.get("files_modified", []),
                result_summary=data.get("result_summary", "")[:500],
                issues_noticed=data.get("issues_noticed", []),
                success=data.get("success", False),
            )
        else:
            impl_result = ImplementationResult(
                task_id=task.id,
                files_modified=[],
                result_summary="No structured output from implementor agent",
                issues_noticed=["No structured output received"],
                success=False,
            )

        task.result_summary = impl_result.result_summary[:500]

        logger.info(
            "Implementor agent completed: success=%s, %s files modified",
            impl_result.success,
            len(impl_result.files_modified),
        )
        return {
            "tasks": task_tree.to_dict(),
            "current_implementation_result": impl_result.to_dict(),
            "messages": [
                f"Implementor: {'Success' if impl_result.success else 'Failed'} - {len(impl_result.files_modified)} files modified"
            ],
        }

    except Exception as e:
        logger.error("Implementor agent exception: %s", e, exc_info=True)
        error_msg = f"Implementor failed executing plan for {current_task_id}: {e}"

        task.attempt_count += 1
        task.last_failure_reason = error_msg
        task.last_failure_stage = "implementor"

        return {
            "tasks": task_tree.to_dict(),
            "current_implementation_result": None,
            "error": error_msg,
            "messages": [f"Implementor exception: {e}"],
        }
    finally:
        os.chdir(original_cwd)
