"""QA agent - verifies that task requirements are satisfied.

Absorbs the former Validator as a pre-step: first verifies that reported
file changes exist on disk, then uses an LLM to assess whether the
implementation meets the task's measurable outcome.

Output: QAResult with passed, feedback, failure_type, issues
"""

import os
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import (
    WorkflowState,
    Task,
    TaskTree,
    QAResult,
    ValidationResult,
    ImplementationResult,
)
from ..llm import planning_llm
from ..tools.read import read_file, read_file_lines


class QAOutput(BaseModel):
    """Structured output for QA agent."""

    passed: bool = Field(description="Whether the task requirements are met")
    feedback: str = Field(description="Detailed assessment (max 500 chars)")
    failure_type: str | None = Field(default=None, description="incomplete, wrong_approach, or plan_issue")
    issues: list[str] = Field(default_factory=list, description="Specific issues found")

logger = get_logger(__name__)


QA_SYSTEM_PROMPT = """
## ROLE
You are a QA agent for a software development project.

## PRIMARY OBJECTIVE
Verify that implemented changes satisfy the task's measurable outcome.

## PROCESS
1. Compare implementation against task's measurable outcome
2. Check if implementation is complete or has missing pieces
3. Determine if code works correctly for intended purpose
4. If failed, classify the failure type: incomplete, wrong_approach, or plan_issue

## ASSESSMENT CRITERIA
- Be thorough 
- Focus on functional requirements
- Set passed=true if measurable outcome is satisfied

## FAILURE TYPES (when passed=false)
- **incomplete**: Implementation started but missing key pieces
- **wrong_approach**: Implementation exists but doesn't solve problem correctly
- **plan_issue**: Implementation follows plan but plan was inadequate
"""


# Max total chars for "ACTUAL IMPLEMENTATION" context fed to QA (avoid huge prompts)
QA_IMPLEMENTATION_CONTEXT_MAX_CHARS = 8192


def _build_implementation_context(
    repo_root: str | Path,
    file_paths: list[str],
    max_chars: int = QA_IMPLEMENTATION_CONTEXT_MAX_CHARS,
    max_lines_per_file: int = 200,
) -> str:
    """Build a single context string for modified files, with a total size cap.

    Files are read from disk (relative to repo_root). If adding content would
    exceed max_chars, remaining files are summarized as diff-stat only (path + line count).

    Returns:
        A string to embed in the QA prompt under "ACTUAL IMPLEMENTATION".
    """
    repo_root = Path(repo_root)
    sections: list[str] = []
    used = 0

    for i, file_path in enumerate(file_paths):
        full_path = repo_root / file_path
        if not full_path.exists() or not full_path.is_file():
            sections.append(f"\n--- {file_path} ---\n(not found or not a file)")
            used += len(sections[-1])
            continue

        try:
            raw = full_path.read_text(encoding="utf-8")
            lines = raw.splitlines()
            line_count = len(lines)
        except Exception as e:
            sections.append(f"\n--- {file_path} ---\nError reading: {e}")
            used += len(sections[-1])
            continue

        # If we're over budget, emit diff-stat for this and remaining files only
        if used >= max_chars:
            sections.append(f"\n--- {file_path} --- ({line_count} lines, content omitted)")
            used += len(sections[-1])
            for rest in file_paths[i + 1 :]:
                try:
                    rp = repo_root / rest
                    lc = len(rp.read_text(encoding="utf-8").splitlines()) if rp.exists() and rp.is_file() else 0
                except Exception:
                    lc = 0
                sections.append(f"\n--- {rest} --- ({lc} lines, content omitted)")
                used += len(sections[-1])
            break

        # Truncate to max_lines_per_file for this file
        if line_count > max_lines_per_file:
            content = "\n".join(lines[:max_lines_per_file])
            content += f"\n... (truncated, {line_count} lines total)"
        else:
            content = raw.rstrip() if raw else "(empty)"

        block = f"\n--- {file_path} ---\n{content}"
        if used + len(block) > max_chars:
            # This file would exceed budget: emit diff-stat for it and rest
            sections.append(f"\n--- {file_path} --- ({line_count} lines, content omitted)")
            used += len(sections[-1])
            for rest in file_paths[i + 1 :]:
                try:
                    rp = repo_root / rest
                    lc = len(rp.read_text(encoding="utf-8").splitlines()) if rp.exists() and rp.is_file() else 0
                except Exception:
                    lc = 0
                sections.append(f"\n--- {rest} --- ({lc} lines, content omitted)")
            break

        sections.append(block)
        used += len(block)

    return "".join(sections) if sections else "(no file content)"


def _run_validation_step(state: WorkflowState) -> ValidationResult | None:
    """Run validator logic (file existence check) as pre-step.

    Returns ValidationResult if we have implementation result to validate,
    else None (caller should handle missing data).
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state.get("tasks", {})
    repo_root = state.get("repo_root")
    current_implementation_result = state.get("current_implementation_result")

    if not current_task_id or not current_implementation_result:
        return None

    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    if not task:
        return None

    impl_result = ImplementationResult.from_dict(current_implementation_result)
    if not impl_result.files_modified:
        return ValidationResult(
            task_id=task.id,
            files_verified=[],
            files_missing=[],
            validation_passed=False,
            validation_issues=["No files were reported as modified by implementor"],
        )

    files_verified = []
    files_missing = []
    validation_issues = []
    original_cwd = os.getcwd()

    try:
        os.chdir(repo_root)
        for file_path in impl_result.files_modified:
            full_path = Path(repo_root) / file_path
            if not full_path.exists():
                files_missing.append(file_path)
                validation_issues.append(f"File does not exist: {file_path}")
                continue
            if not full_path.is_file():
                files_missing.append(file_path)
                validation_issues.append(f"Not a file (directory?): {file_path}")
                continue
            try:
                stat = full_path.stat()
                if stat.st_size == 0:
                    validation_issues.append(f"File is empty: {file_path}")
                    files_verified.append(file_path)
                    continue
            except Exception as e:
                validation_issues.append(f"Cannot stat file: {file_path} ({e})")
                files_missing.append(file_path)
                continue
            try:
                with full_path.open("r", encoding="utf-8") as f:
                    f.read(100)
                files_verified.append(file_path)
            except UnicodeDecodeError:
                validation_issues.append(f"File is binary (not text): {file_path}")
                files_verified.append(file_path)
            except Exception as e:
                validation_issues.append(f"Cannot read file: {file_path} ({e})")
                files_missing.append(file_path)
    finally:
        os.chdir(original_cwd)

    validation_passed = len(files_missing) == 0
    return ValidationResult(
        task_id=task.id,
        files_verified=files_verified,
        files_missing=files_missing,
        validation_passed=validation_passed,
        validation_issues=validation_issues,
    )


@tool
def read_modified_file(path: str, max_lines: int = 100) -> str:
    """Read a modified file to verify implementation.
    
    Args:
        path: Relative path to file
        max_lines: Maximum lines to read (default 100)
    
    Returns:
        File content or error message
    """
    try:
        content = read_file.invoke({"path": path})
        
        # Limit to max_lines
        lines = content.split('\n')
        if len(lines) > max_lines:
            content = '\n'.join(lines[:max_lines])
            content += f"\n... (truncated - {len(lines)} total lines)"
        
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


def qa_node(state: WorkflowState) -> dict:
    """QA agent - verify task requirements are satisfied.

    Runs validation (file existence check) as pre-step, then uses an LLM
    to assess whether the implementation meets the task's measurable outcome.

    Args:
        state: Current workflow state

    Returns:
        State update with current_qa_result set
    """
    logger.info("QA agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state.get("tasks", {})
    repo_root = state.get("repo_root", "")
    current_implementation_plan = state.get("current_implementation_plan")
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
            "tasks": task_tree.to_dict() if task else tasks_dict,
            "current_qa_result": qa_result.to_dict(),
            "messages": ["QA: Failed - no implementation result"],
        }

    # Run validation (absorbed from former Validator agent)
    validation_result = _run_validation_step(state)
    if validation_result is None:
        return {
            "error": "Validation step could not run",
            "messages": ["QA: Validation step failed"],
        }

    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)

    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"QA: Task {current_task_id} not found"],
        }

    # If validation failed, QA automatically fails
    if not validation_result.validation_passed:
        qa_result = QAResult(
            task_id=task.id,
            passed=False,
            feedback="Validation failed - files do not exist or are not readable",
            failure_type="incomplete",
            issues=validation_result.validation_issues,
        )
        
        # Store in task
        task.qa_feedback = qa_result.feedback[:500]
        
        return {
            "tasks": task_tree.to_dict(),
            "current_qa_result": qa_result.to_dict(),
            "messages": ["QA: Failed due to validation failure"],
        }
    
    # Build the QA prompt (use current_task_description from state when available)
    task_desc = state.get("current_task_description") or task.description
    prompt_parts = [
        "="*70,
        "TASK TO VERIFY:",
        "="*70,
        f"ID: {task.id}",
        f"Description: {task_desc}",
        f"Measurable Outcome: {task.measurable_outcome}",
    ]

    if task.tags:
        prompt_parts.append(f"Tags: {', '.join(task.tags)}")

    carry_forward = state.get("carry_forward", [])
    if carry_forward:
        prompt_parts.append("")
        prompt_parts.append("Carry-forward (context for this milestone):")
        for cf in carry_forward[:3]:
            prompt_parts.append(f"  - {str(cf)[:100]}")

    # Add implementation plan summary
    if current_implementation_plan:
        prompt_parts.extend([
            "",
            "="*70,
            "IMPLEMENTATION PLAN (what was supposed to be done):",
            "="*70,
            current_implementation_plan[:1500],  # Limit to 1500 chars
        ])
        if len(current_implementation_plan) > 1500:
            prompt_parts.append("... (truncated)")
    
    # Build implementation context from modified files (deterministic, capped size)
    implementation_context = _build_implementation_context(
        repo_root,
        validation_result.files_verified,
        max_chars=QA_IMPLEMENTATION_CONTEXT_MAX_CHARS,
        max_lines_per_file=200,
    )
    prompt_parts.extend([
        "",
        "="*70,
        "ACTUAL IMPLEMENTATION (files that were modified):",
        "="*70,
        implementation_context,
        "",
        "="*70,
        "",
        "Compare the actual implementation against the task's measurable outcome. "
        "Determine if requirements are satisfied. If not, classify as incomplete, wrong_approach, or plan_issue. "
    ])
    
    try:
        # Use structured output for reliable parsing
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(prompt_parts)),
        ]
        
        structured_llm = planning_llm.with_structured_output(QAOutput)
        data = structured_llm.invoke(messages)
        
        if not data:
            raise ValueError("No structured output received from QA agent")
        
        # Validate failure_type
        valid_failure_types = ["incomplete", "wrong_approach", "plan_issue", None]
        failure_type = data.failure_type
        if failure_type not in valid_failure_types:
            failure_type = "incomplete"
        
        # Create QAResult
        qa_result = QAResult(
            task_id=task.id,
            passed=data.passed,
            feedback=(data.feedback or "")[:500],
            failure_type=failure_type,
            issues=data.issues,
        )
        
        # Store compressed feedback in task
        task.qa_feedback = qa_result.feedback[:500]
        
        logger.info("QA agent completed: passed=%s, failure_type=%s", qa_result.passed, qa_result.failure_type)
        return {
            "tasks": task_tree.to_dict(),
            "current_qa_result": qa_result.to_dict(),
            "messages": [f"QA: {'Passed' if qa_result.passed else 'Failed'} - {qa_result.failure_type or 'requirements met'}"],
        }
        
    except Exception as e:
        logger.error("QA agent exception: %s", e, exc_info=True)
        error_msg = f"QA assessment failed for {current_task_id}: {e}"
        
        # Create failed QA result
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
