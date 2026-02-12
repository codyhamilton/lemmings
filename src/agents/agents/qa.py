"""QA agent - verifies that task requirements are satisfied.

This agent checks if the implemented changes actually satisfy the task's
measurable outcome. It reads the actual file contents and assesses whether
the implementation meets the requirements.

Output: QAResult with passed, feedback, failure_type, issues
"""

from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import WorkflowState, Task, TaskTree, QAResult, ValidationResult
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
You are a QA agent for a Godot/GDScript game project.

## PRIMARY OBJECTIVE
Verify that implemented changes satisfy the task's measurable outcome.

## PROCESS
1. Compare implementation against task's measurable outcome
2. Check if implementation is complete or has missing pieces
3. Determine if code works correctly for intended purpose
4. If failed, classify the failure type

## ASSESSMENT CRITERIA
- Be thorough but fair
- If task says "Add X to Y" and X was added to Y, it passes
- Don't fail for style issues or optimizations
- Focus on functional requirements

## FAILURE TYPES
- **incomplete**: Implementation started but missing key pieces
- **wrong_approach**: Implementation exists but doesn't solve problem correctly
- **plan_issue**: Implementation follows plan but plan was inadequate

## OUTPUT FORMAT
```json
{
    "passed": true|false,
    "feedback": "Detailed assessment of whether task requirements are met (max 500 chars)",
    "failure_type": null|"incomplete"|"wrong_approach"|"plan_issue",
    "issues": ["Issue 1", "Issue 2"]
}
```

## CONSTRAINTS
- Set passed=true if measurable outcome is satisfied, even if implementation could be improved
- Focus on requirements, not perfection
"""


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
    
    This agent uses an LLM to assess whether the implementation meets
    the task's measurable outcome by reading actual file contents.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_qa_result set
    """
    logger.info("QA agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    repo_root = state["repo_root"]
    current_implementation_plan = state.get("current_implementation_plan")
    current_validation_result = state.get("current_validation_result")
    
    if not current_task_id:
        return {
            "error": "No current task set for QA",
            "messages": ["QA: No current task"],
        }
    
    if not current_validation_result:
        return {
            "error": "No validation result available for QA",
            "messages": ["QA: No validation result found"],
        }
    
    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"QA: Task {current_task_id} not found"],
        }
    
    # Load validation result
    validation_result = ValidationResult.from_dict(current_validation_result)
    
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
    
    # Build the QA prompt
    prompt_parts = [
        "="*70,
        "TASK TO VERIFY:",
        "="*70,
        f"ID: {task.id}",
        f"Description: {task.description}",
        f"Measurable Outcome: {task.measurable_outcome}",
    ]
    
    if task.tags:
        prompt_parts.append(f"Tags: {', '.join(task.tags)}")
    
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
    
    # Read verified files
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        
        prompt_parts.extend([
            "",
            "="*70,
            "ACTUAL IMPLEMENTATION (files that were modified):",
            "="*70,
        ])
        
        for file_path in validation_result.files_verified[:10]:  # Limit to 10 files
            prompt_parts.append(f"\n--- {file_path} ---")
            try:
                content = read_modified_file.invoke({"path": file_path, "max_lines": 50})
                prompt_parts.append(content)
            except Exception as e:
                prompt_parts.append(f"Error reading file: {e}")
        
        if len(validation_result.files_verified) > 10:
            prompt_parts.append(f"\n... and {len(validation_result.files_verified) - 10} more files")
    finally:
        os.chdir(original_cwd)
    
    prompt_parts.extend([
        "",
        "="*70,
        "",
        "INSTRUCTIONS:",
        "1. Compare the actual implementation against the task's measurable outcome",
        "2. Determine if the task requirements are satisfied",
        "3. If not satisfied, classify the failure type",
        "4. Output JSON with your assessment",
        "",
        "Remember: Pass if measurable outcome is met, even if implementation could be improved.",
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
