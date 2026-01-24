"""QA agent - verifies that task requirements are satisfied.

This agent checks if the implemented changes actually satisfy the task's
measurable outcome. It reads the actual file contents and assesses whether
the implementation meets the requirements.

Output: QAResult with passed, feedback, failure_type, issues
"""

from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ..state import WorkflowState, Task, TaskTree, QAResult, ValidationResult
from ..llm import planning_llm
from ..normaliser import normalize_agent_output
from ..tools.read import read_file, read_file_lines


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
    
    print("\n" + "="*70)
    print("🔍 QA AGENT (Requirement Satisfaction Check)")
    print("="*70)
    print(f"Task: {task.id}")
    print(f"Description: {task.description}")
    print(f"Measurable Outcome: {task.measurable_outcome}")
    print(f"Verified files: {len(validation_result.files_verified)}")
    print()
    
    # If validation failed, QA automatically fails
    if not validation_result.validation_passed:
        print("❌ Validation failed - QA cannot proceed")
        
        qa_result = QAResult(
            task_id=task.id,
            passed=False,
            feedback="Validation failed - files do not exist or are not readable",
            failure_type="incomplete",
            issues=validation_result.validation_issues,
        )
        
        print(f"\n{'='*70}")
        print("QA RESULT:")
        print(f"{'='*70}")
        print("❌ FAILED - Validation failed")
        print("="*70)
        
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
        # Create a simple LLM call (no agent needed for QA - just assessment)
        from langchain_core.messages import SystemMessage
        
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(prompt_parts)),
        ]
        
        print("💭 Starting QA assessment...\n")
        
        # Stream response
        content = ""
        for chunk in planning_llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                content += chunk.content
        print()
        
        if not content or not content.strip():
            raise ValueError("No content received from QA agent")
        
        # Define expected schema for QA output
        qa_schema = {
            "passed": {
                "type": bool,
                "required": True,
            },
            "feedback": {
                "type": str,
                "required": True,
                "max_length": 500,
            },
            "failure_type": {
                "type": str,
                "required": False,
                "default": None,
            },
            "issues": {
                "type": list,
                "required": False,
                "default": [],
            },
        }
        
        # Use normaliser for JSON extraction and parsing
        norm_result = normalize_agent_output(
            content,
            qa_schema,
            use_llm_summarization=False  # Simple truncation for feedback
        )
        
        if not norm_result.success:
            raise ValueError(f"Normalisation failed: {norm_result.error}")
        
        data = norm_result.data
        
        # Validate failure_type if present
        valid_failure_types = ["incomplete", "wrong_approach", "plan_issue", None]
        failure_type = data.get("failure_type")
        if failure_type not in valid_failure_types:
            print(f"⚠️  Invalid failure_type '{failure_type}', defaulting to 'incomplete'")
            failure_type = "incomplete"
        
        # Create QAResult
        qa_result = QAResult(
            task_id=task.id,
            passed=data["passed"],
            feedback=data["feedback"],
            failure_type=failure_type,
            issues=data.get("issues", []),
        )
        
        # Store compressed feedback in task
        task.qa_feedback = qa_result.feedback[:500]
        
        # Print summary
        print(f"\n{'='*70}")
        print("QA RESULT:")
        print(f"{'='*70}")
        
        if qa_result.passed:
            print("✅ PASSED - Task requirements satisfied")
            print(f"\nFeedback:")
            print(f"  {qa_result.feedback}")
        else:
            print(f"❌ FAILED - Task requirements not satisfied")
            print(f"\nFailure Type: {qa_result.failure_type}")
            print(f"\nFeedback:")
            print(f"  {qa_result.feedback}")
            
            if qa_result.issues:
                print(f"\nIssues ({len(qa_result.issues)}):")
                for issue in qa_result.issues[:10]:
                    print(f"  ! {issue}")
                if len(qa_result.issues) > 10:
                    print(f"  ... and {len(qa_result.issues) - 10} more")
        
        print("="*70)
        
        return {
            "tasks": task_tree.to_dict(),
            "current_qa_result": qa_result.to_dict(),
            "messages": [f"QA: {'Passed' if qa_result.passed else 'Failed'} - {qa_result.failure_type or 'requirements met'}"],
        }
        
    except Exception as e:
        error_msg = f"QA error: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Create failed QA result
        qa_result = QAResult(
            task_id=task.id,
            passed=False,
            feedback=f"QA assessment failed: {e}",
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
