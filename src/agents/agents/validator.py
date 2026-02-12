"""Validator agent - verifies that reported file changes actually exist.

This is a simple file verification agent that checks if the files reported
by the Implementor actually exist on disk and are valid (non-empty, readable).

No LLM needed - just file system checks.

Output: ValidationResult with files_verified, files_missing, validation_passed
"""

import os
from pathlib import Path

from ..logging_config import get_logger
from ..task_states import WorkflowState, Task, TaskTree, ValidationResult, ImplementationResult

logger = get_logger(__name__)


def validator_node(state: WorkflowState) -> dict:
    """Validator - verify reported file changes actually exist.
    
    This is a deterministic file system check (no LLM). Verifies that:
    1. Each reported file exists on disk
    2. Files are non-empty and readable
    3. Files are valid text files (not binary)
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_validation_result set
    """
    logger.info("Validator agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    repo_root = state["repo_root"]
    current_implementation_result = state.get("current_implementation_result")
    
    if not current_task_id:
        return {
            "error": "Validator: No current task set",
            "messages": ["Validator: No current task"],
        }
    
    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not current_implementation_result:
        validation_result = ValidationResult(
            task_id=current_task_id,
            files_verified=[],
            files_missing=[],
            validation_passed=False,
            validation_issues=["No implementation result available"],
        )
        return {
            "current_validation_result": validation_result.to_dict(),
            "error": f"Validator: No implementation result for {current_task_id}",
            "messages": ["Validator: No implementation result found"],
        }
    
    if not task:
        validation_result = ValidationResult(
            task_id=current_task_id,
            files_verified=[],
            files_missing=[],
            validation_passed=False,
            validation_issues=[f"Task {current_task_id} not found"],
        )
        return {
            "current_validation_result": validation_result.to_dict(),
            "error": f"Validator: Task {current_task_id} not found",
            "messages": [f"Validator: Task {current_task_id} not found"],
        }
    
    # Load implementation result
    impl_result = ImplementationResult.from_dict(current_implementation_result)
    
    if not impl_result.files_modified:
        validation_result = ValidationResult(
            task_id=task.id,
            files_verified=[],
            files_missing=[],
            validation_passed=False,
            validation_issues=["No files were reported as modified by implementor"],
        )
        
        return {
            "current_validation_result": validation_result.to_dict(),
            "messages": ["Validator: Failed - no files reported"],
        }
    
    # Verify each reported file
    files_verified = []
    files_missing = []
    validation_issues = []
    
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        
        for file_path in impl_result.files_modified:
            full_path = Path(repo_root) / file_path
            
            # Check 1: File exists
            if not full_path.exists():
                files_missing.append(file_path)
                validation_issues.append(f"File does not exist: {file_path}")
                continue
            
            # Check 2: Is a file (not directory)
            if not full_path.is_file():
                files_missing.append(file_path)
                validation_issues.append(f"Not a file (directory?): {file_path}")
                continue
            
            # Check 3: Non-empty
            try:
                stat = full_path.stat()
                if stat.st_size == 0:
                    # Empty file is suspicious but not necessarily invalid
                    validation_issues.append(f"File is empty: {file_path}")
                    files_verified.append(file_path)
                    continue
            except Exception as e:
                validation_issues.append(f"Cannot stat file: {file_path} ({e})")
                files_missing.append(file_path)
                continue
            
            # Check 4: Readable as text (not binary)
            try:
                with full_path.open('r', encoding='utf-8') as f:
                    # Try to read first 100 chars to verify it's text
                    f.read(100)
                
                files_verified.append(file_path)
                
            except UnicodeDecodeError:
                # Binary file
                validation_issues.append(f"File is binary (not text): {file_path}")
                files_verified.append(file_path)  # Still count as verified, just note it
            except Exception as e:
                validation_issues.append(f"Cannot read file: {file_path} ({e})")
                files_missing.append(file_path)
    finally:
        os.chdir(original_cwd)
    
    # Determine if validation passed
    validation_passed = len(files_missing) == 0
    
    # Create validation result
    validation_result = ValidationResult(
        task_id=task.id,
        files_verified=files_verified,
        files_missing=files_missing,
        validation_passed=validation_passed,
        validation_issues=validation_issues,
    )
    
    logger.info("Validator agent completed: passed=%s, %s verified, %s missing", validation_passed, len(files_verified), len(files_missing))
    return {
        "current_validation_result": validation_result.to_dict(),
        "messages": [f"Validator: {'Passed' if validation_passed else 'Failed'} - {len(files_verified)} verified, {len(files_missing)} missing"],
    }
