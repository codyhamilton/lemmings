"""Summarizer agent - creates concise summaries for dashboard display.

This lightweight agent condenses agent outputs into short summaries (max 80 chars)
for display in the dashboard view.
"""

from typing import Any
from langchain_core.messages import HumanMessage

from ..task_states import (
    WorkflowState, Task, TaskTree, GapAnalysis, 
    ImplementationResult, ValidationResult, QAResult, AssessmentResult
)
from ..llm import planning_llm


SUMMARIZER_PROMPT = """You are a summarizer that creates very concise summaries for a dashboard display.

Your job is to create a short summary (max 80 characters) of what an agent just did or is doing.

Examples:
- "Working on task_003: Add health bar to colony view"
- "Done task_002: Modified 3 files (ColonyView.gd, StatusBar.gd)"
- "Failed task_001: Validation error in DataManager.gd"
- "Analyzing gap for task_004: Check if feature exists"
- "Planning implementation for task_005: Create new resource type"

Be specific but brief. Include:
- Task ID if available
- What was done or is being done
- Key files or outcomes if relevant

Output ONLY the summary text, nothing else. No quotes, no explanation."""


def summarize_agent_activity(
    node_name: str,
    state: WorkflowState,
    agent_output: dict[str, Any] | None = None,
) -> str:
    """Create a concise summary of agent activity.
    
    Args:
        node_name: Name of the node that executed (e.g., "researcher", "implementor")
        state: Current workflow state
        agent_output: Optional structured output from the agent
    
    Returns:
        Short summary string (max 80 chars)
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state.get("tasks", {})
    task_tree = TaskTree.from_dict(tasks_dict) if tasks_dict else TaskTree()
    
    # Get task info if available
    task = None
    task_desc = ""
    if current_task_id:
        task = task_tree.tasks.get(current_task_id)
        if task:
            task_desc = f"task_{current_task_id[-3:]}: {task.description[:40]}"
    
    # Create context based on node type
    context_parts = []
    
    if node_name == "researcher":
        gap_analysis_dict = state.get("current_gap_analysis")
        if gap_analysis_dict:
            gap = GapAnalysis.from_dict(gap_analysis_dict)
            if gap.gap_exists:
                context_parts.append(f"Gap found: {gap.gap_description[:50]}")
            else:
                context_parts.append("No gap - task already satisfied")
        else:
            context_parts.append("Analyzing gap")
    
    elif node_name == "planner":
        plan = state.get("current_implementation_plan")
        if plan:
            # Try to extract file info from plan
            files_mentioned = []
            for line in plan.split("\n")[:20]:  # Check first 20 lines
                if "file" in line.lower() or ".gd" in line.lower():
                    # Extract file names
                    import re
                    matches = re.findall(r'[\w/]+\.gd', line)
                    files_mentioned.extend(matches[:3])
            if files_mentioned:
                context_parts.append(f"Planning for {len(set(files_mentioned))} files")
            else:
                context_parts.append("Creating implementation plan")
        else:
            context_parts.append("Planning implementation")
    
    elif node_name == "implementor":
        impl_result_dict = state.get("current_implementation_result")
        if impl_result_dict:
            impl_result = ImplementationResult.from_dict(impl_result_dict)
            if impl_result.success:
                files = impl_result.files_modified[:2]  # First 2 files
                file_names = [f.split("/")[-1] for f in files]
                if len(impl_result.files_modified) > 2:
                    context_parts.append(f"Modified {len(impl_result.files_modified)} files ({', '.join(file_names)}...)")
                else:
                    context_parts.append(f"Modified {len(impl_result.files_modified)} files ({', '.join(file_names)})")
            else:
                context_parts.append("Implementation failed")
        else:
            context_parts.append("Implementing changes")
    
    elif node_name == "validator":
        val_result_dict = state.get("current_validation_result")
        if val_result_dict:
            val_result = ValidationResult.from_dict(val_result_dict)
            if val_result.validation_passed:
                context_parts.append(f"Validated {len(val_result.files_verified)} files")
            else:
                context_parts.append(f"Validation failed: {len(val_result.files_missing)} missing")
        else:
            context_parts.append("Validating files")
    
    elif node_name == "qa":
        qa_result_dict = state.get("current_qa_result")
        if qa_result_dict:
            qa_result = QAResult.from_dict(qa_result_dict)
            if qa_result.passed:
                context_parts.append("QA passed - requirements satisfied")
            else:
                context_parts.append(f"QA failed: {qa_result.failure_type}")
        else:
            context_parts.append("Checking requirements")
    
    elif node_name == "assessor":
        assessment_dict = state.get("last_assessment")
        if assessment_dict:
            assessment = AssessmentResult.from_dict(assessment_dict)
            if assessment.is_complete:
                context_parts.append("Workflow complete")
            elif assessment.uncovered_gaps:
                context_parts.append(f"Found {len(assessment.uncovered_gaps)} gaps")
            else:
                context_parts.append("Assessing completion")
        else:
            context_parts.append("Assessing workflow")
    
    elif node_name == "prioritizer":
        if current_task_id:
            context_parts.append(f"Selected {task_desc}")
        else:
            context_parts.append("No tasks ready")
    
    elif node_name == "expander":
        tasks_created = state.get("tasks_created_this_iteration", 0)
        if tasks_created > 0:
            context_parts.append(f"Created {tasks_created} new tasks")
        else:
            context_parts.append("No expansion needed")
    
    elif node_name == "intake":
        milestones = state.get("milestones", {})
        if milestones:
            context_parts.append(f"Created {len(milestones)} milestones")
        else:
            context_parts.append("Processing request")
    
    elif node_name in ("mark_complete", "mark_failed"):
        if node_name == "mark_complete":
            context_parts.append("Task completed")
        else:
            context_parts.append("Task failed")
    
    # Build summary string
    if task_desc:
        summary = f"{task_desc} - {', '.join(context_parts) if context_parts else node_name}"
    else:
        summary = ", ".join(context_parts) if context_parts else f"{node_name} executed"
    
    # If summary is too long, use LLM to condense it
    if len(summary) > 80:
        try:
            prompt = f"{SUMMARIZER_PROMPT}\n\nActivity: {node_name}\nContext: {summary}\n\nCreate a concise summary:"
            response = planning_llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            # Remove quotes if present
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]
            if summary.startswith("'") and summary.endswith("'"):
                summary = summary[1:-1]
            # Truncate if still too long
            if len(summary) > 80:
                summary = summary[:77] + "..."
        except Exception:
            # Fallback: just truncate
            summary = summary[:77] + "..."
    
    # Add status prefix
    if node_name in ("mark_complete", "qa") and state.get("current_qa_result"):
        qa_result_dict = state.get("current_qa_result")
        if qa_result_dict:
            qa_result = QAResult.from_dict(qa_result_dict)
            if qa_result.passed:
                return f"✓ {summary}"
    
    if node_name == "mark_failed":
        return f"✗ {summary}"
    
    if node_name in ("researcher", "planner", "implementor", "validator", "qa"):
        return f"▶ {summary}"
    
    return summary
