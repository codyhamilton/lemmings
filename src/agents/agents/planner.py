"""Planner agent - creates detailed implementation plans from gap analysis.

This agent takes a task and its gap analysis, then creates a detailed
implementation plan (PRP format) with specific file changes and code snippets.

Output: Detailed markdown plan with files to create/modify
"""

import json
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ..logging_config import get_logger
from ..task_states import WorkflowState, Task, TaskTree, GapAnalysis
from ..llm import planning_llm

logger = get_logger(__name__)
from ..tools.search import search_files, find_files_by_name
from ..tools.read import read_file_lines
from ..tools.rag import rag_search


PLANNER_SYSTEM_PROMPT = """
## ROLE
You are an implementation planning agent for a Godot/GDScript game project.

## PRIMARY OBJECTIVE
Create a detailed implementation plan (PRP format) that closes the gap identified by the Researcher. The Implementor will execute this plan.

## PROCESS
1. Understand the gap - What needs to be done?
2. Search for patterns - How do similar things work in the codebase?
3. Identify files to change - What files need creation/modification?
4. Design the changes - What specific code changes are needed?
5. Plan the sequence - What order should changes be made?
6. Write the plan - Create detailed PRP with code snippets

## THINKING STEPS (track using write_todos)
- TODO 1: "Understand the gap"
- TODO 2: "Search for patterns"
- TODO 3: "Identify files to change"
- TODO 4: "Design the changes"
- TODO 5: "Plan the sequence"
- TODO 6: "Write the plan"

## TOOLS AVAILABLE
- **rag_search**: Find similar patterns and existing code semantically
- **search_files**: Search for specific patterns/symbols in files
- **find_files_by_name**: Find files by name pattern
- **read_file_lines**: Read existing file content to understand structure

## SEARCH STRATEGY
1. Use rag_search to find similar implementations and patterns
2. Read relevant files to understand structure and conventions
3. Be specific: Include file paths, code snippets, line numbers, explain WHY

## OUTPUT FORMAT
Your plan in PRP (Pull Request Plan) markdown format:

```markdown
# Implementation Plan: [Task Description]

## Task
- ID: task_XXX
- Outcome: [measurable outcome from task]

## Gap Analysis Summary
[Brief summary of what's missing - from gap_description]

## Changes

### Create: `path/to/new/file.gd`
**Purpose**: [Why this file is needed]
**Rationale**: [How this closes the gap]

\\`\\`\\`gdscript
extends Node
class_name NewClass

# Full implementation with comments explaining key decisions
func example():
    pass
\\`\\`\\`

### Modify: `path/to/existing/file.gd`
**Location**: Lines 45-60 (or "After class declaration", "In _ready()", etc.)
**Change**: [What modification to make]
**Rationale**: [Why this change]

\\`\\`\\`gdscript
# Add this code:
func new_function():
    # Implementation
    pass
\\`\\`\\`

### Modify: `path/to/another/file.gd`
**Location**: [Where in the file]
**Change**: [What to change]

\\`\\`\\`gdscript
# Before:
var old_variable = 1

# After:
var new_variable = {"key": "value"}
\\`\\`\\`

## Dependencies
- **Requires**: [What existing code this depends on]
- **Enables**: [What this unblocks or makes possible]

## Implementation Notes
- [Pattern to follow, e.g., "Match the signal pattern in GameState.gd"]
- [Edge cases to handle, e.g., "Ensure backwards compatibility"]
- [Testing considerations, e.g., "Should work with empty arrays"]

## Files Summary
- **Create**: new_file1.gd, new_file2.gd
- **Modify**: existing_file1.gd, existing_file2.gd
```

## CONSTRAINTS
- Include CODE SNIPPETS - the Implementor needs to see actual code
- Be SPECIFIC about locations - line numbers, function names, clear descriptions
- Explain WHY - rationale helps the Implementor understand intent
- Follow existing PATTERNS - use search tools to find how similar things are done
- Use GDScript syntax (not Python or other languages)
- Consider Godot-specific patterns (signals, nodes, scenes)
- Be thorough but concise - detailed but not overwhelming
- Verify files exist before referencing them
"""


def create_planner_agent():
    """Create the planner agent with search tools and todo tracking."""
    middleware = []
    
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    # Tools for finding patterns and understanding codebase
    tools = [
        rag_search,
        search_files,
        find_files_by_name,
        read_file_lines,
    ]
    
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
    )


def _extract_plan_from_output(content: str) -> str:
    """Extract the markdown plan from agent output.
    
    Handles cases where plan is in markdown code blocks or raw text.
    
    Args:
        content: Raw agent output
    
    Returns:
        Extracted markdown plan
    """
    import re
    
    # Try to find markdown in code blocks
    md_match = re.search(r'```markdown\s*(.*?)\s*```', content, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()
    
    # Try to find plan starting with "# Implementation Plan"
    plan_match = re.search(r'(# Implementation Plan:.*)', content, re.DOTALL)
    if plan_match:
        return plan_match.group(1).strip()
    
    # Return full content as fallback
    return content.strip()


def _extract_files_from_plan(plan: str) -> tuple[list[str], list[str]]:
    """Extract files to create and modify from the plan.
    
    Looks for "Create: `path`" and "Modify: `path`" sections.
    
    Args:
        plan: Markdown plan text
    
    Returns:
        Tuple of (files_to_create, files_to_modify)
    """
    import re
    
    files_to_create = []
    files_to_modify = []
    
    # Find all "Create: `path`" entries
    create_matches = re.findall(r'### Create: `([^`]+)`', plan)
    files_to_create.extend(create_matches)
    
    # Find all "Modify: `path`" entries
    modify_matches = re.findall(r'### Modify: `([^`]+)`', plan)
    files_to_modify.extend(modify_matches)
    
    # Also check "Files Summary" section if it exists
    summary_match = re.search(r'## Files Summary\s*\n\s*-\s*\*\*Create\*\*:\s*([^\n]+)', plan)
    if summary_match:
        summary_creates = [f.strip() for f in summary_match.group(1).split(',')]
        files_to_create.extend([f for f in summary_creates if f and f not in files_to_create])
    
    summary_match = re.search(r'## Files Summary\s*\n\s*-\s*\*\*Modify\*\*:\s*([^\n]+)', plan)
    if summary_match:
        summary_modifies = [f.strip() for f in summary_match.group(1).split(',')]
        files_to_modify.extend([f for f in summary_modifies if f and f not in files_to_modify])
    
    return files_to_create, files_to_modify


def _summarize_plan(plan: str, max_chars: int = 500) -> str:
    """Create a compressed summary of the implementation plan.
    
    Args:
        plan: Full markdown plan
        max_chars: Maximum characters for summary
    
    Returns:
        Compressed summary
    """
    try:
        # Use LLM to compress
        prompt = f"""Summarize this implementation plan in under {max_chars} characters, focusing on what will be done:

{plan[:2000]}

Compressed summary ({max_chars} chars max):"""
        
        response = planning_llm.invoke(prompt)
        summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # Fallback to truncation if still too long
        if len(summary) > max_chars:
            summary = summary[:max_chars-3] + "..."
        
        return summary
    except Exception as e:
        # Fallback: extract first paragraph or truncate
        lines = plan.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                truncated = line.strip()[:max_chars-3] + "..."
                return truncated
        return plan[:max_chars-3] + "..."


def planner_node(state: WorkflowState) -> dict:
    """Planner agent - create detailed implementation plan for current task.
    
    Takes the gap analysis and creates a specific, actionable plan with code snippets.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_implementation_plan set
    """
    logger.info("Planner agent starting")
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    repo_root = state["repo_root"]
    current_gap_analysis = state.get("current_gap_analysis")
    
    if not current_task_id:
        return {
            "error": "No current task set for planner",
            "messages": ["Planner: No current task"],
        }
    
    if not current_gap_analysis:
        return {
            "error": "No gap analysis available for planner",
            "messages": ["Planner: No gap analysis found"],
        }
    
    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"Planner: Task {current_task_id} not found"],
        }
    
    # Load gap analysis
    gap_analysis = GapAnalysis.from_dict(current_gap_analysis)
    
    # Load project context if available
    project_context = None
    context_path = Path(repo_root) / "agents" / "context.md"
    if context_path.exists():
        try:
            project_context = context_path.read_text(encoding="utf-8")
        except Exception as e:
            project_context = None
    
    # Build messages with separated context types
    messages = []
    
    # 1. Project context (if available)
    if project_context:
        messages.append(HumanMessage(
            content=f"## PROJECT CONTEXT\n{project_context[:3000]}"  # Limit to 3k chars
        ))
    
    # 2. Task information
    task_context = [
        "## TASK",
        f"**ID**: {task.id}",
        f"**Description**: {task.description}",
        f"**Measurable Outcome**: {task.measurable_outcome}",
    ]
    
    if task.tags:
        task_context.append(f"**Tags**: {', '.join(task.tags)}")
    
    # Add retry context when this is a retry (e.g. QA returned plan_issue)
    if task.attempt_count > 0:
        task_context.append("")
        task_context.append("**RETRY CONTEXT** (previous plan was inadequate - address these issues):")
        if task.qa_feedback:
            task_context.append(f"  QA Feedback: {task.qa_feedback}")
        if task.last_failure_reason:
            task_context.append(f"  Failure Reason: {task.last_failure_reason}")
    
    # Add dependency context (completed tasks)
    if task.depends_on:
        task_context.append("")
        task_context.append("**Dependency Context** (completed tasks):")
        for dep_id in task.depends_on:
            dep_task = task_tree.tasks.get(dep_id)
            if dep_task:
                task_context.append(f"  - {dep_id}: {dep_task.description[:60]}")
                if dep_task.result_summary:
                    task_context.append(f"    Result: {dep_task.result_summary[:100]}")
    
    messages.append(HumanMessage(content="\n".join(task_context)))
    
    # 3. Gap analysis (from Researcher)
    gap_context = [
        "## GAP ANALYSIS (from Researcher)",
        f"**Gap exists**: {gap_analysis.gap_exists}",
        "",
        "**Current State**:",
        gap_analysis.current_state_summary,
        "",
        "**Desired State**:",
        gap_analysis.desired_state_summary,
        "",
        "**Gap Description**:",
        gap_analysis.gap_description,
    ]
    
    if gap_analysis.relevant_files:
        gap_context.append("")
        gap_context.append("**Relevant Files**:")
        for f in gap_analysis.relevant_files:
            gap_context.append(f"  - {f}")
    
    if gap_analysis.keywords:
        gap_context.append(f"\n**Keywords**: {', '.join(gap_analysis.keywords)}")
    
    messages.append(HumanMessage(content="\n".join(gap_context)))
    
    # 4. Instructions (task-specific)
    messages.append(HumanMessage(
        content=f"## INSTRUCTIONS\nRepository root: {repo_root}\n\nUse search tools to find existing patterns and conventions. Create a detailed PRP plan with code snippets. Be specific about file locations and changes. The Implementor will execute this plan, so make it clear and actionable."
    ))
    
    try:
        # Create and run the planner agent
        agent = create_planner_agent()
        
        # Use invoke to get reliable final result
        result = agent.invoke({"messages": messages})
        
        # Extract content from result
        content = ""
        tool_calls_made = []
        
        if result and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    msg_content = str(msg.content)
                    content += msg_content
                
                # Track tool calls for visibility
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        tool_calls_made.append(f"{tool_name}({str(tool_args)[:80]}...)")
        
        if not content or not content.strip():
            raise ValueError("No content received from planner agent")
        
        # Extract markdown plan from output
        implementation_plan = _extract_plan_from_output(content)
        
        if not implementation_plan or len(implementation_plan) < 100:
            raise ValueError(f"Plan too short or empty: {len(implementation_plan)} chars")
        
        # Extract files to create/modify
        files_to_create, files_to_modify = _extract_files_from_plan(implementation_plan)
        
        # Create compressed summary for task storage
        plan_summary = _summarize_plan(implementation_plan, max_chars=500)
        
        # Store summary in task
        task.implementation_plan_summary = plan_summary
        
        logger.info("Planner agent completed: %s new files, %s to modify", len(files_to_create), len(files_to_modify))
        return {
            "tasks": task_tree.to_dict(),
            "current_implementation_plan": implementation_plan,  # Full plan in ephemeral state
            "messages": [f"Planner: Created implementation plan ({len(files_to_create)} new, {len(files_to_modify)} modified)"],
        }
        
    except Exception as e:
        logger.error("Planner agent exception: %s", e, exc_info=True)
        error_msg = f"Planner failed creating implementation plan for {current_task_id}: {e}"
        
        # Increment task attempt count
        task.attempt_count += 1
        task.last_failure_reason = error_msg
        task.last_failure_stage = "planner"
        
        return {
            "tasks": task_tree.to_dict(),
            "current_implementation_plan": None,
            "error": error_msg,
            "messages": [f"Planner exception: {e}"],
        }
