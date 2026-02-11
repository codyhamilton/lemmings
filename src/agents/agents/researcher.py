"""Researcher agent - performs gap analysis for tasks.

This agent is the CRITICAL GATE in the workflow. It determines if there's actually
a gap between current state and desired state for a task.

Key role:
- If gap_exists is False, the task is already satisfied (feature exists)
- If gap_exists is True, we proceed to planning and implementation

Output: GapAnalysis with compressed context (max 1-2k chars per field)
"""

import json
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ..task_states import WorkflowState, Task, TaskStatus, TaskTree, GapAnalysis
from ..llm import planning_llm
from ..normaliser import normalize_agent_output
from ..tools.search import search_files, list_directory, find_files_by_name
from ..tools.read import read_file_lines
from ..tools.rag import rag_search


RESEARCHER_SYSTEM_PROMPT = """
## ROLE
You are a gap analysis agent for a Godot/GDScript game project.

## PRIMARY OBJECTIVE
Determine if a gap exists between current codebase state and task requirements. This is a CRITICAL GATE - if no gap exists, we skip implementation.

## PROCESS
1. Understand task requirement - What does the task ask for?
2. Search for existing implementation - Does it already exist?
3. Assess current state - What do we have now?
4. Identify the gap - What's missing or incomplete?
5. Compile findings - gap_exists? Summaries? Relevant files?

## THINKING STEPS (track using write_todos)
- TODO 1: "Understand task requirement"
- TODO 2: "Search for existing implementation"
- TODO 3: "Assess current state"
- TODO 4: "Identify the gap"
- TODO 5: "Compile findings"

## TOOLS AVAILABLE
- **rag_search**: Find semantically similar code patterns
- **list_directory**: Browse project structure
- **find_files_by_name**: Find files by name pattern
- **search_files**: Search for patterns in files using regex
- **read_file_lines**: Read specific line ranges from files

## SEARCH STRATEGY
1. Start BROAD: Search for keywords, related symbols, use rag_search
2. Validate findings: Read actual code to verify relevance
3. Make determination: Is feature fully/partially/not implemented?

## OUTPUT FORMAT
```json
{
    "gap_exists": true|false,
    "current_state_summary": "What exists now - specific files, symbols, functionality (max 1000 chars)",
    "desired_state_summary": "What task requires - from description and measurable outcome (max 1000 chars)",
    "gap_description": "The delta - what's missing/incomplete or why task is satisfied (max 2000 chars)",
    "relevant_files": ["path/to/file1.gd", "path/to/file2.gd"],
    "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

## CONSTRAINTS
- Be honest: If something exists and works, set gap_exists = false
- Keep summaries compressed (respect max lengths)
- Be specific with file paths and symbols
- Focus on WHAT exists and what's NEEDED, not HOW
- Use tools to VERIFY, don't guess
"""


def create_researcher_agent():
    """Create the researcher agent with search tools and todo tracking."""
    middleware = []
    
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    # Tools for gap analysis
    tools = [
        rag_search,
        list_directory,
        find_files_by_name,
        search_files,
        read_file_lines,
    ]
    
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
    )


def researcher_node(state: WorkflowState) -> dict:
    """Researcher agent - perform gap analysis for current task.
    
    This is the critical gate: if gap_exists is False, we skip implementation.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_gap_analysis set
    """
    current_task_id = state.get("current_task_id")
    tasks_dict = state["tasks"]
    repo_root = state["repo_root"]
    
    if not current_task_id:
        return {
            "error": "No current task set for researcher",
            "messages": ["Researcher: No current task"],
        }
    
    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"Researcher: Task {current_task_id} not found"],
        }
    
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
    
    # 2. Task information (structured)
    task_context = [
        "## TASK TO ANALYZE",
        f"**ID**: {task.id}",
        f"**Description**: {task.description}",
        f"**Measurable Outcome**: {task.measurable_outcome}",
    ]
    
    if task.tags:
        task_context.append(f"**Tags**: {', '.join(task.tags)}")
    
    if task.estimated_complexity:
        task_context.append(f"**Estimated Complexity**: {task.estimated_complexity}")
    
    # Add dependency context (what's been completed)
    if task.depends_on:
        task_context.append("")
        task_context.append("**Dependencies** (completed tasks):")
        for dep_id in task.depends_on:
            dep_task = task_tree.tasks.get(dep_id)
            if dep_task and dep_task.status == TaskStatus.COMPLETE:
                task_context.append(f"  - {dep_id}: {dep_task.description[:80]}")
                if dep_task.result_summary:
                    task_context.append(f"    Result: {dep_task.result_summary[:150]}")
    
    messages.append(HumanMessage(content="\n".join(task_context)))
    
    # 3. Instructions (task-specific)
    messages.append(HumanMessage(
        content=f"## INSTRUCTIONS\nRepository root: {repo_root}\n\nUse your tools to search and verify current state. Determine if a gap exists between current and desired state. Be honest - if the feature exists and works, set gap_exists = false."
    ))
    
    try:
        # Create and run the researcher agent
        agent = create_researcher_agent()
        
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
            raise ValueError("No content received from researcher agent")
        
        # Define expected schema for gap analysis output
        gap_analysis_schema = {
            "gap_exists": {
                "type": bool,
                "required": True,
            },
            "current_state_summary": {
                "type": str,
                "required": True,
                "max_length": 1000,
            },
            "desired_state_summary": {
                "type": str,
                "required": True,
                "max_length": 1000,
            },
            "gap_description": {
                "type": str,
                "required": True,
                "max_length": 2000,
            },
            "relevant_files": {
                "type": list,
                "required": False,
                "default": [],
            },
            "keywords": {
                "type": list,
                "required": False,
                "default": [],
            },
        }
        
        # Use normaliser for JSON extraction and parsing
        norm_result = normalize_agent_output(
            content,
            gap_analysis_schema,
            use_llm_summarization=True  # Use LLM for quality compression
        )
        
        if not norm_result.success:
            raise ValueError(f"Normalisation failed: {norm_result.error}")
        
        data = norm_result.data
        
        # Create GapAnalysis object
        gap_analysis = GapAnalysis(
            task_id=task.id,
            gap_exists=data["gap_exists"],
            current_state_summary=data["current_state_summary"],
            desired_state_summary=data["desired_state_summary"],
            gap_description=data["gap_description"],
            relevant_files=data.get("relevant_files", []),
            keywords=data.get("keywords", []),
        )
        
        # Store compressed gap analysis in task
        task.gap_analysis = gap_analysis.to_dict()
        
        return {
            "tasks": task_tree.to_dict(),
            "current_gap_analysis": gap_analysis.to_dict(),
            "messages": [f"Researcher: Gap analysis complete for {task.id} (gap_exists={gap_analysis.gap_exists})"],
        }
        
    except Exception as e:
        error_msg = f"Researcher failed during gap analysis for {current_task_id}: {e}"
        
        # Increment task attempt count
        task.attempt_count += 1
        task.last_failure_reason = error_msg
        task.last_failure_stage = "researcher"
        
        return {
            "tasks": task_tree.to_dict(),
            "current_gap_analysis": None,
            "error": error_msg,
            "messages": [f"Researcher exception: {e}"],
        }
