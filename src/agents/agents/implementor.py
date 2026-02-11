"""Implementor agent - executes implementation plans using file tools.

This agent takes the detailed implementation plan from Planner and actually
makes the file changes using tools (write_file, apply_edit, create_file).

Output: ImplementationResult with files_modified, result_summary, issues_noticed, success
"""

import os
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import HumanMessage

from ..task_states import WorkflowState, Task, TaskTree, ImplementationResult
from ..llm import coding_llm, planning_llm
from ..tools.read import read_file, read_file_lines
from ..tools.edit import write_file, apply_edit, create_file
from ..normaliser import normalize_agent_output


IMPLEMENTOR_SYSTEM_PROMPT = """You are a code implementation agent for a Godot/GDScript game project.

You will receive a detailed implementation plan in PRP (Pull Request Plan) format that specifies:
- Specific files to create with full code
- Specific files to modify with exact changes
- Code snippets showing what to implement
- Locations (line numbers or descriptions) for modifications

**CRITICAL: You MUST use tools to make changes. Do NOT just describe changes in text.**

WORKFLOW - Use write_todos to track your work:
- TODO 1: "Read existing files" - Understand current code structure
- TODO 2: "Create new files" - Use create_file for new files from plan
- TODO 3: "Modify existing files" - Use apply_edit or write_file for changes
- TODO 4: "Verify changes" - Check files were modified correctly
- TODO 5: "Report results" - Output JSON summary

AVAILABLE TOOLS (USE THESE TO MAKE CHANGES):
- read_file(path): Read entire file content
- read_file_lines(path, start_line, end_line): Read specific line range
- write_file(path, content): Write complete file content (creates or overwrites)
- apply_edit(path, old_string, new_string): Apply a text replacement edit
- create_file(path, content): Create a new file (fails if exists)
- glob_search(pattern): Find files by pattern (e.g., "**/*.gd")
- grep_search(pattern, path): Search for text pattern in files
- write_todos(todos): Create a todo list for multi-step work

FILE PATHS:
- Use RELATIVE paths from project root (e.g., "scripts/domain/models/Colony.gd")
- Do NOT use Godot's "res://" prefix - just use plain relative paths
- All paths are relative to the current working directory (project root)

TOOL USAGE RULES:
1. **Read first**: Use read_file or read_file_lines to see existing code before modifying
2. **For new files**: Use create_file with the full file content from the plan
3. **For small edits**: Use apply_edit with old_text and new_text
   - old_text must match exactly (including whitespace)
   - Include enough context to make the match unique
4. **For large edits**: Use write_file with the complete new file content
5. **Track issues**: If a tool fails, note it in your final report

CRITICAL RULES:
- You MUST call tools to modify files. Text descriptions are NOT sufficient.
- If a tool fails, try to continue with other changes and report the failure
- Be precise with old_text in apply_edit - it must match exactly
- If apply_edit fails due to non-unique match, use write_file instead

After using tools to make all changes, output a JSON summary:
```json
{
    "files_modified": ["path/to/file1.gd", "path/to/file2.gd"],
    "result_summary": "Brief description of what was implemented (max 500 chars)",
    "issues_noticed": ["Issue 1: file not found", "Issue 2: edit failed"],
    "success": true
}
```

Success is true if most changes succeeded (even if some failed). If all changes failed, success is false.
"""


def create_implementor_agent(repo_root: Path):
    """Create the implementor agent with file tools and middleware.
    
    Args:
        repo_root: Repository root path for file search middleware
    
    Returns:
        Configured agent with middleware and tools
    """
    tools = [
        read_file,
        read_file_lines,
        write_file,
        apply_edit,
        create_file,
    ]
    
    # Build middleware list
    middleware = []
    
    # Add todo list middleware for multi-step task tracking
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    # Add file search middleware
    try:
        file_search_middleware = FilesystemFileSearchMiddleware(
            root_path=str(repo_root),
            use_ripgrep=True,
            max_file_size_mb=10,
        )
        middleware.append(file_search_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    # Add summarization middleware to control context length
    try:
        summarization_middleware = SummarizationMiddleware(
            model=planning_llm,  # Use planning LLM for summarization
            trigger=("tokens", 30000),  # Start summarizing at 30k tokens
            keep=("messages", 10),  # Keep last 10 messages
        )
        middleware.append(summarization_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    return create_agent(
        model=coding_llm,
        tools=tools,
        system_prompt=IMPLEMENTOR_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
    )


def implementor_node(state: WorkflowState) -> dict:
    """Implementor agent - execute implementation plan for current task.
    
    Takes the plan from Planner and makes actual file changes using tools.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with current_implementation_result set
    """
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
    
    # Load task tree and get current task
    task_tree = TaskTree.from_dict(tasks_dict)
    task = task_tree.tasks.get(current_task_id)
    
    if not task:
        return {
            "error": f"Task {current_task_id} not found",
            "messages": [f"Implementor: Task {current_task_id} not found"],
        }
    
    # Load project context if available
    project_context = None
    context_path = Path(repo_root) / "agents" / "context.md"
    if context_path.exists():
        try:
            project_context = context_path.read_text(encoding="utf-8")
        except Exception as e:
            project_context = None
    
    # Build the implementation prompt
    prompt_parts = [
        "="*70,
        "WORKING ENVIRONMENT",
        "="*70,
        f"Repository root: {repo_root}",
        "All file paths are relative to this root.",
        "",
    ]
    
    # Add project context
    if project_context:
        prompt_parts.append("="*70)
        prompt_parts.append("PROJECT CONTEXT")
        prompt_parts.append("="*70)
        prompt_parts.append(project_context[:3000])  # Limit to 3k chars
        prompt_parts.append("")
    
    # Add the implementation plan
    prompt_parts.extend([
        "="*70,
        "IMPLEMENTATION PLAN (execute this)",
        "="*70,
        "",
        current_implementation_plan,
        "",
        "="*70,
        "",
        "⚠️  CRITICAL INSTRUCTIONS:",
        "1. Use write_todos to track your work (5 steps)",
        "2. Read existing files first with read_file or read_file_lines",
        "3. Use tools (create_file, write_file, apply_edit) to make ALL changes",
        "4. Do NOT just describe changes - actually call the tools",
        "5. If a tool fails, note the issue and continue with other changes",
        "6. After using tools to make all changes, output the JSON summary",
        "",
        "Remember: The plan specifies WHAT to do. You must use tools to DO it.",
    ])
    
    implementation_prompt = "\n".join(prompt_parts)
    
    # Change to repo root directory for tool execution
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        
        # Create and run the implementor agent
        agent = create_implementor_agent(repo_root=Path(repo_root))
        
        # Track execution
        content = ""
        tool_calls_made = []
        
        try:
            # Use invoke to get full execution with tool calls
            result = agent.invoke(
                {"messages": [HumanMessage(content=implementation_prompt)]}
            )
            
            # Extract content and track tool calls
            if result and "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        msg_content = str(msg.content)
                        content += msg_content
                    
                    # Track tool calls for visibility
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            tool_calls_made.append(tool_name)
            
            # Check intermediate steps
            intermediate_steps = result.get("intermediate_steps", [])
            
        except Exception as e:
            pass  # Agent execution error - continue with fallback
        
        if not content or not content.strip():
            # No JSON output - create basic result from tool calls
            files_modified = []
            issues = []
            success = len(tool_calls_made) > 0
            
            impl_result = ImplementationResult(
                task_id=task.id,
                files_modified=files_modified,
                result_summary=f"Tool calls made: {', '.join(tool_calls_made[:5])}" if tool_calls_made else "No tool calls detected",
                issues_noticed=["No JSON summary output from agent"] + issues,
                success=success,
            )
        else:
            # Define expected schema for implementor output
            implementor_schema = {
                "files_modified": {
                    "type": list,
                    "required": False,
                    "default": [],
                },
                "result_summary": {
                    "type": str,
                    "required": True,
                    "max_length": 1000,
                },
                "issues_noticed": {
                    "type": list,
                    "required": False,
                    "default": [],
                },
                "success": {
                    "type": bool,
                    "required": True,
                },
            }
            
            # Use normaliser for JSON extraction and parsing
            norm_result = normalize_agent_output(
                content,
                implementor_schema,
                use_llm_summarization=False  # Simple truncation
            )
            
            if not norm_result.success:
                # Fallback - extract what we can
                impl_result = ImplementationResult(
                    task_id=task.id,
                    files_modified=[],
                    result_summary=content[:500] if content else "No summary available",
                    issues_noticed=[f"Normalisation failed: {norm_result.error}"],
                    success=len(tool_calls_made) > 0,
                )
            else:
                data = norm_result.data
                
                # Create ImplementationResult from normalized data
                impl_result = ImplementationResult(
                    task_id=task.id,
                    files_modified=data.get("files_modified", []),
                    result_summary=data.get("result_summary", ""),
                    issues_noticed=data.get("issues_noticed", []),
                    success=data.get("success", False),
                )
        
        # Store compressed result in task
        task.result_summary = impl_result.result_summary[:500]  # Ensure 500 char limit
        
        return {
            "tasks": task_tree.to_dict(),
            "current_implementation_result": impl_result.to_dict(),
            "messages": [f"Implementor: {'Success' if impl_result.success else 'Failed'} - {len(impl_result.files_modified)} files modified"],
        }
        
    except Exception as e:
        error_msg = f"Implementor failed executing plan for {current_task_id}: {e}"
        
        # Increment task attempt count
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
        # Restore original working directory
        os.chdir(original_cwd)
