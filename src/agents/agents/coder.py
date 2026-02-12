"""Coder agent - implements changes based on consolidated plan."""

import os
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from pydantic import BaseModel, Field

from ..task_states import AgentState, Change, ChangeStatus, ChangeResult, Requirement
from ..llm import coding_llm, planning_llm
from ..tools.read import read_file, read_file_lines
from ..tools.edit import write_file, apply_edit, create_file


class CoderOutput(BaseModel):
    """Structured output for coder agent."""

    files_modified: list[str] = Field(default_factory=list, description="Files that were modified")
    summary: str = Field(description="Brief description of what was changed")
    success: bool = Field(description="Whether the implementation succeeded")


CODER_SYSTEM_PROMPT = """You are a code implementation agent for a Godot/GDScript game project.

You will receive a detailed implementation prompt in markdown format that contains:
- Requirements to fulfill with measurable outcomes
- Specific files to create/modify
- Code snippets and step-by-step guidance
- Any consistency notes or resolved conflicts

**CRITICAL: You MUST use tools to make changes. Do NOT just describe changes in text.**

WORKFLOW:
1. Read existing files using read_file or read_file_lines to understand current code
2. Use write_file, apply_edit, or create_file tools to actually make the changes
3. After ALL changes are made using tools, output a JSON summary

FILE PATHS:
- Use RELATIVE paths from project root (e.g., "scripts/domain/models/Colony.gd")
- Do NOT use Godot's "res://" prefix - just use plain relative paths
- All paths are relative to the current working directory (project root)

AVAILABLE TOOLS (USE THESE TO MAKE CHANGES):
- read_file(path): Read entire file content
- read_file_lines(path, start_line, end_line): Read specific line range
- write_file(path, content): Write complete file content (creates or overwrites)
- apply_edit(path, old_string, new_string): Apply a text replacement edit
- create_file(path, content): Create a new file with content
- glob_search(pattern): Find files by pattern (e.g., "**/*.gd")
- grep_search(pattern, path): Search for text pattern in files
- write_todos(todos): Create a todo list for multi-step tasks

**IMPORTANT RULES:**
- You MUST call tools to modify files. Text descriptions are NOT sufficient.
- Use read_file first to see existing code before modifying
- Use write_file for new files or complete rewrites
- Use apply_edit for small changes to existing files
- After using tools to make all changes, then output the JSON summary below

After all changes are complete (using tools), output a summary:
```json
{
    "files_modified": ["scripts/path/to/file.gd"],
    "summary": "Brief description of what was changed",
    "success": true
}
```
"""


def create_coder_agent(repo_root: Path | None = None):
    """Create the coder agent with tools using create_agent (LangChain 1.2+).
    
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
    except Exception:
        pass
    
    # Add file search middleware if repo_root is provided
    if repo_root:
        try:
            file_search_middleware = FilesystemFileSearchMiddleware(
                root_path=str(repo_root),
                use_ripgrep=True,
                max_file_size_mb=10,
            )
            middleware.append(file_search_middleware)
        except Exception:
            pass
    
    # Add summarization middleware to control context length
    # This will automatically summarize old messages when approaching token limits
    try:
        summarization_middleware = SummarizationMiddleware(
            model=planning_llm,  # Use planning LLM for summarization (smaller/faster)
            trigger=("tokens", 30000),  # Start summarizing when we hit 30k tokens (out of 40k context)
            keep=("messages", 10),  # Keep last 10 messages (recent context)
        )
        middleware.append(summarization_middleware)
    except Exception:
        pass
    
    return create_agent(
        model=coding_llm,
        tools=tools,
        system_prompt=CODER_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
        response_format=CoderOutput,
    )


def coder_node(state: AgentState) -> dict:
    """Implement the current change.
    
    Phase 5: Code
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with change_results populated
    """
    changes = state["changes"]
    current_idx = state["current_change_index"]
    repo_root = state["repo_root"]
    change_results = dict(state.get("change_results", {}))
    
    if current_idx >= len(changes):
        return {
            "messages": ["Coder: No more changes to implement"],
        }
    
    change = Change.from_dict(changes[current_idx])
    
    # Check if this is a retry after review failure
    existing_review = state.get("reviews", {}).get(change.id)
    is_retry = existing_review and not existing_review.get("passed", False)
    
    # Update change status
    changes[current_idx]["status"] = ChangeStatus.CODING.value
    
    # Display status
    from ..workflow_status import display_agent_header, Phase, get_status
    status = get_status()
    status.set_change_progress(current_idx + 1, len(changes))
    
    display_agent_header(
        "Coder Agent",
        Phase.CODE,
        f"Change {current_idx + 1}/{len(changes)}" + (" (RETRY)" if is_retry else ""),
        {
            "Change": change.description,
            "Requirements": ", ".join(change.requirement_ids),
        }
    )
    
    # Get project context
    project_context = state.get("project_context")
    
    # Build the implementation prompt - the change contains a detailed markdown prompt with code snippets
    # Structure the prompt with clear sections
    prompt_parts = []
    
    # SECTION 1: Working Environment
    prompt_parts.append("="*70)
    prompt_parts.append("WORKING ENVIRONMENT")
    prompt_parts.append("="*70)
    prompt_parts.append(f"Repository root: {repo_root}")
    prompt_parts.append("All file paths below are relative to this root.")
    prompt_parts.append("")
    
    # SECTION 2: Project Context (high-level understanding)
    if project_context:
        prompt_parts.append("="*70)
        prompt_parts.append("PROJECT CONTEXT")
        prompt_parts.append("="*70)
        prompt_parts.append(project_context)
        prompt_parts.append("")
    
    # SECTION 3: Relevant Design Docs (domain knowledge)
    relevant_design_docs = state.get("relevant_design_docs", {})
    if relevant_design_docs:
        prompt_parts.append("="*70)
        prompt_parts.append("RELEVANT DESIGN DOCUMENTATION")
        prompt_parts.append("="*70)
        for doc_file, doc_content in relevant_design_docs.items():
            prompt_parts.append(f"\n--- {doc_file} ---")
            prompt_parts.append(doc_content[:3000])  # Increased from 2000
        prompt_parts.append("")
    
    # SECTION 4: Implementation Instructions (the actual task)
    prompt_parts.append("="*70)
    prompt_parts.append("IMPLEMENTATION INSTRUCTIONS")
    prompt_parts.append("="*70)
    prompt_parts.append("")
    prompt_parts.append(change.implementation_prompt)
    prompt_parts.append("")
    prompt_parts.append("="*70)
    prompt_parts.append("\n")
    prompt_parts.append("⚠️  CRITICAL INSTRUCTIONS:")
    prompt_parts.append("1. You MUST use tools (write_file, apply_edit, create_file) to make changes")
    prompt_parts.append("2. Do NOT just describe changes in text - actually call the tools")
    prompt_parts.append("3. Read files first with read_file or read_file_lines to see existing code")
    prompt_parts.append("4. Then use write_file/create_file/apply_edit to make the actual changes")
    prompt_parts.append("5. Only after using tools to make all changes, output the JSON summary")
    prompt_parts.append("")
    
    if is_retry and existing_review:
        prompt_parts.extend([
            "",
            "⚠️ REVIEW FEEDBACK FROM PREVIOUS ATTEMPT:",
            existing_review.get("feedback", ""),
            "Issues to fix:",
            *[f"- {issue}" for issue in existing_review.get("issues", [])],
            "",
            "Address these issues in your implementation.",
        ])
    
    implementation_prompt = "\n".join(prompt_parts)
    
    # Change to repo root directory for tool execution
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        
        # Create and run the coder agent with middleware
        agent = create_coder_agent(repo_root=Path(repo_root))

        # Invoke agent (response_format ensures structured output after tool calls)
        result = agent.invoke(
            {"messages": [HumanMessage(content=implementation_prompt)]}
        )
        
        # Extract structured output (response_format provides structured_response key)
        data = result.get('structured_response') if isinstance(result, dict) else None
        
        if data and isinstance(data, CoderOutput):
            change_result = ChangeResult(
                change_id=change.id,
                files_modified=data.files_modified,
                summary=data.summary[:500],
                success=data.success,
            )
        elif data and isinstance(data, dict):
            change_result = ChangeResult(
                change_id=change.id,
                files_modified=data.get("files_modified", []),
                summary=data.get("summary", "")[:500],
                success=data.get("success", True),
            )
        else:
            change_result = ChangeResult(
                change_id=change.id,
                files_modified=[],
                summary="No structured output from coder agent",
                success=False,
            )
        
        change_results[change.id] = change_result.to_dict()
        changes[current_idx]["status"] = ChangeStatus.REVIEWING.value

        return {
            "changes": changes,
            "change_results": change_results,
            "messages": [f"Coder completed change: {change.id}"],
        }
        
    except Exception as e:
        changes[current_idx]["status"] = ChangeStatus.FAILED.value
        return {
            "changes": changes,
            "change_results": change_results,
            "messages": [f"Coder error: {e}"],
            "error": f"Implementation failed: {e}",
        }
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
