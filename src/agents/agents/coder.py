"""Coder agent - implements changes based on consolidated plan."""

import json
import os
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    FilesystemFileSearchMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)

from ..task_states import AgentState, Change, ChangeStatus, ChangeResult, Requirement
from ..llm import coding_llm, planning_llm
from ..tools.read import read_file, read_file_lines
from ..tools.edit import write_file, apply_edit, create_file


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
    
    # Verify tools are properly bound
    tool_names = [tool.name for tool in tools if hasattr(tool, 'name')]
    if tool_names:
        print(f"💭 Tools registered: {', '.join(tool_names)}")
    
    # Build middleware list
    middleware = []
    
    # Add todo list middleware for multi-step task tracking
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
        print("💭 Todo list middleware enabled (provides write_todos tool)")
    except Exception as e:
        print(f"⚠️  Could not initialize todo list middleware: {e}")
    
    # Add file search middleware if repo_root is provided
    if repo_root:
        try:
            file_search_middleware = FilesystemFileSearchMiddleware(
                root_path=str(repo_root),
                use_ripgrep=True,
                max_file_size_mb=10,
            )
            middleware.append(file_search_middleware)
            print(f"💭 File search middleware enabled for: {repo_root} (provides glob_search and grep_search tools)")
        except Exception as e:
            print(f"⚠️  Could not initialize file search middleware: {e}")
    
    # Add summarization middleware to control context length
    # This will automatically summarize old messages when approaching token limits
    try:
        summarization_middleware = SummarizationMiddleware(
            model=planning_llm,  # Use planning LLM for summarization (smaller/faster)
            trigger=("tokens", 30000),  # Start summarizing when we hit 30k tokens (out of 40k context)
            keep=("messages", 10),  # Keep last 10 messages (recent context)
        )
        middleware.append(summarization_middleware)
        print("💭 Summarization middleware enabled (trigger: 30k tokens, keep: 10 messages)")
    except Exception as e:
        print(f"⚠️  Could not initialize summarization middleware: {e}")
    
    return create_agent(
        model=coding_llm,
        tools=tools,
        system_prompt=CODER_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
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
    # Debug: Check if implementation_prompt is valid
    if not change.implementation_prompt or len(change.implementation_prompt) < 100:
        print(f"\n⚠️  WARNING: change.implementation_prompt is suspiciously short!")
        print(f"   Length: {len(change.implementation_prompt) if change.implementation_prompt else 0} chars")
        print(f"   Content preview:\n{change.implementation_prompt[:300] if change.implementation_prompt else '(empty)'}")
    else:
        print(f"\n✓ Received implementation_prompt: {len(change.implementation_prompt)} chars")
        # Show first few lines to verify quality
        preview_lines = change.implementation_prompt.split('\n')[:5]
        print(f"   Preview: {chr(10).join(preview_lines)}")
    
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
        
        # Verify agent has tools bound
        if hasattr(agent, "tools"):
            tool_count = len(agent.tools) if agent.tools else 0
            print(f"💭 Agent created with {tool_count} tool(s) bound")
            if tool_count == 0:
                print("⚠️  WARNING: Agent has no tools! This will prevent file modifications.")
        elif hasattr(agent, "get_graph"):
            # For LangGraph agents, tools might be in the graph
            print("💭 Agent created (LangGraph-based, tools should be available)")
        else:
            print("⚠️  WARNING: Could not verify agent tool configuration")
        
        # Verify model supports tool calling
        try:
            # Check if model has bind_tools method (indicates tool calling support)
            if hasattr(coding_llm, "bind_tools"):
                print("💭 Model supports tool calling (bind_tools method available)")
            elif hasattr(coding_llm, "with_structured_output"):
                print("💭 Model may support structured output (with_structured_output available)")
            else:
                print("⚠️  WARNING: Could not verify model tool calling support")
                print("   Model might not support function calling - this could prevent tool usage")
        except Exception as e:
            print(f"⚠️  Could not check model capabilities: {e}")
        
        # Track execution state
        streamed_content = ""
        
        print(f"\n💭 Invoking agent (prompt length: {len(implementation_prompt)} chars)\n")
        print("💭 Tools will log their usage automatically when called.\n")
        
        try:
            # Use invoke to get full execution with tool calls
            # Tools will log themselves when called (see agents/tools/edit.py)
            result = agent.invoke(
                {"messages": [HumanMessage(content=implementation_prompt)]}
            )
            
            # Debug: Check for intermediate steps (tool calls that were attempted)
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                print(f"\n🔍 Debug: Found {len(intermediate_steps)} intermediate step(s)")
                for i, step in enumerate(intermediate_steps, 1):
                    if isinstance(step, tuple) and len(step) >= 2:
                        tool_call, tool_result = step[0], step[1]
                        print(f"   Step {i}: {tool_call}")
                        print(f"   Result: {str(tool_result)[:200]}...")
            else:
                print("\n⚠️  Debug: No intermediate_steps found in result")
                print("   This might indicate:")
                print("   1. Agent didn't attempt any tool calls")
                print("   2. Agent output format doesn't include intermediate_steps")
                print("   3. Check verbose output above for agent reasoning")
            
            # Extract content from result (for JSON summary parsing)
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        content_str = str(msg.content)
                        if content_str.strip():
                            streamed_content += content_str
                    
                    # Debug: Check for tool calls in messages
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"\n🔍 Debug: Found {len(msg.tool_calls)} tool call(s) in message")
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            print(f"   - Tool: {tool_name}")
            
        except Exception as e:
            print(f"\n⚠️  Agent execution error: {e}")
            import traceback
            traceback.print_exc()
        
        # Parse JSON summary if present (agent may output a summary)
        result_data = {"files_modified": [], "summary": streamed_content, "success": True}
        if "```json" in streamed_content:
            try:
                json_str = streamed_content.split("```json")[1].split("```")[0]
                result_data = json.loads(json_str.strip())
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Extract reported files from JSON summary
        reported_files = result_data.get("files_modified", [])
        
        # Create result - let the reviewer verify if files were actually modified
        change_result = ChangeResult(
            change_id=change.id,
            files_modified=reported_files,  # Trust the agent's report, reviewer will verify
            summary=result_data.get("summary", streamed_content[:500]),
            success=True,  # Always mark as success initially, reviewer will catch failures
        )
        
        change_results[change.id] = change_result.to_dict()
        changes[current_idx]["status"] = ChangeStatus.REVIEWING.value
        
        print(f"\n✓ Coder completed")
        if reported_files:
            print(f"  Reported files: {', '.join(reported_files)}")
        print(f"  (Reviewer will verify actual file modifications)")
        
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
