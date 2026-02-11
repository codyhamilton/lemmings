"""Milestone agent - breaks remit into sequential interim states.

This agent focuses solely on planning milestones (user-facing capabilities).
It takes the remit from the Intent agent and creates sequential milestones
without creating tasks.

Output: milestone list
"""

import json
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from typing_extensions import TypedDict

from ..task_states import WorkflowState, Milestone, MilestoneStatus
from ..llm import planning_llm
from ..normaliser import validate_and_normalize_lengths
from ..tools.rag import rag_search


class MilestoneOutput(TypedDict):
    milestones: list[str]


# Schema for normalizer
MILESTONE_SCHEMA = {
    "milestones": {
        "type": list,
        "required": True,
    }
}


MILESTONE_SYSTEM_PROMPT = """
## ROLE
You are a milestone planning agent for a Godot/GDScript game development project.

## PRIMARY OBJECTIVE
Break a remit into sequential milestones representing user-facing interim states.

## PROCESS
1. Review the remit and identified needs
2. Identify logical interim states (what users can DO after each)
3. Order milestones by dependency (earlier enables later)
4. Verify each milestone is testable/observable

## THINKING STEPS (track using write_todos)
- TODO 1: "Understand scope" - What is the full scope from remit?
- TODO 2: "Identify interim states" - What are logical stopping points?
- TODO 3: "Order by dependency" - Which enables which?
- TODO 4: "Verify each is testable" - Can we observe when it's done?
- TODO 5: "Format output" - Create milestone list

## MILESTONE GUIDELINES
- Describe capabilities: "User can X", "System supports Y"
- NOT work descriptions: Avoid "Implement X", use "X is functional"
- Max 200 chars per milestone
- Each milestone should be independently testable
- Think: "What can the user do after this milestone?"

## OUTPUT FORMAT
```json
{
    "milestones": [
        "Milestone 1: User-facing capability description",
        "Milestone 2: Next interim state"
    ]
}
```

## CONSTRAINTS
- Do not create tasks (that's the Expander's job)
- Milestones are states, not activities
- Order matters: earlier milestones enable later ones
- Each milestone describes WHAT users can do, not HOW it's built
"""


def create_milestone_agent():
    """Create the milestone agent with todo list middleware and RAG search."""
    middleware = []
    
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception as e:
        pass  # Middleware initialization failure is non-fatal
    
    # Tools for understanding the codebase
    tools = [rag_search]
    
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=MILESTONE_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
        response_format=MilestoneOutput,
    )


def milestone_node(state: WorkflowState) -> dict:
    """Milestone agent - create sequential milestones from remit.
    
    This is the second step of the workflow. It takes the remit from
    the Intent agent and breaks it into interim states.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with milestones
    """
    remit = state.get("remit", "")
    explicit_needs = state.get("explicit_needs", [])
    implied_needs = state.get("implied_needs", [])
    repo_root = state["repo_root"]
    
    if not remit:
        error_msg = "No remit available for milestone planning"
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone: {error_msg}"],
        }
    
    # Build messages with separated context types
    messages = []
    
    # 1. Remit context (from Intent agent)
    context_parts = [
        "## REMIT",
        remit,
        "",
    ]
    
    # 2. Identified needs (structured context)
    if explicit_needs:
        context_parts.append("## EXPLICIT NEEDS")
        for need in explicit_needs:
            context_parts.append(f"- {need}")
        context_parts.append("")
    
    if implied_needs:
        context_parts.append("## IMPLIED NEEDS")
        for need in implied_needs:
            context_parts.append(f"- {need}")
        context_parts.append("")
    
    messages.append(HumanMessage(content="\n".join(context_parts)))
    
    # 3. Task instruction
    messages.append(HumanMessage(
        content="## TASK\nBreak this remit into sequential milestones. Think carefully about logical interim states."
    ))
    
    try:
        # Create and run the milestone agent
        agent = create_milestone_agent()
        
        # Use invoke for reliable tool execution (LangChain handles tool calls transparently)
        # Tool calls will be automatically executed by the agent executor
        result = agent.invoke({"messages": messages})
        
        final_result = result
        
        # Extract structured output (LangChain provides structured_response when response_format is used)
        # LangChain handles tool calls transparently - tools are executed automatically
        data = None
        
        if isinstance(final_result, dict):
            # Method 1: Check for structured_response (standard for response_format)
            data = final_result.get("structured_response") or final_result.get("structuredResponse")
            
            # Method 2: Check messages for the final AI message with structured output
            if not data and "messages" in final_result:
                messages = final_result["messages"]
                # Look for the last AI message which should contain the structured output
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        content = str(msg.content).strip()
                        # Try to parse as JSON if it looks like structured output
                        try:
                            if content.startswith('{') and content.endswith('}'):
                                parsed = json.loads(content)
                                if "milestones" in parsed:
                                    data = parsed
                                    break
                        except Exception as e:
                            # Not JSON or parse error - continue
                            pass
        
        if not data:
            raise ValueError("No structured output received from milestone agent")
        
        # Only validate lengths (structured output is already valid JSON)
        norm_result = validate_and_normalize_lengths(
            data,
            MILESTONE_SCHEMA,
            use_llm_summarization=True
        )
        
        if norm_result.success:
            data = norm_result.data
        else:
            raise ValueError(f"Length validation failed: {norm_result.error}")
        
        # Create Milestone objects from output
        milestone_data_list = data.get("milestones", [])
        
        if not milestone_data_list:
            raise ValueError("No milestones in output")
        
        milestones = {}
        milestone_order = []
        
        # Create milestones in order
        for i, milestone_desc in enumerate(milestone_data_list):
            milestone_id = f"milestone_{i+1:03d}"  # milestone_001, milestone_002, etc.
            
            milestone = Milestone(
                id=milestone_id,
                description=milestone_desc,
                status=MilestoneStatus.PENDING,
            )
            
            milestones[milestone_id] = milestone.to_dict()
            milestone_order.append(milestone_id)
        
        return {
            "milestones": milestones,
            "milestone_order": milestone_order,
            "active_milestone_id": None,  # Will be set after first milestone expansion
            "tasks": {},  # No tasks yet - expander will create them
            "iteration": 0,
            "tasks_created_this_iteration": 0,
            "messages": [f"Milestone: Created {len(milestones)} milestones"],
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"Milestone failed parsing output: {e}"
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone JSON error: {e}"],
        }
    except Exception as e:
        error_msg = f"Milestone failed creating milestones from remit: {e}"
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone exception: {e}"],
        }
