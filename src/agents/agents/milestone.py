"""Milestone agent - breaks remit into sequential interim states.

This agent focuses solely on planning milestones (user-facing capabilities).
It takes the remit from the Intent agent and creates sequential milestones
without creating tasks.

Output: milestone list
"""

import json
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from typing_extensions import TypedDict

from ..state import WorkflowState, Milestone, MilestoneStatus
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
        print("💭 Milestone: Todo list middleware enabled for reasoning tracking")
    except Exception as e:
        print(f"⚠️  Could not initialize todo list middleware: {e}")
    
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
        print(f"\n❌ {error_msg}")
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone: {error_msg}"],
        }
    
    print("\n" + "="*70)
    print("🗺️  MILESTONE AGENT (Planning Interim States)")
    print("="*70)
    print(f"Remit: {remit}")
    print()
    
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
        
        print("💭 Starting milestone planning with reasoning tracking...\n")
        
        # Use invoke to get reliable final result
        result = agent.invoke({"messages": messages})
        
        # Extract structured output (LangChain provides structured_response when response_format is used)
        data = result.get("structured_response") or result.get("structuredResponse")
        
        if not data:
            raise ValueError("No structured output received from milestone agent (expected structured_response in result)")
        
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
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✓ Created {len(milestones)} milestones:")
        print(f"{'='*70}")
        
        for milestone_id in milestone_order:
            milestone_dict = milestones[milestone_id]
            print(f"  {milestone_id}: {milestone_dict['description']}")
        
        print("="*70)
        
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
        error_msg = f"Failed to parse milestone output: {e}"
        print(f"\n❌ {error_msg}")
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone JSON error: {e}"],
        }
    except Exception as e:
        error_msg = f"Milestone error: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone exception: {e}"],
        }
