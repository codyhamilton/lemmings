"""Intake agent - transforms user request into remit and sequential milestones.

This is the entry point of the workflow. The Intake agent:
1. Understands what the user is REALLY asking for (explicit + implied)
2. Formulates a remit (scope boundary)
3. Creates sequential milestones (interim states, not tasks)
4. Does NOT create tasks - that's done by the expander

Output: remit + milestone list
"""

import json
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from typing_extensions import TypedDict

from ..state import WorkflowState, Milestone, MilestoneStatus
from ..llm import planning_llm
from ..normaliser import validate_and_normalize_lengths
from ..tools.rag import rag_search, perform_rag_search

class IntakeOutput(TypedDict):
    remit: str
    milestones: list[str]

# Schema for normalizer (single source of truth for validation)
INTAKE_SCHEMA = {
    "remit": {
        "type": str,
        "required": True,
        "max_length": 1000,
    },
    "milestones": {
        "type": list,  # Use list, not list[str] - normaliser handles list types
        "required": True,
    }
}

INTAKE_SYSTEM_PROMPT = """You are an intake agent for a game development project using Godot in GDScript.

Your job is to understand the user's request, define the overal request scope and break it into sequential milestones.

## Thinking steps - track using write_todos
1. Consider user intent carefully, using rag search to find application context and planning documentation
2. List implications, determine implicit needs related to the request
3. Formulate remit - Define the scope boundary - what are we achieving?
4. Break into sequential milestones. Create one or more milestones describing usable, functional interim states
5. Output final JSON with remit and milestones

## HOW TO DEFINE MILESTONES
- Think about user-facing capabilities: "User can X", "System supports Y"
- Milestones are short (max 200 chars)
- They describe what users can DO or what features WORK
- Avoid describing work. Not "Implement X", instead "User can Y"
- Think: "What can the user do after this milestone?" not "What work was done?"
- Example milestones: "Able to colonise an asteroid using a coloniser ship", "Colonies track key needs of power, population, life support", "Buildings and placement supported"

## How to define the remit:
- Remit represents the understood intent of the user request
- Takes into account user request, planning documentation and existing state
- Should be expansive but linked to the user intent
- Comprehensive scope - what was asked including explicit + implied necessary needs


Respond in valid JSON with this structure:
```json
{
    "remit": "Comprehensive scope - what we're achieving including explicit + implied needs",
    "milestones": [
        "Short description of interim state (max 200 chars)",
        "Next interim state",
    ]
}
```
"""


def create_intake_agent():
    """Create the intake agent with todo list middleware and RAG search."""
    middleware = []
    
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
        print("💭 Intake: Todo list middleware enabled for reasoning tracking")
    except Exception as e:
        print(f"⚠️  Could not initialize todo list middleware: {e}")
    
    # Tools for understanding the codebase
    tools = [rag_search]
    
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=INTAKE_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
        response_format=IntakeOutput,
    )


def intake_node(state: WorkflowState) -> dict:
    """Intake agent - transform user request into remit and initial tasks.
    
    This is the entry point of the workflow. Creates the initial task tree.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with remit and initial tasks
    """
    user_request = state["user_request"]
    repo_root = state["repo_root"]
    
    print("\n" + "="*70)
    print("🎯 INTAKE AGENT")
    print("="*70)
    print(f"User Request: {user_request}")
    print()
    
    # RAG search the project using the users request as the query
    # Use the library function directly for prefetching (not the tool)
    project_context = perform_rag_search(
        query=user_request,
        n_results=5,
        repo_root=repo_root,
    )
    
    # Build messages: separate RAG context from user request for clarity
    messages = []
    
    # Add RAG context as a separate message if available
    if project_context:
        messages.append(HumanMessage(
            content=f"Relevant codebase context (from semantic search):\n\n{project_context}"
        ))
    
    # Add the actual user request as a separate message
    messages.append(HumanMessage(content=f"User Request: {user_request}. Reason carefully and comprehensively."))
    
    try:
        # Create and run the intake agent
        agent = create_intake_agent()
        
        print("💭 Starting intake analysis with reasoning tracking...\n")
        
        # Use invoke to get reliable final result
        result = agent.invoke({"messages": messages})
        
        # Extract structured output (LangChain provides structured_response when response_format is used)
        data = result.get("structured_response") or result.get("structuredResponse")
        
        if not data:
            raise ValueError("No structured output received from intake agent (expected structured_response in result)")
        
        # Only validate lengths (structured output is already valid JSON)
        norm_result = validate_and_normalize_lengths(
            data,
            INTAKE_SCHEMA,
            use_llm_summarization=True
        )
        
        if norm_result.success:
            data = norm_result.data
        else:
            raise ValueError(f"Length validation failed: {norm_result.error}")
        
        # Extract remit
        remit = data.get("remit", "")
        if not remit:
            raise ValueError("No remit in intake output")
        
        # Create Milestone objects from output
        milestone_data_list = data.get("milestones", [])
        
        if not milestone_data_list:
            raise ValueError("No milestones in intake output")
        
        milestones = {}
        milestone_order = []
        
        # Create milestones in order
        for i, milestone in enumerate(milestone_data_list):
            milestone_id = f"milestone_{i+1:03d}"  # milestone_001, milestone_002, etc.
            description = milestone
            
            milestone = Milestone(
                id=milestone_id,
                description=description,
                status=MilestoneStatus.PENDING,
            )
            
            milestones[milestone_id] = milestone.to_dict()
            milestone_order.append(milestone_id)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"✓ Remit: {remit}")
        print(f"\n✓ Created {len(milestones)} milestones:")
        
        for milestone_id in milestone_order:
            milestone_dict = milestones[milestone_id]
            print(f"  {milestone_id}: {milestone_dict['description']}")
        
        print("="*70)
        
        return {
            "remit": remit,
            "milestones": milestones,
            "milestone_order": milestone_order,
            "active_milestone_id": None,  # Will be set after first milestone expansion
            "tasks": {},  # No tasks yet - expander will create them
            "iteration": 0,
            "tasks_created_this_iteration": 0,
            "messages": [f"Intake: Created remit and {len(milestones)} milestones"],
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse intake output: {e}"
        print(f"\n❌ {error_msg}")
        return {
            "remit": "",
            "milestones": {},
            "milestone_order": [],
            "active_milestone_id": None,
            "tasks": {},
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intake JSON error: {e}"],
        }
    except Exception as e:
        error_msg = f"Intake error: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "remit": "",
            "milestones": {},
            "milestone_order": [],
            "active_milestone_id": None,
            "tasks": {},
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intake exception: {e}"],
        }
