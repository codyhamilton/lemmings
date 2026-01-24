"""Intent agent - interprets user request and formulates comprehensive remit.

This agent focuses solely on understanding what the user is asking for,
including both explicit and implied needs. It produces a remit (scope boundary)
without attempting to plan milestones or tasks.

Output: remit + explicit/implied needs
"""

import json
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from typing_extensions import TypedDict

from ..state import WorkflowState
from ..llm import planning_llm
from ..normaliser import validate_and_normalize_lengths
from ..tools.rag import rag_search, perform_rag_search


class IntentOutput(TypedDict):
    remit: str
    explicit_needs: list[str]
    implied_needs: list[str]
    confidence: str


# Schema for normalizer
INTENT_SCHEMA = {
    "remit": {
        "type": str,
        "required": True,
        "max_length": 1000,
    },
    "explicit_needs": {
        "type": list,
        "required": True,
    },
    "implied_needs": {
        "type": list,
        "required": False,
        "default": [],
    },
    "confidence": {
        "type": str,
        "required": False,
        "default": "medium",
    }
}


INTENT_SYSTEM_PROMPT = """
## ROLE
You are an intent interpretation agent for a Godot/GDScript game development project.

## PRIMARY OBJECTIVE
Produce a comprehensive remit (scope definition) that captures what the user is asking for, including explicit and implied needs.

## PROCESS
1. Analyze the user's request for explicit requirements
2. Identify implied/necessary needs related to the request
3. Use RAG search to understand current codebase state and context
4. Formulate a remit that bounds the scope of work

## THINKING STEPS (track using write_todos)
- TODO 1: "Parse user request" - What is explicitly being asked?
- TODO 2: "Identify implications" - What else is needed for this to work?
- TODO 3: "Search codebase" - What exists now? What patterns to follow?
- TODO 4: "Formulate remit" - Comprehensive scope including explicit + implied needs

## OUTPUT FORMAT
```json
{
    "remit": "Comprehensive scope statement capturing explicit and implied needs (max 1000 chars)",
    "explicit_needs": ["Explicit need 1", "Explicit need 2"],
    "implied_needs": ["Implied need 1", "Implied need 2"],
    "confidence": "high|medium|low"
}
```

## CONSTRAINTS
- Focus ONLY on understanding intent, not planning work
- Remit should be actionable but not prescriptive about HOW
- Include implied needs that are necessary for explicit needs to work
- Use RAG search to understand existing codebase patterns
- Confidence: high if request is clear, medium if some interpretation needed, low if ambiguous
"""


def create_intent_agent():
    """Create the intent agent with todo list middleware and RAG search."""
    middleware = []
    
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
        print("💭 Intent: Todo list middleware enabled for reasoning tracking")
    except Exception as e:
        print(f"⚠️  Could not initialize todo list middleware: {e}")
    
    # Tools for understanding the codebase
    tools = [rag_search]
    
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=INTENT_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
        response_format=IntentOutput,
    )


def intent_node(state: WorkflowState) -> dict:
    """Intent agent - understand user request and formulate remit.
    
    This is the first step of the workflow. It interprets what the user
    is asking for without planning how to achieve it.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with remit and identified needs
    """
    user_request = state["user_request"]
    repo_root = state["repo_root"]
    
    print("\n" + "="*70)
    print("🎯 INTENT AGENT (Understanding Request)")
    print("="*70)
    print(f"User Request: {user_request}")
    print()
    
    # RAG search the project using the user's request as the query
    project_context = perform_rag_search(
        query=user_request,
        n_results=5,
        repo_root=repo_root,
    )
    
    # Build messages with separated context types
    messages = []
    
    # 1. Relevant codebase context (if available)
    if project_context:
        messages.append(HumanMessage(
            content=f"## RELEVANT CODEBASE CONTEXT\n(from semantic search)\n\n{project_context}"
        ))
    
    # 2. User request (task-specific input)
    messages.append(HumanMessage(
        content=f"## USER REQUEST\n{user_request}\n\nAnalyze this request carefully and comprehensively."
    ))
    
    try:
        # Create and run the intent agent
        agent = create_intent_agent()
        
        print("💭 Starting intent analysis with reasoning tracking...\n")
        
        # Use invoke to get reliable final result
        result = agent.invoke({"messages": messages})
        
        # Extract structured output (LangChain provides structured_response when response_format is used)
        data = result.get("structured_response") or result.get("structuredResponse")
        
        if not data:
            raise ValueError("No structured output received from intent agent (expected structured_response in result)")
        
        # Only validate lengths (structured output is already valid JSON)
        norm_result = validate_and_normalize_lengths(
            data,
            INTENT_SCHEMA,
            use_llm_summarization=True
        )
        
        if norm_result.success:
            data = norm_result.data
        else:
            raise ValueError(f"Length validation failed: {norm_result.error}")
        
        # Extract remit and needs
        remit = data.get("remit", "")
        if not remit:
            raise ValueError("No remit in intent output")
        
        explicit_needs = data.get("explicit_needs", [])
        implied_needs = data.get("implied_needs", [])
        confidence = data.get("confidence", "medium")
        
        # Print summary
        print(f"\n{'='*70}")
        print("INTENT ANALYSIS RESULT:")
        print(f"{'='*70}")
        print(f"Remit: {remit}")
        print(f"\nExplicit Needs ({len(explicit_needs)}):")
        for need in explicit_needs:
            print(f"  ✓ {need}")
        
        if implied_needs:
            print(f"\nImplied Needs ({len(implied_needs)}):")
            for need in implied_needs:
                print(f"  → {need}")
        
        print(f"\nConfidence: {confidence}")
        print("="*70)
        
        return {
            "remit": remit,
            "explicit_needs": explicit_needs,
            "implied_needs": implied_needs,
            "messages": [f"Intent: Analyzed request (confidence: {confidence})"],
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse intent output: {e}"
        print(f"\n❌ {error_msg}")
        return {
            "remit": "",
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intent JSON error: {e}"],
        }
    except Exception as e:
        error_msg = f"Intent error: {e}"
        print(f"\n❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "remit": "",
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intent exception: {e}"],
        }
