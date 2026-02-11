"""Intent agent - interprets user request and formulates comprehensive remit.

This agent focuses solely on understanding what the user is asking for,
including both explicit and implied needs. It produces a remit (scope boundary)
without attempting to plan milestones or tasks.

Output: remit + explicit/implied needs
"""

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from typing_extensions import TypedDict

from ..task_states import WorkflowState
from ..llm import planning_llm
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
You are an intent interpretation agent for a software development project.

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

## CONSTRAINTS
- Focus ONLY on understanding intent, not planning work
- Remit should be actionable but not prescriptive about HOW
- Include implied needs that are necessary for explicit needs to work
- Use RAG search to understand existing codebase patterns
- Confidence: high if request is clear, medium if some interpretation needed, low if ambiguous
"""


def create_intent_agent():
    """Create the intent agent."""
    
    return create_agent(
        model=planning_llm,
        tools=[rag_search],
        system_prompt=INTENT_SYSTEM_PROMPT,
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
        messages.append(SystemMessage(f"RELEVANT CODEBASE CONTEXT (from semantic search): {project_context}"))
    
    # 2. User request (task-specific input)
    messages.append(HumanMessage(f"USER REQUEST: {user_request}"))
    
    try:
        # Create and run the intent agent
        agent = create_intent_agent()
        
        # Use invoke for reliable tool execution (LangChain handles tool calls transparently)
        # Tool calls will be automatically executed by the agent executor
        # invoke() should handle the full agent loop: LLM -> tool calls -> tool execution -> LLM -> ... until done
        result = agent.invoke({"messages": messages})
        
        # Extract structured output (LangChain provides structured_response when response_format is used)
        # LangChain handles tool calls transparently - tools are executed automatically
        data = None
        
        if isinstance(result, dict):
            # Method 1: Check for structured_response (standard for response_format)
            data = result.get('structured_response') or result.get('structuredResponse')
            
            # Method 2: Check messages for the final AI message with structured output
            if not data and "messages" in result:
                messages = result["messages"]
                # Look for the last AI message which should contain the structured output
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        content = str(msg.content).strip()
                        # Try to parse as JSON if it looks like structured output
                        try:
                            if content.startswith('{') and content.endswith('}'):
                                parsed = json.loads(content)
                                if "remit" in parsed:
                                    data = parsed
                                    break
                        except Exception as e:
                            # Not JSON or parse error - continue
                            pass
        
        if not data:
            raise ValueError("No structured output received from intent agent")
        
        # Extract remit and needs
        remit = data.get("remit", "")
        if not remit:
            raise ValueError("No remit in intent output")
        
        explicit_needs = data.get("explicit_needs", [])
        implied_needs = data.get("implied_needs", [])
        confidence = data.get("confidence", "medium")
        
        return {
            "remit": remit,
            "explicit_needs": explicit_needs,
            "implied_needs": implied_needs,
            "messages": [f"Intent: Analyzed request (confidence: {confidence})"],
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"Intent failed parsing user request: {e}"
        return {
            "remit": "",
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intent JSON error: {e}"],
        }
    except Exception as e:
        error_msg = f"Intent failed: {e}"
        return {
            "remit": "",
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intent exception: {e}"],
        }
