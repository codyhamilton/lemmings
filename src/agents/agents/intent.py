"""Intent agent - interprets user request and formulates comprehensive remit.

This agent focuses solely on understanding what the user is asking for,
including both explicit and implied needs. It produces a remit (scope boundary)
without attempting to plan milestones or tasks.

Output: remit + explicit/implied needs
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from typing_extensions import TypedDict

from ..logging_config import get_logger
from ..task_states import WorkflowState

logger = get_logger(__name__)
from ..llm import planning_llm
from ..tools.rag import rag_search, perform_rag_search


class IntentOutput(TypedDict):
    remit: str
    explicit_needs: list[str]
    implied_needs: list[str]
    confidence: str


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
    logger.info("Intent agent starting")
    user_request = state["user_request"]
    repo_root = state["repo_root"]
    
    # RAG search the project using the user's request as the query
    project_context = perform_rag_search(
        query=user_request,
        n_results=5,
        repo_root=repo_root,
    )
    
    # 1. Relevant codebase context (if available)
    messages = [SystemMessage(f"RELEVANT CODEBASE CONTEXT (from semantic search): {project_context or 'No relevant codebase context found'}")]
    messages.append(HumanMessage(f"USER REQUEST: {user_request}"))
    
    try:
        result = create_intent_agent().invoke({"messages": messages})
        
        # Extract structured output (response_format provides structured_response key)
        data = result.get('structured_response') if isinstance(result, dict) else None
        
        if not data:
            raise ValueError("No structured output received from intent agent")
        
        # Extract remit and needs
        remit = data.get("remit", "")
        if not remit:
            raise ValueError("No remit in intent output")
        
        explicit_needs = data.get("explicit_needs", [])
        implied_needs = data.get("implied_needs", [])
        confidence = data.get("confidence", "medium")
        
        logger.info("Intent agent completed: remit extracted (confidence=%s)", confidence)
        return {
            "remit": remit,
            "explicit_needs": explicit_needs,
            "implied_needs": implied_needs,
            "messages": [f"Intent: Analyzed request (confidence: {confidence})"],
        }
        
    except Exception as e:
        logger.error("Intent agent exception: %s", e, exc_info=True)
        error_msg = f"Intent failed: {e}"
        return {
            "remit": "",
            "status": "failed",
            "error": error_msg,
            "messages": [f"Intent exception: {e}"],
        }
