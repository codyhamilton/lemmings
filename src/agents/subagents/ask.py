"""ask subagent - targeted factual questions about the codebase.

Wrapped as a LangChain tool for use by ScopeAgent and TaskPlanner.
Returns short, specific answers (not lengthy explanations).
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ..logging_config import get_logger
from ..llm import planning_llm
from ..tools.rag import rag_search
from ..tools.search import search_files, find_files_by_name
from ..tools.read import read_file_lines

logger = get_logger(__name__)

ASK_SYSTEM_PROMPT = """
## ROLE
You are a factual lookup subagent. Answer specific questions about the codebase concisely.

## PRIMARY OBJECTIVE
Given a factual question, find the answer and return it briefly. No lengthy explanations.

## PROCESS
1. Search for the relevant information (rag_search, search_files, find_files_by_name)
2. Read the minimal code needed to answer
3. Return a short, direct answer

## TOOLS AVAILABLE
- rag_search: Semantic search for relevant code
- search_files: Regex search for symbols/keywords
- find_files_by_name: Find files by glob pattern
- read_file_lines: Read specific line ranges

## OUTPUT STYLE
- Short and targeted (1-5 sentences typically)
- Direct answer first, minimal context
- Include specific file/line if relevant
- Max ~500 chars unless the question demands more

## CONSTRAINTS
- Use tools to verify - don't guess
- If the answer is "no" or "not found", say so clearly
- Be precise: "Colony.gd line 42" not "somewhere in the colony code"
"""


def _create_ask_agent():
    """Create the ask subagent with code lookup tools."""
    tools = [
        rag_search,
        search_files,
        find_files_by_name,
        read_file_lines,
    ]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=ASK_SYSTEM_PROMPT,
    )


def _extract_final_response(result: dict) -> str:
    """Extract the final AI response from agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    content_parts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            content_parts.append(str(msg.content))
    return "\n\n".join(content_parts) if content_parts else "No response generated."


_ask_agent = None


def _get_ask_agent():
    global _ask_agent
    if _ask_agent is None:
        _ask_agent = _create_ask_agent()
    return _ask_agent


@tool
def ask(query: str) -> str:
    """Answer a specific factual question about the codebase.

    Use this for quick lookups. Returns a short, targeted answer.

    Use when you need to know:
    - Does X exist?
    - Where is Y defined?
    - What does Z do? (brief)
    - Is there code for W?

    Args:
        query: Factual question (e.g., "Does any colony-related code exist?",
               "What properties does Entity have?", "Where is GameState defined?")

    Returns:
        Short, specific answer (typically 1-5 sentences)
    """
    try:
        logger.info("ask subagent invoked (once per call): %s", query[:80])
        agent = _get_ask_agent()
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        response = _extract_final_response(result)
        logger.debug("ask completed, response length: %d", len(response))
        return response
    except Exception as e:
        logger.warning("ask failed: %s", e)
        return f"ask error: {e}"
