"""explain_code subagent - deep codebase research with detailed structural explanation.

Wrapped as a LangChain tool for use by ScopeAgent and TaskPlanner.
Returns detailed response with key files, classes, functions, and how they connect.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ..logging_config import get_logger
from ..llm import planning_llm
from ..tools.rag import rag_search
from ..tools.search import search_files, list_directory, find_files_by_name
from ..tools.read import read_file_lines

logger = get_logger(__name__)

EXPLAIN_CODE_SYSTEM_PROMPT = """
## ROLE
You are a codebase research subagent. You find and explain code to answer questions about how the codebase works.

## PRIMARY OBJECTIVE
Given a query, find relevant code and explain: what it does, how it works, what connects to it, and key files/classes/functions.

## PROCESS
1. Search for relevant code (rag_search, search_files, find_files_by_name)
2. Read the actual code to understand it
3. Synthesise a clear explanation with structure
4. Include key file paths, class names, function names

## TOOLS AVAILABLE
- rag_search: Semantic search for relevant code patterns
- search_files: Regex search for symbols/keywords
- find_files_by_name: Find files by glob pattern
- list_directory: Browse project structure
- read_file_lines: Read specific line ranges

## OUTPUT STYLE
- Detailed and structural (not terse)
- Include file paths and line references
- Explain relationships: "X calls Y", "A extends B"
- Summarise key findings at the end
- Keep under ~2000 chars unless the query demands more

## CONSTRAINTS
- Use tools to verify - don't guess
- Be accurate about file paths and symbols
- If nothing relevant found, say so clearly
"""


def _create_explain_code_agent():
    """Create the explain_code subagent with code research tools."""
    tools = [
        rag_search,
        search_files,
        find_files_by_name,
        list_directory,
        read_file_lines,
    ]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=EXPLAIN_CODE_SYSTEM_PROMPT,
    )


def _extract_final_response(result: dict) -> str:
    """Extract the final AI response from agent result."""
    messages = result.get("messages", [])
    # Find last AI message with content (final response after tool calls)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    # Fallback: concatenate all AI message content
    content_parts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            content_parts.append(str(msg.content))
    return "\n\n".join(content_parts) if content_parts else "No response generated."


_explain_agent = None


def _get_explain_agent():
    global _explain_agent
    if _explain_agent is None:
        _explain_agent = _create_explain_code_agent()
    return _explain_agent


@tool
def explain_code(query: str) -> str:
    """Find and explain code relevant to the query.

    Use this for deep codebase research. Returns a detailed explanation of how
    the code works, what connects to it, and key files/classes/functions.

    Use when you need to understand:
    - How a system or pattern works
    - What exists for a given concept
    - How components connect
    - Architecture and structure

    Args:
        query: Natural language question (e.g., "What game entity systems exist?",
               "How does GameState track entities?", "How does the Entity base class work?")

    Returns:
        Detailed explanation with file paths, classes, functions, and relationships
    """
    try:
        logger.debug("explain_code subagent invoked: %s", query[:80])
        agent = _get_explain_agent()
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        response = _extract_final_response(result)
        logger.debug("explain_code completed, response length: %d", len(response))
        return response
    except Exception as e:
        logger.warning("explain_code failed: %s", e)
        return f"explain_code error: {e}"
