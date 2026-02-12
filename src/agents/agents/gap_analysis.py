"""Gap Analysis agent - assesses codebase for high-level gaps per need.

This agent sits between Intent and Milestone. It uses the remit and identified
needs (explicit + implied) to assess the codebase and determine high-level gaps
for each need. Output feeds the Milestone agent for grounded planning.

Output: need_gaps (list of NeedGap dicts, one per need)
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import WorkflowState, NeedGap
from ..llm import planning_llm
from ..tools.search import search_files, list_directory, find_files_by_name
from ..tools.rag import rag_search

logger = get_logger(__name__)


class NeedGapOutput(BaseModel):
    """Single need gap from gap analysis."""

    need: str = Field(description="Description of the need (from intent)")
    need_type: str = Field(description="explicit or implied")
    gap_exists: bool = Field(description="Whether a gap exists for this need")
    current_state_summary: str = Field(description="What exists now (max 500 chars)")
    desired_state_summary: str = Field(description="What's needed (max 500 chars)")
    gap_description: str = Field(
        description="High-level delta - what's missing (max 1000 chars)"
    )
    relevant_areas: list[str] = Field(
        default_factory=list, description="Key files/dirs/areas"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Search terms for downstream agents"
    )


class GapAnalysisOutput(BaseModel):
    """Gap analysis structured output."""

    need_gaps: list[NeedGapOutput] = Field(
        description="One NeedGap per need assessed"
    )


GAP_ANALYSIS_SYSTEM_PROMPT = """
## ROLE
You are a remit-level gap analysis agent for a software development project.

## PRIMARY OBJECTIVE
Assess the codebase against each need (explicit and implied) from the remit.
For each need, determine the high-level gap between current state and desired state.
Output feeds the Milestone agent for grounded planning.

## PROCESS
1. Review the remit and identified needs
2. For each explicit need: search codebase, assess current vs desired, produce NeedGap
3. For each implied need: same process
4. Be honest: if a need is already satisfied, set gap_exists = false
5. Output need_gaps for each need (one per need)

## THINKING STEPS (track using write_todos)
- TODO 1: "Review remit and needs"
- TODO 2: "Search codebase for explicit needs"
- TODO 3: "Search codebase for implied needs"
- TODO 4: "Assess current vs desired for each"
- TODO 5: "Compile need_gaps"

## TOOLS AVAILABLE
- **rag_search**: Find semantically similar code patterns
- **list_directory**: Browse project structure
- **find_files_by_name**: Find files by name pattern
- **search_files**: Search for patterns in files using regex

## CONSTRAINTS
- One NeedGap per need; if no needs exist, produce one remit-level NeedGap
- Be honest: if something exists and works, set gap_exists = false
- Keep summaries compressed (respect max lengths)
- High-level assessment only - not task-level detail
- Use tools to VERIFY, don't guess
"""


def create_gap_analysis_agent():
    """Create the gap analysis agent with search tools."""
    middleware = []
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception:
        pass  # Middleware initialization failure is non-fatal

    tools = [
        rag_search,
        list_directory,
        find_files_by_name,
        search_files,
    ]

    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=GAP_ANALYSIS_SYSTEM_PROMPT,
        middleware=middleware,
        response_format=GapAnalysisOutput,
    )


def gap_analysis_node(state: WorkflowState) -> dict:
    """Gap Analysis agent - assess codebase for high-level gaps per need.

    Runs after Intent, before Milestone. Produces need_gaps for each
    explicit and implied need.

    Args:
        state: Current workflow state

    Returns:
        State update with need_gaps
    """
    logger.info("Gap Analysis agent starting")
    remit = state.get("remit", "")
    explicit_needs = state.get("explicit_needs", [])
    implied_needs = state.get("implied_needs", [])
    repo_root = state.get("repo_root", "")

    if not remit:
        error_msg = "No remit available for gap analysis"
        return {
            "need_gaps": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Gap Analysis: {error_msg}"],
        }

    # Build list of needs to assess
    needs_with_type = []
    for need in explicit_needs:
        if need and need.strip():
            needs_with_type.append((need.strip(), "explicit"))
    for need in implied_needs:
        if need and need.strip():
            needs_with_type.append((need.strip(), "implied"))

    # If no needs, use remit as single need to assess
    if not needs_with_type:
        needs_with_type = [(remit[:200], "explicit")]

    # Build messages
    messages = []
    context_parts = [
        "## REMIT",
        remit,
        "",
        "## NEEDS TO ASSESS",
    ]
    for need, need_type in needs_with_type:
        context_parts.append(f"- [{need_type}] {need}")
    context_parts.append("")
    messages.append(HumanMessage(content="\n".join(context_parts)))
    messages.append(
        HumanMessage(
            content=f"## INSTRUCTIONS\nRepository root: {repo_root}\n\n"
            "Use your tools to search and verify current state for each need. "
            "Assess what exists vs what's needed. Be honest - if a need is already "
            "satisfied, set gap_exists = false. Output need_gaps for each need above."
        )
    )

    def _synthetic_need_gaps() -> list[dict]:
        """Produce synthetic need_gaps when agent fails (e.g. no generations)."""
        return [
            NeedGap(
                need=need,
                need_type=need_type,
                gap_exists=True,
                current_state_summary="(Assessment unavailable)",
                desired_state_summary=need[:200],
                gap_description="Gap assumed; assessment failed or unavailable.",
                relevant_areas=[],
                keywords=[],
            ).to_dict()
            for need, need_type in needs_with_type
        ]

    try:
        agent = create_gap_analysis_agent()
        result = agent.invoke({"messages": messages})

        # Extract structured output (response_format provides structured_response key)
        data = result.get('structured_response') if isinstance(result, dict) else None

        if not data:
            logger.warning("Gap Analysis: No structured output, using synthetic need_gaps")
            return {
                "need_gaps": _synthetic_need_gaps(),
                "messages": ["Gap Analysis: Used synthetic gaps (no structured output)"],
            }

        # Convert structured output to NeedGap dicts (apply length constraints)
        if isinstance(data, GapAnalysisOutput):
            raw_gaps = [ng.model_dump() for ng in data.need_gaps]
        else:
            raw_gaps = data.get("need_gaps", [])
        if not isinstance(raw_gaps, list):
            raw_gaps = []

        need_gaps = []
        for i, item in enumerate(raw_gaps):
            if not isinstance(item, dict):
                continue
            need = item.get("need", f"Need {i+1}")
            need_type = item.get("need_type", "explicit")
            if need_type not in ("explicit", "implied"):
                need_type = "explicit"
            gap_exists = item.get("gap_exists", True)
            if isinstance(gap_exists, str):
                gap_exists = gap_exists.lower() in ("true", "yes", "1")
            current = str(item.get("current_state_summary", ""))[:500]
            desired = str(item.get("desired_state_summary", ""))[:500]
            gap_desc = str(item.get("gap_description", ""))[:1000]
            relevant = item.get("relevant_areas", [])
            if not isinstance(relevant, list):
                relevant = []
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []

            need_gaps.append(
                NeedGap(
                    need=need,
                    need_type=need_type,
                    gap_exists=gap_exists,
                    current_state_summary=current,
                    desired_state_summary=desired,
                    gap_description=gap_desc,
                    relevant_areas=relevant,
                    keywords=keywords,
                ).to_dict()
            )

        logger.info("Gap Analysis agent completed: %s need gaps", len(need_gaps))
        return {
            "need_gaps": need_gaps,
            "messages": [f"Gap Analysis: Assessed {len(need_gaps)} needs"],
        }

    except Exception as e:
        err_str = str(e)
        if "No generations found in stream" in err_str or "No content" in err_str:
            logger.warning("Gap Analysis: Agent produced no generations, using synthetic need_gaps")
            return {
                "need_gaps": _synthetic_need_gaps(),
                "messages": ["Gap Analysis: Used synthetic gaps (agent yielded no generations)"],
            }
        logger.error("Gap Analysis agent exception: %s", e, exc_info=True)
        error_msg = f"Gap Analysis failed: {e}"
        return {
            "need_gaps": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Gap Analysis exception: {e}"],
        }
