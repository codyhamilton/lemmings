"""Scope agents: InitialScope (entry) and ScopeReview (re-plan on Assessor escalation).

Shared: definitions, output schema, output-to-state mapping.
InitialScope: interpret user request and define initial scope; must produce at least one milestone.
ScopeReview: revise scope from prior work and divergence; may output empty milestones (scope complete).
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.tools.search import list_directory

from ..logging_config import get_logger
from ..task_states import WorkflowState
from ..llm import planning_llm
from ..subagents import explain_code, ask, web_search

logger = get_logger(__name__)


# =============================================================================
# Shared: output schema and definitions
# =============================================================================

class MilestoneItem(BaseModel):
    """Single milestone with description and area sketch."""

    description: str = Field(description="User-testable outcome (max 200 chars)")
    sketch: str = Field(
        default="",
        description="Rough areas of work: themes like 'data models', 'state integration' (max 150 chars)",
    )


class ScopeAgentOutput(BaseModel):
    """Structured output from scope agents (initial or review)."""

    remit: str = Field(description="Interpreted scope - what the user really wants (max 1000 chars)")
    explicit_needs: list[str] = Field(
        default_factory=list,
        description="Explicit requirements from the request",
    )
    implied_needs: list[str] = Field(
        default_factory=list,
        description="Implied/necessary needs for explicit needs to work",
    )
    milestones: list[MilestoneItem] = Field(
        description="Ordered list of user-testable outcomes with area sketches",
    )


SCOPE_DEFINITIONS = """
### REMIT
- A broad statement of the scope of work the user is asking for
- Defines boundary of outcome, not implementation

### Gap Analysis
- Specific list of gaps between current state and desired outcome
- Expressed in terms of features, not implementation

### Explicit/Implied Needs
- Represent logical conclusions of the remit. "If X then Y".
- Explicit are directly concluded from the user statement.
- Implied are necessary for explicit needs to work.
- The ability to test outcomes are always at least implied.

### MILESTONES
- Broad, sequential steps (1..n) from current state to desired state. No smaller than a sprint.
- Self-contained and testable.
- Describe outcomes, not implementation
"""


def _scope_output_to_state(scope_data: ScopeAgentOutput) -> dict:
    """Map scope agent output directly to state (milestones_list, active_milestone_index)."""
    milestones_list = []
    for i, m in enumerate(scope_data.milestones):
        if not m.description or not str(m.description).strip():
            continue
        milestones_list.append({
            "id": f"milestone_{i+1:03d}",
            "description": str(m.description)[:200],
            "sketch": (m.sketch or "")[:150],
            "order": i,
        })
    return {
        "remit": (scope_data.remit or "")[:1000],
        "milestones_list": milestones_list,
        "active_milestone_index": 0 if milestones_list else -1,
    }


# =============================================================================
# Initial scope agent (entry)
# =============================================================================

INITIAL_SCOPE_SYSTEM_PROMPT = """
## ROLE
You are a scope definition agent for a software development project. You interpret user requests,
assess the current state, and define milestones as user-testable outcomes.

## PRIMARY OBJECTIVE
Given a user request, produce:
1. A remit (scope boundary) - what the user really wants
2. A list of explicit and implied needs
3. A list of sequential milestones - each a discrete, testable user outcome

## Steps
1. Interpret the user request - figure out what they mean in the context of the current project
2. Produce a list of explicit and implied needs to satisfy the remit
3. Research current state and identify feature gaps between current and needs
4. Produce a list of (1..n) self-contained milestones to close the gaps. You must produce at least one milestone.

## DEFINITIONS
""" + SCOPE_DEFINITIONS + """
## Guidance
- Prefer logical completeness over strict interpretation.
- Don't worry about implementation details.
"""


def _create_initial_scope_agent():
    """Create the InitialScope agent with subagent tools."""
    tools = [explain_code, ask, web_search, list_directory]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=INITIAL_SCOPE_SYSTEM_PROMPT,
        response_format=ScopeAgentOutput,
    )


def _build_initial_messages(state: WorkflowState) -> list:
    """Build input messages for InitialScope (user request and repo only)."""
    user_request = state.get("user_request", "")
    repo_root = state.get("repo_root", "")
    parts = [
        "## USER REQUEST",
        user_request,
        "",
        f"## REPOSITORY ROOT: {repo_root}",
        "",
        "## INSTRUCTIONS",
        "Use explain_code and ask to research the codebase. Understand what exists and what's needed. "
        "Define the remit and milestones as user-testable outcomes. Output the structured JSON.",
    ]
    return [HumanMessage(content="\n".join(parts))]


def initial_scope_agent_node(state: WorkflowState) -> dict:
    """InitialScope - define remit and milestones from user request (entry point).

    Must produce at least one milestone. No prior work context.
    """
    logger.info("InitialScopeAgent starting")
    user_request = state.get("user_request", "")

    if not user_request:
        return {
            "remit": "",
            "status": "failed",
            "error": "No user request for InitialScopeAgent",
            "messages": ["InitialScopeAgent: No user request"],
        }

    messages = _build_initial_messages(state)

    try:
        agent = _create_initial_scope_agent()
        result = agent.invoke({"messages": messages})

        data = result.get("structured_response") if isinstance(result, dict) else None
        if not data:
            # Log what we did get so we can see if output was in another key or malformed
            result_keys = list(result.keys()) if isinstance(result, dict) else []
            last_content = ""
            if isinstance(result, dict) and "messages" in result:
                msgs = result.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "content") and isinstance(last.content, str):
                        last_content = last.content[:500] + ("..." if len(last.content) > 500 else "")
            logger.warning(
                "InitialScopeAgent: no structured_response. result_keys=%s last_content_snippet=%s",
                result_keys,
                last_content[:200] if last_content else "(none)",
            )
            raise ValueError(
                f"No structured output from InitialScopeAgent (result keys: {result_keys}). "
                "Model may have returned thinking/tool output instead of the required JSON, or output may be in an unexpected format."
            )

        scope_data = ScopeAgentOutput.model_validate(data)
        state_update = _scope_output_to_state(scope_data)

        if not state_update["milestones_list"]:
            raise ValueError("InitialScopeAgent produced no milestones")

        logger.info(
            "InitialScopeAgent completed: remit (%d chars), %d milestones",
            len(state_update["remit"]),
            len(state_update["milestones_list"]),
        )

        return {
            **state_update,
            "tasks": {},
            "iteration": 0,
            "tasks_created_this_iteration": 0,
            "carry_forward": [],
            "messages": [f"InitialScopeAgent: Defined remit and {len(state_update['milestones_list'])} milestones"],
        }

    except Exception as e:
        logger.error("InitialScopeAgent exception: %s", e, exc_info=True)
        return {
            "remit": "",
            "milestones_list": [],
            "active_milestone_index": -1,
            "status": "failed",
            "error": f"InitialScopeAgent failed: {e}",
            "messages": [f"InitialScopeAgent exception: {e}"],
        }


# =============================================================================
# Scope review agent (re-plan on Assessor escalation)
# =============================================================================

SCOPE_REVIEW_SYSTEM_PROMPT = """
## ROLE
You are a scope review agent for a software development project. You revise scope based on
prior work and divergence feedback from assessment.

## PRIMARY OBJECTIVE
Given the original user request, what has been accomplished, and any divergence notes, produce:
1. A revised remit (scope boundary)
2. A list of explicit and implied needs
3. Remaining milestones - what is left to do. Output an empty list if scope is complete.

## Steps
1. Review the prior work and divergence notes
2. Revise the remit and remaining milestones given this context. Keep completed work.
3. If all user outcomes are satisfied, output empty milestones. Otherwise list remaining milestones (1..n).

## DEFINITIONS
""" + SCOPE_DEFINITIONS + """
## Guidance
- Prefer logical completeness over strict interpretation.
- Empty milestones means scope is complete; use when prior work satisfies the remit.
"""


def _create_scope_review_agent():
    """Create the ScopeReview agent with subagent tools."""
    tools = [explain_code, ask, web_search]
    return create_agent(
        model=planning_llm,
        tools=tools,
        system_prompt=SCOPE_REVIEW_SYSTEM_PROMPT,
        response_format=ScopeAgentOutput,
    )


def _build_review_messages(state: WorkflowState) -> list:
    """Build input messages for ScopeReview (user request, repo, done_list, last_assessment)."""
    user_request = state.get("user_request", "")
    repo_root = state.get("repo_root", "")
    prior_work = state.get("done_list", [])
    divergence_analysis = state.get("last_assessment")

    parts = [
        "## USER REQUEST",
        user_request,
        "",
        f"## REPOSITORY ROOT: {repo_root}",
        "",
        "## RE-PLAN CONTEXT (Assessor escalation - revise scope)",
    ]
    if prior_work:
        parts.append("### What has been accomplished:")
        for i, item in enumerate(prior_work[:10], 1):
            if isinstance(item, dict):
                desc = item.get("description", item.get("task_description", str(item)))
                result = item.get("result", item.get("result_summary", ""))
                parts.append(f"  {i}. {desc[:80]}... → {result[:60]}...")
            else:
                parts.append(f"  {i}. {str(item)[:120]}")
        parts.append("")
    if divergence_analysis and isinstance(divergence_analysis, dict):
        notes = divergence_analysis.get("assessment_notes", divergence_analysis.get("divergence_analysis", ""))
        if notes:
            parts.append("### Divergence detected:")
            parts.append(notes[:500])
            parts.append("")
    parts.append("Revise the remit and remaining milestones given this context. Keep completed work.")
    parts.append("")
    parts.append(
        "## INSTRUCTIONS\n"
        "Use explain_code and ask if needed. Define the revised remit and remaining milestones. "
        "Output the structured JSON. Use empty milestones if scope is complete."
    )

    return [HumanMessage(content="\n".join(parts))]


def scope_review_agent_node(state: WorkflowState) -> dict:
    """ScopeReview - revise scope from prior work and divergence (re-plan).

    May output empty milestones; then sets status=complete.
    """
    logger.info("ScopeReviewAgent starting")
    user_request = state.get("user_request", "")

    if not user_request:
        return {
            "remit": "",
            "status": "failed",
            "error": "No user request for ScopeReviewAgent",
            "messages": ["ScopeReviewAgent: No user request"],
        }

    messages = _build_review_messages(state)

    try:
        agent = _create_scope_review_agent()
        result = agent.invoke({"messages": messages})

        data = result.get("structured_response") if isinstance(result, dict) else None
        if not data:
            raise ValueError("No structured output from ScopeReviewAgent")

        scope_data = ScopeAgentOutput.model_validate(data)
        state_update = _scope_output_to_state(scope_data)

        if not state_update["milestones_list"]:
            logger.info("ScopeReviewAgent: no remaining milestones (scope complete)")
            state_update["status"] = "complete"

        logger.info(
            "ScopeReviewAgent completed: remit (%d chars), %d milestones",
            len(state_update["remit"]),
            len(state_update["milestones_list"]),
        )

        return {
            **state_update,
            "tasks": {},
            "iteration": 0,
            "tasks_created_this_iteration": 0,
            "carry_forward": [],
            "messages": [f"ScopeReviewAgent: Revised scope, {len(state_update['milestones_list'])} remaining milestones"],
        }

    except Exception as e:
        logger.error("ScopeReviewAgent exception: %s", e, exc_info=True)
        return {
            "remit": "",
            "milestones_list": [],
            "active_milestone_index": -1,
            "status": "failed",
            "error": f"ScopeReviewAgent failed: {e}",
            "messages": [f"ScopeReviewAgent exception: {e}"],
        }
