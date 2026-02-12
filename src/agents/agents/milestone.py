"""Milestone agent - breaks remit into sequential interim states.

This agent focuses solely on planning milestones (user-facing capabilities).
It takes the remit from the Intent agent and creates sequential milestones
without creating tasks.

Output: milestone list
"""

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from typing_extensions import TypedDict

from ..logging_config import get_logger
from ..task_states import WorkflowState, Milestone, MilestoneStatus
from ..llm import planning_llm
from ..normaliser import validate_and_normalize_lengths
from ..tools.rag import rag_search

logger = get_logger(__name__)


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
You are a milestone planning agent for a software development project.

## PRIMARY OBJECTIVE
Consider the given remit and identify appropriate milestones toward its completion

## PROCESS
1. Roughly size the remit using t-shirt sizing
2. Divide the remit into milestones no smaller than a sprint (1 milestone is ok)
3. Ensure each milestone is a functional user state, defines what they can do
4. Order milestones sequentially

## MILESTONE GUIDELINES
- Should represent a logical state.
- Describe state or capability: "User can X", "System supports Y"
- Does not describe implementation, just what it does.
"""


def create_milestone_agent():
    """Create the milestone agent."""

    return create_agent(
        model=planning_llm,
        tools=[rag_search],
        system_prompt=MILESTONE_SYSTEM_PROMPT,
        response_format=MilestoneOutput,
    )


def milestone_node(state: WorkflowState) -> dict:
    """Milestone agent - create sequential milestones from remit.
    
    This runs after Gap Analysis. It takes the remit and need_gaps
    (gap assessment per need) to create grounded interim states.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with milestones
    """
    logger.info("Milestone agent starting")
    remit = state.get("remit", "")
    explicit_needs = state.get("explicit_needs", [])
    implied_needs = state.get("implied_needs", [])
    need_gaps = state.get("need_gaps", [])
    
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
        context_parts.append("### EXPLICIT NEEDS")
        for need in explicit_needs:
            context_parts.append(f"- {need}")
        context_parts.append("")
    
    if implied_needs:
        context_parts.append("### IMPLIED NEEDS")
        for need in implied_needs:
            context_parts.append(f"- {need}")
        context_parts.append("")

    # 3. Gap assessment (from Gap Analysis agent)
    if need_gaps:
        context_parts.append("### GAP ASSESSMENT BY NEED")
        for gap in need_gaps:
            need = gap.get("need", "")
            need_type = gap.get("need_type", "explicit")
            gap_exists = gap.get("gap_exists", True)
            current = gap.get("current_state_summary", "")
            desired = gap.get("desired_state_summary", "")
            gap_desc = gap.get("gap_description", "")
            status_str = "SATISFIED" if not gap_exists else "GAP EXISTS"
            context_parts.append(f"- [{need_type}] {need} ({status_str})")
            if gap_exists and gap_desc:
                context_parts.append(f"  Gap: {gap_desc[:300]}")
            elif not gap_exists and current:
                context_parts.append(f"  Current: {current[:200]}")
        context_parts.append("")
    
    messages.append(HumanMessage(content="\n".join(context_parts)))
    
    # 4. Task instruction
    task_instruction = (
        "## TASK\nBreak this remit into sequential milestones. Think carefully about logical interim states."
    )
    if need_gaps:
        task_instruction += (
            " Use the gap assessment above to inform which needs require work."
            " Prioritize needs with gaps; satisfied needs may be skipped or simplified."
        )
    messages.append(HumanMessage(content=task_instruction))
    
    try:
        # Create and run the milestone agent
        agent = create_milestone_agent()
        
        # Use invoke for reliable tool execution (LangChain handles tool calls transparently)
        # Tool calls will be automatically executed by the agent executor
        result = agent.invoke({"messages": messages})
        
        # Extract structured output (response_format provides structured_response key)
        data = result.get('structured_response') if isinstance(result, dict) else None
        
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
        
        logger.info("Milestone agent completed: created %s milestones", len(milestones))
        return {
            "milestones": milestones,
            "milestone_order": milestone_order,
            "active_milestone_id": None,  # Will be set after first milestone expansion
            "tasks": {},  # No tasks yet - expander will create them
            "iteration": 0,
            "tasks_created_this_iteration": 0,
            "messages": [f"Milestone: Created {len(milestones)} milestones"],
        }
        
    except Exception as e:
        logger.error("Milestone agent exception: %s", e, exc_info=True)
        error_msg = f"Milestone failed creating milestones from remit: {e}"
        return {
            "milestones": {},
            "milestone_order": [],
            "status": "failed",
            "error": error_msg,
            "messages": [f"Milestone exception: {e}"],
        }
