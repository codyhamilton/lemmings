"""Expander agent - discovers new tasks through IF X THEN Y reasoning.

This agent looks at completed and open tasks and identifies what else is needed
to fully satisfy the remit. It applies expansion patterns to discover missing
pieces and creates new tasks with proper dependencies.

Key role: Continuous expansion during workflow execution, not just at start.
"""

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from ..task_states import WorkflowState, Task, TaskStatus, TaskTree
from ..llm import planning_llm
from ..tools.rag import rag_search

logger = get_logger(__name__)


class IdentifiedNeedOutput(BaseModel):
    """Identified need from expansion analysis."""

    need: str = Field(description="Description of what's missing")
    covered_by_open_task: str | None = Field(
        default=None,
        description="task_id if covered by open task, else null",
    )
    reasoning: str = Field(description="Why this is needed (IF X THEN Y)")


class NewTaskOutput(BaseModel):
    """New task from expander."""

    description: str = Field(description="What needs to be done")
    measurable_outcome: str = Field(description="How we know it's complete")
    tags: list[str] = Field(default_factory=list, description="Tags for grouping")
    estimated_complexity: str | None = Field(
        default=None,
        description="simple, moderate, or complex",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs this depends on (same milestone only)",
    )
    reasoning: str = Field(description="Why this task is needed")


class ExpanderOutput(BaseModel):
    """Expander structured output."""

    analysis: str = Field(
        default="No analysis provided",
        description="Summary of what was reviewed and patterns applied",
    )
    identified_needs: list[IdentifiedNeedOutput] = Field(
        default_factory=list,
        description="Needs identified with coverage check",
    )
    new_tasks: list[NewTaskOutput] = Field(
        default_factory=list,
        description="New tasks for uncovered needs",
    )


EXPANDER_SYSTEM_PROMPT = """
## ROLE
You are a task discovery agent for a software development project.

## PRIMARY OBJECTIVE
Expand a single milestone into tasks using "If X Therefore Y" reasoning to identify what's needed to achieve the milestone goal.

## PROCESS
1. Review milestone description - What goal are we achieving?
2. Review completed work from previous milestones - What context exists?
3. Review existing tasks in milestone - What is already planned?
4. Apply expansion patterns - What else is needed? (IF X THEN Y)
5. Check coverage - Are identified needs already covered?
6. Create new tasks - For uncovered needs only
7. Establish dependencies - How do new tasks relate? (within milestone only)

## THINKING STEPS (track using write_todos)
- TODO 1: "Review milestone description"
- TODO 2: "Review completed work from previous milestones"
- TODO 3: "Review existing tasks in milestone"
- TODO 4: "Apply expansion patterns"
- TODO 5: "Check coverage"
- TODO 6: "Create new tasks"
- TODO 7: "Establish dependencies"

EXPANSION PATTERNS (IF X THEN Y):

1. **Data-driven**:
   - IF action X exists → THEN data model for X must exist
   - IF we store Y → THEN we need data structure for Y
   - IF we track state Z → THEN Z must be initialized somewhere

2. **UI-driven**:
   - IF mechanic X exists → THEN UI to interact with X must exist
   - IF we display data Y → THEN UI component for Y must exist
   - IF user action Z → THEN button/input for Z must exist

3. **Integration-driven**:
   - IF feature A and B both exist → THEN integration point must exist
   - IF system X depends on Y → THEN connection logic must exist
   - IF we modify Z → THEN anything referencing Z may need updates

4. **Completion-driven**:
   - IF user flow starts at X → THEN it must end at Y
   - IF we create resource A → THEN we need cleanup/disposal for A
   - IF we open dialog B → THEN we need close/cancel for B

5. **Signal-driven**:
   - IF state changes → THEN something must observe/react
   - IF event occurs → THEN listeners must be registered
   - IF we emit signal X → THEN handler for X must exist

6. **Validation-driven**:
   - IF we accept input X → THEN we need validation for X
   - IF we store data Y → THEN we need bounds checking for Y
   - IF action Z can fail → THEN we need error handling for Z

## TASK COVERAGE CHECK
Before creating a new task, check if it's already covered by open tasks:
- Compare descriptions (similar intent?)
- Check tags (same category?)
- Look at measurable outcomes (same goal?)

Only create new tasks for truly uncovered needs.

## DEPENDENCY REASONING
When creating new tasks, establish dependencies WITHIN THE SAME MILESTONE:
- Tasks can ONLY depend on other tasks in the same milestone
- Data models created BEFORE things that use them
- UI created AFTER mechanics they interact with
- Integration AFTER systems being integrated exist
- DO NOT create dependencies on tasks from previous milestones

## CONSTRAINTS
- Only create tasks for the milestone being expanded
- Tasks can ONLY depend on other tasks in the same milestone
- Only create tasks for needs NOT covered by open tasks
- Be specific with dependencies - use actual task IDs
- Keep new tasks at same granularity level as existing tasks
- If everything is covered, return empty new_tasks array
- Each new task must have clear "IF X THEN Y" reasoning
- Use completed work from previous milestones as CONTEXT only, not dependencies
- Apply ALL expansion patterns systematically within milestone scope
- Focus on LOGICAL needs to achieve the milestone's interim state
"""


def create_expander_agent():
    """Create the expander agent with RAG search."""
    middleware = []
    try:
        todo_middleware = TodoListMiddleware()
        middleware.append(todo_middleware)
    except Exception:
        pass  # Middleware initialization failure is non-fatal

    return create_agent(
        model=planning_llm,
        tools=[rag_search],
        system_prompt=EXPANDER_SYSTEM_PROMPT,
        middleware=middleware,
        response_format=ExpanderOutput,
    )


def expander_node(state: WorkflowState) -> dict:
    """Expander agent - expand a milestone into tasks.
    
    This agent expands a single milestone into tasks. It only expands the next
    milestone (after the active one is complete). Tasks can only depend on
    other tasks within the same milestone.
    
    Args:
        state: Current workflow state
    
    Returns:
        State update with new tasks added to tree for the milestone
    """
    logger.info("Expander agent starting")
    remit = state["remit"]
    tasks_dict = state["tasks"]
    milestones = state.get("milestones", {})
    milestone_order = state.get("milestone_order", [])
    active_milestone_id = state.get("active_milestone_id")
    iteration = state.get("iteration", 0)
    completed_task_ids = state.get("completed_task_ids", [])
    
    # Load task tree
    task_tree = TaskTree.from_dict(tasks_dict)
    
    # Determine which milestone to expand
    # When active_milestone_id exists, expand it (e.g. after advance_milestone or gap re-expansion).
    # When no active, expand first milestone.
    milestone_to_expand_id = None
    if not active_milestone_id:
        # No active milestone - expand first milestone
        if milestone_order:
            milestone_to_expand_id = milestone_order[0]
        else:
            return {
                "tasks": task_tree.to_dict(),
                "tasks_created_this_iteration": 0,
                "messages": ["Expander: No milestones defined"],
            }
    else:
        # Active exists - expand the active milestone
        if active_milestone_id in milestone_order:
            milestone_to_expand_id = active_milestone_id
        else:
            return {
                "tasks": task_tree.to_dict(),
                "tasks_created_this_iteration": 0,
                "messages": [f"Expander: Active milestone {active_milestone_id} not found"],
            }
    
    # Get milestone info
    milestone_dict = milestones.get(milestone_to_expand_id)
    if not milestone_dict:
        return {
            "tasks": task_tree.to_dict(),
            "tasks_created_this_iteration": 0,
            "messages": [f"Expander: Milestone {milestone_to_expand_id} not found"],
        }
    
    milestone_desc = milestone_dict.get("description", "")
    
    # Get completed tasks from previous milestones (for context)
    previous_completed_tasks = []
    if active_milestone_id:
        # Get all completed tasks from previous milestones
        for tid in completed_task_ids:
            if tid in task_tree.tasks:
                task = task_tree.tasks[tid]
                # Check if task is from a previous milestone
                if task.milestone_id and task.milestone_id in milestone_order:
                    task_milestone_index = milestone_order.index(task.milestone_id)
                    expand_milestone_index = milestone_order.index(milestone_to_expand_id)
                    if task_milestone_index < expand_milestone_index:
                        previous_completed_tasks.append(task)
    
    # Get existing tasks in milestone being expanded
    existing_milestone_tasks = task_tree.get_tasks_by_milestone(milestone_to_expand_id)
    open_milestone_tasks = [
        t for t in existing_milestone_tasks
        if t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS)
    ]
    
    stats = task_tree.get_statistics()
    
    # Build the expansion prompt
    prompt_parts = [
        f"Remit (scope boundary): {remit}",
        "",
        f"MILESTONE TO EXPAND: {milestone_to_expand_id}",
        f"Milestone Description: {milestone_desc}",
        "",
        "="*70,
        "COMPLETED WORK FROM PREVIOUS MILESTONES (for context only):",
        "="*70,
    ]
    
    if previous_completed_tasks:
        for task in previous_completed_tasks[-15:]:  # Last 15 for context
            summary = task_tree.get_task_summary(task.id, max_chars=250)
            prompt_parts.append(f"\n{summary}")
            if task.result_summary:
                prompt_parts.append(f"  Result: {task.result_summary[:150]}")
    else:
        prompt_parts.append("No previous milestones completed yet")
    
    prompt_parts.extend([
        "",
        "="*70,
        f"EXISTING TASKS IN MILESTONE {milestone_to_expand_id} (what is already planned):",
        "="*70,
    ])
    
    if open_milestone_tasks:
        for task in open_milestone_tasks[:20]:  # Limit to 20
            summary = task_tree.get_task_summary(task.id, max_chars=250)
            prompt_parts.append(f"{summary}")
    else:
        prompt_parts.append("No existing tasks in this milestone yet")
    
    # Add any gaps from assessor
    last_assessment = state.get("last_assessment")
    if last_assessment:
        uncovered_gaps = last_assessment.get("uncovered_gaps", [])
        if uncovered_gaps:
            prompt_parts.extend([
                "",
                "="*70,
                "UNCOVERED GAPS (from Assessor):",
                "="*70,
            ])
            for gap in uncovered_gaps[:10]:
                prompt_parts.append(f"- {gap}")
    
    full_instructions = [
        "",
        "="*70,
        "",
        "INSTRUCTIONS:",
        "1. Use write_todos to track your reasoning through milestone expansion",
        "2. Focus on achieving the milestone's interim state: " + milestone_desc,
        "3. Use completed work from previous milestones as CONTEXT (what exists now)",
        "4. Apply expansion patterns (data, UI, integration, completion, signal, validation)",
        "5. Check if identified needs are covered by existing tasks in this milestone",
        "6. Create new tasks ONLY for uncovered needs in THIS milestone",
        "7. Establish dependencies using actual task IDs FROM THIS MILESTONE ONLY",
        "8. Output analysis, identified_needs, and new_tasks",
        "",
        "CRITICAL RULES:",
        "- All new tasks must belong to milestone: " + milestone_to_expand_id,
        "- Tasks can ONLY depend on other tasks in the same milestone",
        "- Use previous milestone work as context, not dependencies",
        "- Stay within the milestone scope - don't expand beyond it",
    ]
    simplified_instructions = [
        "",
        "="*70,
        "",
        "INSTRUCTIONS (simplified retry):",
        "Expand this milestone into concrete tasks. Output analysis, identified_needs, and new_tasks.",
        "Each task in new_tasks needs: description, measurable_outcome, depends_on (list of task IDs).",
        "All tasks must belong to milestone " + milestone_to_expand_id + ".",
    ]
    
    last_error = None
    try:
        for attempt in range(2):
            is_retry = attempt == 1
            instructions = simplified_instructions if is_retry else full_instructions
            prompt_parts_this = prompt_parts + instructions
            
            agent = create_expander_agent()
            result = agent.invoke({"messages": [HumanMessage(content="\n".join(prompt_parts_this))]})
            
            # Extract structured output (response_format provides structured_response key)
            data = result.get('structured_response') if isinstance(result, dict) else None

            if not data:
                last_error = "No expansion output"
                if is_retry:
                    return {
                        "tasks": task_tree.to_dict(),
                        "tasks_created_this_iteration": 0,
                        "error": f"Expander failed after retry: {last_error}",
                        "messages": [f"Expander: {last_error} (retry also failed)"],
                    }
                logger.info("Expander: No expansion output, retrying with simplified prompt")
                continue

            # Extract from structured output
            if isinstance(data, ExpanderOutput):
                analysis = data.analysis
                identified_needs = [n.model_dump() for n in data.identified_needs]
                new_task_data = [t.model_dump() for t in data.new_tasks]
            else:
                analysis = data.get("analysis", "")
                identified_needs = data.get("identified_needs") or []
                new_task_data = data.get("new_tasks") or []
            
            # Create new tasks
            new_tasks_created = 0
            if new_task_data:
                # Generate new task IDs
                max_task_num = 0
                for task_id in task_tree.tasks.keys():
                    num_str = task_id.replace("task_", "").lstrip("0")
                    if num_str:
                        max_task_num = max(max_task_num, int(num_str))
                
                for i, task_data in enumerate(new_task_data):
                    new_task_id = f"task_{max_task_num + i + 1:03d}"
                    
                    # Get max_attempts from state
                    max_task_retries = state.get("max_task_retries", 3)
                    
                    # Create task
                    new_task = Task(
                        id=new_task_id,
                        description=task_data.get("description", ""),
                        measurable_outcome=task_data.get("measurable_outcome", ""),
                        status=TaskStatus.PENDING,
                        depends_on=[],  # Will be set below
                        milestone_id=milestone_to_expand_id,  # Associate with milestone
                        created_by="expander",
                        created_at_iteration=iteration,
                        tags=task_data.get("tags") or [],
                        estimated_complexity=task_data.get("estimated_complexity"),
                        max_attempts=max_task_retries,
                    )
                    
                    # Parse dependencies - only allow dependencies within the same milestone
                    depends_on = task_data.get("depends_on") or []
                    if isinstance(depends_on, list):
                        # Normalize and verify dependencies exist AND are in the same milestone
                        valid_deps = []
                        for dep in depends_on:
                            if isinstance(dep, str):
                                # Handle both "task_1" and "task_001" formats
                                dep_num = dep.replace("task_", "").lstrip("0")
                                if dep_num:
                                    normalized_dep = f"task_{int(dep_num):03d}"
                                    if normalized_dep in task_tree.tasks:
                                        dep_task = task_tree.tasks[normalized_dep]
                                        # Verify dependency is in the same milestone
                                        if dep_task.milestone_id == milestone_to_expand_id:
                                            valid_deps.append(normalized_dep)
                        
                        new_task.depends_on = valid_deps
                    
                    # Add to tree
                    try:
                        task_tree.add_task(new_task)
                        new_tasks_created += 1
                        
                        # Update status if no dependencies
                        if not new_task.depends_on:
                            new_task.status = TaskStatus.READY
                        
                    except ValueError:
                        pass  # Task creation failed, skip it
            
            if new_tasks_created > 0:
                logger.info("Expander agent completed: created %s tasks for %s", new_tasks_created, milestone_to_expand_id)
                return {
                    "tasks": task_tree.to_dict(),
                    "tasks_created_this_iteration": new_tasks_created,
                    "messages": [f"Expander: Created {new_tasks_created} new tasks for {milestone_to_expand_id}"],
                }
            
            last_error = "Zero tasks created"
            if is_retry:
                return {
                    "tasks": task_tree.to_dict(),
                    "tasks_created_this_iteration": 0,
                    "error": f"Expander failed after retry: {last_error}",
                    "messages": [f"Expander: {last_error} (retry also produced no tasks)"],
                }
            logger.info("Expander: %s, retrying with simplified prompt", last_error)
    
    except Exception as e:
        logger.error("Expander agent exception: %s", e, exc_info=True)
        context = f" expanding {milestone_to_expand_id}" if milestone_to_expand_id else ""
        error_msg = f"Expander failed{context}: {e}"
        return {
            "tasks": task_tree.to_dict(),
            "tasks_created_this_iteration": 0,
            "error": error_msg,
            "messages": [f"Expander exception: {e}"],
        }

