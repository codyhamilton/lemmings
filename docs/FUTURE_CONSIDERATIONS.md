# Future Considerations

Ideas and optimisations that are out of scope for the current redesign but worth capturing. These emerged during design discussions and address known limitations or opportunities.

For the current architecture, see [../WORKFLOW_ARCHITECTURE.md](../WORKFLOW_ARCHITECTURE.md).
For the reasoning behind the current design, see [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md).

---

## Dynamic Assessment Frequency

### Current: Fixed Interval

The Assessor runs every N tasks (default N=5). This is simple and predictable but doesn't adapt to the actual uncertainty level of execution.

### Proposed: Uncertainty-Driven Triggers

Collect uncertainty signals from agent outputs and accumulate them into a score that can pull forward the next assessment:

| Signal | Source | Weight | Rationale |
|--------|--------|--------|-----------|
| QA failure | QA agent | High | Direct evidence something went wrong |
| Minor drift verdict | Assessor | Medium | Already flagged, next assessment sooner |
| High carry-forward churn | TaskPlanner | Medium | If lookahead keeps changing, direction is uncertain |
| Many explain_code calls | TaskPlanner | Low | Struggling to understand codebase, possible misalignment |
| Task abort | TaskPlanner | Immediate | Triggers assessment regardless of counter |

**Implementation sketch**:
```python
# In WorkflowState
uncertainty_score: float  # Accumulated uncertainty, 0.0 = fully confident

# After each task, increment based on signals
if qa_failed:
    uncertainty_score += 0.4
if carry_forward_changed_significantly:
    uncertainty_score += 0.2

# Trigger assessment when score exceeds threshold
if uncertainty_score >= 1.0 or tasks_since_last_review >= review_interval:
    run_assessor()
    uncertainty_score = 0.0  # Reset after assessment
```

This allows the system to assess sooner when things are uncertain and less often when things are going smoothly. The fixed interval acts as a backstop.

### Open Questions

- How to calibrate weights without real execution data?
- Should the threshold be adaptive (tighter early in a milestone, looser later)?
- Should confidence signals propagate backward (high confidence in QA reduces accumulated uncertainty)?

---

## Execution Observability

### The Need

To continuously optimise the system, we need insight into:
1. **Why** the system goes off the rails (not just that it did)
2. **Where** token budget is spent (which agents, which tool calls)
3. **How** task quality correlates with planning depth

### Proposed: Event Stream + Analytics

Every agent invocation produces structured events:

```python
@dataclass
class AgentEvent:
    timestamp: str
    agent: str                    # "task_planner", "implementor", etc.
    event_type: str               # "start", "tool_call", "output", "error"
    milestone_id: str
    task_description: str | None
    details: dict                 # Agent-specific data

    # For tool calls
    tool_name: str | None         # "explain_code", "rag_search", etc.
    tool_input_summary: str | None
    tool_output_summary: str | None

    # For outputs
    action: str | None            # "implement", "skip", "abort", etc.
    token_count: int | None       # Tokens used in this invocation
```

### Analytics Questions

With this event stream, we could answer:

- **Divergence analysis**: "When the Assessor flags drift, what happened in the preceding 5 tasks?" Correlate drift events with tool call patterns, QA failure rates, and carry-forward churn.

- **Cost analysis**: "How many tokens does a typical simple task cost vs a complex one?" Identify if subagent calls are worth their token cost.

- **Quality analysis**: "Do tasks with explain_code research have higher QA pass rates?" Validate that research improves planning quality.

- **Bottleneck analysis**: "Where does the system spend most of its time?" Identify if certain agents or tool calls are disproportionately slow.

### Implementation Path

1. **Phase 1**: Add structured logging to all agent nodes (agent name, action, token count, success/failure). Low effort, immediate value.
2. **Phase 2**: Add tool call logging within agents (tool name, input summary, output summary). Requires middleware or tool wrappers.
3. **Phase 3**: Build analytics dashboard that reads event logs and produces summaries. Could be a separate tool or web UI.

---

## Parallel Task Execution

### Current: Serial

Tasks execute one at a time: TaskPlanner → Implementor → QA → next task.

### Opportunity

When the TaskPlanner's carry-forward contains independent tasks (no shared files, no logical dependencies), they could execute in parallel:

```
TaskPlanner identifies: [task_A (modifies GameState.gd), task_B (creates Colony.gd)]
→ These are independent
→ Execute both in parallel: Implementor_A + Implementor_B → QA_A + QA_B
```

### Challenges

1. **File conflicts**: Two Implementors modifying the same file simultaneously. Requires conflict detection or file-level locking.
2. **Context divergence**: Each parallel branch sees a different version of the codebase. Merging results may create inconsistencies.
3. **Token budget**: Running two agents in parallel doubles token usage per time unit.
4. **Complexity**: LangGraph supports parallel execution via `Send()`, but managing parallel task state adds significant complexity.

### Recommendation

Defer until serial execution is proven and optimised. The sliding window model is already efficient for serial execution. Parallel execution is a throughput optimisation, not a correctness improvement.

---

## Human-in-the-Loop Checkpoints

### Current: Fully Autonomous

The system runs from start to finish without human input.

### Opportunity

Insert optional approval gates at key decision points:

1. **After ScopeAgent**: "Here's my understanding and milestones. Approve?" User can adjust remit or milestones before any execution starts.
2. **After Assessor drift**: "I've detected drift. Here are the options: A) continue, B) re-plan, C) manual guidance." User can steer correction.
3. **Before abort**: "TaskPlanner wants to abort this task because [reason]. Approve or provide guidance?"

### Implementation

LangGraph supports `interrupt()` for human-in-the-loop. The agent pauses, the state is persisted, and execution resumes when the user provides input.

```python
from langgraph.types import interrupt

def scope_agent_node(state):
    # ... produce milestones ...
    if state.get("require_approval"):
        user_response = interrupt({
            "question": "Review milestones before execution",
            "milestones": milestones,
        })
        if user_response.get("approved"):
            return {"milestones": milestones}
        else:
            # Re-plan with user feedback
            return scope_agent_with_feedback(state, user_response)
```

### Recommendation

Implement as opt-in configuration. Default behaviour is fully autonomous. When `require_approval: true`, insert interrupt points at ScopeAgent output and Assessor escalations.

---

## Adaptive Review Interval

### Current: Fixed N=5

The assessment interval is a constant. This might be too frequent for well-understood problem domains and too infrequent for novel ones.

### Proposed: Confidence-Based Adaptation

After each assessment:
- If verdict was "aligned" and uncertainty score was low → increase interval (N = min(N+1, 10))
- If verdict was "minor_drift" → decrease interval (N = max(N-2, 3))
- If verdict was "major_divergence" → reset to minimum (N = 3)

This creates a natural rhythm: the system monitors closely at the start (uncertain), relaxes as confidence builds, and tightens again if problems are detected.

---

## ScopeAgent Learning Across Sessions

### Current: Stateless

Each workflow run starts fresh. The ScopeAgent has no memory of previous runs.

### Opportunity

If the ScopeAgent could access summaries of previous runs (what worked, what diverged, which milestones were revised), it could make better initial plans:

- "Last time I tried to decompose a similar feature into 5 milestones. The last 2 were revised during execution. For this type of feature, 3 milestones is more realistic."
- "The codebase uses X pattern for features like this. Start with that assumption."

### Implementation

LangGraph's `Store` API supports cross-thread persistence. After each workflow completes, store a run summary:

```python
store.put(("workflow_summaries",), key=run_id, value={
    "user_request": request,
    "milestones_planned": original_milestones,
    "milestones_actual": final_milestones,
    "divergence_events": divergence_log,
    "total_tasks": task_count,
    "success": final_status,
})
```

The ScopeAgent retrieves recent summaries as context for planning.

### Recommendation

Defer to after the core architecture is proven. Cross-session learning is valuable but adds complexity (what to remember, when to forget, how to avoid reinforcing bad patterns).

---

## Quality Regression Detection

### The Problem

The system might establish a pattern early (e.g., "use dictionaries for entity storage") and then later create code that conflicts with it (e.g., "use arrays for colony storage"). QA checks individual tasks against their specs but doesn't check cross-task consistency.

### Possible Approaches

1. **Architecture sentinel**: A lightweight check that reads the codebase's established patterns and flags deviations. Could run as part of periodic assessment.

2. **Style/pattern tests**: The TaskPlanner could create pattern tests alongside implementation tasks (e.g., "all entity storage uses dictionaries"). QA would run these.

3. **Post-milestone review**: After a milestone completes, a review agent scans all changes for consistency. Heavier but more thorough.

### Recommendation

Start with approach 1 (lightest weight). The Assessor could include a "pattern consistency" check in its periodic review, using `explain_code` to sample recent changes and flag deviations.
