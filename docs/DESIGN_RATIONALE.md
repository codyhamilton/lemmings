# Design Rationale

This document captures the reasoning behind the workflow architecture redesign. The "why" is as informative as the "what" -- understanding the tradeoffs helps guide future changes.

For the architecture itself, see [../WORKFLOW_ARCHITECTURE.md](../WORKFLOW_ARCHITECTURE.md).
For future optimisations, see [FUTURE_CONSIDERATIONS.md](FUTURE_CONSIDERATIONS.md).

---

## From 11 Nodes to 5: Why Consolidate?

### The Problem with High Decomposition

The previous architecture had 11 agent nodes: Intent, Gap Analysis, Milestone, Expander, Prioritizer, Researcher, Planner, Implementor, Validator, QA, Assessor. Each had a clear single responsibility, following the principle of cognitive load reduction through specialisation.

However, this created a **minimum complexity floor**. Every task, no matter how simple, went through:
1. Researcher (full gap analysis with tool-calling loop)
2. Planner (full PRP generation with tool-calling loop)
3. Implementor (execution)
4. Validator (file checks)
5. QA (assessment)

For a trivial task like "add a property to an existing class," this was 4-5 LLM calls when 2 would suffice. The system couldn't scale down.

Conversely, it couldn't scale up either. For a very complex task, the fixed pipeline of Researcher → Planner gave exactly one shot at research and one shot at planning. There was no mechanism for the planner to say "I need more research on this specific aspect."

### The Subagent Solution

The LangChain subagent-as-tool pattern resolves both problems. Instead of fixed pipeline stages, we give the TaskPlanner subagent tools that it calls when needed:

- **Simple task**: TaskPlanner reads the task, does a quick `rag_search`, writes the PRP. One LLM invocation. No subagent.
- **Complex task**: TaskPlanner calls `explain_code` for thorough research, reads the results, plans. Two or more LLM invocations.
- **Very complex task**: TaskPlanner calls `explain_code` multiple times for different aspects, synthesises, plans. The LLM decides the research depth.

The key insight is that **the LLM is better at deciding when to research than a fixed graph edge**. A fixed pipeline always researches. An agent with tools researches when it's uncertain.

### What We Lose

1. **Static discoverability**: LangGraph can inspect subgraph state for statically discoverable subgraphs. Subagents called inside tool functions are not statically discoverable, so `get_state(subgraphs=True)` won't return their state. This is an acceptable tradeoff -- we gain flexibility and lose some debugging visibility.

2. **Clear per-node telemetry**: With separate nodes, each step produces a distinct state update visible in LangGraph Studio. With subagents, the research step is hidden inside the TaskPlanner's tool calls. Mitigated by logging subagent calls explicitly.

3. **Independent testability per stage**: Previously you could test the Researcher independently. Now research is an internal tool call. Mitigated by testing the subagent tools independently (they're still separate functions).

---

## Why Sliding Window Over Upfront Decomposition

### The Extended Planning Fallacy

The previous architecture used an Expander agent to decompose milestones into complete task DAGs upfront. This suffered from two fundamental problems:

1. **Speculative decomposition**: Tasks were defined before any task-level research. The Expander would create "Create Colony.gd with properties X, Y, Z" and later the Researcher would discover Colony.gd already exists. The task description was wrong from birth.

2. **Token pressure at scale**: For a large milestone, the Expander needed to hold the full milestone scope, existing tasks, and produce a complete DAG with dependencies -- all in one context window. At 32-64k tokens, this meant either shallow tasks or limited quantity.

Extended task planning, even for humans in large teams, follows a universal pattern: **the further ahead you plan, the less accurate the plan**. The system should embrace this rather than fight it.

### The Sliding Window Alternative

Instead of planning all tasks upfront, the TaskPlanner works with a limited lookahead:

1. Think a few steps ahead (lookahead)
2. Pick a bite-sized chunk to execute now
3. Execute it
4. Reassess with fresh context

The carry-forward tasks are "notes from my past self" -- the TaskPlanner can keep, modify, or drop them. They're rough descriptions (~100 chars), not detailed plans. Detailed planning only happens for the task being executed NOW.

This means:
- **No speculative detail**: We never waste tokens planning something we might not do
- **Constant token pressure**: A 5-task milestone and a 50-task milestone use the same context per round
- **Self-correcting**: Each round benefits from the latest implementation context
- **Research-informed**: Tasks are defined after research, not before

### What We Lose

1. **Upfront visibility**: Users can't see "here are all 15 tasks" before execution starts. Milestones provide coarse visibility, but detailed task lists emerge during execution.

2. **Dependency graph**: The previous architecture maintained a DAG of task dependencies with explicit `depends_on` and `blocks` relationships. The sliding window relies on the TaskPlanner's judgment about sequencing. For simple milestones this is fine; for complex milestones with true parallelism opportunities, we lose the ability to identify independent work streams. (In practice, with a single-threaded execution model, this doesn't matter yet.)

3. **Progress estimation**: Without a full task list, you can't say "we're 60% done." The done list grows, but the total is unknown until the milestone is complete.

---

## Why the ScopeAgent Consolidates Three Agents

### Previous: Intent → Gap Analysis → Milestone (3 agents)

Each had a single responsibility:
- Intent: Understand what the user wants
- Gap Analysis: Assess what exists vs what's needed
- Milestone: Break into sequential phases

### Problem: Artificial Separation

Understanding scope, assessing gaps, and defining milestones are one coherent cognitive act. You can't define milestones without understanding the gaps, and you can't assess gaps without understanding the intent. Separating them created:

1. **Context loss between agents**: Intent produced a remit, Gap Analysis read it and produced need_gaps, Milestone read both. Each handoff compressed information.
2. **No feedback loop**: If Gap Analysis discovered something that changed the interpretation of intent, there was no way to go back.
3. **Three LLM calls for what is conceptually one question**: "Given this request and this codebase, what are we trying to achieve and what are the major phases?"

### Solution: One Agent with Subagent Tools

The ScopeAgent uses `explain_code` and `ask` to research the codebase, then defines the remit and milestones in one pass. The LLM can interleave research and planning naturally:

1. Read request
2. `explain_code("What game entity systems exist?")` -- understand current state
3. `ask("Does colony code exist?")` -- targeted check
4. Define remit based on understanding
5. Define milestones based on gaps identified

If step 3 reveals something surprising, the LLM naturally adjusts its remit in step 4. No handoff, no context loss.

---

## Why Milestones Are User Outcomes, Not Implementation Steps

### The Tension

Milestones need to be:
- Defined from gap analysis alone (without implementation knowledge)
- Self-contained enough to constrain TaskPlanner scope
- Coarse enough to avoid becoming a task list

If milestones are implementation steps ("Set up data models", "Create UI components"), they require implementation knowledge to define and tend to prescribe HOW rather than WHAT. They become a task list with extra steps.

### The Resolution

Milestones describe user-testable outcomes: "User can create and name colonies." This framing:

1. **Doesn't require implementation knowledge**: You don't need to know the codebase to know that "user can create colonies" is a meaningful interim state.
2. **Is naturally self-contained**: Each outcome bounds a scope the TaskPlanner can think within.
3. **Is testable**: The Assessor can evaluate "can the user do this?" without understanding implementation details.
4. **Is sequenceable**: User outcomes have natural ordering ("user can create colonies" before "user can view colony statistics").

The ScopeAgent also produces rough "area sketches" per milestone (e.g., "data models, state hooks") as orientation for the TaskPlanner, but these are explicitly non-binding.

---

## Why Periodic Assessment Over Per-Task Assessment

### The Over-Monitoring Problem

The previous architecture ran the Assessor after every single task. This was problematic because:

1. **Premature judgment**: After 1 of 10 tasks, progress naturally looks incomplete. The Assessor would need to constantly say "keep going" -- wasted tokens for a trivially obvious answer.

2. **Short-term vs long-term confusion**: A task that creates infrastructure (data models, utilities) looks unrelated to the user outcome. Two tasks later, the UI task that depends on that infrastructure makes the connection clear. Per-task assessment would flag the infrastructure task as "drifting" when it's actually foundation work.

3. **Cost**: An LLM call after every task, when the correct answer is "continue" 80% of the time.

### The Periodic Alternative

Assessment every N tasks (default 5) provides enough data points for meaningful judgment without over-monitoring. The Assessor sees a batch of work and can evaluate trajectory rather than individual steps.

Additional triggers (completion claims, aborts) ensure critical moments are never missed.

### The Counter-Argument

"What if we go off the rails on task 2 and don't catch it until task 7?" This is a real risk. The mitigation is:

1. The TaskPlanner has its own internal coherence -- it's research-informed, so individual tasks are grounded in reality.
2. QA catches implementation-level issues after every task.
3. The cost of 5 slightly-wrong tasks is lower than the cost of constantly second-guessing progress.

Future work on dynamic assessment frequency (confidence-based triggers) can reduce this window. See [FUTURE_CONSIDERATIONS.md](FUTURE_CONSIDERATIONS.md).

---

## The Failure Escalation Design

### Principle: Handle Problems at the Lowest Level

The three-tier escalation emerged from asking: "What kinds of problems can occur, and who has the authority/context to fix them?"

| Problem | Who Sees It | Who Fixes It | Example |
|---------|-------------|-------------|---------|
| Code doesn't match plan | QA | TaskPlanner | Missing function, wrong file location |
| Task is impossible/conflicting | TaskPlanner | Assessor (evaluate) | Dependency doesn't exist, circular requirement |
| We're building the wrong thing | Assessor | ScopeAgent | Milestones don't match what we're learning |

Each tier has increasing scope and decreasing frequency:
- QA runs after every implementation
- TaskPlanner evaluates on every retry
- Assessor evaluates every ~5 tasks

### The Single Retry Path

The previous architecture had three QA failure types that routed to three different retry targets:
- `wrong_approach` → Researcher
- `plan_issue` → Planner  
- `incomplete` → Implementor

This was complex and created 3 separate retry nodes. The new design routes all QA failures to the TaskPlanner, which decides what to do internally. Benefits:

1. **Simpler graph**: One retry path instead of three
2. **Smarter retry**: The TaskPlanner has full context to decide whether to re-research, re-plan, or just adjust. The graph doesn't need to classify the failure type for routing.
3. **Escalation option**: If the TaskPlanner determines the task is fundamentally unfeasible after receiving QA feedback, it can abort. The previous architecture had no escalation path from a QA failure.

### TaskPlanner Abort

A new capability: the TaskPlanner can abort a task and provide escalation context. This covers cases where:
- Research reveals the task conflicts with an established pattern
- The task depends on something that doesn't exist and can't be created within the milestone scope
- Multiple retry attempts have failed and the TaskPlanner concludes the approach is wrong

The abort routes to the Assessor, which evaluates whether this is an isolated task failure or a sign of broader misalignment. This closes a gap in the previous architecture where fundamental conflicts could only surface as repeated task failures.

---

## Alternatives Considered

### Alternative: Two-Phase Decomposition (Sketch Then Detail)

Keep scope and decomposition as separate agents, but add research to the decomposition step.

**Rejected because**: For a large milestone with 10+ tasks, the decomposition agent needs to hold research results for all of them plus the task definitions. At 32-64k tokens, this limits decomposition to ~5-7 tasks per pass. You'd need multiple passes, arriving back at the iterative model anyway.

### Alternative: Map-Reduce Decomposition

Use LangGraph's map-reduce pattern to decompose milestones in parallel: scope agent creates milestones, each milestone is independently researched and decomposed.

**Rejected because**: The parent agent still needs to receive and merge results. The deeper the hierarchy, the more lossy the summaries. For a local model at 32-64k, you'd get maybe one level of useful delegation before context compression loses critical detail.

### Alternative: Full DAG Task Tree with Just-In-Time Research

Keep the Expander and DAG-based task tree, but add research to the Expander so tasks are research-informed.

**Rejected because**: This preserves the complexity of DAG management (cycle detection, dependency resolution, status propagation) while still requiring all tasks to be defined upfront within a milestone. The sliding window is simpler and more adaptive.

### Alternative: Per-Task Assessment with Fast-Path

Keep per-task assessment but add a deterministic fast-path: if there are remaining carry-forward tasks and no abort signals, skip the LLM call and continue.

**Considered for future**: This preserves the ability to catch problems early while reducing token cost for the common case. It's a middle ground between per-task and periodic assessment. May be worth implementing if the periodic model proves too slow to catch divergence.
