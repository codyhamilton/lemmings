# Architecture

## What this is

An autonomous AI agent that takes a development remit and executes it. It is not conversational. It is given a task, works through it via a plan-execute-verify loop, and completes.

In the short term it has a fixed lifecycle: user command in, execution, eventual completion. Longer term it becomes a service that runs independently, with UIs connecting to observe and provide guidance.

## Design principles

- **Elegance**: Prefer fewer, clearer abstractions over many small ones.
- **Simplicity**: Each module does one thing. If it does two things, split it.
- **Focus**: No speculative features. Build what the current lifecycle needs.
- **Readability**: Code should be obvious. If it needs a comment explaining *why* it exists, reconsider the structure.

## Module structure

```
src/agents/
  main.py              # Entry point: start graph, start UI, pull loop
  graph.py             # LangGraph workflow definition
  task_states.py       # Core data models (WorkflowState, Task, Milestone, etc.)
  agents/              # Individual agent nodes (intent, researcher, coder, etc.)
  tools/               # Tools available to agents (read, edit, search, etc.)
  stream/              # Stream processing (service output layer)
    handler.py         # Normalize raw graph chunks -> stable models, dispatch
    messages.py        # AI text stream: per-node block parsing, subscribe API
    status.py          # Status events: task/milestone changes, subscribe API
  state/               # Derived state: milestones, tasks, status views
    status_history.py  # StatusEvent, StatusEventType, StatusHistory
  ui/                  # UI clients (consumers of stream APIs)
    console.py         # Console UI: stateless pipe, merged text + status output
  rag/                 # Retrieval-augmented generation (indexing, retrieval)
  llm.py               # LLM configuration
  config.py            # Agent configuration
```

## Separation of concerns

### The agent (graph + agents + tools)

The graph is the core. It takes a `WorkflowState`, runs agents through a plan-execute-verify loop, and produces state updates. It does not know about UIs, streams, or display. It is a pure state machine.

### Stream processing (stream/)

The stream layer sits between the graph and any consumer. It has two jobs:

1. **handler.py**: Adapt raw LangGraph chunk shapes into stable, normalized models (`MessageChunk`, `StatusUpdate`). This is the only place that knows LangGraph's stream format. When LangGraph changes, only handler.py changes.

2. **messages.py / status.py**: Process normalized chunks into meaningful events with subscribe APIs. Messages handles per-node text block parsing. Status handles task and milestone state changes. Both emit events to subscribers.

The stream layer is the service's output API. Any consumer (console, future web UI, tests) subscribes to it.

**Streams**: Four logical streams feed the console:
- **messages**: Agent text (thinking, reasoning, prose)
- **task**: Task/milestone/iteration events (task complete, milestone advance, etc.)
- **graph**: Node transitions (which agent is running; node start/complete/failed)
- **tools**: Tool invocations (future; tools currently do not print)

Task and graph are split from status updates. StatusStreamHandler uses `node_name` from each chunk (the updates key) to detect node transitions; nodes do not return `current_node`.

### State (state/)

State that is derived from or projected out of the graph's `WorkflowState`. Currently this is milestones and tasks. The status stream is a **view** into this state: it watches graph updates and emits both task-level events (task complete, milestone advance) and node-level events (node start/complete/failed).

**Node transitions matter**: Which agent is running is important for console UI and debugging. The status stream emits node start/complete/failed events using `node_name` from each updates chunk. Task-level events (task complete, milestone advance) are separate; subscribers can filter by type (`subscribe_task`, `subscribe_graph`).

This module will grow as state becomes more abstract and less linear (e.g. richer task projections, progress summaries). For now it contains only what the status stream needs to do its job.

### UI (ui/)

UI modules are **clients** of the stream layer. They subscribe to message and status streams and render output. They do not reach into the graph or hold workflow state.

**ConsoleUI** is the simplest implementation: a stateless pipe that merges text and status into one stdout stream.

### What main.py does

1. Initiate the graph (create initial state, configure).
2. Initiate the UI (which subscribes to stream handlers).
3. Pull chunks from `graph.stream()` and pass each to the stream handler.
4. Accumulate state from `updates` chunks (stream does not return final state; merge each state_update into final_state).
5. Pass accumulated final_state to `render_final_summary` (including `work_report` from report agent).

That's it. No state wiring, no event translation, no message history management.

## Data flow

```
graph.stream()
    |
    v
stream/handler.py       -- normalize raw chunks to MessageChunk / StatusUpdate
    |           |
    v           v
messages.py   status.py  -- process, track state, emit events
    |           |
    v           v
  subscribers            -- ConsoleUI, future web UI, tests
```

## Current lifecycle

Fixed lifecycle, no interactive input during execution:

1. User provides a command (remit).
2. Agent plans (intent -> gap analysis -> milestones -> task expansion).
3. Agent executes (research -> plan -> implement -> validate -> QA per task).
4. Agent assesses and iterates until stable.
5. Report agent produces narrative summary (runs before every exit).
6. Agent completes.

## Future: service model

The agent becomes a service, agnostic to whether anyone is connected. It runs a loop: plan, execute, verify, repeat.

- **Input channel**: Users send messages that go through intake, queue, and update milestones/plans when agents are available. The planning process integrates guidance into its loop.
- **Output streams**: Same subscribe API as now. Clients connect and disconnect freely.
- **Lifecycle**: No fixed end. The agent works its remit, accepts new guidance, re-plans.

This is not the current scope. The architecture supports it by keeping the stream layer as the boundary between agent and consumers, and by not coupling the graph to any specific UI.
