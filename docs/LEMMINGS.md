# About Lemmings

The high level goal is to build a hybrid local/cloud AI development orchestrator capable of long-running development work with asynchronous oversight.

At its core is a planning-execution loop typical of all orchestrated agentic workflows, which more or less naively builds to a given scope.

Around this is a series of layers providing information persistance and asynchronous, out-of-loop supervision

These layers, broadly, are:

1. **Planning-Execution loop** - this is the core loop carrying out naive task planning and implementation against a mostly fixed remit
2. **Contextual Knowledge Layer** - This is a store of knowledge about the application. Designs, future scope, code conventions, decisions on patterns, documentation related to the project available for quick reference.
3. **Supervisor Layer** - This is a layer consisting of agents whose responsibility is to inspect the repository, inner layer progress, the work that has been carried out so far and assess, revise, guide. They receive information from the repository and signals, and write to the contextual knowledge layer.
4. **Global/User Knowledge Layer** - This is a layer containing persisted knowledge with relevance beyond the repository. It is built up over time via internalised lessons from supervisor agents
5. **Organisational Layer** - this can best be thought of as agents who maintain organisational knowledge about broad organisation rules and patterns.

# Layers

## Planning-Execution Loop

This is fundamentally a LangGraph instance, or multiple, running an internal loop against a given remit until it considers its scope complete.

## Contextual Knowledge Layer

At its simplest implementation, this could just be a directory with a bunch of markdown files in a repository which are organised into topical folders and indexed using a vector db.

**Scope:**
* Project intent and scope
* Architecture, solution designs
* Dependencies, how to do various things (e.g. integration, testing)
* Lessons learned (what works and doesn't)
* Repository specific conventions and patterns
* Project domain knowledge, e.g about the problem domain

## Supervisors

This layer contains various role-specifc agents whose purpose is to inspect, review and adjust direction of the inner loop.

They are explicitly designed to be non-interrupting. Some may be periodical, some on signal only, but they occur outside the implementation loop.

They interact with the inner loop through two channels:

1. **Direct signals**: Supervisors send directives to the scope planner, which creates specific tasks with concrete direction. For example, if an approach needs changing, a task is created to rewrite the old approach -- not just ambient knowledge update.
2. **Indirect influence**: Supervisors update the contextual knowledge store, which shapes future planning decisions. The execution loop naturally absorbs updated conventions, patterns, and designs on its next planning round.

Both channels work together. Direct signals ensure critical changes are acted on explicitly. Indirect influence ensures the execution loop's future decisions are informed by the latest understanding.

You can think of the supervisors as being the way to keep execution on the rails, and the way you automatically change direction when there is dissonance between plans and reality. It also provides a way to discover and backlog "polish", cleaning up designs, simplifying, keeping coding standards consistent. Execution loops focus on functional behaviour, this makes sure it's been solved cleanest way.

Example supervisors
* Code style review
* Design review
* Internaliser (pull up knowledge from task to project level)

### Code Style Supervisor

Reviewing alignment of code conventions across the recent work and project, finds inconsistencies and gaps. It can also set conventions (writing to contextual store), based on its own decisions (either permanently or temporarily) and putting up an async signal to user to resolve permanently. It reads from task change logs, state, repository and outputs gaps, updates to repo knowledge, can research best practices. The inner loop reads these gap reports to add to scope.

With this reviewer, we iteratively build out a consistent set of conventions which are baked into knowledge and provided to executions in inner loop, and also reports inconsistencies which get appended into build scope

### Design Supervisor

Looks at the overall solution design and reviews it in the context of our existing knowledge, what we have learned, what the inner loop has found. It writes back to knowledge, which is then internalised into the inner loop on its next planning review (and updating designs triggers this earlier)

This is the way we refine our design iteratively, using experience to make better decisions, making small or drastic changes to the plan while the inner loop continues. 

### Internaliser (Knowledge Supervisor)

Sees all of the thinking, outputs of the inner loop and finds anything novel - something an agent discovers or decides, summarises and write it in to repo store as a list of items to think about. 

We will then take this list and consider, drop items, keep them as appropriate. 

This provides a way of turning ephemeral task knowledge into long term referenceable memory where useful

# Architecture

The architecture scales through three phases. The core principle is that a huge portion of the desired benefits can be achieved through simple, coupled in-graph async agents in Phase 1. Proving the model at this scale then funds dealing with the complexities of distribution.

## Phase 1: Single Instance (In-Graph)

A single LangGraph system running all layers inside one instance.

* The execution loop (existing graph nodes)
* Supervisor agents (async nodes within the same graph, triggered by signals)
* Contextual Store (repo-level docs folder in markdown, indexed by vector DB)
* Global Store (docs folder in user home)

Supervisors run asynchronously but _within the graph_, meaning they share execution context. They can read the full graph state, see the execution loop's reasoning, and write back through well-defined state channels. This tight coupling is a feature at this scale -- it avoids the coordination problems of distributed systems while proving the supervision model works.

**What works well at this scale:**
* Supervisors have full local context -- no information asymmetry between layers
* Knowledge store as markdown files is sufficient for single-repo, single-user use
* Signalling is in-process -- no network latency, no message loss
* Supervisor-to-supervisor conflicts are visible in shared state
* Escalation routing can use graph state directly to classify scope

**Challenges at this scale:**
* Graph complexity grows with each supervisor type added
* All agents compete for the same compute (mitigated by hybrid local/cloud model)
* Knowledge store has no staleness tracking -- managed by convention and agent judgment

## Phase 2: Distributed

The execution loop becomes an independently scalable set of instances picking up user or automated remits. Supervisors run as a separate stack on configurable signals.

**What this enables:**
* Execution loop can scale up eagerly as long as available scope exists
* Supervisor layer is independently configurable, runs less frequently as knowledge matures
* Task commits (git push) become natural trigger points for supervisor work

**New challenges at this scale:**
* Knowledge store as flat files breaks down -- needs a persisted database with staleness tracking, conflict resolution, and relevance scoring
* Information asymmetry emerges -- supervisors no longer share execution context, must work from repository state and signals alone
* Supervisor-to-supervisor coordination becomes a real problem -- concurrent writes to knowledge store can conflict
* Escalation routing can no longer inspect local state to classify scope -- needs an explicit classification mechanism
* Prioritisation of competing demands (functional work, polish backlog, directives) can no longer rely on shared state visibility

## Phase 3: Organisational

Out of scope for implementation planning, noted for architectural consideration.

* Multiple repositories, multiple execution loops, multiple supervisor stacks
* Organisational-level supervisor agents maintaining cross-repo conventions
* Organisational knowledge store (its own database, not repo-level)
* Everything from Phase 2 applies, plus cross-repo consistency and conflict resolution

# Signalling

Signalling is the way layers let each other know they need to do something. There's really two critical signal classes, _escalations_ and _directives_

## Escalations

Escalations can happen at any layer and they basically bubble up to handling layers, all the way up to a user.

What we really mean by escalations are requests for direction.

Requests for direction can come about due to insufficient confidence on a path forward or because critical assumptions in a plan have been found to be false.

Requests for direction are effectively non-blocking. If an implementation agent is unsure, it should be able to signal for direction while still executing its best guess. A user will be able to provide direction at their leisure and then updates are made to contextual docs and a signal is sent to the execution loop to include changed plans in scope on next planning interval.

It might mean some duplication of work, but that's ok. It can be minimised by having the task planner be aware of pending direction and prioritising other work. At least it's not stopped for an unknown period of time.

The other more important one is about the failure of critical assumptions. One of the key gaps with the naive execution-loop on its own is it is generally unable to copy cleanly with a total failure of a key design assumption. It then flails and hacks around it, eager to keep to the plan, and the plan doesn't typically have a method of total review on failing the assumption.

This signal is like a spark of lightning which travels up the chain of responsibility until it finds a handler with the scope to adjust. E.g, if a particular task is poorly designed, the signal ends at the task planner who adjusts it. If the problem wasn't the task but the whole approach, it goes further up to the scope design agent. If the problem isn't just the approach but the remit, it has to go further to the supervisor layer which might decide it actually needs human input

The supervisor layer at this point provides an adjusted directive to the scope designer which then ringfences the probem or provides an acceptable temporary solution until user guidance is provided.

## Directives
These are signals in the opposite direction to escalations. 

A code review agent, asynchronously reviewing code changes, might find a large number of code quality anomalies, where style doesn't match our guidelines. Maybe the code was originally with it, but user or other direction now means its outside!

The supervisor is able to provide a directive (mostly async, sync would mean it's something extremely urgent) down to the scope agent telling it to include new items in scope, which then alter the plan. This would be aided by the review agent updating the project contextual permanent store which itself indirectly affects decision making in the execution loop.

Rather than all directives being necessarily interrupting, they can be queued and picked up on the next planning round (and also banked up directives could bring that on sooner), so the execution loop naturally adjusts as direction changes. This could work with any of the supervisor agents.

In effect this creates a system of backlogging. Supervisor agents, being backward looking come up with feedback on already delivered work, which is then backlogged and automatically subject to prioritisation in the execution loop.

**Prioritisation model**: Feature-first. Functional directives (approach changes, new requirements) are prepended to the task list. Cleanup directives (style fixes, polish) are appended. This ensures forward progress on the remit is never stalled by refinement work. Prioritisation is a core feature of the execution loop's scope planner.

The execution loop will periodically re-enter scope planning, which is when these directives are picked up. We could use a system of hints (like criticality ratings) to "push forward" planning to an earlier iteration.

# Other concepts

## Automatic Polishing

One of the key issues with naive planner-executor loops is they tend to stop once the product is functional, and skip the iterative refinement process typical of human development. 

In most human development, the first functional prototype is complete in about half the total build time. Then it is refined, bugs dealt with, cleaned up. The final result may look very little like the original functional prototype. 

In Agentic workflows, they typically are more likely to have a functional and somewhat better designed prototype, but they stop at that point. 

The supervisor agents - particularly design and code review, are designed to provide an asynchronous method of refinement. As the executor loop builds out, these agents review, find improved ways of solving the already-built problems, signals pass the message to the scope planner agent, the refinements are added into scope. The supervisor agents also have automatically updated guidance in the contextual knowledge store, so the implementor agent now has different guidance for code style and patterns when tackling the same problem.

This provides a form of iterative design, building, then reviewing in the background, then backlogging improvements. 

This automatic polish backlog has a separate benefit, in that if we are waiting on user interaction we can undertake benefical low-risk improvement work while we wait, or if more time is available after remit is complete the backlog can be worked on.

## Task Commits

Each task in the execution loop, which is signed off by QA is commited into the branch. Each execution loop operates in its own branch. This provides a window in task work which supervisors can use, and which can actually trigger supervisor work on push in a distributed deployment model. It also means we can quickly reference what work was done when and why, by providing contextual information in the git commit to backtrack this in a permanently referencable way.

## Hybrid Local-Cloud Agents

One cost goal is to make informed choice about use of local vs cloud agents for various purposes. A good example is keeping the planner-execution loop in local agents (or private cloud for distributed models) for fast iteration while using cloud providers for certain supervisors, or having a more dynamic model of agent choice.

The idea is to bring as much work as possible into smaller models and use our layered knowledge store to ensure highly concentrated, opinionated context is provided.

# Gaps

This is a high level plan. Known gaps are listed below, categorised by which scale phase they become critical.

## Phase 1 (In-Graph)

* **LLM sensitivity to written guidance**: The entire supervisor model depends on LLMs effectively integrating knowledge store contents into planning decisions. If the LLM treats contextual documents as background noise rather than actionable direction, supervisors are inert. Prompt design and knowledge format are critical.
* **Convergence criterion for polish loops**: Supervisors continuously find improvements and queue them. Without a stopping condition, refinement loops could run indefinitely. Need a diminishing-returns signal or explicit "good enough" threshold.
* **Prioritisation implementation**: The feature-first model is defined but the mechanism for the scope planner to weigh functional work, polish backlog, and supervisor directives needs to be designed within the existing execution loop.

## Phase 2+ (Distributed)

* **Supervisor-to-supervisor coordination**: Independent supervisors can produce conflicting guidance. A design supervisor might recommend a pattern change while the code style supervisor simultaneously flags inconsistency based on the old pattern. In Phase 1 this is visible in shared state; in Phase 2 it requires explicit conflict resolution.
* **Escalation routing classification**: Determining whether a failure is a task issue, design issue, or remit issue requires judgment that is itself hard. Under-classifying means the execution loop keeps flailing at a systemic problem; over-classifying means small hiccups trigger unnecessary re-scoping. In Phase 1 the graph has enough context to classify reasonably; in Phase 2 this needs a dedicated mechanism.
* **Information asymmetry**: Supervisors working from repository state and signals alone lack the execution loop's internal reasoning about tradeoffs and invariants. The internaliser helps but captures what seems _novel_, not necessarily what is _important_. Critical implementation decisions may be too obvious to flag but too nuanced to see from outside.
* **Knowledge store at scale**: Flat document store breaks down with multiple users, concurrent writes, and growing content. Needs staleness tracking, conflict resolution, relevance scoring, and pruning. At sufficient scale, _how does an agent know_ whether a piece of knowledge is still relevant?
