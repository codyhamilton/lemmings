# Multi-Model Architecture Plan

## Executive Summary

This document outlines a plan to extend the Lemmings agent from a single GPU-bound model (Qwen via text-generation-webui) to a multi-model architecture where:

- **Primary graph work** stays on the GPU-bound Qwen model
- **Supporting activities** run on CPU models (Ollama) and cloud models (OpenRouter)
- Some activities run **sync** (in the main graph flow), others **async** (background/supervisor)

---

## Current State

| Component | Model | Location |
|-----------|-------|----------|
| ScopeAgent, TaskPlanner, QA, Assessor, Report | `planning_llm` | GPU (Qwen) |
| Implementor, Coder | `coding_llm` | GPU (Qwen) |
| Subagents (ask, explain_code) | `planning_llm` | GPU (Qwen) |
| Summarizer (dashboard/status) | `planning_llm` | GPU (Qwen) |
| SummarizationMiddleware (Implementor/Coder) | `planning_llm` | GPU (Qwen) |
| Normaliser (field summarization) | `planning_llm` | GPU (Qwen) |

All LLM calls currently go through `llm.py` → `get_llm()` → `ChatOpenAI` pointing at `http://127.0.0.1:5000/v1` (text-generation-webui).

---

## Target Model Roles

| Role | Use Case | Preferred Backend | Execution |
|------|----------|-------------------|------------|
| **Primary** | Main graph agents (Scope, TaskPlanner, Implementor, QA, Assessor, Report) | GPU (Qwen) | Sync |
| **Summarization** | Dashboard summaries, status summaries, SummarizationMiddleware, normaliser | CPU (small model) | Sync |
| **Deep research** | External web/docs research, broad topic exploration | Cloud (OpenRouter) | Sync or async |
| **Supervisor** | Periodic state monitoring, high-level adjustments | CPU (larger) or Cloud | Async |
| **Tool-specific** | Subagents (ask, explain_code), future specialized tools | Configurable per tool | Sync |

---

## 1. CPU Models: Ollama vs Direct LangGraph

### Recommendation: **Use Ollama for CPU models**

| Approach | Pros | Cons |
|----------|------|------|
| **Ollama** | Same OpenAI-compatible API as text-generation-webui; LangChain `ChatOllama` or `ChatOpenAI` with `base_url`; easy model switching; runs well on CPU; single process, no GPU contention | Extra process to run; model loading/unloading |
| **Direct from LangGraph** | No extra process | LangGraph/LangChain don't run models directly—they call APIs. You'd need to run *something* (Ollama, llama.cpp server, etc.) anyway |
| **llama.cpp Python bindings** | In-process, no HTTP | More integration work; less standard; harder to swap models |

**Conclusion**: Ollama is the standard, well-supported way to run small models on CPU. It exposes an OpenAI-compatible API, so you can use `ChatOpenAI(base_url="http://localhost:11434/v1", model="tinyllama")` or `ChatOllama(model="tinyllama")`. No need to run models "directly from LangGraph"—LangGraph always calls an LLM API.

### Ollama CPU Usage

- Ollama automatically uses CPU when no GPU is available
- On a machine with GPU, you can run Ollama on CPU by:
  - Using a separate machine/container without GPU, or
  - Setting `OLLAMA_NUM_GPU=0` (see [ollama/ollama#9624](https://github.com/ollama/ollama/issues/9624))
- Small models (TinyLlama 1.1B, Phi-2 2.7B, Qwen2.5-0.5B) run reasonably on CPU for summarization

### Suggested CPU Models for Summarization

| Model | Size | Notes |
|-------|------|-------|
| **tinyllama** | 1.1B | Fast, good for short summaries |
| **qwen2.5:0.5b** | 0.5B | Very fast, decent quality |
| **phi3:mini** | 3.8B | Better quality, slower on CPU |
| **gemma2:2b** | 2B | Good balance |

---

## 2. Cloud Models: OpenRouter

### Integration

OpenRouter is OpenAI-compatible. Use `ChatOpenAI` with:

```python
ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="anthropic/claude-3.5-sonnet",  # or openai/gpt-4o, etc.
)
```

### Use Cases

- **Deep external research**: When a task needs web/docs lookup beyond the codebase
- **Supervisor** (optional): If you want a powerful model to periodically review state
- **Fallback**: If local models are overloaded or unavailable

### Cost / Latency

- API calls cost money and add network latency
- Use for high-value, occasional tasks (research, supervisor), not for every summarization

---

## 3. LangGraph Multi-Model Support

### Per-Node Model Selection

LangGraph supports different models per node by **passing a different LLM into each agent**. There is no special "bind model per node" API—each node/agent is constructed with its own model reference.

**Current pattern** (all use `planning_llm` or `coding_llm`):

```python
# scope_agent.py
create_agent(model=planning_llm, ...)
```

**Multi-model pattern**:

```python
# scope_agent.py – stays on primary
create_agent(model=primary_llm, ...)

# summarizer.py – uses CPU model
create_agent(model=summarizer_llm, ...)  # Ollama tinyllama
```

So the change is: **inject the appropriate LLM into each agent at construction time**. No LangGraph API changes needed.

### Async Execution

- **Sync**: Normal graph nodes run one after another. Summarizer, subagents, etc. run when their node executes.
- **Async**: For a "supervisor" that runs in the background:
  - **Option A**: Run a separate `asyncio` task that periodically reads state (e.g. from a checkpointer) and invokes a cloud/CPU model. It can push suggestions back via state or a side-channel.
  - **Option B**: Use LangGraph's `astream` / `ainvoke` and run the supervisor in a separate async task that shares the same event loop.
  - **Option C**: Use a "deferred" or parallel subgraph if the supervisor is part of the graph (e.g. fan-out to supervisor + main path, then fan-in).

For a true background supervisor that "monitors and adjusts," Option A is most flexible: a separate process or async task that polls state and can inject guidance.

---

## 4. Proposed Architecture

### 4.1 LLM Registry (`llm.py`)

Replace the current `default_llm`, `planning_llm`, etc. with a registry:

```python
# llm.py
from langchain_core.language_models import BaseChatModel

class LLMRegistry:
    primary: BaseChatModel      # GPU Qwen (main graph)
    summarizer: BaseChatModel   # CPU Ollama (tinyllama/qwen2.5:0.5b)
    research: BaseChatModel     # OpenRouter (optional)
    supervisor: BaseChatModel    # CPU or OpenRouter (optional, async)

registry: LLMRegistry = None

def get_llm(role: str = "primary") -> BaseChatModel:
    """Get LLM for a given role. Roles: primary, summarizer, research, supervisor."""
    ...
```

### 4.2 Config Schema (`config.json`)

```json
{
  "llm": {
    "primary": {
      "provider": "openai_compatible",
      "model": "Qwen3-8B-exl2-6_0",
      "base_url": "http://127.0.0.1:5000/v1",
      "api_key": "not-needed"
    },
    "summarizer": {
      "provider": "ollama",
      "model": "tinyllama",
      "base_url": "http://127.0.0.1:11434/v1"
    },
    "research": {
      "provider": "openrouter",
      "model": "anthropic/claude-3.5-sonnet",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "${OPENROUTER_API_KEY}"
    },
    "supervisor": {
      "provider": "ollama",
      "model": "phi3:mini",
      "base_url": "http://127.0.0.1:11434/v1"
    }
  }
}
```

### 4.3 Component → Model Mapping

| Component | Model Role | Notes |
|-----------|------------|-------|
| ScopeAgent, TaskPlanner, QA, Assessor, Report | `primary` | Unchanged |
| Implementor, Coder | `primary` | Keep coding_llm = primary for now |
| Summarizer (dashboard/status) | `summarizer` | CPU model |
| SummarizationMiddleware | `summarizer` | CPU model |
| Normaliser | `summarizer` | CPU model |
| ask, explain_code | `primary` | Or add `research` for explain_code if doing external research |
| Deep research tool (future) | `research` | OpenRouter |
| Supervisor (future) | `supervisor` | Async, CPU or cloud |

---

## 5. Implementation Phases

### Phase 1: LLM Registry + Summarizer on CPU (Low Risk)

1. Add `LLMRegistry` and config schema for `primary` and `summarizer`
2. Install `langchain-ollama` if using `ChatOllama`, or keep `ChatOpenAI` with Ollama base_url
3. Wire Summarizer, SummarizationMiddleware, and Normaliser to `summarizer` LLM
4. Ensure Ollama is running with e.g. `tinyllama` or `qwen2.5:0.5b`
5. Fallback: if summarizer LLM unavailable, fall back to primary

**Deliverable**: Summarization no longer blocks the GPU model.

### Phase 2: OpenRouter for Research (Optional)

1. Add `research` config and `OPENROUTER_API_KEY` support
2. Create a "deep research" tool or subagent that uses `research` LLM
3. Use only when the task explicitly needs external/web research

**Deliverable**: External research can use cloud models without affecting main graph.

### Phase 3: Async Supervisor (Future)

1. Design supervisor contract: what it reads (state snapshot), what it can suggest (milestone changes, task reprioritization)
2. Implement as async task: poll state every N steps or time interval, call `supervisor` LLM, apply suggestions if safe
3. Keep it non-blocking: main graph continues; supervisor runs in background

**Deliverable**: Optional background supervisor for long-running workflows.

### Phase 4: Tool-Specific Models (Future)

1. Allow tools/subagents to declare their preferred model (e.g. `ask` → primary, `deep_research` → research)
2. Pass model into `create_agent()` per tool

**Deliverable**: Flexible per-tool model assignment.

---

## 6. Running CPU Models: Practical Setup

### Option A: Same Machine, GPU + CPU

- **text-generation-webui** (port 5000): Qwen on GPU
- **Ollama** (port 11434): TinyLlama/Phi on CPU
  - Set `OLLAMA_NUM_GPU=0` if you want Ollama to avoid GPU and leave it for text-generation-webui
  - Or use a small model that fits in CPU RAM without needing GPU

### Option B: Separate Machines

- Machine 1: text-generation-webui + Qwen (GPU)
- Machine 2: Ollama (CPU-only, e.g. in Docker without GPU)
- Config: `summarizer.base_url = "http://machine2:11434/v1"`

### Option C: All Local, Single GPU

- Run only text-generation-webui with Qwen
- Summarizer falls back to primary (no Ollama)
- Simplest, but summarization competes for GPU

---

## 7. Dependencies

```txt
# Already have
langchain-openai  # ChatOpenAI for OpenAI-compatible APIs

# Add for Ollama (optional, ChatOpenAI + base_url also works)
langchain-ollama  # ChatOllama for native Ollama integration
```

Using `ChatOpenAI(base_url="http://localhost:11434/v1")` works with Ollama's OpenAI-compatible API, so `langchain-ollama` is optional. `ChatOllama` can provide better defaults and validation.

---

## 8. Summary

| Question | Answer |
|----------|--------|
| **CPU models: Ollama or direct from LangGraph?** | Use **Ollama**. LangGraph doesn't run models; it calls APIs. Ollama is the standard way to run small models on CPU with an OpenAI-compatible API. |
| **How to run multiple models in LangGraph?** | Pass a different LLM into each agent at construction. No special API—each node gets the model it needs. |
| **Sync vs async?** | Main graph stays sync. Summarizer, research, etc. run sync when their node runs. Supervisor runs async in a separate task. |
| **OpenRouter?** | Use `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"` and `api_key`. Use for research and optional supervisor. |
| **First step?** | Phase 1: Add LLM registry, config for summarizer, wire Summarizer + SummarizationMiddleware + Normaliser to a CPU model (Ollama). |
