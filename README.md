# Multi-Agent Orchestration System

A LangGraph-based multi-agent system for coordinating local LLM agents to handle development tasks efficiently.

## New: RAG-Based Context Management 🚀

The system now includes **Retrieval-Augmented Generation (RAG)** for fast, semantic code search:

- **30-60x faster** than shell-based tools (1-2s vs 30-60s)
- **Semantic search** finds relevant code by meaning, not just keywords
- **Pre-loaded context** for planner with repo overview and relevant docs
- **Automatic fallback** to shell tools if RAG unavailable

### Quick Start with RAG

```bash
# 1. Build the RAG index (one-time setup, ~5-10s)
rag build

# 2. Run agents - index updates automatically before each run
agents "Add a new resource type 'titanium' to the economy"

# Optional: Watch for file changes and auto-update (10s debounce)
rag build -w
# or
rag watch

# Manual update if needed
rag update

# Check index stats
rag stats
```

**Index Location**: `.rag_index/` (local to repo, git-ignored)
**Auto-Update**: Index automatically updates when running agents (only changed files)
**File Watching**: Use `-w` flag or `watch` command for real-time updates

## Architecture Overview

### Sequential Execution (Critical)

**This system executes agents SEQUENTIALLY, not in parallel.**

- **Why?** We have a single GPU and single LLM instance (Qwen2.5-Coder-7B via TabbyAPI/ExLlamaV2)
- **Why divide agents?** To limit context per agent, not for parallelism
- **How?** LangGraph executes nodes sequentially by default - each node completes before the next starts

**Example:** If processing 10 changes:
```
Change 1: coder → reviewer → next_change
Change 2: coder → reviewer → next_change
...
Change 10: coder → reviewer → validator
```

All changes are processed **one at a time** to ensure:
- Efficient GPU usage (only one agent uses GPU at a time)
- Predictable state transitions
- Proper context accumulation
- No context overflow

### Pipeline Stages

The workflow follows a 7-phase pipeline:

1. **UNDERSTAND** (planner) - Break request into requirements with keywords
2. **RESEARCH** (researcher) - Find context for each requirement (sequential loop)
3. **PLAN** (requirement_planner) - Create change summary per requirement (sequential loop)
4. **CONSOLIDATE** (consolidator) - Reconcile into unified change list
5. **CODE** (coder) - Implement each change (sequential loop)
6. **REVIEW** (reviewer) - Validate each change (sequential loop, with retry)
7. **VALIDATE** (validator) - Final validation (can trigger refinement)

### Self-Healing Mechanisms

The system includes multiple retry and recovery mechanisms:

- **Phase retries**: Each phase can retry up to `MAX_PHASE_RETRIES` (2) times on failure
- **Coder retries**: Up to `MAX_CODER_RETRIES` (3) times after review failure
- **Skip on persistent failure**: If retries exhausted, skip and continue
- **Refinement loop**: Validator can trigger full refinement with accumulated learnings
- **Error accumulation**: Errors are tracked and passed to refinement iterations

### Context Efficiency

Each agent receives only the context it needs:

- **Planner**: User request + **RAG repo overview** + **RAG relevant docs** + refinement context
- **Researcher**: Single requirement + **RAG semantic search** (replaces 3-pass shell tool loop)
- **Requirement Planner**: Single requirement + its context
- **Consolidator**: All requirements + contexts (necessary for reconciliation)
- **Coder**: Single change + its context (can query RAG for patterns)
- **Reviewer**: Single change + its requirements (NOT full plan)
- **Validator**: All requirements + all changes + reviews (necessary for validation)

### Visual Progress Tracking

The system provides real-time visual feedback:

- Pipeline progress bar showing current phase
- Requirement/change progress (X/Y)
- Retry indicators
- Refinement iteration tracking
- Streaming output from each agent

## Usage

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Run a task
agents "Add a new resource type 'titanium' to the economy"

# With verbose output
agents -v "Fix the bug in colony view"

# Set max refinement iterations
agents --max-iterations 5 "Complex refactoring task"
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (tested on RTX 3080 Ti)
- CUDA 12.1+ compatible drivers
- TabbyAPI running locally with ExLlamaV2 backend
- Dependencies in `requirements.txt` (includes ChromaDB and sentence-transformers for RAG)

## Installation

### 1. Install Python Dependencies

```bash
# From the lemmings project root
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the package in editable mode and registers the `rag` and `agents` console commands.

### 2. Install TabbyAPI

TabbyAPI is the OpenAI-compatible server that provides ExLlamaV2 inference.

```bash
# TabbyAPI is included in the project (as a git submodule or directory)
# If it's a submodule, initialize it:
# git submodule update --init --recursive

# Install TabbyAPI and ExLlamaV2
cd tabbyAPI
pip install -e .

# Return to project root
cd ..
```

### 3. Download the Model

The `models/` directory is at the project root. Download a pre-quantized EXL2 model:

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface-hub

# Models directory is already at the project root: models/
# Download your preferred model, for example:

# Qwen3-8B (currently configured)
huggingface-cli download <model-repo> \
  --local-dir models/Qwen3-8B-exl2-6_0

# Or other models:
# StarCoder2-7B: bartowski/starcoder2-7b-exl2
# CodeQwen1.5-7B: bartowski/CodeQwen1.5-7B-exl2
# Qwen2.5-Coder-7B: bartowski/Qwen2.5-Coder-7B-Instruct-exl2
```

**Note:** Update `model_name` in `tabby_config.yml` to match your downloaded model folder name.

### 4. Configure TabbyAPI

The project includes a `tabby_config.yml` file optimized for 12GB VRAM with 32K context:

The project includes `tabby_config.yml` and `tabby_config_qwen3.yml` files at the root, optimized for 12GB VRAM with 32K context:

```yaml
model:
  model_dir: /home/codyh/workspace/lemmings/models  # Absolute path to models directory
  model_name: Qwen3-8B-exl2-6_0  # Model name (adjust to match your model)
  prompt_template: tool_calls/chatml_with_headers  # or qwen3 for qwen3 config
  gpu_split_auto: true

network:
  host: 127.0.0.1
  port: 5000

sampling:
  temperature: 0.6  # Recommended for Qwen3 thinking mode
  top_p: 0.95
  top_k: 20
  min_p: 0.0
```

**Key Optimizations:**
- **Q4 KV cache**: Dramatically reduces VRAM usage for context storage
- **Flash Attention 2**: Enabled by default in ExLlamaV2 (faster, lower memory)
- **6.5 bpw quantization**: Bartowski's recommended quant - excellent quality for coding
- **32K context**: Optimal balance of quality/capacity with safe VRAM margins (~1.5GB headroom)

### 5. Start TabbyAPI Server

```bash
# In a separate terminal, start TabbyAPI
cd tabbyAPI
python -m tabbyAPI.main --config ../tabby_config.yml
# or use the qwen3 config:
python -m tabbyAPI.main --config ../tabby_config_qwen3.yml
```

The server will start on `http://127.0.0.1:5000` and provide an OpenAI-compatible API.

### 6. Build the RAG Index

```bash
# In the original terminal with the virtual environment activated
# (from the lemmings project root)

# Build the index (one-time, ~5-10s)
rag build

# This will:
# - Scan all .gd, .json, .tscn, and .md files
# - Create semantic embeddings using sentence-transformers
# - Store in <repo>/.rag_index/ (git-ignored)

# Optional: Keep index updated automatically
rag build -w   # Build then watch for changes
# or
rag watch      # Just watch (if already built)
```

### 7. Run the Agent System

```bash
# Run with RAG-powered context retrieval
# Index auto-updates before each run (only changed files)
agents "Add a new resource type 'titanium' to the economy"

# The index will automatically update incrementally before running
# No manual index updates needed!
```

## Performance Expectations

With the optimized TabbyAPI + ExLlamaV2 setup on 12GB VRAM:

| Metric | Expected Performance |
|--------|---------------------|
| Max context | 32K tokens |
| Quantization | 6.5 bpw (bartowski recommended) |
| GPU utilization | 100% (fully on-GPU) |
| Prompt processing | 150-250 tokens/sec |
| Generation speed | 30-45 tokens/sec |
| VRAM usage | ~10.5-11 GB peak |
| VRAM headroom | ~1-1.5 GB safety margin |
| Code quality | Excellent (near 8.0 bpw quality) |

## Performance Comparison

### RAG vs Shell Tools (Researcher Agent)

| Metric | Shell Tools (Old) | RAG (New) |
|--------|------------------|-----------|
| Research time | 30-60s | 1-2s |
| Tool calls | ~42 (3 passes × 14) | 0 (direct retrieval) |
| Search type | Keyword-based | Semantic |
| Accuracy | Good | Excellent |
| First-time setup | None | ~5-10s (index build) |

**Result**: 30-60x faster context retrieval with better semantic understanding.

## Troubleshooting

**RAG index issues:**
- **"Index not found"**: Run `rag build` to create the index
- **Auto-update not working**: Index updates automatically when running agents (only changed files)
- **Out of disk space**: Index is ~10-50MB. Stored in `<repo>/.rag_index/` (git-ignored)
- **Embedding model download**: First run downloads `all-MiniLM-L6-v2` (~90MB) from HuggingFace
- **File watching**: Use `rag build -w` for real-time updates (requires `watchdog` package)
- **Commands not found**: Make sure you've run `pip install -e .` in the agents directory to register the console scripts

**Out of Memory (OOM) errors:**
- Reduce `max_seq_len` in `tabby_config.yml` (try 24576 or 16384)
- Use a lower bpw model (try 5.0 bpw or 4.25 bpw instead of 6.5 bpw)
- Enable `gpu_split_auto: true` if you have multiple GPUs

**Slow generation:**
- Ensure Flash Attention 2 is enabled (default in ExLlamaV2)
- Check GPU isn't being used by other processes
- Verify CUDA drivers are up to date

**Connection errors:**
- Ensure TabbyAPI is running before starting the agent system
- Check that port 5000 is not blocked by firewall
- Verify `base_url` in `agents/llm.py` matches TabbyAPI configuration
