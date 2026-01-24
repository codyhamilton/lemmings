"""LLM configuration for TabbyAPI with ExLlamaV2 backend."""

import os
from langchain_openai import ChatOpenAI


def get_llm(
    model: str = "Qwen3-8B-exl2-6_0",
    base_url: str = "http://127.0.0.1:5000/v1",
    num_ctx: int = 38400,
    temperature: float = None,  # None = use config defaults (recommended for Qwen3)
    verbose: bool = False,
) -> ChatOpenAI:
    """Get a configured ChatOpenAI instance pointing to TabbyAPI.
    
    Args:
        model: The model name (should match the model loaded in TabbyAPI)
        base_url: The TabbyAPI base URL (OpenAI-compatible endpoint)
        num_ctx: Context window size (TabbyAPI handles this via max_seq_len config)
        temperature: Sampling temperature (None = use config defaults, recommended for Qwen3 thinking mode)
        verbose: Enable verbose logging to see API requests/responses
    
    Returns:
        Configured ChatOpenAI instance
    """
    llm_kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": "not-needed",  # TabbyAPI doesn't require auth by default
        "max_tokens": 16384,  # Max tokens to generate per response
        "verbose": verbose,  # Enable verbose output if requested
    }
    
    # Only set temperature if explicitly provided (otherwise use config defaults)
    # This is important for Qwen3 thinking mode which has specific recommendations
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Enable debug logging if verbose is set or DEBUG_LANGCHAIN env var is set
    if verbose or os.getenv("DEBUG_LANGCHAIN", "").lower() in ("true", "1", "yes"):
        try:
            from langchain.globals import set_debug, set_verbose
            set_debug(True)
            set_verbose(True)
            print("🔍 LangChain debug logging enabled - you'll see detailed execution traces")
        except ImportError:
            try:
                from langchain_core.globals import set_debug
                set_debug(True)
                print("🔍 LangChain debug logging enabled")
            except ImportError:
                print("⚠️  Could not enable debug logging - langchain.globals not available")
    
    return llm


# Check if verbose/debug mode is requested
_verbose = os.getenv("DEBUG_LANGCHAIN", "").lower() in ("true", "1", "yes")

# Default LLM instance for the orchestration system
# Uses Qwen3-8B with thinking mode (uses config defaults - recommended)
default_llm = get_llm(verbose=_verbose)


# Smaller/faster model for simpler tasks like planning
# Uses config defaults (recommended for Qwen3 thinking mode)
planning_llm = get_llm(verbose=_verbose)


# Model for code generation
# Uses config defaults (recommended for Qwen3 thinking mode)
coding_llm = get_llm(verbose=_verbose)


# Model for review
# Uses config defaults (recommended for Qwen3 thinking mode)
review_llm = get_llm(verbose=_verbose)
