"""LLM configuration for text-generation-webui with OpenAI-compatible API."""

from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug, set_verbose
from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)

default_llm: ChatOpenAI = None
planning_llm: ChatOpenAI = None
coding_llm: ChatOpenAI = None
review_llm: ChatOpenAI = None

def get_llm(
    model: str = config["llm"]["model"],
    base_url: str = config["llm"]["base_url"],
    verbose: bool = config["verbose"],
    debug: bool = config["debug"],
) -> ChatOpenAI:
    """Get a configured ChatOpenAI instance pointing to text-generation-webui.
    
    text-generation-webui now properly returns tool calls in the tool_calls property
    of the API response, so LangChain can handle them natively without transformation.
    
    Args:
        model: The model name (should match the model loaded in text-generation-webui)
        base_url: The text-generation-webui API base URL (OpenAI-compatible endpoint)
        num_ctx: Context window size (handled by text-generation-webui config)
        verbose: Enable verbose logging to see API requests/responses
    
    Returns:
        Configured ChatOpenAI instance that handles tool calls natively
    """
    llm_kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": "not-needed",  # text-generation-webui doesn't require auth by default
        "max_tokens": config["llm"]["max_tokens"],
        "verbose": verbose,  # Enable verbose output if requested
    }
    # num_ctx (context window) is server-side config in text-generation-webui; OpenAI API doesn't support it
    
    # Use ChatOpenAI directly - tool calls are now handled natively by LangChain
    llm = ChatOpenAI(**llm_kwargs)
    
    # Enable debug logging if verbose is set or DEBUG_LANGCHAIN env var is set
    if verbose:
        set_verbose(True)
    if debug:
        set_debug(True)
        logger.info("LangChain debug logging enabled")
    
    return llm

def initialise_llms():
    global default_llm, planning_llm, coding_llm, review_llm
    default_llm = get_llm()
    planning_llm = default_llm
    coding_llm = default_llm
    review_llm = default_llm