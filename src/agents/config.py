import json
import os

from .logging_config import get_logger

logger = get_logger(__name__)
config: dict = None

def _initialise_config() -> dict:
    """Get application configuration.

    Loads configuration from a JSON file and also optionally from env vars
    """

    config = {
        "log_level": "INFO",
        "log_file": None,
        "llm": {
            "model": "Qwen3-8B-exl2-6_0",
            "base_url": "http://127.0.0.1:5000/v1",
            "num_ctx": 65536,
            "max_tokens": 16384,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "typical_p": 1.0,
            "min_p": 0.0,
        },
    }

    # Attempt to load config.json; allow missing files, only log if verbose
    try:
        with open("config.json", "r") as f:
            config.update(json.load(f))
    except Exception as e:
        logger.debug("Could not load config.json: %s (using defaults)", e)

    # Env override for LLM base URL (e.g. for e2e with different port)
    if os.getenv("LEMMINGS_LLM_BASE_URL"):
        config["llm"]["base_url"] = os.getenv("LEMMINGS_LLM_BASE_URL")

    # Log level from env (default INFO)
    if os.getenv("LEMMINGS_LOG_LEVEL"):
        config["log_level"] = os.getenv("LEMMINGS_LOG_LEVEL").upper()

    # Derive verbose/debug from log level for LangChain and UI
    config["verbose"] = config["log_level"] == "DEBUG"
    config["debug"] = config["log_level"] == "DEBUG"
    # Show thinking in console by default; set LEMMINGS_NO_THINKING=1 to disable
    config["show_thinking"] = os.getenv("LEMMINGS_NO_THINKING", "").strip().lower() not in ("1", "true", "yes")
    if os.getenv("LEMMINGS_LOG_FILE") is not None:
        config["log_file"] = os.getenv("LEMMINGS_LOG_FILE")

    # Return config
    return config

config = _initialise_config()