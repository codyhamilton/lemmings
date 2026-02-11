import json
import os

config: dict = None

def _initialise_config() -> dict:
    """Get application configuration.

    Loads configuration from a JSON file and also optionally from env vars
    """

    config = {
        "debug": False,
        "verbose": False,
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
        if config.get("verbose", False):
            print(f"[config] Could not load config.json: {e} (using defaults)")

    # Load override env vars for debug and verbose
    if os.getenv("LEMMINGS_DEBUG", "false").lower() == "true":
        config["debug"] = True
    if os.getenv("LEMMINGS_VERBOSE", "false").lower() == "true":
        config["verbose"] = True

    # Return config
    return config

config = _initialise_config()