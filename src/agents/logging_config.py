"""Centralized logging configuration for the agent system."""

import logging
import os
from typing import Optional

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger with level and output destination.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file. If None, logs to stdout only.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Use env vars as overrides if set
    env_level = os.getenv("LEMMINGS_LOG_LEVEL")
    if env_level:
        log_level = getattr(logging, env_level.upper(), log_level)

    env_file = os.getenv("LEMMINGS_LOG_FILE")
    if env_file is not None:
        log_file = env_file

    root = logging.getLogger("agents")
    root.setLevel(log_level)

    # Clear any existing handlers
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    # Stdout handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            root.warning("Could not open log file %s, logging to stdout only", log_file)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance.
    """
    if name.startswith("agents.") or name == "agents":
        return logging.getLogger(name)
    return logging.getLogger(f"agents.{name}")
