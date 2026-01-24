"""Callback handlers for streaming + tool logging.

These callbacks are more reliable than inferring tool calls/results from agent messages.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .workflow_status import print_thinking_line, print_thinking


def _compact(obj: Any, limit: int = 240) -> str:
    s = str(obj)
    s = " ".join(s.split())  # collapse whitespace/newlines
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


class StreamingToolLogger(BaseCallbackHandler):
    """Prints tool start/end (with inputs) and optionally LLM events."""

    def __init__(self, label: str = "") -> None:
        self.label = label.strip()

    def _prefix(self) -> str:
        return f"{self.label}: " if self.label else ""

    # ---- Tools ----
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name", "tool")
        print_thinking_line(f"{self._prefix()}{name}({ _compact(input_str) })")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        print_thinking_line(f"{self._prefix()}tool_result: {_compact(output)}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        print_thinking_line(f"{self._prefix()}tool_error: {error}")

    # ---- LLM (optional; useful to prove streaming is alive) ----
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        # Avoid dumping prompts; just show that the LLM call started.
        model = serialized.get("name") or serialized.get("id") or "llm"
        print_thinking_line(f"{self._prefix()}llm_start({model})")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print_thinking_line(f"{self._prefix()}llm_end")

