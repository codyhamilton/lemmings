"""Minimal workflow status helpers for callbacks and legacy agent display.

Streaming output is handled by the stream layer and ConsoleUI. This module
provides stubs for code that still imports display_agent_header, Phase, get_status,
print_thinking_line, print_thinking.
"""

from enum import Enum
import sys


class Phase(str, Enum):
    CODE = "code"
    UNDERSTAND = "understand"
    RESEARCH = "research"
    PLAN = "plan"
    CONSOLIDATE = "consolidate"
    REVIEW = "review"
    VALIDATE = "validate"
    REFINE = "refine"
    COMPLETE = "complete"


class _Status:
    def set_change_progress(self, current: int, total: int) -> None:
        pass


def get_status() -> _Status:
    return _Status()


def display_agent_header(agent_name: str, phase: Phase, subtitle: str, details: dict) -> None:
    """Minimal display; stream layer handles real output."""
    print(f"\n[{agent_name}] {subtitle}", flush=True)


def print_thinking_line(msg: str) -> None:
    print(msg, flush=True)


def print_thinking(msg: str) -> None:
    print(msg, end="", flush=True)
