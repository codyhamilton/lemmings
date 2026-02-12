"""Minimal workflow status helpers for callbacks and legacy agent display.

Streaming output is handled by the stream layer and ConsoleUI. This module
provides no-op stubs - only console UI and main.py error paths should print.
"""

from enum import Enum


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
    """No-op; stream layer handles real output via ConsoleUI."""

def print_thinking_line(msg: str) -> None:
    """No-op; stream layer handles real output."""

def print_thinking(msg: str) -> None:
    """No-op; stream layer handles real output."""
