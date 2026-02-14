"""Subagent tools for ScopeAgent and TaskPlanner.

These tools wrap research capabilities that parent agents can invoke when needed.
- explain_code: Deep codebase research with detailed structural explanation
- ask: Targeted factual questions about the codebase
- web_search: Search the internet for external information
"""

from .explain_code import explain_code
from .ask import ask
from .web_search import web_search

__all__ = [
    "explain_code",
    "ask",
    "web_search",
]
