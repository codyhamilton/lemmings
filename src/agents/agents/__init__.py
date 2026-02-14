"""Agent modules for the orchestration system."""

# Lazy imports to avoid circular import issues when agents package is loaded
# These are only imported when actually needed (e.g., when importing from agents.agents)

__all__ = [
    "initial_scope_agent_node",
    "scope_review_agent_node",
    "task_planner_node",
    "implementor_node",
    "qa_node",
    "assessor_node",
    "report_node",
]


def __getattr__(name):
    """Lazy import agent nodes when accessed."""
    if name == "initial_scope_agent_node":
        from .scope_agent import initial_scope_agent_node
        return initial_scope_agent_node
    elif name == "scope_review_agent_node":
        from .scope_agent import scope_review_agent_node
        return scope_review_agent_node
    elif name == "task_planner_node":
        from .task_planner import task_planner_node
        return task_planner_node
    elif name == "implementor_node":
        from .implementor import implementor_node
        return implementor_node
    elif name == "qa_node":
        from .qa import qa_node
        return qa_node
    elif name == "assessor_node":
        from .assessor import assessor_node
        return assessor_node
    elif name == "report_node":
        from .report import report_node
        return report_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
