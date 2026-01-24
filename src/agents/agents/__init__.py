"""Agent modules for the orchestration system."""

# Lazy imports to avoid circular import issues when agents package is loaded
# These are only imported when actually needed (e.g., when importing from agents.agents)

__all__ = [
    "intent_node",
    "milestone_node",
    "expander_node",
    "prioritizer_node",
    "researcher_node",
    "planner_node",
    "implementor_node",
    "validator_node",
    "qa_node",
    "assessor_node",
]


def __getattr__(name):
    """Lazy import agent nodes when accessed."""
    if name == "intent_node":
        from .intent import intent_node
        return intent_node
    elif name == "milestone_node":
        from .milestone import milestone_node
        return milestone_node
    elif name == "expander_node":
        from .expander import expander_node
        return expander_node
    elif name == "prioritizer_node":
        from .prioritizer import prioritizer_node
        return prioritizer_node
    elif name == "researcher_node":
        from .researcher import researcher_node
        return researcher_node
    elif name == "planner_node":
        from .planner import planner_node
        return planner_node
    elif name == "implementor_node":
        from .implementor import implementor_node
        return implementor_node
    elif name == "validator_node":
        from .validator import validator_node
        return validator_node
    elif name == "qa_node":
        from .qa import qa_node
        return qa_node
    elif name == "assessor_node":
        from .assessor import assessor_node
        return assessor_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
