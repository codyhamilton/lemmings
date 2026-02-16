"""Tool lifecycle events: start/end emitted from LangChain callbacks for console and storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from langchain_core.callbacks import BaseCallbackHandler


def _compact(obj: Any, limit: int = 240) -> str:
    s = str(obj)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


@dataclass
class ToolStartEvent:
    """Emitted when a tool begins execution."""
    name: str
    args_repr: str
    milestone_id: str | None = None
    task_id: str | None = None


@dataclass
class ToolEndEvent:
    """Emitted when a tool finishes (success or error)."""
    name: str
    args_repr: str
    output: str
    milestone_id: str | None = None
    task_id: str | None = None


class ToolEventStream:
    """Emits tool start/end events to subscribers. Used by ToolEventEmitter and consumed by ConsoleUI."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[ToolStartEvent | ToolEndEvent], None]] = []

    def subscribe(self, callback: Callable[[ToolStartEvent | ToolEndEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ToolStartEvent | ToolEndEvent], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit_start(
        self,
        name: str,
        args_repr: str,
        milestone_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        event = ToolStartEvent(
            name=name,
            args_repr=args_repr,
            milestone_id=milestone_id,
            task_id=task_id,
        )
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    def emit_end(
        self,
        name: str,
        args_repr: str,
        output: str,
        milestone_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        event = ToolEndEvent(
            name=name,
            args_repr=args_repr,
            output=output,
            milestone_id=milestone_id,
            task_id=task_id,
        )
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass


class ToolEventEmitter(BaseCallbackHandler):
    """LangChain callback that emits ToolStartEvent and ToolEndEvent to a ToolEventStream."""

    def __init__(self, stream: ToolEventStream) -> None:
        super().__init__()
        self._stream = stream
        self._pending: dict[str, tuple[str, str]] = {}  # run_id -> (name, args_repr)

    def _context_from_kwargs(self, kwargs: Any) -> tuple[str | None, str | None]:
        config = kwargs.get("config") or {}
        configurable = config.get("configurable") or {}
        milestone_id = configurable.get("milestone_id")
        task_id = configurable.get("task_id")
        return (milestone_id, task_id)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name", "tool")
        args_repr = _compact(input_str)
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._pending[str(run_id)] = (name, args_repr)
        milestone_id, task_id = self._context_from_kwargs(kwargs)
        self._stream.emit_start(name, args_repr, milestone_id=milestone_id, task_id=task_id)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        name, args_repr = "tool", ""
        if run_id is not None:
            key = str(run_id)
            if key in self._pending:
                name, args_repr = self._pending.pop(key)
        output_str = str(output) if not isinstance(output, str) else output
        milestone_id, task_id = self._context_from_kwargs(kwargs)
        self._stream.emit_end(name, args_repr, output_str, milestone_id=milestone_id, task_id=task_id)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        name, args_repr = "tool", ""
        if run_id is not None:
            key = str(run_id)
            if key in self._pending:
                name, args_repr = self._pending.pop(key)
        milestone_id, task_id = self._context_from_kwargs(kwargs)
        self._stream.emit_end(name, args_repr, str(error), milestone_id=milestone_id, task_id=task_id)
