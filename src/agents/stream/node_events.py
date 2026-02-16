"""Node lifecycle events: start/end emitted from LangChain chain callbacks for console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from langchain_core.callbacks import BaseCallbackHandler

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NodeStartEvent:
    """Emitted when a graph node starts (on_chain_start with langgraph_node in metadata)."""
    node_name: str


@dataclass
class NodeEndEvent:
    """Emitted when a graph node ends (on_chain_end or on_chain_error)."""
    node_name: str
    summary: str
    failed: bool = False


class NodeEventStream:
    """Emits node start/end events to subscribers. Used by NodeEventEmitter and consumed by ConsoleUI."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[NodeStartEvent | NodeEndEvent], None]] = []

    def subscribe(self, callback: Callable[[NodeStartEvent | NodeEndEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[NodeStartEvent | NodeEndEvent], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit_start(self, node_name: str) -> None:
        event = NodeStartEvent(node_name=node_name)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    def emit_end(self, node_name: str, summary: str, failed: bool = False) -> None:
        event = NodeEndEvent(node_name=node_name, summary=summary, failed=failed)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass


def _inputs_look_like_workflow_state(inputs: dict[str, Any]) -> bool:
    """True if inputs are the full workflow state (so this run is the workflow node, not a nested chain)."""
    if not inputs or not isinstance(inputs, dict):
        return False
    return "user_request" in inputs and "repo_root" in inputs


class NodeEventEmitter(BaseCallbackHandler):
    """LangChain callback that emits NodeStartEvent and NodeEndEvent from on_chain_start/on_chain_end/on_chain_error.

    Only emits for the run that represents the workflow node: metadata has langgraph_node and inputs
    look like workflow state (user_request, repo_root). Nested chain/llm runs are ignored.
    """

    def __init__(self, stream: NodeEventStream) -> None:
        super().__init__()
        self._stream = stream
        self._pending: dict[str, str] = {}  # run_id -> node_name

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: List[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        node_name = meta.get("langgraph_node")
        is_workflow_node = (
            node_name is not None and _inputs_look_like_workflow_state(inputs)
        )
        logger.debug(
            "chain_start run_id=%s parent_run_id=%s langgraph_node=%s is_workflow_node=%s metadata_keys=%s",
            run_id,
            parent_run_id,
            node_name,
            is_workflow_node,
            list(meta.keys()),
        )
        if is_workflow_node:
            if run_id is not None:
                self._pending[str(run_id)] = node_name
            self._stream.emit_start(node_name)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        node_name = None
        if run_id is not None:
            key = str(run_id)
            if key in self._pending:
                node_name = self._pending.pop(key)
        logger.debug(
            "chain_end run_id=%s parent_run_id=%s node_name=%s",
            run_id,
            parent_run_id,
            node_name,
        )
        if node_name is not None:
            self._stream.emit_end(node_name, "Complete", failed=False)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        node_name = None
        if run_id is not None:
            key = str(run_id)
            if key in self._pending:
                node_name = self._pending.pop(key)
        logger.debug(
            "chain_error run_id=%s parent_run_id=%s node_name=%s error=%s",
            run_id,
            parent_run_id,
            node_name,
            error,
        )
        if node_name is not None:
            self._stream.emit_end(node_name, str(error), failed=True)
