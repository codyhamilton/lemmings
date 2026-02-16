"""Normalize raw LangGraph stream chunks into stable models and dispatch to handlers.

This is the only place that knows LangGraph chunk shapes. When LangGraph changes,
only this module changes. Downstream handlers receive MessageChunk or StatusUpdate only.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MessageChunk:
    """Normalized message chunk: node id and text content."""
    node_id: str
    content: str
    is_tool_result: bool = False


@dataclass
class StatusUpdate:
    """Normalized status update: node name and state update dict."""
    node_name: str
    state_update: Dict[str, Any]


def _normalize_message_chunk(raw: Any) -> Optional[MessageChunk]:
    """Extract node_id and content from raw message-mode chunk. Returns None if not message chunk."""
    node_id = "default"
    content = ""

    if isinstance(raw, tuple) and len(raw) >= 2:
        # LangGraph often yields (content_or_msg, metadata) or (node_info, (chunk, metadata))
        first, second = raw[0], raw[1]
        metadata = None
        content_or_msg = None

        if isinstance(second, dict):
            metadata = second
            content_or_msg = first
        elif isinstance(second, tuple) and len(second) >= 2:
            # (node_info, (chunk, metadata))
            content_or_msg = second[0]
            metadata = second[1] if len(second) > 1 else {}
            if isinstance(first, dict) and "node" in first:
                node_id = first.get("node", node_id)
            elif isinstance(first, str):
                node_id = first

        if metadata is not None:
            node_id = metadata.get("langgraph_node", metadata.get("node", node_id))

        if content_or_msg is not None:
            msg_cls = getattr(content_or_msg, "__class__", None)
            cls_name = getattr(msg_cls, "__name__", str(type(content_or_msg)))
            is_tool_result = cls_name == "ToolMessage"
            if logger.isEnabledFor(logging.DEBUG):
                content_preview = ""
                if hasattr(content_or_msg, "content"):
                    c = getattr(content_or_msg, "content", None)
                    if isinstance(c, str):
                        content_preview = c[:200] + ("..." if len(c) > 200 else "")
                    elif isinstance(c, list):
                        content_preview = f"list[{len(c)} items]"
                    else:
                        content_preview = str(c)[:200] + ("..." if len(str(c)) > 200 else "")
                else:
                    content_preview = str(content_or_msg)[:200] + ("..." if len(str(content_or_msg)) > 200 else "")
                logger.debug(
                    "Stream chunk: raw_type=%s content_or_msg_type=%s content_preview=%s",
                    type(raw).__name__,
                    cls_name,
                    content_preview,
                )
            if isinstance(content_or_msg, str):
                content = content_or_msg
            elif hasattr(content_or_msg, "content"):
                parts = getattr(content_or_msg, "content", [])
                if isinstance(parts, list):
                    for item in parts:
                        if isinstance(item, dict) and "text" in item:
                            content += item.get("text", "")
                        elif isinstance(item, str):
                            content += item
                else:
                    content = str(parts)
            else:
                content = str(content_or_msg)

            return MessageChunk(node_id=node_id, content=content, is_tool_result=is_tool_result)

    if isinstance(raw, str):
        return MessageChunk(node_id=node_id, content=raw, is_tool_result=False)

    return None


def _is_update_chunk(chunk: Any) -> bool:
    """True if chunk is an updates-mode dict (node_name -> state_update)."""
    return (
        isinstance(chunk, dict)
        and len(chunk) > 0
        and all(isinstance(k, str) for k in chunk.keys())
    )


# Public alias for state accumulation in main.py
is_update_chunk = _is_update_chunk


class StreamHandler:
    """Normalizes raw stream chunks and dispatches to message and status handlers. Stateless."""

    def __init__(
        self,
        message_handler: Optional[Any] = None,
        status_handler: Optional[Any] = None,
    ):
        self._message_handler = message_handler
        self._status_handler = status_handler

    def handle(self, chunk: Any) -> None:
        """Normalize chunk and pass to the appropriate handler."""
        # With multiple stream modes, LangGraph yields (metadata, mode, data) tuples
        raw = chunk
        if isinstance(chunk, tuple) and len(chunk) >= 3:
            mode, raw = chunk[1], chunk[2]
            if mode == "updates" and isinstance(raw, dict):
                for node_name, state_update in raw.items():
                    if isinstance(state_update, dict) and self._status_handler is not None:
                        self._status_handler.process_status_update(
                            StatusUpdate(node_name=node_name, state_update=state_update)
                        )
                return
            if mode == "messages":
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Stream handle: mode=messages raw_type=%s", type(raw).__name__)
                msg = _normalize_message_chunk(raw)
                if msg is not None and self._message_handler is not None:
                    logger.debug("Dispatching message chunk from node %s", msg.node_id)
                    self._message_handler.handle(msg)
                return

        if _is_update_chunk(chunk):
            logger.debug("Dispatching status update chunk for %s", list(chunk.keys()))
            for node_name, state_update in chunk.items():
                if isinstance(state_update, dict) and self._status_handler is not None:
                    self._status_handler.process_status_update(
                        StatusUpdate(node_name=node_name, state_update=state_update)
                    )
            return

        msg = _normalize_message_chunk(chunk)
        if msg is not None and self._message_handler is not None:
            logger.debug("Dispatching message chunk from node %s", msg.node_id)
            self._message_handler.handle(msg)

    def finalize(self) -> None:
        """Finalize both handlers."""
        logger.debug("Finalizing stream handlers")
        if self._message_handler is not None and hasattr(self._message_handler, "finalize"):
            self._message_handler.finalize()
        if self._status_handler is not None and hasattr(self._status_handler, "finalize"):
            self._status_handler.finalize()
