"""AI message stream handler: block parsing and event emission per node.

Accepts normalized MessageChunk (node_id + content) from the stream handler.
Emits StreamEvent with node_id. Consumers subscribe for text and block transitions.
"""

from enum import Enum
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass

from .handler import MessageChunk


# Blocks to output as events
class BlockType(Enum):
    THINK = "think"
    TOOL_CALL = "tool_call"
    CODE_INLINE = "code_inline"
    CODE_BLOCK = "code_block"


BLOCKS = {
    BlockType.THINK: {"start_tag": "<think>", "end_tag": "</think>"},
    BlockType.TOOL_CALL: {"start_tag": "<tool_call>", "end_tag": "</tool_call>"},
    BlockType.CODE_INLINE: {"start_tag": "`", "end_tag": "`"},
    BlockType.CODE_BLOCK: {"start_tag": "```", "end_tag": "```"},
}


class StreamEventType(Enum):
    TEXT_CHUNK = "text"
    BLOCK_START = "block_start"
    BLOCK_END = "block_end"


@dataclass
class Block:
    text: str
    type: BlockType
    start_index: int
    end_index: int
    event_type: StreamEventType = StreamEventType.BLOCK_START


@dataclass
class StreamEvent:
    type: StreamEventType
    block: Optional[BlockType]
    text: str
    node_id: Optional[str] = None
    is_tool_result: bool = False


@dataclass
class _NodeStreamState:
    """Per-node state for block parsing and buffering."""
    open_blocktype: Optional[BlockType] = None
    buffer: str = ""


class AIMessageStreamHandler:
    """Processes message chunks per node; emits StreamEvents with node_id. Subscribe to get events."""

    def __init__(self) -> None:
        self._node_state: Dict[str, _NodeStreamState] = {}
        self._event_subscribers: List[Callable[[StreamEvent], None]] = []

    def _get_node_state(self, node_id: str) -> _NodeStreamState:
        if node_id not in self._node_state:
            self._node_state[node_id] = _NodeStreamState()
        return self._node_state[node_id]

    def handle(self, chunk: MessageChunk) -> None:
        """Handle a normalized MessageChunk (node_id, content)."""
        if not isinstance(chunk, MessageChunk):
            return
        node_id = chunk.node_id
        content = chunk.content or ""
        state = self._get_node_state(node_id)
        is_tool_result = getattr(chunk, "is_tool_result", False)
        self._process_text(node_id, content, state, is_tool_result=is_tool_result)

    def close_node(self, node_id: str) -> None:
        """Remove node from internal state. No open-block cleanup yet."""
        self._node_state.pop(node_id, None)

    def subscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        if callback not in self._event_subscribers:
            self._event_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    def finalize(self) -> None:
        """Cleanup; no pending messages collected."""
        pass

    def _process_text(
        self,
        node_id: str,
        chunk: str,
        state: _NodeStreamState,
        is_tool_result: bool = False,
    ) -> None:
        index = 0
        text = state.buffer + chunk
        state.buffer = ""

        while index < len(text):
            block = None
            if state.open_blocktype is not None:
                block = self._find_closing_block(text, state.open_blocktype, index)
            else:
                block = self._find_next_block(text, index)

            if block is not None:
                self._emit_event(
                    node_id, text[index : block.start_index], StreamEventType.TEXT_CHUNK,
                    is_tool_result=is_tool_result,
                )
                self._emit_event(node_id, block.text, block.event_type, block.type)
                index = block.end_index
                state.open_blocktype = (
                    block.type if block.event_type == StreamEventType.BLOCK_START else None
                )
            else:
                partial_index = self._find_partial_index(text, index, state.open_blocktype)
                if partial_index < len(text):
                    state.buffer = text[partial_index:]
                self._emit_event(
                    node_id, text[index:partial_index], StreamEventType.TEXT_CHUNK,
                    is_tool_result=is_tool_result,
                )
                index = len(text)

    def _find_partial_index(
        self, text: str, start_index: int, open_blocktype: Optional[BlockType]
    ) -> int:
        if open_blocktype is not None:
            search_tags = [BLOCKS[open_blocktype]["end_tag"]]
        else:
            search_tags = [
                BLOCKS[BlockType.THINK]["start_tag"],
                BLOCKS[BlockType.TOOL_CALL]["start_tag"],
            ]
        if open_blocktype not in (BlockType.THINK, BlockType.TOOL_CALL):
            if text.endswith("``"):
                return len(text) - 2
            if text.endswith("`"):
                return len(text) - 1
        max_len = max(len(t) for t in search_tags)
        index = max(start_index, len(text) - max_len + 1)
        for search_tag in search_tags:
            i = text.rfind(search_tag[0], index)
            if i != -1 and text[i:] == search_tag[: len(text) - i]:
                return i
        return len(text)

    def _find_next_block(self, text: str, start_index: int) -> Optional[Block]:
        code_index = text.find("`", start_index, len(text) - 1)
        tag_end_index = len(text) if code_index == -1 else code_index
        block = None
        for block_type in (BlockType.THINK, BlockType.TOOL_CALL):
            found = text.find(BLOCKS[block_type]["start_tag"], start_index, tag_end_index)
            if found > -1 and (block is None or found < block.start_index):
                tag = BLOCKS[block_type]["start_tag"]
                block = Block(
                    text=tag,
                    type=block_type,
                    start_index=found,
                    end_index=found + len(tag),
                )
        if block is None and code_index > -1:
            if text[code_index : code_index + 3] == "```":
                block = Block(
                    text="```",
                    type=BlockType.CODE_BLOCK,
                    start_index=code_index,
                    end_index=code_index + 3,
                )
            elif (
                code_index + 1 < len(text)
                and text[code_index + 1] != "`"
                and not (
                    code_index > 0
                    and text[code_index - 1] == text[code_index + 1]
                    and text[code_index - 1] in ("'", '"')
                )
            ):
                block = Block(
                    text="`",
                    type=BlockType.CODE_INLINE,
                    start_index=code_index,
                    end_index=code_index + 1,
                )
        return block

    def _find_closing_block(
        self, text: str, block_type: BlockType, start_index: int
    ) -> Optional[Block]:
        if block_type == BlockType.CODE_INLINE:
            index = start_index
            while index < len(text) - 1:
                match = text.find("`", index, len(text) - 1)
                if match == -1:
                    return None
                if match + 1 < len(text) and text[match + 1] == "`":
                    if match + 2 < len(text) and text[match + 2] == "`":
                        index = match + 3
                    else:
                        index = match + 2
                else:
                    return Block(
                        text="`",
                        event_type=StreamEventType.BLOCK_END,
                        type=BlockType.CODE_INLINE,
                        start_index=match,
                        end_index=match + 1,
                    )
            return None
        search_tag = BLOCKS[block_type]["end_tag"]
        closing_index = text.find(search_tag, start_index)
        if closing_index > -1:
            return Block(
                text=search_tag,
                event_type=StreamEventType.BLOCK_END,
                type=block_type,
                start_index=closing_index,
                end_index=closing_index + len(search_tag),
            )
        return None

    def _emit_event(
        self,
        node_id: str,
        text: str,
        event_type: StreamEventType,
        block: Optional[BlockType] = None,
        is_tool_result: bool = False,
    ) -> None:
        if event_type == StreamEventType.TEXT_CHUNK and len(text) == 0:
            return
        event = StreamEvent(
            type=event_type,
            block=block,
            text=text,
            node_id=node_id,
            is_tool_result=is_tool_result,
        )
        for sub in self._event_subscribers:
            try:
                sub(event)
            except Exception:
                pass
