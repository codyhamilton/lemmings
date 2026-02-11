"""Stream handlers for processing graph streams and state updates.

This module combines both AI message stream handling and status stream handling
for processing LangGraph streams and creating structured events for UI consumption.
"""

from enum import Enum
from typing import Optional, List, Callable
from dataclasses import dataclass

# Blocks to output as events
class BlockType(Enum):
    THINK = "think"
    TOOL_CALL = "tool_call"
    CODE_INLINE= "code_inline"
    CODE_BLOCK = "code_block"

BLOCKS = {
    BlockType.THINK: {
        "start_tag": "<think>",
        "end_tag": "</think>"
    },
    BlockType.TOOL_CALL: {
        "start_tag": "<tool_call>",
        "end_tag": "</tool_call>"
    },
    BlockType.CODE_INLINE: {
        "start_tag": "`",
        "end_tag": "`"
    },
    BlockType.CODE_BLOCK: {
        "start_tag": "```",
        "end_tag": "```"
    }
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
    text: str # In block events, the output text is the block, e.g. <think>...</think> to allow raw processing

class AIMessageStreamHandler:
    """Processes AIMessage streams and formats them as strings for WorkflowState.messages.
    
    Identifies blocks (thinking, tool_use) within AI message streams and emits
    events. This turns a contiguous stream of text into a series of text and
    transition events.

    Consumers can subscribe to track the flow of AI output
    """
    
    def __init__(self):
        # Blocks cannot nest. If a block is open, we treat blocks as text until the block is closed.
        self._open_blocktype: Optional[BlockType] = None
        
        # Event subscribers 
        self._event_subscribers: List[Callable[[StreamEvent], None]] = []
        
        # Buffer for incomplete tags that might span chunks
        self._buffer: str = ""

    def handle_chunk(self, chunk: str) -> None:
        """Handle a chunk of text.
        
        Args:
            chunk: Text chunk
        """
        index = 0
        text = self._buffer + chunk
        self._buffer = ""

        while index < len(text):
            block = None

            if self._open_blocktype is not None:
                block = self._find_closing_block(text, self._open_blocktype, index)
            else:
                block = self._find_next_block(text, index)

            if block is not None:
                self._emit_event(text[index:block.start_index], StreamEventType.TEXT_CHUNK)
                self._emit_event(block.text, block.event_type, block.type)
                index = block.end_index
                self._open_blocktype = block.type if block.event_type == StreamEventType.BLOCK_START else None
            else:
                partial_index = self._find_partial_index(text, index)
                if partial_index < len(text):
                    self._buffer = text[partial_index:]
                self._emit_event(text[index:partial_index], StreamEventType.TEXT_CHUNK)
                index = len(text)
    
    def _find_partial_index(self, text: str, start_index: int) -> int:
        """Find the partial index in the text.
        
        Args:
            text: Text to search
            start_index: Index to start searching from
        
        Returns: 
            Partial index if found, len(text) otherwise
        """

        if self._open_blocktype is not None:
            search_tags = [BLOCKS[self._open_blocktype]["end_tag"]]
        else:
            search_tags = [BLOCKS[BlockType.THINK]["start_tag"], BLOCKS[BlockType.TOOL_CALL]["start_tag"]]

        if self._open_blocktype not in [BlockType.THINK, BlockType.TOOL_CALL]:
            if text.endswith('``'):
                return len(text) - 2
            elif text.endswith('`'):
                return len(text) - 1

        # Start searching from the end of the text minus the length of the longest search tag
        max_search_tag_length = max(len(search_tag) for search_tag in search_tags)
        index=max(start_index, len(text) - max_search_tag_length + 1)

        for search_tag in search_tags:
            potential_closing_tag_index = text.rfind(search_tag[0], index)
            if potential_closing_tag_index != -1 and text[potential_closing_tag_index:] == search_tag[:len(text) - potential_closing_tag_index]:
                return potential_closing_tag_index
        
        return len(text)

    def _find_next_block(self, text: str, start_index: int) -> Optional[Block]:
        """Find the next block in the text.
        
        Args:
            text: Text to search
            start_index: Index to start searching from
        
        Returns:
            Block if found, None otherwise
        """
        code_index = text.find('`', start_index, len(text) - 1)
        tag_end_index = len(text) if code_index == -1 else code_index

        block = None
        for block_type in [BlockType.THINK, BlockType.TOOL_CALL]:
            found_tag_index = text.find(BLOCKS[block_type]["start_tag"], start_index, tag_end_index)
            if found_tag_index > -1 and (block is None or found_tag_index < block.start_index):
                block = Block(text=BLOCKS[block_type]["start_tag"], type=block_type, start_index=found_tag_index, end_index=found_tag_index + len(BLOCKS[block_type]["start_tag"]))
        
        if block is None and code_index > -1:
            if text[code_index:code_index + 3] == '```':
                block = Block(text="```", type=BlockType.CODE_BLOCK, start_index=code_index, end_index=code_index + 3)
            # Single backtick is a code inline block if it's not immediately quoted
            # Also ignore double backticks with nothing in between, empty code blocks
            elif text[code_index + 1] == '`' or (code_index > 0 and text[code_index - 1] == text[code_index + 1] and text[code_index - 1] in ["'", '"']):
                return None
            else:
                block = Block(text="`", type=BlockType.CODE_INLINE, start_index=code_index, end_index=code_index + 1)

        return block

    def _find_closing_block(self, text: str, block_type: BlockType, start_index: int) -> Optional[Block]:
        """Find the closing block in the text.
        
        Args:
            text: Text to search
            block_type: Block type to search for
            start_index: Index to start searching from
        """
        # If we have an inline block, we need to iterate to avoid mismatching a fenced code block
        if block_type == BlockType.CODE_INLINE:
            index = start_index
            while index < len(text) - 1:
                match = text.find('`', index, len(text) - 1)
                if match == -1:
                    return None
                # Jump past double/triple backticks
                elif text[match + 1] == '`':
                    if match + 2 < len(text) and text[match + 2] == '`':
                        index = match + 3
                    else:
                        index = match + 2
                else:
                    return Block(text="`", event_type=StreamEventType.BLOCK_END, type=BlockType.CODE_INLINE, start_index=match, end_index=match + 1)
            return None

        # For other blocks, we can just find an exact match
        search_tag = BLOCKS[block_type]["end_tag"]
        closing_index = text.find(search_tag, start_index)
        if closing_index > -1:
            return Block(text=search_tag, event_type= StreamEventType.BLOCK_END, type=block_type, start_index=closing_index, end_index=closing_index + len(search_tag))
        return None

    def _emit_event(self, text: str, event_type: StreamEventType, block: Optional[BlockType] = None) -> None:
        """Emit a stream event.
        
        Args:
            text: Text content
            type: Event type
            block: Block type
        """
        # Don't emit empty text chunks
        if event_type == StreamEventType.TEXT_CHUNK and len(text) == 0:
            return

        event = StreamEvent(type=event_type, text=text, block=block)
        for subscriber in self._event_subscribers:
            subscriber(event)