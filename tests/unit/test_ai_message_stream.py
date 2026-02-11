"""Unit tests for AIMessageStreamHandler.handle_chunk.

Validates streaming chunk parsing: tags (<think>, <tool_call>), code blocks (inline `, fenced ```),
and buffering of incomplete tags across chunk boundaries.

The handler BREAKS UP incoming chunks on block boundaries; it does NOT join adjacent text
chunks. So ["hello", " ", "world"] produces three separate TEXT_CHUNK events. The only
exception is partial-match buffering: when a chunk ends with a possible tag prefix (e.g.
"<" or "<thi"), that suffix is buffered and the next chunk is processed as (buffer + chunk),
so the next delivery may emit multiple events from that combined string.
"""

import pytest
from agents.state.ai_message_stream import (
    AIMessageStreamHandler,
    StreamEventType,
    BlockType,
    StreamEvent,
)


def run_chunks(chunks):
    """Feed chunks to handler and return list of emitted events (type, block, text)."""
    handler = AIMessageStreamHandler()
    events = []

    def collect(e: StreamEvent):
        events.append((e.type, e.block, e.text))

    handler._event_subscribers.append(collect)
    for chunk in chunks:
        handler.handle_chunk(chunk)
    return events


# Test cases: (id, chunks, expected_events)
# expected_events: list of (StreamEventType, BlockType | None, text: str)
HANDLE_CHUNK_CASES = [
    # --- Plain text ---
    (
        "plain_text_single_chunk",
        ["hello world"],
        [(StreamEventType.TEXT_CHUNK, None, "hello world")],
    ),
    (
        "plain_text_multiple_chunks",
        ["hello ", "world"],
        [
            (StreamEventType.TEXT_CHUNK, None, "hello "),
            (StreamEventType.TEXT_CHUNK, None, "world"),
        ],
    ),
    (
        "three_separate_text_chunks_no_joining",
        ["hello", " ", "world"],
        [
            (StreamEventType.TEXT_CHUNK, None, "hello"),
            (StreamEventType.TEXT_CHUNK, None, " "),
            (StreamEventType.TEXT_CHUNK, None, "world"),
        ],
    ),
    (
        "empty_chunk_emits_nothing",
        [""],
        [],
    ),
    (
        "multiple_empty_chunks",
        ["", "", "x", ""],
        [(StreamEventType.TEXT_CHUNK, None, "x")],
    ),
    # --- Think block ---
    (
        "single_chunk_complete_think",
        ["pre <think>inner</think> post"],
        [
            (StreamEventType.TEXT_CHUNK, None, "pre "),
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "inner"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " post"),
        ],
    ),
    (
        "multichunk_think_content",
        ["pre ", "<think>", "think ", "content", "</think>", " post"],
        [
            (StreamEventType.TEXT_CHUNK, None, "pre "),
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "think "),
            (StreamEventType.TEXT_CHUNK, None, "content"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " post"),
        ],
    ),
    (
        "think_opening_tag_split_across_chunks",
        ["pre <thi", "nk>body</think> after"],
        [
            (StreamEventType.TEXT_CHUNK, None, "pre "),
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "body"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " after"),
        ],
    ),
    (
        "think_opening_split_then_content_only",
        ["hello <th", "ink> world"],
        [
            (StreamEventType.TEXT_CHUNK, None, "hello "),
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, " world"),
        ],
    ),
    (
        "think_closing_tag_split_across_chunks",
        ["<think>body<", "/think> after"],
        [
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "body"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " after"),
        ],
    ),
    (
        "partial_angle_bracket_false_positive",
        ["Did you know 3 <", " 5?"],
        [
            (StreamEventType.TEXT_CHUNK, None, "Did you know 3 "),
            (StreamEventType.TEXT_CHUNK, None, "< 5?"),
        ],
    ),
    (
        "partial_tool_call_start_then_not_tag",
        ["a <tool_", "xyz"],
        [
            (StreamEventType.TEXT_CHUNK, None, "a "),
            (StreamEventType.TEXT_CHUNK, None, "<tool_xyz"),
        ],
    ),
    # --- Tool call block ---
    (
        "single_chunk_complete_tool_call",
        ["x <tool_call>args</tool_call> y"],
        [
            (StreamEventType.TEXT_CHUNK, None, "x "),
            (StreamEventType.BLOCK_START, BlockType.TOOL_CALL, "<tool_call>"),
            (StreamEventType.TEXT_CHUNK, None, "args"),
            (StreamEventType.BLOCK_END, BlockType.TOOL_CALL, "</tool_call>"),
            (StreamEventType.TEXT_CHUNK, None, " y"),
        ],
    ),
    (
        "tool_call_opening_split",
        ["a <tool_", "call>inner</tool_call> b"],
        [
            (StreamEventType.TEXT_CHUNK, None, "a "),
            (StreamEventType.BLOCK_START, BlockType.TOOL_CALL, "<tool_call>"),
            (StreamEventType.TEXT_CHUNK, None, "inner"),
            (StreamEventType.BLOCK_END, BlockType.TOOL_CALL, "</tool_call>"),
            (StreamEventType.TEXT_CHUNK, None, " b"),
        ],
    ),
    # --- Inline code ---
    (
        "inline_code_single_chunk",
        ["see `code` here"],
        [
            (StreamEventType.TEXT_CHUNK, None, "see "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, "code"),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " here"),
        ],
    ),
    (
        "inline_code_quoted",
        ["there's a problem with the '`' character"],
        [
            (StreamEventType.TEXT_CHUNK, None, "there's a problem with the '`' character"),
        ],
    ),
    (
        "fenced_block_inside_inline_code",
        ["it's a ` ``` code ``` block ` inside a code block"],
        [
            (StreamEventType.TEXT_CHUNK, None, "it's a "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " ``` code ``` block "),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " inside a code block"),
        ],
    ),
    (
        "fenced_block_inside_inline_code_across_chunks",
        ["it's a ` ``` code `", "`` block ` inside a code block"],
        [
            (StreamEventType.TEXT_CHUNK, None, "it's a "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " ``` code "),
            (StreamEventType.TEXT_CHUNK, None, "``` block "),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " inside a code block"),
        ],
    ),
    (
        "inline_code_single_chunk_with_quotes",
        ["see `\"code\"` here"],
        [
            (StreamEventType.TEXT_CHUNK, None, "see "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, "\"code\""),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " here"),
        ],
    ),
    (
        "inline_code_split_across_chunks",
        ["see ", "`", "code", "`", " here"],
        [
            (StreamEventType.TEXT_CHUNK, None, "see "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, "code"),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " here"),
        ],
    ),
    (
        "double_backtick_in_middle_inline_then_inline",
        ["a `` b"],
        [
            (StreamEventType.TEXT_CHUNK, None, "a `` b")
        ],
    ),
    (
        "trailing_single_backtick_buffered",
        ["hello `"],
        [(StreamEventType.TEXT_CHUNK, None, "hello ")],
    ),
    (
        "trailing_single_backtick_then_next_chunk_completes_inline",
        ["hello `", "x` world"],
        [
            (StreamEventType.TEXT_CHUNK, None, "hello "),
            (StreamEventType.BLOCK_START, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, "x"),
            (StreamEventType.BLOCK_END, BlockType.CODE_INLINE, "`"),
            (StreamEventType.TEXT_CHUNK, None, " world"),
        ],
    ),
    (
        "trailing_double_backtick_buffered",
        ["hello ``"],
        [(StreamEventType.TEXT_CHUNK, None, "hello ")],
    ),
    (
        "trailing_double_backtick_then_fenced_in_next_chunk",
        ["code ``", "`\nbody\n```"],
        [
            (StreamEventType.TEXT_CHUNK, None, "code "),
            (StreamEventType.BLOCK_START, BlockType.CODE_BLOCK, "```"),
            (StreamEventType.TEXT_CHUNK, None, "\nbody\n"),
            (StreamEventType.BLOCK_END, BlockType.CODE_BLOCK, "```"),
        ],
    ),
    # --- Fenced code block ---
    (
        "fenced_code_single_chunk",
        ["text ```\ncode\n``` more"],
        [
            (StreamEventType.TEXT_CHUNK, None, "text "),
            (StreamEventType.BLOCK_START, BlockType.CODE_BLOCK, "```"),
            (StreamEventType.TEXT_CHUNK, None, "\ncode\n"),
            (StreamEventType.BLOCK_END, BlockType.CODE_BLOCK, "```"),
            (StreamEventType.TEXT_CHUNK, None, " more"),
        ],
    ),
    (
        "fenced_opening_split",
        ["x ``", "`\nbody\n``` y"],
        [
            (StreamEventType.TEXT_CHUNK, None, "x "),
            (StreamEventType.BLOCK_START, BlockType.CODE_BLOCK, "```"),
            (StreamEventType.TEXT_CHUNK, None, "\nbody\n"),
            (StreamEventType.BLOCK_END, BlockType.CODE_BLOCK, "```"),
            (StreamEventType.TEXT_CHUNK, None, " y"),
        ],
    ),
    # --- Mixed ---
    (
        "text_then_think_then_text",
        ["Hi ", "<think>ok</think>", " bye"],
        [
            (StreamEventType.TEXT_CHUNK, None, "Hi "),
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "ok"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " bye"),
        ],
    ),
    (
        "think_before_backtick_not_nested",
        ["<think>inner ` not code</think> post"],
        [
            (StreamEventType.BLOCK_START, BlockType.THINK, "<think>"),
            (StreamEventType.TEXT_CHUNK, None, "inner ` not code"),
            (StreamEventType.BLOCK_END, BlockType.THINK, "</think>"),
            (StreamEventType.TEXT_CHUNK, None, " post"),
        ],
    ),
]


@pytest.mark.parametrize("case_id,chunks,expected", HANDLE_CHUNK_CASES, ids=[c[0] for c in HANDLE_CHUNK_CASES])
def test_handle_chunk(case_id, chunks, expected):
    """Parameterized: feed chunks and assert emitted events match expected (type, block, text)."""
    actual = run_chunks(chunks)
    assert len(actual) == len(expected), (
        f"event count: got {len(actual)}, expected {len(expected)}\n"
        f"  actual:   {actual}\n  expected: {expected}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a[0] == e[0], f"event[{i}].type: got {a}, expected {e}"
        assert a[1] == e[1], f"event[{i}].block: got {a}, expected {e}"
        assert a[2] == e[2], f"event[{i}].text: got {a}, expected {e}"
