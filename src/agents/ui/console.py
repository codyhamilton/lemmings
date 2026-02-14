"""Console UI: stateless pipe that merges message and status streams to stdout."""

import json
import sys

from ..stream.messages import AIMessageStreamHandler, StreamEvent, StreamEventType, BlockType
from ..stream.status import StatusStreamHandler
from ..state import StatusEvent, StatusEventType
from ..task_states import WorkflowState, TaskTree, TaskStatus, AssessmentResult

TERMINAL_GREY = "\033[90m"
TERMINAL_RESET = "\033[0m"
TERMINAL_BLUE = "\033[94m"


class ConsoleUI:
    """Stateless pipe: subscribes to message and status handlers, prints merged stream to stdout."""

    def __init__(
        self,
        message_handler: AIMessageStreamHandler,
        status_handler: StatusStreamHandler,
        show_thinking: bool = True,
    ) -> None:
        self._message_handler = message_handler
        self._status_handler = status_handler
        self.show_thinking = show_thinking
        self._is_thinking = False
        self._in_tool_call = False
        self._tool_call_buffer: list[str] = []

        message_handler.subscribe(self._on_stream_event)
        status_handler.subscribe(self._on_status_event)

    def _on_stream_event(self, event: StreamEvent) -> None:
        if event.type == StreamEventType.TEXT_CHUNK:
            if self._in_tool_call:
                self._tool_call_buffer.append(event.text)
                return
            if self._is_thinking and not self.show_thinking:
                return
            print(event.text, end="", flush=True)
            return
        if event.type == StreamEventType.BLOCK_START and event.block == BlockType.THINK:
            self._is_thinking = True
            if self.show_thinking:
                # Output block tag so stream structure is visible (e.g. <think>)
                print(f"{TERMINAL_GREY}{event.text} ", end="", flush=True)
                print("💭 ", end="", flush=True)
            return
        if event.type == StreamEventType.BLOCK_END and event.block == BlockType.THINK:
            self._is_thinking = False
            if self.show_thinking:
                # Output closing tag so block boundary is visible (e.g. </think>)
                print(f"{event.text}{TERMINAL_RESET}", end="", flush=True)
            return
        if event.type == StreamEventType.BLOCK_START and event.block == BlockType.TOOL_CALL:
            self._in_tool_call = True
            self._tool_call_buffer = []
            # Output block tag so stream structure is visible (e.g. <tool_call>)
            print(f"{TERMINAL_BLUE}{event.text} ", end="", flush=True)
            return
        if event.type == StreamEventType.BLOCK_END and event.block == BlockType.TOOL_CALL:
            raw = "".join(self._tool_call_buffer)
            self._in_tool_call = False
            self._tool_call_buffer = []
            try:
                parsed = json.loads(raw)
                print(json.dumps(parsed, indent=2), end="", flush=True)
            except (json.JSONDecodeError, TypeError):
                print(raw, end="", flush=True)
            # Output closing tag (e.g. </tool_call>)
            print(f" {event.text}{TERMINAL_RESET}", flush=True)
            return
        if event.type in (StreamEventType.BLOCK_START, StreamEventType.BLOCK_END):
            if self._is_thinking and not self.show_thinking:
                return
            print(event.text, end="", flush=True)

    def _on_status_event(self, event: StatusEvent) -> None:
        match event.type:
            case StatusEventType.NODE_START:
                print("=" * 70)
                print(f"🤖 {event.node_name or 'node'}: {event.summary or 'Starting'}")
            case StatusEventType.NODE_COMPLETE:
                print()
                print(f"✅ {event.node_name or 'node'}: {event.summary or 'Complete'}")
            case StatusEventType.NODE_FAILED:
                print()
                print(f"❌ {event.node_name or 'node'}: {event.data.get('error', event.summary or 'Failed')}")
            case StatusEventType.TASK_COMPLETE:
                print(f"✅ Task {event.data.get('task_id', '')}: {event.summary or 'completed'}")
            case StatusEventType.TASK_FAILED:
                print(f"❌ Task {event.data.get('task_id', '')}: {event.data.get('error', event.summary or 'failed')}")
            case StatusEventType.MILESTONE_ADVANCE:
                print(f"🔄 {event.summary or 'Milestone advanced'}")
            case StatusEventType.ITERATION_INCREMENT:
                print(f"🔄 {event.summary or 'Iteration started'}")
            case StatusEventType.STATE_UPDATE:
                if event.summary:
                    print(f"🔍 {event.summary}")

    def print_workflow_start(self, user_request: str, repo_root: str) -> None:
        print("\n" + "=" * 70)
        print("🤖 ITERATIVE TASK TREE WORKFLOW")
        print("=" * 70)
        print(f"Repository: {repo_root}")
        print()
        print("Request:")
        print(f"  {user_request}")
        print("\nWorkflow stages: Intake → Expander → Prioritizer → Researcher → Planner → Implementor → Validator → QA → Assessor")
        print("=" * 70)
        sys.stdout.flush()

    def render_final_summary(self, state: WorkflowState) -> None:
        work_report = state.get("work_report")
        status = state.get("status", "unknown")
        iteration = state.get("iteration", 0)
        remit = state.get("remit", "")

        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        if work_report:
            print("\n" + work_report)
            print()
        if status == "complete":
            print("✅ Status: COMPLETED SUCCESSFULLY")
        elif status == "failed":
            print("❌ Status: FAILED")
            if state.get("error"):
                print(f"   Error: {state['error']}")
        else:
            print(f"📌 Status: {status}")
        if remit:
            print(f"\nRemit: {remit[:200]}{'...' if len(remit) > 200 else ''}")
        if iteration > 0:
            print(f"   Expansion iterations: {iteration}")

        from ..task_states import get_milestones_list, get_active_milestone_index
        milestones_list = get_milestones_list(state)
        active_index = get_active_milestone_index(state)
        if milestones_list:
            print("\n" + "─" * 70)
            print("🎯 MILESTONES")
            print("─" * 70)
            for i, m in enumerate(milestones_list):
                if not isinstance(m, dict):
                    continue
                mid = m.get("id", "")
                st = "complete" if i < active_index else ("active" if i == active_index else "pending")
                icon = {"pending": "⏸️", "active": "🔄", "complete": "✅"}.get(st, "❓")
                print(f"{icon} {mid}: {m.get('description', mid)}")

        tasks = state.get("tasks", {})
        if tasks:
            tree = TaskTree.from_dict(tasks)
            stats = tree.get_statistics()
            print("\n" + "─" * 70)
            print("📋 TASKS")
            print("─" * 70)
            print(f"Total: {stats['total']} | ✅ Complete: {stats['complete']} | ❌ Failed: {stats['failed']} | 🚫 Blocked: {stats['blocked']}")

        last_assessment = state.get("last_assessment")
        if last_assessment:
            assessment = AssessmentResult.from_dict(last_assessment)
            print("\n" + "─" * 70)
            print("📊 ASSESSMENT")
            print("─" * 70)
            print(f"Overall Complete: {'✅' if assessment.is_complete else '❌'}")
            if assessment.uncovered_gaps:
                for gap in assessment.uncovered_gaps[:5]:
                    print(f"   - {gap[:80]}...")

        print("=" * 70)
        sys.stdout.flush()

    def cleanup(self) -> None:
        self._message_handler.unsubscribe(self._on_stream_event)
        self._status_handler.unsubscribe(self._on_status_event)
        sys.stdout.flush()
