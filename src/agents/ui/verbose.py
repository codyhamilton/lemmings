"""Verbose terminal UI for workflow output.

This module provides a terminal-based verbose UI that renders messages
from MessageHistory with detailed formatting.
"""

import sys

from .base import UIBase
from .message_history import MessageHistory, Message, MessageType
from .status_history import StatusHistory, StatusEvent, StatusEventType
from .state_manager import UIStateManager
from .stream_handlers import AIMessageStreamHandler, StreamTransitionEvent, StreamTransitionEventType
from ..task_states import WorkflowState, TaskTree, TaskStatus, AssessmentResult

TERMINAL_GREY = "\033[90m"
TERMINAL_RESET = "\033[0m"
TERMINAL_BLUE = "\033[94m"

class VerboseUI(UIBase):
    """Terminal-based verbose UI that renders messages with detailed formatting.
    
    Subscribes to MessageHistory and renders messages as they arrive.
    Provides detailed formatting for different message types and workflow summaries.
    """
    
    def __init__(
        self,
        message_history: MessageHistory,
        status_history: StatusHistory,
        ui_state_manager: UIStateManager,
        stream_handler: AIMessageStreamHandler,
        show_thinking: bool = False
    ):
        """Initialize verbose UI.
        
        Args:
            message_history: MessageHistory to subscribe to
            status_history: Optional StatusHistory to subscribe to
            ui_state_manager: Optional UIStateManager to subscribe to
            show_thinking: Whether to show thinking blocks (default: False)
            stream_handler: AIMessageStreamHandler to subscribe to for transition events
        """
        self.ui_state_manager = ui_state_manager
        self.show_thinking = show_thinking
        self.is_thinking = False
        
        message_history.subscribe(self.on_message)
        message_history.subscribe_completion(self.on_message_complete)
        status_history.subscribe(self.on_status_event)
        stream_handler.subscribe_transitions(self.on_transition_event)
        stream_handler.subscribe_text_chunks(self.on_message_chunk)
    
    def on_message_chunk(self, message_chunk: str) -> None:
        """Handle message chunk from stream handler.
        
        Args:
            message_chunk: Message chunk that was received
        """
        if self.is_thinking and not self.show_thinking:
            pass
        else:
            print(message_chunk, end="", flush=True)
    
    def on_message(self, message: Message) -> None:
        """Handle message updates from MessageHistory.
        
        Args:
            message: Message that was created/updated
        """
        # Render message based on type
        if message.type == MessageType.TOOL_RESULT:
            self._render_tool_result(message)
        elif message.type == MessageType.USER:
            self._render_user_message(message)
        elif message.type == MessageType.SYSTEM:
            self._render_system_message(message)
    
    def render_final_summary(self, state: WorkflowState) -> None:
        """Render final workflow summary.
        
        Args:
            state: Final workflow state
        """
        self._print_final_summary(state)
    
    def on_status_event(self, event: StatusEvent) -> None:
        """Handle status events from StatusHistory.
        
        Args:
            event: Status event that was created
        """
        # Render status events based on type
        match event.type:
            case StatusEventType.NODE_START:
                print("=" * 70)
                print(f"🤖 {event.node_name.upper()}: Starting task {event.data.get('task_id')}")
            case StatusEventType.NODE_FAILED:
                print(f"❌ {event.node_name.upper()}: Task {event.data.get('task_id')} failed: {event.data.get('error', 'Unknown error')}")
            case StatusEventType.TASK_COMPLETE:
                print(f"✅ {event.node_name.upper()}: Task {event.data.get('task_id')} completed successfully")
            case StatusEventType.TASK_FAILED:
                print(f"❌ {event.node_name.upper()}: Task {event.data.get('task_id')} failed: {event.data.get('error', 'Unknown error')}")
            case StatusEventType.MILESTONE_ADVANCE:
                print(f"🔄 {event.node_name.upper()}: Milestone {event.data.get('current_milestone_id')} advanced from {event.data.get('previous_milestone_id')}")
            case StatusEventType.ITERATION_INCREMENT:
                print(f"🔄 {event.node_name.upper()}: Iteration {event.data.get('iteration')} started")
            case StatusEventType.STATE_UPDATE:
                print(f"🔍 {event.node_name.upper()}: {event.summary}")
    
    # TODO determine if we actually need this
    def cleanup(self) -> None:
        """Cleanup UI resources."""
        # Unsubscribe from message history
        self.message_history.unsubscribe(self.on_message)
        self.message_history.unsubscribe(self.on_message_complete)
        # Unsubscribe from status history if subscribed
        if self.status_history:
            self.status_history.unsubscribe(self.on_status_event)
        # Unsubscribe from UI state manager if subscribed
        if self.ui_state_manager:
            self.ui_state_manager.unsubscribe(self.on_ui_state_update)
        # Unsubscribe from stream handler if subscribed
        if self.stream_handler:
            self.stream_handler.unsubscribe_transitions(self.on_transition_event)
        # Ensure output is flushed
        sys.stdout.flush()
    
    # =============================================================================
    # Message Rendering
    # =============================================================================
    
    def on_transition_event(self, event: StreamTransitionEvent) -> None:
        """Handle transition events from stream handler.
        
        This is called by AIMessageStreamHandler when transitions are detected
        in the stream (think_start, think_end, tool_call_start, tool_call_end, text_chunk).
        
        Args:
            event: StreamTransitionEvent containing event type and text
        """
        match event.event_type:
            case StreamTransitionEventType.THINK_START:
                self.is_thinking = True
                print(f"{TERMINAL_GREY}💭 Thinking... ", end="", flush=True)
            case StreamTransitionEventType.THINK_END:
                self.is_thinking = False
                print(f"{TERMINAL_RESET}", end="", flush=True)
            case StreamTransitionEventType.TOOL_CALL_START:
                print(f"{TERMINAL_BLUE}🔧 Call: ", end="", flush=True)
            case StreamTransitionEventType.TOOL_CALL_END:
                print(f"{TERMINAL_RESET}", flush=True)
    
    def _render_tool_result(self, message: Message) -> None:
        """Render tool result.
        
        Args:
            message: Tool result message to render
        """
        print("="*70)
        print(f"✓ Tool result: {message.content}", flush=True)
    
    def _render_user_message(self, message: Message) -> None:
        """Render user message.
        
        Args:
            message: User message to render
        """
        print("="*70)
        print(f"👤 User: {message.content}", flush=True)
    
    def _render_system_message(self, message: Message) -> None:
        """Render system message.
        
        Args:
            message: System message to render
        """
        print("="*70)
        print(f"⚙️  System: {message.content}", flush=True)
    
    # =============================================================================
    # Summary Methods (moved from main.py)
    # =============================================================================
    
    def _print_milestone_summary(self, milestones: dict, milestone_order: list) -> None:
        """Print a summary of milestones.
        
        Args:
            milestones: Dictionary of milestones
            milestone_order: Ordered list of milestone IDs
        """
        if not milestones:
            return
        
        print("\n" + "─" * 70)
        print("🎯 MILESTONES")
        print("─" * 70)
        
        for milestone_id in milestone_order:
            milestone_dict = milestones.get(milestone_id, {})
            status = milestone_dict.get("status", "pending")
            description = milestone_dict.get("description", milestone_id)
            
            status_icon = {
                "pending": "⏸️",
                "active": "🔄",
                "complete": "✅",
            }.get(status, "❓")
            
            print(f"{status_icon} {milestone_id}: {description}")
    
    def _print_task_summary(self, tasks: dict) -> None:
        """Print a summary of tasks.
        
        Args:
            tasks: Dictionary of tasks
        """
        if not tasks:
            return
        
        task_tree = TaskTree.from_dict(tasks)
        stats = task_tree.get_statistics()
        
        print("\n" + "─" * 70)
        print("📋 TASKS")
        print("─" * 70)
        print(f"Total: {stats['total']} | "
              f"✅ Complete: {stats['complete']} | "
              f"🔄 Ready: {stats['ready']} | "
              f"⏸️  Pending: {stats['pending']} | "
              f"❌ Failed: {stats['failed']} | "
              f"🚫 Blocked: {stats['blocked']}")
        
        # Show some example tasks
        all_tasks = list(task_tree.tasks.values())
        for task in all_tasks[:10]:
            status_icon = {
                TaskStatus.READY: "🟢",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETE: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.PENDING: "⏸️",
            }.get(task.status, "❓")
            
            print(f"\n{status_icon} {task.id}: {task.description[:60]}...")
            if task.measurable_outcome:
                print(f"   Outcome: {task.measurable_outcome[:50]}...")
        
        if len(all_tasks) > 10:
            print(f"\n... and {len(all_tasks) - 10} more tasks")
    
    # TODO Stats should be calculated, but summary can be an agent output
    def _print_final_summary(self, state: WorkflowState) -> None:
        """Print the final workflow summary.
        
        Args:
            state: Final workflow state
        """
        status = state.get("status", "unknown")
        iteration = state.get("iteration", 0)
        remit = state.get("remit", "")
        
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        
        # Status
        if status == "complete":
            print("✅ Status: COMPLETED SUCCESSFULLY")
        elif status == "failed":
            print("❌ Status: FAILED")
            if state.get("error"):
                print(f"   Error: {state['error']}")
        else:
            print(f"📌 Status: {status}")
        
        if remit:
            print(f"\nRemit: {remit[:200]}...")
        
        if iteration > 0:
            print(f"   Expansion iterations: {iteration}")
        
        # Milestones summary
        milestones = state.get("milestones", {})
        milestone_order = state.get("milestone_order", [])
        if milestones:
            self._print_milestone_summary(milestones, milestone_order)
        
        # Tasks summary
        tasks = state.get("tasks", {})
        if tasks:
            self._print_task_summary(tasks)
        
        # Assessment summary
        last_assessment = state.get("last_assessment")
        if last_assessment:
            assessment = AssessmentResult.from_dict(last_assessment)
            print("\n" + "─" * 70)
            print("📊 ASSESSMENT")
            print("─" * 70)
            print(f"Overall Complete: {'✅' if assessment.is_complete else '❌'}")
            if assessment.uncovered_gaps:
                print(f"\n⚠️  {len(assessment.uncovered_gaps)} uncovered gaps:")
                for gap in assessment.uncovered_gaps[:5]:
                    print(f"   - {gap[:80]}...")
            if assessment.assessment_notes:
                print(f"\nNotes: {assessment.assessment_notes[:200]}...")
        
        print("=" * 70)
    
    # TODO Automatically generate ASCII workflow diagram from the graph
    def print_workflow_start(self, user_request: str, repo_root: str) -> None:
        """Print workflow start header.
        
        Args:
            user_request: User's request
            repo_root: Repository root path
        """
        print("\n" + "=" * 70)
        print("🤖 ITERATIVE TASK TREE WORKFLOW")
        print("=" * 70)
        print(f"Repository: {repo_root}")
        print()
        print("Request:")
        print(f"  {user_request}")
        # ASCII diagram of the worfklow stages, showing executor loop
        print("\nWorkflow stages:")
        print("  1. 📥 Intake - Understand request and create milestones")
        print("  2. 🌱 Expander - Discover tasks via IF X THEN Y reasoning")
        print("  3. 🎯 Prioritizer - Select next ready task")
        print("  4. 🔍 Researcher - Analyze gap between current and desired state")
        print("  5. 📝 Planner - Create implementation plan")
        print("  6. 💻 Implementor - Execute changes")
        print("  7. ✅ Validator - Verify files exist")
        print("  8. 🔍 QA - Verify requirements satisfied")
        print("  9. 📊 Assessor - Check completion and gaps")
        print()
        
        print("=" * 70)
        sys.stdout.flush()
