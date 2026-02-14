from typing import Any, Callable, Dict, List, Optional

from ..logging_config import get_logger
from ..state import StatusEvent, StatusEventType, StatusHistory

logger = get_logger(__name__)
from ..task_states import TaskTree

from .handler import StatusUpdate


# Event types for stream split: task vs graph
_GRAPH_EVENT_TYPES = frozenset({
    StatusEventType.NODE_START,
    StatusEventType.NODE_COMPLETE,
    StatusEventType.NODE_FAILED,
})
_TASK_EVENT_TYPES = frozenset({
    StatusEventType.TASK_COMPLETE,
    StatusEventType.TASK_FAILED,
    StatusEventType.MILESTONE_ADVANCE,
    StatusEventType.ITERATION_INCREMENT,
    StatusEventType.STATE_UPDATE,
})


class StatusStreamHandler:
    """Processes normalized status updates; owns previous_state. Emits task-level and graph-level events.

    Stream split:
    - Task stream: task completed, task failed, milestone advanced, iteration started, ephemeral state
    - Graph stream: node started, node completed, node failed
    """

    def __init__(self, status_history: StatusHistory) -> None:
        self.status_history = status_history
        self.previous_state: Optional[Dict[str, Any]] = None
        self.current_node_name: Optional[str] = None
        self.node_start_times: Dict[str, float] = {}
        self._subscribers: List[Callable[[StatusEvent], None]] = []
        self._task_subscribers: List[Callable[[StatusEvent], None]] = []
        self._graph_subscribers: List[Callable[[StatusEvent], None]] = []

    def subscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        """Subscribe to all status events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def subscribe_task(self, callback: Callable[[StatusEvent], None]) -> None:
        """Subscribe to task stream only (task/milestone/iteration/state events)."""
        if callback not in self._task_subscribers:
            self._task_subscribers.append(callback)

    def subscribe_graph(self, callback: Callable[[StatusEvent], None]) -> None:
        """Subscribe to graph stream only (node start/complete/failed)."""
        if callback not in self._graph_subscribers:
            self._graph_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._task_subscribers:
            self._task_subscribers.remove(callback)
        if callback in self._graph_subscribers:
            self._graph_subscribers.remove(callback)

    def _append_event(self, event: StatusEvent) -> None:
        self.status_history.append(event)
        logger.debug("Status event: type=%s node=%s", event.type.value, event.node_name)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Subscriber callback failed: %s", e)
        if event.type in _TASK_EVENT_TYPES:
            for cb in self._task_subscribers:
                try:
                    cb(event)
                except Exception as e:
                    logger.warning("Task subscriber callback failed: %s", e)
        if event.type in _GRAPH_EVENT_TYPES:
            for cb in self._graph_subscribers:
                try:
                    cb(event)
                except Exception as e:
                    logger.warning("Graph subscriber callback failed: %s", e)

    def process_status_update(self, update: StatusUpdate) -> None:
        """Process a normalized status update. Merges into internal previous_state.

        Uses node_name from StatusUpdate (the chunk key) - nodes do not return current_node.
        When we receive StatusUpdate(node_name=N, ...), N just ran. Emit: previous complete,
        N start, process update, N complete.
        """
        node_name = update.node_name
        state_update = update.state_update
        if self.previous_state:
            current_state = {**self.previous_state, **state_update}
        else:
            current_state = state_update.copy()

        # Node transitions: use node_name from StatusUpdate, not state_update.get("current_node")
        # When we get N's chunk, N just ran. Previous node's complete was already emitted.
        if node_name != self.current_node_name:
            self._emit_node_start(node_name, current_state)
            self.current_node_name = node_name

        # Detect task status changes
        self._detect_task_changes(current_state)

        # Detect milestone changes
        self._detect_milestone_changes(current_state)

        # Detect iteration changes
        self._detect_iteration_changes(current_state)

        # Detect ephemeral state updates (gap_analysis, implementation_result, etc.)
        self._detect_ephemeral_state_changes(state_update, current_state)

        # N just ran - emit N complete
        self._emit_node_complete(node_name, current_state)

        # Update previous state
        self.previous_state = current_state.copy()
    
    def _emit_node_start(self, node_name: str, state: Dict[str, Any]) -> None:
        """Emit node start event.
        
        Args:
            node_name: Name of the node
            state: Current state
        """
        import time
        self.node_start_times[node_name] = time.time()

        event = StatusEvent(
            type=StatusEventType.NODE_START,
            node_name=node_name,
            summary="Starting",
            data={
                "task_id": state.get("current_task_id"),
            }
        )
        self._append_event(event)

    def _emit_node_complete(self, node_name: str, state: Dict[str, Any]) -> None:
        """Emit node complete event.
        
        Args:
            node_name: Name of the node
            state: Current state
        """
        error = state.get("error")
        if error:
            event_type = StatusEventType.NODE_FAILED
            summary = error
            data = {
                "task_id": state.get("current_task_id"),
                "error": error,
            }
        else:
            event_type = StatusEventType.NODE_COMPLETE
            summary = "Complete"
            data = {
                "task_id": state.get("current_task_id"),
            }

        event = StatusEvent(
            type=event_type,
            node_name=node_name,
            summary=summary,
            data=data
        )
        self._append_event(event)

    def _detect_task_changes(self, current_state: Dict[str, Any]) -> None:
        """Detect task status changes.
        
        Args:
            current_state: Current workflow state
        """
        if not self.previous_state:
            return
        
        prev_tasks = self.previous_state.get("tasks", {})
        curr_tasks = current_state.get("tasks", {})
        prev_completed = set(self.previous_state.get("completed_task_ids", []))
        curr_completed = set(current_state.get("completed_task_ids", []))
        prev_failed = set(self.previous_state.get("failed_task_ids", []))
        curr_failed = set(current_state.get("failed_task_ids", []))
        
        # Detect newly completed tasks
        newly_completed = curr_completed - prev_completed
        for task_id in newly_completed:
            task_tree = TaskTree.from_dict(curr_tasks)
            task = task_tree.tasks.get(task_id)
            if task:
                summary = f"Task {task_id} completed: {task.description[:60]}"
                event = StatusEvent(
                    type=StatusEventType.TASK_COMPLETE,
                    node_name="mark_complete",
                    summary=summary,
                    data={
                        "task_id": task_id,
                        "description": task.description,
                    }
                )
                self._append_event(event)

        # Detect newly failed tasks
        newly_failed = curr_failed - prev_failed
        for task_id in newly_failed:
            task_tree = TaskTree.from_dict(curr_tasks)
            task = task_tree.tasks.get(task_id)
            error = current_state.get("error", "Unknown error")
            if task:
                summary = f"Task {task_id} failed: {error[:60]}"
                event = StatusEvent(
                    type=StatusEventType.TASK_FAILED,
                    node_name="mark_failed",
                    summary=summary,
                    data={
                        "task_id": task_id,
                        "description": task.description,
                        "error": error,
                    }
                )
                self._append_event(event)

    def _detect_milestone_changes(self, current_state: Dict[str, Any]) -> None:
        """Detect milestone changes.
        
        Args:
            current_state: Current workflow state
        """
        if not self.previous_state:
            return
        
        prev_active = self.previous_state.get("active_milestone_id")
        curr_active = current_state.get("active_milestone_id")
        
        # Detect milestone advance
        if prev_active and curr_active and prev_active != curr_active:
            milestones = current_state.get("milestones", {})
            prev_milestone = milestones.get(prev_active, {})
            curr_milestone = milestones.get(curr_active, {})
            
            summary = f"Advanced from {prev_active} to {curr_active}"
            event = StatusEvent(
                type=StatusEventType.MILESTONE_ADVANCE,
                node_name="advance_milestone",
                summary=summary,
                data={
                    "previous_milestone_id": prev_active,
                    "current_milestone_id": curr_active,
                    "previous_description": prev_milestone.get("description", ""),
                    "current_description": curr_milestone.get("description", ""),
                }
            )
            self._append_event(event)

    def _detect_iteration_changes(self, current_state: Dict[str, Any]) -> None:
        """Detect iteration changes.
        
        Args:
            current_state: Current workflow state
        """
        if not self.previous_state:
            return
        
        prev_iteration = self.previous_state.get("iteration", 0)
        curr_iteration = current_state.get("iteration", 0)
        
        if curr_iteration > prev_iteration:
            tasks_created = current_state.get("tasks_created_this_iteration", 0)
            summary = f"Iteration {curr_iteration} started"
            if tasks_created > 0:
                summary += f" - {tasks_created} tasks created"
            
            event = StatusEvent(
                type=StatusEventType.ITERATION_INCREMENT,
                node_name="increment_iteration",
                summary=summary,
                data={
                    "iteration": curr_iteration,
                    "tasks_created": tasks_created,
                }
            )
            self._append_event(event)

    def _detect_ephemeral_state_changes(
        self,
        state_update: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> None:
        """Detect ephemeral state changes (gap_analysis, implementation_result, etc.).
        
        Args:
            state_update: State update dictionary
            current_state: Current full state
        """
        # Check for new ephemeral state updates
        ephemeral_fields = [
            "current_gap_analysis",
            "current_implementation_plan",
            "current_implementation_result",
            "current_validation_result",
            "current_qa_result",
            "last_assessment",
        ]
        
        for field in ephemeral_fields:
            if field in state_update:
                # Ephemeral state was updated - this is handled by node complete events
                # but we can emit a state update event for tracking
                if self.current_node_name:
                    # Create a summary based on the field
                    summary = self._create_ephemeral_summary(field, state_update[field], current_state)
                    if summary:
                        event = StatusEvent(
                            type=StatusEventType.STATE_UPDATE,
                            node_name=self.current_node_name,
                            summary=summary,
                            data={
                                "field": field,
                                "task_id": current_state.get("current_task_id"),
                            }
                        )
                        self._append_event(event)

    def _create_ephemeral_summary(
        self,
        field: str,
        value: Any,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """Create a summary for ephemeral state update.
        
        Args:
            field: Name of the ephemeral field
            value: Value of the field
            state: Current state
            
        Returns:
            Summary string or None if no summary needed
        """
        if not value:
            return None
        
        if field == "current_gap_analysis":
            from ..task_states import GapAnalysis
            gap = GapAnalysis.from_dict(value)
            if gap.gap_exists:
                return f"Gap analysis: {gap.gap_description[:60]}"
            else:
                return "No gap found - task already satisfied"
        
        elif field == "current_implementation_result":
            from ..task_states import ImplementationResult
            result = ImplementationResult.from_dict(value)
            if result.success:
                files_count = len(result.files_modified)
                return f"Implementation: Modified {files_count} file(s)"
            else:
                return f"Implementation failed: {result.issues_noticed[0] if result.issues_noticed else 'Unknown error'}"
        
        elif field == "current_validation_result":
            from ..task_states import ValidationResult
            result = ValidationResult.from_dict(value)
            if result.validation_passed:
                return f"Validation: {len(result.files_verified)} files verified"
            else:
                return f"Validation failed: {len(result.files_missing)} files missing"
        
        elif field == "current_qa_result":
            from ..task_states import QAResult
            result = QAResult.from_dict(value)
            if result.passed:
                return "QA: Requirements satisfied"
            else:
                return f"QA failed: {result.failure_type}"
        
        return None
    
    def finalize(self) -> None:
        """Finalize any remaining events.
        
        Should be called when stream ends to ensure all nodes are marked complete.
        """
        if self.current_node_name and self.previous_state:
            self._emit_node_complete(self.current_node_name, self.previous_state)