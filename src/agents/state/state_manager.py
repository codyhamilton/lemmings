"""UI State Manager for workflow state synchronization.

This module provides:
- UIStateManager: Background thread that listens to graph events via queue
- Maintains UIState (derived from WorkflowState)
- Provides subscription mechanism for UIs
- Throttles updates (capped stream)

Architecture:
- Graph execution layer emits events to UI State Manager via queue
- UI State Manager (listener thread) captures changes and updates UI state
- UI subscribes to UI state (infrequent updates) or capped stream (up to max refresh rate)
"""

import queue
import threading
import time
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..task_states import WorkflowState, get_milestones_list, get_active_milestone_id


@dataclass
class UIStateEvent:
    """Event emitted from graph to UI State Manager.
    
    Events represent state changes that need to be reflected in UI.
    """
    event_type: str  # "state_update", "node_start", "node_complete", etc.
    node_name: Optional[str] = None
    state_update: Optional[Dict[str, Any]] = None  # Partial state update dict
    full_state: Optional[WorkflowState] = None  # Full state snapshot (if available)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UIState:
    """UI state derived from WorkflowState.
    
    This is a simplified/derived state optimized for UI rendering.
    It's maintained by UIStateManager and updated from WorkflowState events.
    """
    
    def __init__(self, initial_state: Optional[WorkflowState] = None):
        """Initialize UI state from WorkflowState.
        
        Args:
            initial_state: Initial workflow state to derive from
        """
        if initial_state:
            self._update_from_workflow_state(initial_state)
        else:
            # Initialize with empty state
            self.current_node: Optional[str] = None
            self.node_statuses: Dict[str, str] = {}
            self.current_task_id: Optional[str] = None
            self.status: str = "running"
            self.error: Optional[str] = None
            self.milestones: Dict[str, Dict[str, Any]] = {}
            self.active_milestone_id: Optional[str] = None
            self.milestone_order: List[str] = []
            self.tasks: Dict[str, Dict[str, Any]] = {}
            self.iteration: int = 0
            self.is_stable: bool = False
            self.last_updated: float = time.time()
    
    def _update_from_workflow_state(self, state: WorkflowState) -> None:
        """Update UI state from WorkflowState.
        
        Args:
            state: WorkflowState to derive from
        """
        self.current_node = state.get("current_node")
        self.node_statuses = state.get("node_statuses", {}).copy()
        self.current_task_id = state.get("current_task_id")
        self.status = state.get("status", "running")
        self.error = state.get("error")
        milestones_list = get_milestones_list(state)
        self.milestones = {m["id"]: m for m in milestones_list if isinstance(m, dict) and m.get("id")}
        self.active_milestone_id = get_active_milestone_id(state)
        self.milestone_order = [m["id"] for m in milestones_list if isinstance(m, dict) and m.get("id")]
        self.tasks = state.get("tasks", {}).copy()
        self.iteration = state.get("iteration", 0)
        self.is_stable = state.get("is_stable", False)
        self.last_updated = time.time()
    
    def update_from_partial(self, update: Dict[str, Any]) -> None:
        """Update UI state from partial state update.
        
        Args:
            update: Partial state update dictionary
        """
        if "current_node" in update:
            self.current_node = update["current_node"]
        if "node_statuses" in update:
            self.node_statuses.update(update["node_statuses"])
        if "current_task_id" in update:
            self.current_task_id = update["current_task_id"]
        if "status" in update:
            self.status = update["status"]
        if "error" in update:
            self.error = update["error"]
        if "milestones_list" in update:
            ml = update["milestones_list"] or []
            self.milestones = {m["id"]: m for m in ml if isinstance(m, dict) and m.get("id")}
            self.milestone_order = [m["id"] for m in ml if isinstance(m, dict) and m.get("id")]
        if "milestones" in update:
            self.milestones.update(update["milestones"])
        if "active_milestone_index" in update:
            idx = update["active_milestone_index"]
            ml = self.milestone_order
            self.active_milestone_id = ml[idx] if 0 <= idx < len(ml) else None
        if "active_milestone_id" in update:
            self.active_milestone_id = update["active_milestone_id"]
        if "milestone_order" in update:
            self.milestone_order = update["milestone_order"]
        if "tasks" in update:
            self.tasks.update(update["tasks"])
        if "iteration" in update:
            self.iteration = update["iteration"]
        if "is_stable" in update:
            self.is_stable = update["is_stable"]
        
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert UI state to dictionary.
        
        Returns:
            Dictionary representation of UI state
        """
        return {
            "current_node": self.current_node,
            "node_statuses": self.node_statuses.copy(),
            "current_task_id": self.current_task_id,
            "status": self.status,
            "error": self.error,
            "milestones": self.milestones.copy(),
            "active_milestone_id": self.active_milestone_id,
            "milestone_order": self.milestone_order.copy(),
            "tasks": self.tasks.copy(),
            "iteration": self.iteration,
            "is_stable": self.is_stable,
            "last_updated": self.last_updated,
        }


class UIStateManager:
    """Manages UI state in a background thread.
    
    Listens to graph events via queue and maintains UIState.
    Provides subscription mechanism for UIs with throttled updates.
    
    Architecture:
    - Graph emits events to event_queue
    - UIStateManager (listener thread) processes events and updates UIState
    - UIs subscribe to UIStateManager and receive callbacks on state changes
    - Updates are throttled to max_refresh_rate (default: 10 Hz)
    """
    
    def __init__(
        self,
        initial_state: Optional[WorkflowState] = None,
        max_refresh_rate: float = 10.0,  # Max updates per second
    ):
        """Initialize UI State Manager.
        
        Args:
            initial_state: Initial workflow state
            max_refresh_rate: Maximum refresh rate in Hz (updates per second)
        """
        self.ui_state = UIState(initial_state)
        self.event_queue: queue.Queue[UIStateEvent] = queue.Queue()
        self.max_refresh_rate = max_refresh_rate
        self.min_update_interval = 1.0 / max_refresh_rate if max_refresh_rate > 0 else 0.0
        
        # Subscribers
        self._subscribers: List[Callable[[UIState], None]] = []
        
        # Threading
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Throttling
        self._last_update_time: float = 0.0
        self._pending_update = False
    
    def start(self) -> None:
        """Start the UI State Manager listener thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self._listener_thread.start()
    
    def stop(self) -> None:
        """Stop the UI State Manager listener thread."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        # Put a sentinel to wake up the queue
        try:
            self.event_queue.put_nowait(UIStateEvent(event_type="stop"))
        except queue.Full:
            pass
        
        if self._listener_thread:
            self._listener_thread.join(timeout=2.0)
    
    def emit_event(self, event: UIStateEvent) -> None:
        """Emit an event from graph to UI State Manager.
        
        This is called by the graph execution layer to send events.
        
        Args:
            event: UIStateEvent to emit
        """
        if not self._running:
            return
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            # If queue is full, drop the event (or could use blocking put with timeout)
            pass
    
    def subscribe(self, callback: Callable[[UIState], None]) -> None:
        """Subscribe to UI state updates.
        
        Args:
            callback: Function called with UIState when state changes
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[UIState], None]) -> None:
        """Unsubscribe from UI state updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def get_state(self) -> UIState:
        """Get current UI state.
        
        Returns:
            Current UIState (thread-safe read)
        """
        return self.ui_state
    
    def _listener_loop(self) -> None:
        """Main listener loop running in background thread."""
        while self._running and not self._stop_event.is_set():
            try:
                # Get event from queue with timeout
                try:
                    event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if we need to send a throttled update
                    if self._pending_update:
                        self._maybe_notify_subscribers()
                    continue
                
                # Handle stop event
                if event.event_type == "stop":
                    break
                
                # Process event
                self._process_event(event)
                
                # Mark that we have a pending update
                self._pending_update = True
                
                # Check if we should notify subscribers (throttled)
                self._maybe_notify_subscribers()
                
            except Exception as e:
                # Don't let errors in listener break the system
                # Log error if logging is available
                pass
        
        self._running = False
    
    def _process_event(self, event: UIStateEvent) -> None:
        """Process an event and update UI state.
        
        Args:
            event: UIStateEvent to process
        """
        if event.full_state:
            # Full state update
            self.ui_state._update_from_workflow_state(event.full_state)
        elif event.state_update:
            # Partial state update
            self.ui_state.update_from_partial(event.state_update)
        
        # Handle specific event types
        if event.event_type == "node_start":
            if event.node_name:
                self.ui_state.current_node = event.node_name
                self.ui_state.node_statuses[event.node_name] = "active"
        elif event.event_type == "node_complete":
            if event.node_name:
                self.ui_state.node_statuses[event.node_name] = "complete"
        elif event.event_type == "node_failed":
            if event.node_name:
                self.ui_state.node_statuses[event.node_name] = "failed"
    
    def _maybe_notify_subscribers(self) -> None:
        """Notify subscribers if enough time has passed (throttling).
        
        This implements the capped stream - updates are throttled to max_refresh_rate.
        """
        current_time = time.time()
        time_since_last_update = current_time - self._last_update_time
        
        if time_since_last_update >= self.min_update_interval:
            self._notify_subscribers()
            self._last_update_time = current_time
            self._pending_update = False
    
    def _notify_subscribers(self) -> None:
        """Notify all subscribers of state change.
        
        This is called after throttling check, so it's safe to call frequently.
        """
        # Create a snapshot of current state for subscribers
        # We pass the UIState directly, but subscribers should treat it as read-only
        for callback in self._subscribers:
            try:
                callback(self.ui_state)
            except Exception:
                # Don't let subscriber errors break the system
                pass
