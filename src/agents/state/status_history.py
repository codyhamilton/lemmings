"""Status history for workflow state change tracking.

This module provides:
- StatusEventType: Enum for different status event types
- StatusEvent: Dataclass representing a status event in history
- StatusHistory: Class that owns and manages the unified status event history
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Dict, Any
import time
import uuid


class StatusEventType(str, Enum):
    """Type of status event in the history."""
    NODE_START = "node_start"  # Node execution started
    NODE_COMPLETE = "node_complete"  # Node execution completed
    NODE_FAILED = "node_failed"  # Node execution failed
    TASK_COMPLETE = "task_complete"  # Task marked complete
    TASK_FAILED = "task_failed"  # Task marked failed
    MILESTONE_ADVANCE = "milestone_advance"  # Milestone advanced
    ITERATION_INCREMENT = "iteration_increment"  # Expansion iteration incremented
    STATE_UPDATE = "state_update"  # General state change


@dataclass
class StatusEvent:
    """A status event in the unified status history.
    
    Status events represent workflow state changes, node executions,
    task completions, and other orchestration-level events.
    """
    type: StatusEventType
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique event ID
    node_name: Optional[str] = None  # Which node generated this (if applicable)
    timestamp: float = field(default_factory=time.time)
    summary: str = ""  # Human-readable summary (from summarizer)
    data: Dict[str, Any] = field(default_factory=dict)  # Structured data (task_id, error, etc.)
    is_complete: bool = True  # Status events are typically complete immediately


class StatusHistory:
    """Manages unified status event history state.
    
    Owns the status event history and provides subscription mechanism for UIs.
    Events represent workflow state changes and node execution status.
    """
    
    def __init__(self):
        """Initialize empty status history."""
        self.events: List[StatusEvent] = []
        self._subscribers: List[Callable[[StatusEvent], None]] = []
    
    def append(self, event: StatusEvent) -> None:
        """Add a new status event to history and notify subscribers.
        
        Args:
            event: StatusEvent to add
        """
        self.events.append(event)
        self._notify_subscribers(event)
    
    def get_history(self) -> List[StatusEvent]:
        """Get current status event history.
        
        Returns:
            List of all events in order
        """
        return self.events.copy()
    
    def get_event(self, event_id: str) -> Optional[StatusEvent]:
        """Get a specific event by ID.
        
        Args:
            event_id: ID of event to retrieve
            
        Returns:
            StatusEvent if found, None otherwise
        """
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    
    def get_recent_events(self, limit: int = 10) -> List[StatusEvent]:
        """Get recent events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events (most recent last)
        """
        return self.events[-limit:] if len(self.events) > limit else self.events.copy()
    
    def subscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        """Subscribe to status event updates.
        
        Args:
            callback: Function called with StatusEvent on append
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[StatusEvent], None]) -> None:
        """Unsubscribe from status event updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, event: StatusEvent) -> None:
        """Notify all subscribers of status event.
        
        Args:
            event: StatusEvent that was added
        """
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                # Don't let subscriber errors break the system
                pass
