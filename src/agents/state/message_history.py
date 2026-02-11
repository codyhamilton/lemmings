"""Message history for unified message state management.

This module provides:
- MessageType: Enum for different message types
- Message: Dataclass representing a message in history
- MessageHistory: Class that reads from WorkflowState.messages and provides UI subscription

MessageHistory no longer maintains its own state - it reads from WorkflowState.messages
and converts string messages to Message objects for UI consumption.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Dict, Any
import time
import uuid
from ..task_states import WorkflowState


class MessageType(str, Enum):
    """Type of message in the history."""
    USER = "user"              # User input message
    SYSTEM = "system"          # System message
    AI_MESSAGE = "ai_message"  # Regular AI message content
    THINKING = "thinking"      # Thinking block within AI message
    TOOL_USE = "tool_use"     # Tool call block within AI message
    TOOL_RESULT = "tool_result"  # Tool execution result


@dataclass
class Message:
    """A message in the unified message history.
    
    Messages can be streamed (content updated incrementally) or complete.
    """
    id: str  # Unique message ID
    type: MessageType
    node_name: Optional[str] = None  # Which node generated this (if AI)
    timestamp: float = field(default_factory=time.time)
    content: str = ""  # Message content (accumulates for streaming messages)
    is_complete: bool = False  # True when message is finalized
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context (tool name, etc.)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())


class MessageHistory:
    """Reads from WorkflowState.messages and provides UI subscription mechanism.
    
    No longer maintains its own state - reads from WorkflowState.messages (list[str])
    and converts to Message objects for UI consumption. Provides subscription mechanism
    for UIs to receive updates when state changes.
    """
    
    def __init__(self):
        """Initialize message history reader."""
        self._subscribers: List[Callable[[Message], None]] = []
        self._completion_subscribers: List[Callable[[Message], None]] = []
        self._last_message_count: int = 0
        self._parsed_messages: Dict[str, Message] = {}  # Cache parsed messages by index
        self._current_state: Optional[WorkflowState] = None
    
    def update_from_state(self, state: WorkflowState) -> None:
        """Update from WorkflowState and notify subscribers of new messages.
        
        Args:
            state: Current workflow state
        """
        self._current_state = state
        messages = state.get("messages", [])
        current_count = len(messages)
        
        # Check for new messages
        if current_count > self._last_message_count:
            # New messages added - parse and notify
            for i in range(self._last_message_count, current_count):
                msg_str = messages[i]
                # Parse string message to Message object
                message = self._parse_message_string(msg_str, i)
                if message:
                    self._parsed_messages[str(i)] = message
                    self._notify_subscribers(message)
                    # If message appears complete (ends with newline or is final), notify completion
                    if msg_str.endswith('\n') or i == current_count - 1:
                        message.is_complete = True
                        self._notify_completion_subscribers(message)
        
        self._last_message_count = current_count
    
    def _parse_message_string(self, msg_str: str, index: int) -> Optional[Message]:
        """Parse a string message from WorkflowState.messages into a Message object.
        
        Args:
            msg_str: String message from state
            index: Index in messages list
            
        Returns:
            Message object or None if parsing fails
        """
        if not msg_str:
            return None
        
        # Detect message type from content
        msg_type = MessageType.AI_MESSAGE
        content = msg_str
        
        if msg_str.startswith("🔧 Tool:"):
            msg_type = MessageType.TOOL_USE
            # Extract tool name and args from formatted string
            lines = msg_str.strip().split('\n')
            if len(lines) >= 2:
                tool_line = lines[0]  # "🔧 Tool: tool_name"
                args_line = lines[1] if len(lines) > 1 else ""  # "Args: {...}"
                content = f"{tool_line}\n{args_line}"
        elif msg_str.startswith("[THINKING]"):
            msg_type = MessageType.THINKING
            # Remove the [THINKING] prefix
            content = msg_str.replace("[THINKING]", "").strip()
        elif msg_str.startswith("👤 User:"):
            msg_type = MessageType.USER
            content = msg_str.replace("👤 User:", "").strip()
        elif msg_str.startswith("⚙️  System:"):
            msg_type = MessageType.SYSTEM
            content = msg_str.replace("⚙️  System:", "").strip()
        
        message = Message(
            id=str(index),
            type=msg_type,
            content=content,
            is_complete=msg_str.endswith('\n') if msg_str else False,
        )
        return message
    
    def get_history(self) -> List[Message]:
        """Get current message history from WorkflowState.
        
        Returns:
            List of all messages in order
        """
        if not self._current_state:
            return []
        messages = self._current_state.get("messages", [])
        return [self._parse_message_string(msg_str, i) for i, msg_str in enumerate(messages) if self._parse_message_string(msg_str, i)]
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a specific message by ID.
        
        Args:
            message_id: ID of message to retrieve (index as string)
            
        Returns:
            Message if found, None otherwise
        """
        return self._parsed_messages.get(message_id)
    
    def subscribe(self, callback: Callable[[Message], None]) -> None:
        """Subscribe to message updates.
        
        Args:
            callback: Function called with Message on append/update/complete
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Message], None]) -> None:
        """Unsubscribe from message updates.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
        if callback in self._completion_subscribers:
            self._completion_subscribers.remove(callback)
    
    def _notify_subscribers(self, message: Message) -> None:
        """Notify all subscribers of message update.
        
        Args:
            message: Message that was updated
        """
        for callback in self._subscribers:
            try:
                callback(message)
            except Exception:
                # Don't let subscriber errors break the system
                pass
    
    def _notify_completion_subscribers(self, message: Message) -> None:
        """Notify completion subscribers when a message is finalized.
        
        Args:
            message: Message that was completed
        """
        for callback in self._completion_subscribers:
            try:
                callback(message)
            except Exception:
                # Don't let subscriber errors break the system
                pass
