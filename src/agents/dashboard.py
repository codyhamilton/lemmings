"""Dashboard renderer for workflow visualization.

Provides a Textual-based dashboard showing:
- Milestone progress
- Graph visualization with node status
- Summarized agent activity
"""

import threading
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.containers import Container, Vertical
from textual.reactive import reactive

from .state import WorkflowState, TaskTree, Milestone, MilestoneStatus
from .agents.summarizer import summarize_agent_activity


class NodeStatus(str, Enum):
    """Status of a graph node."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class DashboardState:
    """Tracks dashboard state for rendering."""
    current_node: Optional[str] = None
    node_statuses: dict[str, NodeStatus] = field(default_factory=dict)
    active_task_summary: Optional[str] = None
    last_updates: deque = field(default_factory=lambda: deque(maxlen=10))
    show_detailed: bool = False


class MilestoneWidget(Static):
    """Widget displaying milestone progress."""
    
    milestones_dict: reactive[dict] = reactive({})
    milestone_order: reactive[list] = reactive([])
    active_milestone_id: reactive[Optional[str]] = reactive(None)
    
    def render(self) -> str:
        """Render milestone indicators."""
        if not self.milestones_dict or not self.milestone_order:
            return ""
        
        lines = []
        lines.append("[bold]Milestones:[/bold]")
        
        milestone_indicators = []
        for milestone_id in self.milestone_order:
            milestone_data = self.milestones_dict.get(milestone_id, {})
            if not milestone_data:
                milestone_indicators.append("[ ]")
                continue
            
            try:
                milestone = Milestone.from_dict(milestone_data)
                
                if milestone.status == MilestoneStatus.COMPLETE:
                    milestone_indicators.append("[green][✓][/green]")
                elif milestone_id == self.active_milestone_id:
                    milestone_indicators.append("[yellow][▶][/yellow]")
                else:
                    milestone_indicators.append("[ ]")
            except Exception:
                milestone_indicators.append("[ ]")
        
        if milestone_indicators:
            lines.append("  " + " ".join(milestone_indicators))
        
        # Show active milestone description
        if self.active_milestone_id:
            active_milestone_data = self.milestones_dict.get(self.active_milestone_id, {})
            if active_milestone_data:
                try:
                    active_milestone = Milestone.from_dict(active_milestone_data)
                    lines.append(f"  [dim]Active: {active_milestone.description}[/dim]")
                except Exception:
                    pass
        
        return "\n".join(lines) if lines else ""
    
    def watch_milestones_dict(self) -> None:
        """Update when milestones change."""
        self.refresh()
    
    def watch_milestone_order(self) -> None:
        """Update when milestone order changes."""
        self.refresh()
    
    def watch_active_milestone_id(self) -> None:
        """Update when active milestone changes."""
        self.refresh()


class GraphWidget(Static):
    """Widget displaying graph visualization."""
    
    current_node: reactive[Optional[str]] = reactive(None)
    node_statuses: reactive[dict] = reactive({})
    
    def render(self) -> str:
        """Render graph visualization."""
        lines = []
        lines.append("[bold]Graph:[/bold]")
        
        node_display_names = {
            "intake": "Intake",
            "expander": "Expander",
            "prioritizer": "Prioritizer",
            "researcher": "Researcher",
            "planner": "Planner",
            "implementor": "Implementor",
            "validator": "Validator",
            "qa": "QA",
            "assessor": "Assessor",
            "mark_complete": "✓",
            "mark_failed": "✗",
        }
        
        def get_node_display(node_name: str) -> str:
            """Get formatted node display."""
            display_name = node_display_names.get(node_name, node_name)
            status = self.node_statuses.get(node_name, NodeStatus.PENDING)
            
            if node_name == self.current_node:
                color = "yellow"
                prefix = "▶"
            elif status == NodeStatus.COMPLETE:
                color = "green"
                prefix = "✓"
            elif status == NodeStatus.FAILED:
                color = "red"
                prefix = "✗"
            else:
                color = "grey70"
                prefix = "·"
            
            return f"[{color}]{prefix} {display_name}[/{color}]"
        
        # Render simplified graph - adjust for terminal width
        width = self.size.width if self.size else 80
        if width < 70:
            # Compact vertical layout
            lines.append("  " + get_node_display("intake"))
            lines.append("    ↓")
            lines.append("  " + get_node_display("expander"))
            lines.append("    ↓")
            lines.append("  " + get_node_display("prioritizer"))
            lines.append("    ↓")
            lines.append("  " + get_node_display("researcher") + " → " + 
                        get_node_display("planner") + " → " + get_node_display("implementor"))
            lines.append("    ↓")
            lines.append("  " + get_node_display("validator") + " → " + 
                        get_node_display("qa") + " → " + get_node_display("assessor"))
        else:
            # Full horizontal layout
            line1 = "  " + get_node_display("intake") + " ──→ " + get_node_display("expander") + " ──→ " + get_node_display("prioritizer")
            if len(line1) <= width:
                lines.append(line1)
                lines.append("                              │")
                lines.append("                              ↓")
                line2 = "  " + get_node_display("researcher") + " ──→ " + get_node_display("planner") + " ──→ " + get_node_display("implementor")
                if len(line2) <= width:
                    lines.append(line2)
                    lines.append("              │                    │              │")
                    lines.append("              └────────────────────┴──────────────┘")
                    lines.append("                              ↓")
                    lines.append("  " + get_node_display("validator") + " ──→ " + get_node_display("qa") + " ──→ " + get_node_display("assessor"))
                else:
                    # Fallback to compact
                    lines.append("  " + get_node_display("researcher") + " → " + get_node_display("planner") + " → " + get_node_display("implementor"))
                    lines.append("    ↓")
                    lines.append("  " + get_node_display("validator") + " → " + get_node_display("qa") + " → " + get_node_display("assessor"))
            else:
                # Too narrow for full layout, use compact
                lines.append("  " + get_node_display("intake") + " → " + get_node_display("expander") + " → " + get_node_display("prioritizer"))
                lines.append("    ↓")
                lines.append("  " + get_node_display("researcher") + " → " + get_node_display("planner") + " → " + get_node_display("implementor"))
                lines.append("    ↓")
                lines.append("  " + get_node_display("validator") + " → " + get_node_display("qa") + " → " + get_node_display("assessor"))
        
        return "\n".join(lines)
    
    def watch_current_node(self) -> None:
        """Update when current node changes."""
        self.refresh()
    
    def watch_node_statuses(self) -> None:
        """Update when node statuses change."""
        self.refresh()


class ActivityWidget(Static):
    """Widget displaying current activity."""
    
    current_node: reactive[Optional[str]] = reactive(None)
    active_task_summary: reactive[Optional[str]] = reactive(None)
    
    def render(self) -> str:
        """Render current activity."""
        if self.current_node:
            node_display = self.current_node.replace("_", " ").title()
            if self.active_task_summary:
                return f"[bold]Current:[/bold] {node_display} - {self.active_task_summary}"
            else:
                return f"[bold]Current:[/bold] {node_display}"
        else:
            return "[bold]Current:[/bold] [dim]Waiting...[/dim]"
    
    def watch_current_node(self) -> None:
        """Update when current node changes."""
        self.refresh()
    
    def watch_active_task_summary(self) -> None:
        """Update when task summary changes."""
        self.refresh()


class RecentUpdatesWidget(Static):
    """Widget displaying recent activity updates."""
    
    last_updates: reactive[list] = reactive([])
    
    def render(self) -> str:
        """Render recent updates."""
        lines = []
        lines.append("[bold]Recent:[/bold]")
        
        if self.last_updates:
            for update in list(self.last_updates)[-5:]:
                lines.append(f"  {update}")
        else:
            lines.append("  [dim]No recent activity[/dim]")
        
        return "\n".join(lines)
    
    def watch_last_updates(self) -> None:
        """Update when last updates change."""
        self.refresh()


class DashboardApp(App):
    """Textual app for workflow dashboard."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    MilestoneWidget, GraphWidget, ActivityWidget, RecentUpdatesWidget {
        padding: 1;
        margin: 1;
        border: solid $primary;
        height: auto;
    }
    
    #header {
        text-align: center;
        padding: 1;
        border: solid $primary;
    }
    
    #footer {
        text-align: center;
        padding: 1;
        border: solid $primary;
    }
    """
    
    BINDINGS = [
        ("escape", "toggle_detailed", "Toggle detailed view"),
    ]
    
    def __init__(self, initial_state: WorkflowState):
        """Initialize dashboard app.
        
        Args:
            initial_state: Initial workflow state
        """
        super().__init__()
        self.state = initial_state
        self.dashboard_state = DashboardState()
        self._app_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Widget references (set after compose)
        self.milestone_widget: Optional[MilestoneWidget] = None
        self.graph_widget: Optional[GraphWidget] = None
        self.activity_widget: Optional[ActivityWidget] = None
        self.updates_widget: Optional[RecentUpdatesWidget] = None
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard UI."""
        with Container(id="main"):
            yield Static("🤖 [bold]Workflow Dashboard[/bold]", id="header")
            
            with Vertical():
                yield MilestoneWidget(id="milestones")
                yield GraphWidget(id="graph")
                yield ActivityWidget(id="activity")
                yield RecentUpdatesWidget(id="updates")
            
            yield Static("[dim]Press ESC to toggle detailed view[/dim]", id="footer")
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Get widget references
        self.milestone_widget = self.query_one("#milestones", MilestoneWidget)
        self.graph_widget = self.query_one("#graph", GraphWidget)
        self.activity_widget = self.query_one("#activity", ActivityWidget)
        self.updates_widget = self.query_one("#updates", RecentUpdatesWidget)
        
        # Initialize with current state
        self._update_widgets()
        self._running = True
    
    def _update_widgets(self) -> None:
        """Update all widgets with current state."""
        if not self._running:
            return
        
        # Update milestones
        if self.milestone_widget:
            milestones_dict = self.state.get("milestones", {})
            milestone_order = self.state.get("milestone_order", [])
            active_milestone_id = self.state.get("active_milestone_id")
            
            self.milestone_widget.milestones_dict = milestones_dict
            self.milestone_widget.milestone_order = milestone_order
            self.milestone_widget.active_milestone_id = active_milestone_id
        
        # Update graph
        if self.graph_widget:
            self.graph_widget.current_node = self.dashboard_state.current_node
            # Convert string statuses to NodeStatus enums
            node_statuses = {}
            for node, status in self.dashboard_state.node_statuses.items():
                if isinstance(status, str):
                    try:
                        node_statuses[node] = NodeStatus(status)
                    except ValueError:
                        node_statuses[node] = NodeStatus.PENDING
                else:
                    node_statuses[node] = status
            self.graph_widget.node_statuses = node_statuses
        
        # Update activity
        if self.activity_widget:
            self.activity_widget.current_node = self.dashboard_state.current_node
            self.activity_widget.active_task_summary = self.dashboard_state.active_task_summary
        
        # Update recent updates
        if self.updates_widget:
            self.updates_widget.last_updates = list(self.dashboard_state.last_updates)
    
    def action_toggle_detailed(self) -> None:
        """Toggle detailed view mode."""
        self.dashboard_state.show_detailed = not self.dashboard_state.show_detailed
        from .workflow_status import set_output_suppression
        set_output_suppression(
            suppress=not self.dashboard_state.show_detailed,
            show_detailed=self.dashboard_state.show_detailed
        )
    
    def update_state(self, state: WorkflowState, node_name: Optional[str] = None) -> None:
        """Update dashboard with new workflow state.
        
        Args:
            state: Current workflow state
            node_name: Name of node that just executed (if any)
        """
        self.state = state
        
        # Update current node
        if node_name:
            self.dashboard_state.current_node = node_name
            # Mark previous node as complete
            for prev_node in self.dashboard_state.node_statuses:
                if self.dashboard_state.node_statuses[prev_node] == NodeStatus.ACTIVE:
                    self.dashboard_state.node_statuses[prev_node] = NodeStatus.COMPLETE
            # Mark new node as active
            self.dashboard_state.node_statuses[node_name] = NodeStatus.ACTIVE
        
        # Generate summary for last node execution
        if node_name:
            summary = summarize_agent_activity(node_name, state)
            self.dashboard_state.last_updates.append(summary)
            self.dashboard_state.active_task_summary = summary
        
        # Update widgets (call from thread-safe context if app is running)
        try:
            if self.is_running:
                self.call_from_thread(self._update_widgets)
            else:
                # If not running yet, update directly (will be applied when app starts)
                self._update_widgets()
        except Exception:
            # Fallback: update directly if call_from_thread fails
            try:
                self._update_widgets()
            except Exception:
                pass
    
    def render(self) -> None:
        """Render the dashboard (called from external code for compatibility)."""
        # Textual handles rendering automatically, but we can trigger a refresh
        if self._running:
            try:
                if self.is_running:
                    self.call_from_thread(self._update_widgets)
                else:
                    self._update_widgets()
            except Exception:
                # If update fails, try direct update
                try:
                    self._update_widgets()
                except Exception:
                    pass
    
    def run_in_thread(self) -> None:
        """Run the app in a separate thread."""
        def run_app():
            try:
                self.run()
            except Exception:
                # If Textual can't run in thread (e.g., terminal control issues),
                # we'll handle it gracefully
                pass
        
        self._app_thread = threading.Thread(target=run_app, daemon=True)
        self._app_thread.start()
        # Give it a moment to start
        import time
        time.sleep(0.2)
    
    def stop_app(self) -> None:
        """Stop the app."""
        self._running = False
        try:
            if hasattr(self, 'is_running') and self.is_running:
                self.exit()
        except Exception:
            pass
    
    def cleanup(self) -> None:
        """Cleanup dashboard resources."""
        self.stop_app()
        if self._app_thread and self._app_thread.is_alive():
            self._app_thread.join(timeout=2.0)


class DashboardRenderer:
    """Wrapper for dashboard renderer (maintains compatibility with existing code)."""
    
    def __init__(self, state: WorkflowState):
        """Initialize dashboard renderer.
        
        Args:
            state: Initial workflow state
        """
        self.app = DashboardApp(state)
        self.app.run_in_thread()
        # Wait a moment for app to start
        import time
        time.sleep(0.1)
    
    def update_state(self, state: WorkflowState, node_name: Optional[str] = None):
        """Update dashboard with new workflow state.
        
        Args:
            state: Current workflow state
            node_name: Name of node that just executed (if any)
        """
        self.app.update_state(state, node_name)
    
    def render(self):
        """Render the dashboard."""
        self.app.render()
    
    def cleanup(self):
        """Cleanup dashboard resources."""
        self.app.cleanup()


def parse_milestones_from_plan(plan_path: str) -> dict[str, dict]:
    """Parse milestones from plan file.
    
    Args:
        plan_path: Path to plan file
    
    Returns:
        Dictionary of milestone_id -> milestone data
    """
    milestones = {}
    
    try:
        with open(plan_path, 'r') as f:
            content = f.read()
        
        # Parse frontmatter for todos
        import re
        todo_pattern = r'- id: (\w+)\s+content: (.+?)\s+status: (\w+)'
        matches = re.findall(todo_pattern, content, re.MULTILINE)
        
        # Extract milestone-related todos
        milestone_todos = {}
        for todo_id, content, status in matches:
            if 'milestone' in content.lower():
                milestone_todos[todo_id] = {
                    'id': todo_id,
                    'content': content,
                    'status': status
                }
        
        # Try to extract milestone descriptions from plan content
        # This is a simple heuristic - may need adjustment based on actual plan format
        milestone_desc_pattern = r'milestone[_\s]*(\d+)[:\s]+(.+?)(?:\n|$)'
        desc_matches = re.findall(milestone_desc_pattern, content, re.IGNORECASE | re.MULTILINE)
        
        for i, (num, desc) in enumerate(desc_matches):
            milestone_id = f"milestone_{num.zfill(3)}"
            milestones[milestone_id] = {
                'id': milestone_id,
                'description': desc.strip()[:200],
                'status': 'pending',
                'order': i
            }
    
    except (FileNotFoundError, IOError, Exception) as e:
        # If plan file doesn't exist or can't be parsed, return empty dict
        pass
    
    return milestones
