"""Workflow status tracking and visual progress display."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import sys


class Phase(str, Enum):
    """Pipeline phases."""
    UNDERSTAND = "understand"
    RESEARCH = "research"
    PLAN = "plan"
    CONSOLIDATE = "consolidate"
    CODE = "code"
    REVIEW = "review"
    VALIDATE = "validate"
    REFINE = "refine"
    COMPLETE = "complete"


PHASE_INFO = {
    Phase.UNDERSTAND: ("🎯", "Understanding request"),
    Phase.RESEARCH: ("🔍", "Finding context"),
    Phase.PLAN: ("📝", "Planning changes"),
    Phase.CONSOLIDATE: ("🔄", "Consolidating plan"),
    Phase.CODE: ("💻", "Implementing changes"),
    Phase.REVIEW: ("✅", "Reviewing changes"),
    Phase.VALIDATE: ("🏁", "Validating results"),
    Phase.REFINE: ("⟳", "Refining approach"),
    Phase.COMPLETE: ("✨", "Complete"),
}


@dataclass
class WorkflowStatus:
    """Tracks the current state of the workflow for visual display."""
    
    current_phase: Phase = Phase.UNDERSTAND
    iteration: int = 0
    
    # Progress within phases
    total_requirements: int = 0
    current_requirement: int = 0
    
    total_changes: int = 0
    current_change: int = 0
    
    # Retry tracking
    current_retry: int = 0
    max_retries: int = 3
    
    # Error accumulation for learning
    phase_errors: dict = field(default_factory=dict)
    
    def set_phase(self, phase: Phase) -> None:
        """Update the current phase and display."""
        self.current_phase = phase
        self._display_status()
    
    def set_requirement_progress(self, current: int, total: int) -> None:
        """Update requirement progress."""
        self.current_requirement = current
        self.total_requirements = total
    
    def set_change_progress(self, current: int, total: int) -> None:
        """Update change progress."""
        self.current_change = current
        self.total_changes = total
    
    def record_error(self, phase: Phase, error: str) -> None:
        """Record an error for potential retry/learning."""
        if phase not in self.phase_errors:
            self.phase_errors[phase] = []
        self.phase_errors[phase].append(error)
    
    def get_error_context(self) -> str:
        """Get accumulated error context for refinement."""
        if not self.phase_errors:
            return ""
        
        parts = ["Previous errors encountered:"]
        for phase, errors in self.phase_errors.items():
            parts.append(f"\n{phase.value}:")
            for err in errors[-3:]:  # Last 3 errors per phase
                parts.append(f"  - {err[:200]}")
        return "\n".join(parts)
    
    def _display_status(self) -> None:
        """Display the current workflow status."""
        # Build the pipeline visualization
        phases = [
            Phase.UNDERSTAND, Phase.RESEARCH, Phase.PLAN, 
            Phase.CONSOLIDATE, Phase.CODE, Phase.REVIEW, Phase.VALIDATE
        ]
        
        # Determine current phase index (handle COMPLETE specially)
        if self.current_phase == Phase.COMPLETE:
            current_idx = len(phases)  # All phases complete
        elif self.current_phase in phases:
            current_idx = phases.index(self.current_phase)
        else:
            current_idx = 0  # Fallback
        
        # Build phase indicators
        indicators = []
        for phase in phases:
            emoji, _ = PHASE_INFO[phase]
            phase_idx = phases.index(phase)
            if phase == self.current_phase:
                indicators.append(f"[{emoji}]")
            elif phase_idx < current_idx:
                indicators.append(f" ✓ ")
            else:
                indicators.append(f" · ")
        
        # Add COMPLETE indicator if we're in complete phase
        if self.current_phase == Phase.COMPLETE:
            complete_emoji, _ = PHASE_INFO[Phase.COMPLETE]
            indicators.append(f"[{complete_emoji}]")
        
        # Print the status bar
        print("\n┌" + "─" * 68 + "┐")
        print(f"│ Pipeline: {' → '.join(indicators)[:60]:60} │")
        
        # Current phase info
        emoji, desc = PHASE_INFO[self.current_phase]
        print(f"│ {emoji} {desc:62} │")
        
        # Progress details
        if self.current_phase in (Phase.RESEARCH, Phase.PLAN):
            progress = f"Requirement {self.current_requirement}/{self.total_requirements}"
            print(f"│   {progress:62} │")
        elif self.current_phase in (Phase.CODE, Phase.REVIEW):
            progress = f"Change {self.current_change}/{self.total_changes}"
            if self.current_retry > 0:
                progress += f" (retry {self.current_retry}/{self.max_retries})"
            print(f"│   {progress:62} │")
        
        # Iteration info
        if self.iteration > 0:
            print(f"│   Refinement iteration: {self.iteration:38} │")
        
        print("└" + "─" * 68 + "┘")
        sys.stdout.flush()


# Global status instance
_status: Optional[WorkflowStatus] = None


def get_status() -> WorkflowStatus:
    """Get or create the global workflow status."""
    global _status
    if _status is None:
        _status = WorkflowStatus()
    return _status


def reset_status() -> WorkflowStatus:
    """Reset the workflow status for a new run."""
    global _status
    _status = WorkflowStatus()
    return _status


# ANSI color codes
GREY = "\033[90m"
RESET = "\033[0m"
DIM = "\033[2m"

# Global flag for output suppression (set by dashboard)
_suppress_output = False
_show_detailed = False


def set_output_suppression(suppress: bool, show_detailed: bool = False):
    """Set output suppression mode.
    
    Args:
        suppress: Whether to suppress verbose output
        show_detailed: Whether detailed view is active (overrides suppression)
    """
    global _suppress_output, _show_detailed
    _suppress_output = suppress
    _show_detailed = show_detailed


def should_suppress_output() -> bool:
    """Check if output should be suppressed.
    
    Returns:
        True if output should be suppressed (dashboard mode and not detailed view)
    """
    return _suppress_output and not _show_detailed


def print_thinking(text: str) -> None:
    """Print thinking/reasoning output in grey.
    
    Use this for:
    - Tool calls and their results
    - Intermediate reasoning steps
    - Agent's internal thoughts
    """
    if should_suppress_output():
        return
    print(f"{GREY}{text}{RESET}", flush=True)


def print_thinking_line(text: str) -> None:
    """Print a thinking line with grey prefix."""
    print_thinking(f"  💭 {text}")


def display_agent_header(
    agent_name: str,
    phase: Phase,
    subtitle: str = "",
    details: dict | None = None,
) -> None:
    """Display a consistent agent header with workflow context.
    
    Args:
        agent_name: Name of the agent
        phase: Current pipeline phase
        subtitle: Optional subtitle
        details: Optional dict of key-value details to show
    """
    if should_suppress_output():
        return
    
    status = get_status()
    status.set_phase(phase)
    
    emoji, phase_desc = PHASE_INFO[phase]
    
    print("\n" + "=" * 70)
    print(f"{emoji} {agent_name.upper()}")
    print("=" * 70)
    
    if subtitle:
        print(subtitle)
    
    if details:
        for key, value in details.items():
            if value:
                # Truncate long values
                str_val = str(value)
                if len(str_val) > 60:
                    str_val = str_val[:57] + "..."
                print(f"  {key}: {str_val}")
    
    print()
