"""UI module for workflow user interfaces.

This module provides:
- UIBase: Base class for all UIs
- VerboseUI: Terminal-based verbose UI
- DashboardUI: Dashboard UI (Textual-based)
- UIStateManager: Background thread that manages UI state
"""

from .base import UIBase
from .verbose import VerboseUI
from .state_manager import UIStateManager, UIState, UIStateEvent

__all__ = ["UIBase", "VerboseUI", "UIStateManager", "UIState", "UIStateEvent"]
