"""Status screen — pure TUI dashboard using existing widgets; no subprocess needed."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Static

from tui.state import AppState
from tui.widgets.leaderboard import Leaderboard
from tui.widgets.recent_activity import RecentActivity
from tui.widgets.queue_summary import QueueSummary
from tui.widgets.status_strip import StatusStrip


class StatusScreen(Screen):
    DEFAULT_CSS = """
    StatusScreen {
        layout: vertical;
        background: #0F111A;
    }
    StatusScreen > #st-title {
        height: 1;
        background: #0F111A;
        padding: 0 1;
        color: #8BE9FD;
    }
    StatusScreen > StatusStrip {
        height: 1;
        margin: 0;
    }
    StatusScreen > #st-mid {
        height: 1fr;
        layout: horizontal;
    }
    StatusScreen > #st-mid > RecentActivity {
        width: 1fr;
        margin: 1 1 0 1;
    }
    StatusScreen > #st-mid > Leaderboard {
        width: 36;
        margin: 1 1 0 0;
    }
    StatusScreen > QueueSummary {
        margin: 0 1 0 1;
    }
    """

    BINDINGS = [
        ("q",      "back", "Back"),
        ("escape", "back", "Back"),
        ("r",      "refresh", "Refresh"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []

    def compose(self) -> ComposeResult:
        queue_path = (
            self._layout.queue_md if self._layout else None
        )
        yield Static("Status", id="st-title")
        yield StatusStrip(id="st-strip")
        with Horizontal(id="st-mid"):
            yield RecentActivity(id="st-recent")
            yield Leaderboard(id="st-lb")
        yield QueueSummary(queue_md=queue_path, id="st-queue")
        yield Footer()

    def update_state(self, state: AppState) -> None:
        for wid, cls in (
            ("#st-strip", StatusStrip),
            ("#st-recent", RecentActivity),
            ("#st-lb", Leaderboard),
            ("#st-queue", QueueSummary),
        ):
            try:
                w = self.query_one(wid, cls)
                if hasattr(w, "update_state"):
                    w.update_state(state)
            except Exception:
                pass

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self.update_state(self.app.state)
