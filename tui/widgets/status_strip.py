"""Status strip — single-line header for the home screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from tui.events import Phase
from tui.state import AppState


class StatusStrip(Horizontal):
    DEFAULT_CSS = """
    StatusStrip {
        height: 1;
        background: #0F111A;
        padding: 0 1;
    }
    StatusStrip > #ss-brand { width: auto; color: #8BE9FD; }
    StatusStrip > #ss-comp  { width: auto; margin-left: 2; }
    StatusStrip > #ss-right { width: 1fr; content-align: right middle; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()

    def compose(self) -> ComposeResult:
        yield Static("AutoMedal", id="ss-brand")
        yield Static("", id="ss-comp")
        yield Static("", id="ss-right")

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._paint()

    def _paint(self) -> None:
        st = self._state
        comp = st.competition
        slug = (comp.slug if comp else "") or "(no competition)"

        phase_label = st.phase.value.upper() if st.phase != Phase.IDLE else ""
        parts: list[str] = []
        if st.best_val_loss < float("inf"):
            parts.append(f"best={st.best_val_loss:.4f}")
        if st.iteration > 0:
            parts.append(f"iter {st.iteration}/{st.total_iterations}")
        if phase_label:
            parts.append(phase_label)
        right = "  ".join(parts)

        try:
            self.query_one("#ss-comp", Static).update(f"· {slug}")
            self.query_one("#ss-right", Static).update(right)
        except Exception:
            pass
