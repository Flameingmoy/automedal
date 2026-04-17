"""Session totals — iterations, success rate, elapsed time."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.state import AppState


class SessionStats(Vertical):
    DEFAULT_CSS = """
    SessionStats { height: 1fr; padding: 0 1; }
    SessionStats > #ss-title { height: 1; color: #8BE9FD; }
    SessionStats > #ss-body { height: 1fr; }
    """

    def __init__(self, started_ts: float | None = None, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()
        self._started = started_ts or time.time()

    def compose(self) -> ComposeResult:
        yield Static("session", id="ss-title")
        yield Static("—", id="ss-body")

    def on_mount(self) -> None:
        self.set_interval(1.0, lambda: self._paint())

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._paint()

    def _paint(self) -> None:
        st = self._state
        kept = sum(1 for e in st.experiments.values() if e.status == "kept")
        reverted = sum(1 for e in st.experiments.values() if e.status == "reverted")
        crashed = sum(1 for e in st.experiments.values() if e.status == "crash")
        total = kept + reverted + crashed
        rate = (kept / total * 100.0) if total else 0.0
        elapsed = int(time.time() - self._started)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        body = (
            f"iter     {st.iteration}/{st.total_iterations}\n"
            f"kept     {kept}\n"
            f"reverted {reverted}\n"
            f"crashed  {crashed}\n"
            f"success  {rate:.0f}%\n"
            f"elapsed  {h:02d}:{m:02d}:{s:02d}"
        )
        try:
            self.query_one("#ss-body", Static).update(body)
        except Exception:
            pass
