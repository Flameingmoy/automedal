"""Current-experiment card: hypothesis, budget bar, live val_loss."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.events import Phase
from tui.state import AppState


def _bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100.0 * width)
    return "#" * filled + "-" * (width - filled)


class CurrentExperiment(Vertical):
    DEFAULT_CSS = """
    CurrentExperiment { height: 8; padding: 0 1; }
    CurrentExperiment > Static { height: 1; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()

    def compose(self) -> ComposeResult:
        yield Static("current experiment")
        yield Static("(no pending)", id="ce-hyp")
        yield Static("budget: [-] 0%", id="ce-budget")
        yield Static("val_loss: n/a", id="ce-loss")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._paint)

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._paint()

    def _paint(self) -> None:
        st = self._state
        try:
            hyp = st.queue.current_hypothesis or "(no pending)"
            if st.current_exp_id and st.current_exp_id in st.experiments:
                exp = st.experiments[st.current_exp_id]
                hyp = exp.hypothesis or hyp
            self.query_one("#ce-hyp", Static).update(hyp[:80])

            pct = 0.0
            if st.phase == Phase.EXPERIMENT and st.training_started_ts:
                elapsed = time.time() - st.training_started_ts
                budget_total = max(1.0, st.train_budget_minutes * 60)
                pct = min(100.0, elapsed / budget_total * 100.0)
            self.query_one("#ce-budget", Static).update(f"budget: {_bar(pct, 12)} {pct:3.0f}%")

            loss_txt = "val_loss: n/a"
            if st.current_exp_id in st.experiments:
                v = st.experiments[st.current_exp_id].val_loss
                if v is not None:
                    loss_txt = f"val_loss: {v:.4f}"
            self.query_one("#ce-loss", Static).update(loss_txt)
        except Exception:
            pass
