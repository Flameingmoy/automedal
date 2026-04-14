"""Sprite panel — shows the current phase sprite + phase label + heartbeat pulse."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.events import Phase
from tui.state import AppState
from tui.sprite_loader import load_sprite
from tui.themes.palette import DEFAULT_THEME, Theme


class SpritePanel(Vertical):
    DEFAULT_CSS = """
    SpritePanel { height: 16; padding: 0 1; }
    SpritePanel > #sprite-art { height: 12; }
    SpritePanel > #sprite-label { height: 1; content-align: center middle; }
    SpritePanel > #sprite-sub { height: 1; content-align: center middle; color: $text-muted; }
    """

    def __init__(self, theme: Theme = DEFAULT_THEME, **kw) -> None:
        super().__init__(**kw)
        self.theme = theme
        self._state = AppState()
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(" ", id="sprite-art")
        yield Static("idle", id="sprite-label")
        yield Static("—", id="sprite-sub")

    def on_mount(self) -> None:
        self.set_interval(0.6, self._tick)
        self._paint()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % 2
        self._paint()

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._paint()

    def _paint(self) -> None:
        sprite = load_sprite(self._state.phase, size=24, frame=self._frame, theme=self.theme)
        try:
            self.query_one("#sprite-art", Static).update(sprite)
            color = self.theme.phase_colors.get(self._state.phase, self.theme.fg)
            self.query_one("#sprite-label", Static).update(f"[b {color}]{self._state.phase.value.upper()}[/]")
            sub = ""
            if self._state.phase == Phase.EXPERIMENT and self._state.training_started_ts:
                elapsed = time.time() - self._state.training_started_ts
                budget = self._state.train_budget_minutes * 60
                remaining = max(0, int(budget - elapsed))
                sub = f"budget: {remaining//60:02d}:{remaining%60:02d}"
            elif self._state.phase == Phase.FROZEN and self._state.frozen_reason:
                sub = self._state.frozen_reason
            elif self._state.current_exp_id:
                sub = f"exp {self._state.current_exp_id}"
            self.query_one("#sprite-sub", Static).update(sub or "—")
        except Exception:
            pass
