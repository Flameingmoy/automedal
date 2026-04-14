"""Phase-colored live log stream. Consumes RawLine events directly."""

from __future__ import annotations

from textual.widgets import RichLog

from tui.themes.palette import DEFAULT_THEME, Theme


class LiveStream(RichLog):
    DEFAULT_CSS = "LiveStream { height: 1fr; border: round $panel; }"

    def __init__(self, theme: Theme = DEFAULT_THEME, **kw) -> None:
        super().__init__(max_lines=5000, wrap=False, highlight=False, markup=False, **kw)
        self.theme = theme
        self._filter = ""
        self._paused = False

    def set_filter(self, text: str) -> None:
        self._filter = text.lower()

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.auto_scroll = not self._paused

    def push_line(self, line: str) -> None:
        if self._filter and self._filter not in line.lower():
            return
        style = None
        low = line.lower()
        if "[harness]" in low:
            style = self.theme.accent
        elif "training done" in low or "val_loss" in low:
            style = self.theme.phase_colors.get(__import__("tui.events", fromlist=["Phase"]).Phase.EXPERIMENT)
        elif "error" in low or "traceback" in low or "failed" in low:
            style = "#FF5555"
        if style:
            self.write(f"[{style}]{line}[/]", shrink=False)
        else:
            self.write(line, shrink=False)
