"""Fullscreen raw log-stream view, toggled by ctrl+o."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import RichLog


class RawStreamScreen(Screen):
    BINDINGS = [
        ("escape", "app.pop_screen", "back"),
        ("ctrl+o", "app.pop_screen", "back"),
    ]

    def compose(self) -> ComposeResult:
        yield RichLog(id="raw-log", max_lines=20000, wrap=False, highlight=False, markup=False)

    def push_line(self, line: str) -> None:
        try:
            self.query_one("#raw-log", RichLog).write(line)
        except Exception:
            pass
