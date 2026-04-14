"""Top-5 experiments by val_loss with medals for top-3."""

from __future__ import annotations

from textual.widgets import Static
from textual.containers import Vertical
from textual.app import ComposeResult

from tui.state import AppState
from tui.themes.palette import DEFAULT_THEME, Theme


MEDALS = ("🥇", "🥈", "🥉")


class Leaderboard(Vertical):
    DEFAULT_CSS = """
    Leaderboard { height: auto; border: round $panel; padding: 0 1; }
    Leaderboard > #lb-title { height: 1; color: $accent; }
    Leaderboard > #lb-body { height: auto; }
    """

    def __init__(self, theme: Theme = DEFAULT_THEME, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()
        self.theme = theme

    def compose(self) -> ComposeResult:
        yield Static("leaderboard", id="lb-title")
        yield Static("no experiments yet", id="lb-body")

    def update_state(self, state: AppState) -> None:
        self._state = state
        top = state.top_n(5)
        if not top:
            body = "no experiments yet"
        else:
            lines = []
            for i, exp in enumerate(top):
                medal = MEDALS[i] if i < 3 else f" #{i+1}"
                lines.append(f"{medal} #{exp.exp_id} {exp.val_loss:.4f} {exp.slug[:20]}")
            body = "\n".join(lines)
        try:
            self.query_one("#lb-body", Static).update(body)
            title = "leaderboard"
            if state.new_best_toast_for:
                title = f"[b {self.theme.medal_gold}]🥇 NEW GOLD · #{state.new_best_toast_for}[/]"
            self.query_one("#lb-title", Static).update(title)
        except Exception:
            pass
