"""Recent experiments — last 5 journal entries sorted by exp_id (newest first)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.state import AppState


class RecentActivity(Vertical):
    DEFAULT_CSS = """
    RecentActivity { height: 9; border: round #6272A4; padding: 0 1; background: #0F111A; }
    RecentActivity > #ra-title { height: 1; color: #8BE9FD; }
    RecentActivity > #ra-body  { height: 7; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()

    def compose(self) -> ComposeResult:
        yield Static("recent activity", id="ra-title")
        yield Static("(no experiments yet)", id="ra-body")

    def update_state(self, state: AppState) -> None:
        self._state = state
        self._paint()

    def _paint(self) -> None:
        exps = sorted(
            self._state.experiments.values(),
            key=lambda e: e.exp_id,
            reverse=True,
        )[:5]

        if not exps:
            body = "(no experiments yet)"
        else:
            lines: list[str] = []
            for e in exps:
                glyph = {"kept": "✓", "reverted": "✗", "crash": "!"}.get(e.status, "·")
                loss_str = f"{e.val_loss:.4f}" if e.val_loss is not None else "   n/a"
                # Compute delta vs best_so_far when available
                delta_str = ""
                if e.best_so_far is not None and e.val_loss is not None:
                    delta = e.val_loss - e.best_so_far
                    sign = "+" if delta >= 0 else ""
                    delta_str = f"({sign}{delta:.4f})"
                slug = (e.slug or "")[:28]
                lines.append(
                    f"#{e.exp_id}  {glyph}  {slug:<28}  {loss_str}  {delta_str}"
                )
            body = "\n".join(lines)

        try:
            self.query_one("#ra-body", Static).update(body)
        except Exception:
            pass
