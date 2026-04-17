"""Select-competition screen — native Textual DataTable picker.

Replaces the interactive input() flow in scout/select.py when launched from
the TUI. Shell `automedal select` still uses the old input() path.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

from tui.state import AppState


class SelectCompetitionScreen(Screen):
    DEFAULT_CSS = """
    SelectCompetitionScreen {
        layout: vertical;
        background: #0F111A;
    }
    SelectCompetitionScreen > #sc-title {
        height: 1;
        background: #0F111A;
        padding: 0 1;
        color: #8BE9FD;
    }
    SelectCompetitionScreen > #sc-hint {
        height: 1;
        color: #6272A4;
        padding: 0 1;
    }
    SelectCompetitionScreen > DataTable {
        height: 1fr;
        margin: 1;
    }
    SelectCompetitionScreen > #sc-status {
        height: 1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("enter",  "select_row",  "Bootstrap"),
        ("q",      "back",        "Back"),
        ("escape", "back",        "Back"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, candidates_json: Path | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []
        self._candidates_json = candidates_json
        self._candidates: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("Select a competition  (Enter to bootstrap, q to cancel)", id="sc-title")
        yield Static("", id="sc-hint")
        yield DataTable(id="sc-table", cursor_type="row")
        yield Static("", id="sc-status")
        yield Footer()

    def on_mount(self) -> None:
        self._load_candidates()

    def _resolve_json(self) -> Path | None:
        if self._candidates_json and self._candidates_json.exists():
            return self._candidates_json
        # Try Layout path
        if self._layout is not None:
            p = self._layout.cwd / "scout" / "outputs" / "competition_candidates.json"
            if p.exists():
                return p
        # Fallback: project-root relative (dev mode)
        from pathlib import Path as _P
        fallback = _P("scout") / "outputs" / "competition_candidates.json"
        if fallback.exists():
            return fallback
        return None

    def _load_candidates(self) -> None:
        table = self.query_one("#sc-table", DataTable)
        table.add_columns("#", "Score", "Category", "Metric", "Teams", "Slug")

        json_path = self._resolve_json()
        if json_path is None:
            self.query_one("#sc-hint", Static).update(
                "[yellow]No candidates.json found — run 'discover' first[/]"
            )
            return

        try:
            data = json.loads(json_path.read_text())
            self._candidates = data.get("candidates", [])
        except Exception as exc:
            self.query_one("#sc-hint", Static).update(f"[red]Error loading candidates: {exc}[/]")
            return

        for i, c in enumerate(self._candidates, 1):
            comp = c["competition"]
            table.add_row(
                str(i),
                str(c.get("final_score", "?")),
                (comp.get("category", "?") or "?")[:15],
                (comp.get("evaluationMetric", "?") or "?")[:12],
                str(comp.get("teamCount", 0)),
                comp.get("ref", "?"),
                key=str(i),
            )

        self.query_one("#sc-hint", Static).update(
            f"  {len(self._candidates)} competitions loaded — use ↑↓ to navigate"
        )
        if self._candidates:
            table.focus()

    def action_select_row(self) -> None:
        table = self.query_one("#sc-table", DataTable)
        row_key = table.cursor_row
        if row_key is None or row_key >= len(self._candidates):
            return
        candidate = self._candidates[row_key]
        slug = candidate["competition"].get("ref", "")
        if not slug:
            return
        self.query_one("#sc-status", Static).update(f"  Bootstrapping {slug}…")
        self.app.spawn_command("init", [slug])
        self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()

    def update_state(self, state: AppState) -> None:
        pass
