"""Discover screen — streams `automedal discover` then shows a DataTable of results."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, RichLog, Static

from tui.state import AppState

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]|\x1B\][^\x07]*\x07")


def _clean(line: str) -> str:
    """Strip ANSI and unprintable control chars before rendering."""
    line = _ANSI_RE.sub("", line)
    return "".join(ch for ch in line if ch == "\t" or ch >= " ")


class DiscoverScreen(Screen):
    DEFAULT_CSS = """
    DiscoverScreen {
        layout: vertical;
        background: #0F111A;
    }
    DiscoverScreen > #dc-title {
        height: 1;
        background: #0F111A;
        padding: 0 1;
        color: #8BE9FD;
    }
    DiscoverScreen > #dc-hint {
        height: 1;
        color: #6272A4;
        padding: 0 1;
    }
    DiscoverScreen > DataTable {
        height: 1fr;
        margin: 1 1 0 1;
    }
    DiscoverScreen > #dc-log {
        height: 8;
        margin: 0 1 0 1;
        border: round #6272A4;
    }
    """

    BINDINGS = [
        ("s",      "select_screen", "Select"),
        ("r",      "refresh",       "Re-discover"),
        ("q",      "back",          "Back"),
        ("escape", "back",          "Back"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []
        self._proc: asyncio.subprocess.Process | None = None
        self._candidates: list[dict] = []
        self._json_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Static("Discover competitions", id="dc-title")
        yield Static("  Running discover…  (s=select, r=re-run, q=back)", id="dc-hint")
        yield DataTable(id="dc-table", cursor_type="none")
        yield RichLog(id="dc-log", markup=False, highlight=False, wrap=True, max_lines=500)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#dc-table", DataTable)
        table.add_columns("#", "Score", "Category", "Metric", "Teams", "Slug")
        asyncio.get_event_loop().create_task(self._run_discover())

    async def _run_discover(self) -> None:
        log = self.query_one("#dc-log", RichLog)
        log.write("Starting discover…")
        cmd = [sys.executable, "-m", "automedal", "discover"] + self._args
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"},
            )
            assert self._proc.stdout is not None
            async for raw in self._proc.stdout:
                text = _clean(raw.decode(errors="replace").rstrip())
                if text:
                    log.write(text)
            await self._proc.wait()
        except Exception as exc:
            log.write(f"[error] {exc}")
        log.write("─── discover done ───")
        self._load_table()

    def _resolve_json(self) -> Path | None:
        if self._layout is not None:
            p = self._layout.cwd / "scout" / "outputs" / "competition_candidates.json"
            if p.exists():
                return p
        fallback = Path("scout") / "outputs" / "competition_candidates.json"
        if fallback.exists():
            return fallback
        return None

    def _load_table(self) -> None:
        json_path = self._resolve_json()
        if json_path is None:
            self.query_one("#dc-hint", Static).update("  [yellow]No candidates.json found[/]")
            return
        self._json_path = json_path
        try:
            data = json.loads(json_path.read_text())
            self._candidates = data.get("candidates", [])
        except Exception as exc:
            self.query_one("#dc-hint", Static).update(f"  [red]Error: {exc}[/]")
            return

        table = self.query_one("#dc-table", DataTable)
        table.clear()
        table.cursor_type = "row" if self._candidates else "none"
        for i, c in enumerate(self._candidates, 1):
            comp = c["competition"]
            table.add_row(
                str(i),
                str(c.get("final_score", "?")),
                (comp.get("category", "?") or "?")[:15],
                (comp.get("evaluationMetric", "?") or "?")[:12],
                str(comp.get("teamCount", 0)),
                comp.get("ref", "?"),
            )

        self.query_one("#dc-hint", Static).update(
            f"  {len(self._candidates)} competitions  —  s=select highlighted, r=re-run, q=back"
        )
        if self._candidates:
            table.focus()

    def action_select_screen(self) -> None:
        from tui.screens.select_competition import SelectCompetitionScreen
        self.app.push_screen(
            SelectCompetitionScreen(
                layout=self._layout,
                candidates_json=self._json_path,
                name="select",
            )
        )

    def action_refresh(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        log = self.query_one("#dc-log", RichLog)
        log.clear()
        table = self.query_one("#dc-table", DataTable)
        table.clear()
        asyncio.get_event_loop().create_task(self._run_discover())

    def action_back(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.app.pop_screen()

    def update_state(self, state: AppState) -> None:
        pass
