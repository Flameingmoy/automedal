"""AutoMedalApp — composes sources → bus → state reducer → widgets."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.widgets import Header, Footer

from tui.bus import EventBus
from tui.events import CompetitionInfo, HeartBeat, RawLine, Phase
from tui.screens.dashboard import DashboardScreen
from tui.screens.raw_stream import RawStreamScreen
from tui.state import AppState, PhaseMachine
from tui.themes import tcss_path
from tui.themes.palette import DEFAULT_THEME

from tui.sources import log_tail, journal, results as results_src, memory, gpu, demo as demo_src


def _load_competition_info(repo_root: Path) -> CompetitionInfo:
    p = repo_root / "configs" / "competition.yaml"
    if not p.exists():
        return CompetitionInfo()
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        comp = data.get("competition", {}) or {}
        return CompetitionInfo(slug=str(comp.get("slug") or ""), title=str(comp.get("title") or ""))
    except Exception:
        return CompetitionInfo()


class AutoMedalApp(App):
    CSS_PATH = str(tcss_path("dark"))
    TITLE = "AutoMedal"
    BINDINGS = [
        ("q", "quit", "quit"),
        ("ctrl+o", "toggle_raw", "raw stream"),
        ("p", "toggle_pause", "pause"),
        ("slash", "focus_filter", "filter"),
    ]

    def __init__(
        self,
        repo_root: Path,
        *,
        log_file: Optional[Path] = None,
        demo_fixture: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.log_file = log_file or (repo_root / "agent_loop.log")
        self.demo_fixture = demo_fixture
        self.bus = EventBus()
        self.state = AppState()
        self.state.competition = _load_competition_info(repo_root)
        self._source_tasks: list[asyncio.Task] = []

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(name="dashboard"))
        self._start_sources()
        self.set_interval(1.0, self._heartbeat)

    def _start_sources(self) -> None:
        loop = asyncio.get_event_loop()
        if self.demo_fixture:
            self._source_tasks.append(loop.create_task(demo_src.run(self.bus, self.demo_fixture)))
        else:
            self._source_tasks.append(loop.create_task(log_tail.run(self.bus, self.log_file)))
            self._source_tasks.append(loop.create_task(journal.run(self.bus, self.repo_root / "journal")))
            self._source_tasks.append(loop.create_task(results_src.run(self.bus, self.repo_root / "agent" / "results.tsv")))
            self._source_tasks.append(loop.create_task(memory.run(self.bus, self.repo_root)))
            self._source_tasks.append(loop.create_task(gpu.run(self.bus)))
        self._source_tasks.append(loop.create_task(self._consume()))

    def _heartbeat(self) -> None:
        self.bus.publish_nowait(HeartBeat(ts=time.time()))

    async def _consume(self) -> None:
        q = self.bus.subscribe()
        while True:
            event = await q.get()
            self.state = PhaseMachine.reduce(self.state, event)
            self._broadcast(event)

    def _broadcast(self, event) -> None:
        dashboard = self._find_dashboard()
        if dashboard is None:
            return
        for wid in ("sprite", "metric", "leaderboard", "explog", "current", "gpu", "session"):
            try:
                w = dashboard.query_one(f"#{wid}")
                if hasattr(w, "update_state"):
                    w.update_state(self.state)
            except Exception:
                pass
        if isinstance(event, RawLine):
            try:
                dashboard.query_one("#stream").push_line(event.text)
            except Exception:
                pass

        comp = self.state.competition
        try:
            if comp.slug:
                self.sub_title = f"{comp.slug} · run {self.state.iteration}/{self.state.total_iterations}"
        except Exception:
            pass

    def _find_dashboard(self):
        for s in reversed(self.screen_stack):
            if isinstance(s, DashboardScreen):
                return s
        return None

    def action_toggle_raw(self) -> None:
        top = self.screen_stack[-1] if self.screen_stack else None
        if isinstance(top, RawStreamScreen):
            self.pop_screen()
        else:
            self.push_screen(RawStreamScreen(name="raw"))

    def action_toggle_pause(self) -> None:
        dash = self._find_dashboard()
        if dash is None:
            return
        try:
            dash.query_one("#stream").toggle_pause()
        except Exception:
            pass

    def action_focus_filter(self) -> None:
        # Filter UX is TODO; binding reserved so /  doesn't fall through to OS.
        self.bell()
