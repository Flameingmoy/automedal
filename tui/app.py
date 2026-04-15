"""AutoMedalApp — composes sources → bus → state reducer → screens."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

from textual.app import App

from tui.bus import EventBus
from tui.events import CompetitionInfo, HeartBeat, RawLine
from tui.screens.dashboard import DashboardScreen
from tui.screens.home import HomeScreen
from tui.screens.raw_stream import RawStreamScreen
from tui.state import AppState, PhaseMachine
from tui.themes import tcss_path
from tui.themes.palette import DEFAULT_THEME

from tui.sources import log_tail, journal, results as results_src, memory, gpu, demo as demo_src


def _load_competition_info(config_yaml: Path) -> CompetitionInfo:
    if not config_yaml.exists():
        return CompetitionInfo()
    try:
        import yaml
        data = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
        comp = data.get("competition", {}) or {}
        return CompetitionInfo(
            slug=str(comp.get("slug") or ""),
            title=str(comp.get("title") or ""),
        )
    except Exception:
        return CompetitionInfo()


class AutoMedalApp(App):
    CSS_PATH = str(tcss_path("dark"))
    TITLE = "AutoMedal"
    BINDINGS = [
        ("ctrl+o", "toggle_raw",    "raw stream"),
        ("p",      "toggle_pause",  "pause"),
        ("slash",  "focus_filter",  "filter"),
    ]
    # Note: 'q' is intentionally NOT bound at the app level;
    # HomeScreen binds it to quit, DashboardScreen binds it to "back/home".

    def __init__(
        self,
        repo_root: Path,
        *,
        log_file: Optional[Path] = None,
        demo_fixture: Optional[Path] = None,
        start_dashboard: bool = False,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root

        # Resolve paths via Layout when available
        try:
            from automedal.paths import Layout
            layout = Layout(cwd=repo_root)
            self._layout = layout
            self.log_file = log_file or layout.log_file
            self._config_yaml = layout.config_yaml
            self._journal_dir = layout.journal_dir
            self._results_tsv = layout.results_tsv
            self._memory_root = layout.cwd
        except ImportError:
            # TUI used standalone without automedal package
            self._layout = None
            self.log_file = log_file or (repo_root / "agent_loop.log")
            self._config_yaml = repo_root / "configs" / "competition.yaml"
            self._journal_dir = repo_root / "journal"
            self._results_tsv = repo_root / "agent" / "results.tsv"
            self._memory_root = repo_root

        self.demo_fixture = demo_fixture
        self.start_dashboard = start_dashboard  # True → skip home, go straight to dashboard

        self.bus = EventBus()
        self.state = AppState()
        self.state.competition = _load_competition_info(self._config_yaml)

        self._source_tasks: list[asyncio.Task] = []
        self._run_proc: Optional[asyncio.subprocess.Process] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        if self.start_dashboard or self.demo_fixture:
            # Legacy / demo mode: go straight to dashboard (old ./am tui behaviour)
            self.push_screen(HomeScreen(name="home"))
            self.push_screen(DashboardScreen(name="dashboard"))
        else:
            self.push_screen(HomeScreen(name="home"))

        self._start_sources()
        self.set_interval(1.0, self._heartbeat)

    def _start_sources(self) -> None:
        loop = asyncio.get_event_loop()
        if self.demo_fixture:
            self._source_tasks.append(
                loop.create_task(demo_src.run(self.bus, self.demo_fixture))
            )
        else:
            self._source_tasks.append(
                loop.create_task(log_tail.run(self.bus, self.log_file))
            )
            self._source_tasks.append(
                loop.create_task(journal.run(self.bus, self._journal_dir))
            )
            self._source_tasks.append(
                loop.create_task(results_src.run(self.bus, self._results_tsv))
            )
            self._source_tasks.append(
                loop.create_task(memory.run(self.bus, self._memory_root))
            )
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

    # ── state broadcast ───────────────────────────────────────────────────

    def _broadcast(self, event) -> None:
        """Push new state to whatever screen is currently on top."""
        # Update ALL screens in the stack so they're ready when popped back
        for screen in self.screen_stack:
            if hasattr(screen, "update_state"):
                try:
                    screen.update_state(self.state)
                except Exception:
                    pass

        # Raw lines go to the dashboard stream wherever it is in the stack
        if isinstance(event, RawLine):
            dash = self._find_dashboard()
            if dash is not None:
                try:
                    dash.push_line(event.text)
                except Exception:
                    pass

        # Update app subtitle from competition info
        comp = self.state.competition
        try:
            if comp.slug:
                self.sub_title = (
                    f"{comp.slug} · {self.state.iteration}/{self.state.total_iterations}"
                )
        except Exception:
            pass

    def _find_dashboard(self) -> Optional[DashboardScreen]:
        for s in reversed(self.screen_stack):
            if isinstance(s, DashboardScreen):
                return s
        return None

    # ── command spawning ──────────────────────────────────────────────────

    def spawn_command(self, cmd: str, args: list[str]) -> None:
        """Dispatch a command from the home screen."""
        asyncio.get_event_loop().create_task(self._spawn_async(cmd, args))

    def _custom_screen(self, cmd: str, args: list[str]):
        """Return a custom Screen for cmd if one exists, else None."""
        layout = self._layout
        try:
            from tui.screens.select_competition import SelectCompetitionScreen
            from tui.screens.setup_wizard import SetupWizardScreen
            from tui.screens.discover import DiscoverScreen
            from tui.screens.status import StatusScreen
            from tui.screens.init_progress import InitProgressScreen
            from tui.screens.confirm_clean import ConfirmCleanModal
            from tui.screens.doctor import DoctorScreen

            mapping = {
                "select":    lambda: SelectCompetitionScreen(layout=layout, args=args, name="select"),
                "setup":     lambda: SetupWizardScreen(layout=layout, args=args, name="setup"),
                "discover":  lambda: DiscoverScreen(layout=layout, args=args, name="discover"),
                "status":    lambda: StatusScreen(layout=layout, args=args, name="status"),
                "init":      lambda: InitProgressScreen(layout=layout, args=args, name="init"),
                "bootstrap": lambda: InitProgressScreen(layout=layout, args=args, name="bootstrap"),
                "clean":     lambda: ConfirmCleanModal(layout=layout, args=args, name="clean"),
                "doctor":    lambda: DoctorScreen(layout=layout, args=args, name="doctor"),
            }
            factory = mapping.get(cmd)
            return factory() if factory else None
        except ImportError:
            return None

    async def _spawn_async(self, cmd: str, args: list[str]) -> None:
        if cmd == "run":
            await self._spawn_run(args)
        else:
            screen = self._custom_screen(cmd, args)
            if screen is not None:
                self.push_screen(screen)
            else:
                from tui.screens.command_output import CommandOutputScreen
                self.push_screen(CommandOutputScreen(cmd=cmd, args=args, name="cmd-output"))

    async def _spawn_run(self, args: list[str]) -> None:
        n = args[0] if args else "50"

        env = dict(os.environ)
        if self._layout is not None:
            env.update(self._layout.as_env())
        pi_bin = env.get("AUTOMEDAL_PI_BIN", "")
        if not pi_bin:
            try:
                from automedal.pi_runtime import ensure_pi
                env["AUTOMEDAL_PI_BIN"] = str(ensure_pi())
            except Exception:
                pass

        # Spawn `python -m automedal run N` — dashboard tails the log file
        self._run_proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "automedal", "run", n,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        self.push_screen(DashboardScreen(name="dashboard"))

        # Wait for the subprocess to finish, then return to home
        await self._run_proc.wait()
        self._run_proc = None

        # Pop back to home if dashboard is still on top
        if self.screen_stack and isinstance(self.screen_stack[-1], DashboardScreen):
            self.pop_screen()

    # ── actions ───────────────────────────────────────────────────────────

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
        self.bell()

    def action_show_help(self) -> None:
        from tui.screens.command_output import CommandOutputScreen
        self.push_screen(CommandOutputScreen(cmd="help", args=[], name="help-output"))
