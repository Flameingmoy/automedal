"""Dashboard screen — live monitoring panels + stream."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen

from tui.state import AppState
from tui.widgets.sprite_panel import SpritePanel
from tui.widgets.metric_chart import MetricChart
from tui.widgets.leaderboard import Leaderboard
from tui.widgets.experiment_log import ExperimentLog
from tui.widgets.current_experiment import CurrentExperiment
from tui.widgets.gpu_stats import GpuStats
from tui.widgets.session_stats import SessionStats
from tui.widgets.live_stream import LiveStream

_WIDGET_IDS = ("sprite", "metric", "leaderboard", "explog", "current", "gpu", "session")


class DashboardScreen(Screen):
    BINDINGS = [
        # 'q' returns to home (or quits if home isn't in the stack)
        ("q",      "back_or_quit", "Home"),
        # ctrl+c cancels an active run and returns to home
        ("ctrl+c", "cancel_run",   "Cancel run"),
    ]

    def compose(self) -> ComposeResult:
        yield SpritePanel(id="sprite")
        yield MetricChart(id="metric")
        yield Leaderboard(id="leaderboard")
        yield ExperimentLog(id="explog")
        with Vertical(id="right-stack"):
            yield CurrentExperiment(id="current")
            yield GpuStats(id="gpu")
            yield SessionStats(id="session")
        yield LiveStream(id="stream")

    def update_state(self, state: AppState) -> None:
        """Called by AutoMedalApp._broadcast whenever state changes."""
        for wid in _WIDGET_IDS:
            try:
                w = self.query_one(f"#{wid}")
                if hasattr(w, "update_state"):
                    w.update_state(state)
            except Exception:
                pass

    def push_line(self, text: str) -> None:
        """Forward a raw log line to the live stream widget."""
        try:
            self.query_one("#stream", LiveStream).push_line(text)
        except Exception:
            pass

    def action_back_or_quit(self) -> None:
        """Return to home screen if it exists, otherwise quit the app."""
        from tui.screens.home import HomeScreen
        has_home = any(isinstance(s, HomeScreen) for s in self.app.screen_stack)
        if has_home:
            self.app.pop_screen()
        else:
            self.app.exit()

    def action_cancel_run(self) -> None:
        """SIGTERM the active run subprocess and return to home."""
        import asyncio
        proc = getattr(self.app, "_run_proc", None)
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                # Show a banner in the log stream
                self.push_line("[harness] cancel requested — waiting for iteration to finish…")
            except Exception:
                pass
        else:
            # No active run — just go back
            self.action_back_or_quit()
