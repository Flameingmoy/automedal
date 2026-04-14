"""Dashboard screen — panels + live stream."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from tui.widgets.sprite_panel import SpritePanel
from tui.widgets.metric_chart import MetricChart
from tui.widgets.leaderboard import Leaderboard
from tui.widgets.experiment_log import ExperimentLog
from tui.widgets.current_experiment import CurrentExperiment
from tui.widgets.gpu_stats import GpuStats
from tui.widgets.session_stats import SessionStats
from tui.widgets.live_stream import LiveStream


class DashboardScreen(Screen):
    def compose(self) -> ComposeResult:
        with Horizontal(id="top-row"):
            yield SpritePanel(id="sprite")
            yield MetricChart(id="metric")
            yield Leaderboard(id="leaderboard")
        with Horizontal(id="middle-row"):
            yield ExperimentLog(id="explog")
            with Vertical(id="right-stack"):
                yield CurrentExperiment(id="current")
                yield GpuStats(id="gpu")
                yield SessionStats(id="session")
        yield LiveStream(id="stream")
