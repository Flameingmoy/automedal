"""val_loss over iterations. Uses textual-plotext if available, else a simple Static sparkline."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.state import AppState

try:
    from textual_plotext import PlotextPlot
    _HAS_PLOTEXT = True
except ImportError:
    _HAS_PLOTEXT = False


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[0] * len(values)
    span = hi - lo
    out = []
    for v in values:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


class MetricChart(Vertical):
    DEFAULT_CSS = """
    MetricChart { height: 1fr; padding: 0 1; }
    MetricChart > #metric-title { height: 1; }
    MetricChart > #metric-body { height: 1fr; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()

    def compose(self) -> ComposeResult:
        yield Static("val_loss", id="metric-title")
        if _HAS_PLOTEXT:
            yield PlotextPlot(id="metric-body")
        else:
            yield Static("", id="metric-body")

    def update_state(self, state: AppState) -> None:
        self._state = state
        values = [v for _, v in state.val_losses if v is not None]
        try:
            if _HAS_PLOTEXT:
                plot = self.query_one("#metric-body", PlotextPlot)
                plot.plt.clear_figure()
                if values:
                    plot.plt.plot(list(range(1, len(values) + 1)), values, marker="hd")
                    plot.plt.ylabel("loss")
                    if state.best_val_loss != float("inf"):
                        plot.plt.hline(state.best_val_loss)
                plot.refresh()
            else:
                sp = _spark(values[-60:])
                best = "" if state.best_val_loss == float("inf") else f"best={state.best_val_loss:.4f}"
                self.query_one("#metric-body", Static).update(f"{sp}\n{best}")
            title = "val_loss"
            if values:
                title = f"val_loss  latest={values[-1]:.4f}  n={len(values)}"
            self.query_one("#metric-title", Static).update(title)
        except Exception:
            pass
