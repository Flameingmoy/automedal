"""Three bars for GPU util / VRAM / temp. Hides if no sample has ever arrived."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.state import AppState


def _bar(pct: float, width: int = 16) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100.0 * width)
    return "█" * filled + "·" * (width - filled)


class GpuStats(Vertical):
    DEFAULT_CSS = """
    GpuStats { height: 7; border: round $panel; padding: 0 1; }
    GpuStats > #gpu-title { height: 1; color: $accent; }
    GpuStats > #gpu-body { height: 3; }
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._state = AppState()

    def compose(self) -> ComposeResult:
        yield Static("gpu", id="gpu-title")
        yield Static("(no nvidia-smi)", id="gpu-body")

    def update_state(self, state: AppState) -> None:
        self._state = state
        if state.gpu is None:
            text = "(no nvidia-smi)"
        else:
            g = state.gpu
            vram_pct = (g.mem_used_mb / g.mem_total_mb * 100.0) if g.mem_total_mb else 0.0
            text = (
                f"util {g.util_pct:5.1f}% [{_bar(g.util_pct)}]\n"
                f"vram {vram_pct:5.1f}% [{_bar(vram_pct)}]  {int(g.mem_used_mb)}/{int(g.mem_total_mb)} MB\n"
                f"temp {g.temp_c:5.1f}°C"
            )
        try:
            self.query_one("#gpu-body", Static).update(text)
        except Exception:
            pass
