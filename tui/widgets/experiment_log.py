"""Scrollable table of experiments with status badges."""

from __future__ import annotations

from textual.widgets import DataTable

from tui.state import AppState


STATUS_GLYPH = {
    "running": "[yellow]RUN[/]",
    "kept": "[green]KEPT ✓[/]",
    "reverted": "[red]REV ✗[/]",
    "crash": "[red]CRASH[/]",
}


class ExperimentLog(DataTable):
    def __init__(self, **kw) -> None:
        super().__init__(zebra_stripes=True, cursor_type="row", **kw)
        self._state = AppState()

    def on_mount(self) -> None:
        self.add_columns("#", "status", "slug", "val_loss", "Δ")

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.clear()
        prev_loss = None
        for exp_id in state.experiment_order[-50:]:
            exp = state.experiments.get(exp_id)
            if exp is None:
                continue
            status = STATUS_GLYPH.get(exp.status, exp.status)
            loss = f"{exp.val_loss:.4f}" if exp.val_loss is not None else "—"
            if exp.val_loss is not None and prev_loss is not None:
                d = exp.val_loss - prev_loss
                delta = f"[green]{d:+.4f}[/]" if d < 0 else f"[red]{d:+.4f}[/]"
            else:
                delta = "—"
            self.add_row(exp_id, status, exp.slug or "—", loss, delta)
            if exp.val_loss is not None:
                prev_loss = exp.val_loss
