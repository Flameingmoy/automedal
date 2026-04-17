"""Confirm-clean modal — asks y/n before wiping memory files and results.tsv."""

from __future__ import annotations

import asyncio
import sys

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Footer, Static

from tui.state import AppState


class ConfirmCleanModal(ModalScreen):
    DEFAULT_CSS = """
    ConfirmCleanModal {
        align: center middle;
    }
    ConfirmCleanModal > #cc-box {
        width: 60;
        height: 9;
        background: #0F111A;
        border: round #FF5555;
        padding: 1 2;
        layout: vertical;
    }
    ConfirmCleanModal > #cc-box > #cc-title {
        height: 1;
        color: #FF5555;
    }
    ConfirmCleanModal > #cc-box > #cc-msg {
        height: 3;
        color: #E1E4E8;
    }
    ConfirmCleanModal > #cc-box > #cc-hint {
        height: 2;
        color: #6272A4;
    }
    """

    BINDINGS = [
        ("y",      "confirm",  "Yes, wipe"),
        ("n",      "cancel",   "No, cancel"),
        ("q",      "cancel",   "Cancel"),
        ("escape", "cancel",   "Cancel"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical
        with Vertical(id="cc-box"):
            yield Static("⚠  Clean — are you sure?", id="cc-title")
            yield Static(
                "This will wipe knowledge.md, experiment_queue.md,\n"
                "research_notes.md, journal/, and results.tsv.",
                id="cc-msg",
            )
            yield Static("  [y] confirm wipe    [n / Esc] cancel", id="cc-hint")

    def action_confirm(self) -> None:
        asyncio.get_event_loop().create_task(self._do_clean())

    async def _do_clean(self) -> None:
        try:
            cmd = [sys.executable, "-m", "automedal", "clean", "--yes"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def update_state(self, state: AppState) -> None:
        pass
