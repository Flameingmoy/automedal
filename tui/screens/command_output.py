"""Generic command output screen — streams a subprocess's stdout into a RichLog."""

from __future__ import annotations

import asyncio
import sys

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static


class CommandOutputScreen(Screen):
    """Runs `automedal <cmd> <args>` and streams output.  `q` / Escape returns to home."""

    DEFAULT_CSS = """
    CommandOutputScreen {
        layout: vertical;
        background: $background;
    }
    CommandOutputScreen > #co-title {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $accent;
    }
    CommandOutputScreen > RichLog {
        border: round $panel;
        margin: 1;
    }
    """

    BINDINGS = [
        ("q",      "back", "Back"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, cmd: str, args: list[str], **kw) -> None:
        super().__init__(**kw)
        self._cmd = cmd
        self._args = list(args)
        self._proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"automedal {self._cmd} {' '.join(self._args)}  (q to return)",
            id="co-title",
        )
        yield RichLog(id="co-log", markup=False, highlight=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        asyncio.get_event_loop().create_task(self._stream())

    async def _stream(self) -> None:
        log = self.query_one("#co-log", RichLog)
        cmd_args = [sys.executable, "-m", "automedal", self._cmd] + self._args
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert self._proc.stdout is not None
            async for raw in self._proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                log.write(line)
            await self._proc.wait()
        except Exception as exc:
            log.write(f"[error] {exc}")
        log.write("─── done  (press q to return) ───")

    def action_back(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.app.pop_screen()
