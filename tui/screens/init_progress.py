"""Init progress screen — staged checklist for `automedal init <slug>`."""

from __future__ import annotations

import asyncio
import sys

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from tui.state import AppState


# Map line-prefix patterns from bootstrap.py to step names
_STEPS = [
    ("validate",  "1. Validate slug"),
    ("download",  "2. Download data"),
    ("sniff",     "3. Sniff CSV schema"),
    ("build",     "4. Build config"),
    ("render",    "5. Render templates"),
    ("prepare",   "6. Run prepare.py"),
]

_PENDING  = "⏳"
_OK       = "✓"
_FAIL     = "✗"


class InitProgressScreen(Screen):
    DEFAULT_CSS = """
    InitProgressScreen {
        layout: vertical;
        background: $background;
    }
    InitProgressScreen > #ip-title {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $accent;
    }
    InitProgressScreen > #ip-steps {
        height: auto;
        margin: 1 2;
    }
    InitProgressScreen > #ip-log {
        height: 1fr;
        margin: 0 1;
        border: round $panel;
    }
    """

    BINDINGS = [
        ("q",      "back", "Back"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._slug = args[0] if args else ""
        self._args = args or []
        self._proc: asyncio.subprocess.Process | None = None
        self._step_widgets: list[Static] = []
        self._current_step = -1

    def compose(self) -> ComposeResult:
        slug = self._slug or "<no slug>"
        yield Static(f"Init: {slug}  (q to return)", id="ip-title")
        from textual.containers import Vertical
        with Vertical(id="ip-steps"):
            for _, label in _STEPS:
                w = Static(f"  {_PENDING}  {label}", classes="ip-step")
                self._step_widgets.append(w)
                yield w
        yield RichLog(id="ip-log", markup=False, highlight=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        if not self._slug:
            log = self.query_one("#ip-log", RichLog)
            log.write("[error] No slug provided — use 'init <slug>'")
            return
        asyncio.get_event_loop().create_task(self._run_init())

    async def _run_init(self) -> None:
        log = self.query_one("#ip-log", RichLog)
        cmd = [sys.executable, "-m", "automedal", "init", self._slug, "--yes"]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert self._proc.stdout is not None
            async for raw in self._proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                log.write(line)
                self._update_steps(line)
            rc = await self._proc.wait()
        except Exception as exc:
            log.write(f"[error] {exc}")
            rc = 1

        marker = "✓ Done" if rc == 0 else "✗ Failed"
        log.write(f"─── {marker}  (press q to return) ───")
        self._finalize(rc == 0)

    def _update_steps(self, line: str) -> None:
        lower = line.lower()
        for i, (key, _) in enumerate(_STEPS):
            if key in lower and i > self._current_step:
                # Mark previous as done
                if self._current_step >= 0:
                    self._set_step(self._current_step, _OK)
                self._current_step = i
                self._set_step(i, _PENDING)
                break

    def _finalize(self, success: bool) -> None:
        glyph = _OK if success else _FAIL
        for i in range(self._current_step + 1):
            self._set_step(i, glyph if i == self._current_step else _OK)

    def _set_step(self, idx: int, glyph: str) -> None:
        if 0 <= idx < len(self._step_widgets):
            _, label = _STEPS[idx]
            try:
                self._step_widgets[idx].update(f"  {glyph}  {label}")
            except Exception:
                pass

    def action_back(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self.app.pop_screen()

    def update_state(self, state: AppState) -> None:
        pass
