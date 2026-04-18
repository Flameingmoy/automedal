"""Command input box with Tab autocomplete and live hints — sits at the bottom of HomeScreen."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Static

COMMANDS = [
    "run", "init", "discover", "select", "doctor", "status",
    "clean", "prepare", "render", "setup", "help", "quit",
]

_TRAILING_DIGITS = re.compile(r"^([a-zA-Z]+)(\d+)$")


def normalize(text: str) -> str:
    """Forgiving parser: 'run30' → 'run 30'; leaves 'run 30' untouched."""
    text = text.strip()
    if not text:
        return text
    first, _, rest = text.partition(" ")
    m = _TRAILING_DIGITS.match(first)
    if m and m.group(1).lower() in COMMANDS:
        fixed = f"{m.group(1)} {m.group(2)}"
        return f"{fixed} {rest}".strip() if rest else fixed
    return text


class CommandInput(Vertical):
    """An `>` prompt with inline suggester, live hints row, and a Submitted message."""

    DEFAULT_CSS = """
    CommandInput {
        height: 4;
        border: round #6272A4;
        padding: 0 1;
        background: #0F111A;
    }
    CommandInput > #ci-row {
        height: 1;
    }
    CommandInput > #ci-row > #ci-prompt { width: 2; color: #50FA7B; }
    CommandInput > #ci-row > #ci-input  { width: 1fr; border: none; background: transparent; }
    CommandInput > #ci-hints { height: 1; color: #6272A4; padding: 0 0 0 2; }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        with Horizontal(id="ci-row"):
            yield Static(">", id="ci-prompt")
            yield Input(
                placeholder="type a command (e.g. run 30)",
                id="ci-input",
            )
        yield Static(f"  {'  '.join(COMMANDS[:6])}", id="ci-hints")

    def on_mount(self) -> None:
        self.query_one("#ci-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        text = normalize(event.value or "")
        word = (text.split() or [""])[0].lower()
        if not word:
            hint = "  ".join(COMMANDS[:6])
        else:
            matches = [c for c in COMMANDS if c.startswith(word)]
            hint = "  ".join(matches[:6]) if matches else "(no matches)"
        try:
            self.query_one("#ci-hints", Static).update(f"  {hint}")
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = normalize(event.value)
        if value:
            self.post_message(self.Submitted(value))
            inp = self.query_one("#ci-input", Input)
            inp.value = ""
            try:
                self.query_one("#ci-hints", Static).update(f"  {'  '.join(COMMANDS[:6])}")
            except Exception:
                pass

    def on_key(self, event) -> None:
        if event.key == "tab":
            event.stop()
            self._autocomplete()

    def _autocomplete(self) -> None:
        inp = self.query_one("#ci-input", Input)
        text = normalize(inp.value)
        if not text:
            return
        first, _, rest = text.partition(" ")
        word = first.lower()
        matches = [c for c in COMMANDS if c.startswith(word)]
        if len(matches) == 1:
            inp.value = matches[0] + (f" {rest}" if rest else "")
            inp.cursor_position = len(inp.value)
