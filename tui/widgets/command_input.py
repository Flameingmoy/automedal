"""Command input box with Tab autocomplete — sits at the bottom of HomeScreen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Input, Static

COMMANDS = [
    "run", "init", "discover", "select", "doctor", "status",
    "clean", "prepare", "render", "setup", "help", "quit",
]


class CommandInput(Horizontal):
    """An `>` prompt with tab-completion and a Submitted message."""

    DEFAULT_CSS = """
    CommandInput {
        height: 3;
        border: round $panel;
        padding: 0 1;
        align: left middle;
    }
    CommandInput > #ci-prompt { width: 2; color: $accent; }
    CommandInput > #ci-input  { width: 1fr; border: none; background: transparent; }
    """

    class Submitted(Message):
        """Emitted when the user presses Enter."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        yield Static(">", id="ci-prompt")
        yield Input(placeholder="type a command…", id="ci-input")

    def on_mount(self) -> None:
        self.query_one("#ci-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if value:
            self.post_message(self.Submitted(value))
            # Clear after submit
            self.query_one("#ci-input", Input).value = ""

    def on_key(self, event) -> None:
        if event.key == "tab":
            event.stop()
            self._autocomplete()

    def _autocomplete(self) -> None:
        inp = self.query_one("#ci-input", Input)
        text = inp.value.strip()
        if not text:
            return
        word = text.split()[0].lower()
        matches = [c for c in COMMANDS if c.startswith(word)]
        if len(matches) == 1:
            rest = text[len(word):]
            inp.value = matches[0] + rest
            inp.cursor_position = len(inp.value)
        elif len(matches) > 1:
            # Show hint but don't complete (ambiguous)
            pass
