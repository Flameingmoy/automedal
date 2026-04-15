"""Home screen — command-centre landing page for the AutoMedal TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static

from tui.state import AppState
from tui.widgets.command_input import CommandInput
from tui.widgets.recent_activity import RecentActivity
from tui.widgets.status_strip import StatusStrip


_QUICK_HELP = (
    "  [r] run 50  [d] discover  [i] init <slug>  [s] status  "
    "[q] quit  [Tab] autocomplete"
)


class HomeScreen(Screen):
    """Landing page with status strip, recent activity, and command palette."""

    DEFAULT_CSS = """
    HomeScreen {
        layout: vertical;
        background: $background;
    }
    HomeScreen > #home-help {
        height: 1;
        color: $text-muted;
        background: $panel;
        padding: 0 1;
    }
    HomeScreen > RecentActivity {
        margin: 1 1 0 1;
    }
    HomeScreen > CommandInput {
        margin: 1 1 0 1;
    }
    """

    BINDINGS = [
        ("r", "quick_run",      "Run"),
        ("d", "quick_discover", "Discover"),
        ("i", "quick_init",     "Init"),
        ("s", "quick_status",   "Status"),
        ("q", "quit",           "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield StatusStrip(id="home-status")
        yield Static(_QUICK_HELP, id="home-help")
        yield RecentActivity(id="home-recent")
        yield CommandInput(id="home-cmd")
        yield Footer()

    def update_state(self, state: AppState) -> None:
        try:
            self.query_one("#home-status", StatusStrip).update_state(state)
        except Exception:
            pass
        try:
            self.query_one("#home-recent", RecentActivity).update_state(state)
        except Exception:
            pass

    # ── command input handling ─────────────────────────────────────────────

    def on_command_input_submitted(self, event: CommandInput.Submitted) -> None:
        self._dispatch_text(event.value)

    def _dispatch_text(self, text: str) -> None:
        parts = text.strip().split()
        if not parts:
            return
        cmd, *args = parts
        cmd = cmd.lower()

        if cmd == "quit":
            self.app.exit()
        elif cmd == "help":
            self.app.action_show_help()
        else:
            self.app.spawn_command(cmd, args)

    # ── quick-action keybindings ───────────────────────────────────────────

    def action_quick_run(self) -> None:
        self._prefill("run 50")

    def action_quick_discover(self) -> None:
        self.app.spawn_command("discover", [])

    def action_quick_init(self) -> None:
        self._prefill("init ")

    def action_quick_status(self) -> None:
        self.app.spawn_command("status", [])

    def action_quit(self) -> None:
        self.app.exit()

    def _prefill(self, text: str) -> None:
        """Put text into the command input and focus it."""
        try:
            inp = self.query_one("#home-cmd", CommandInput)
            from textual.widgets import Input
            i = inp.query_one("#ci-input", Input)
            i.value = text
            i.cursor_position = len(text)
            i.focus()
        except Exception:
            pass
