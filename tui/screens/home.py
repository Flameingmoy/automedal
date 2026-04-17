"""Home screen — command-centre landing page for the AutoMedal TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static

from tui.state import AppState
from tui.widgets.command_input import CommandInput
from tui.widgets.recent_activity import RecentActivity
from tui.widgets.status_strip import StatusStrip


class HomeScreen(Screen):
    """Landing page with status strip, recent activity, and command palette."""

    DEFAULT_CSS = """
    HomeScreen {
        layout: vertical;
        background: #0F111A;
    }
    HomeScreen > #home-logo {
        height: auto;
        padding: 1 1 0 1;
        color: #FFD700;
    }
    HomeScreen > #home-logo.hidden { display: none; }
    HomeScreen > #home-banner {
        height: auto;
        color: #FFD700;
        padding: 0 1;
    }
    HomeScreen > #home-banner.hidden { display: none; }
    HomeScreen > RecentActivity {
        margin: 1 1 0 1;
    }
    HomeScreen > CommandInput {
        margin: 1 1 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("", id="home-logo")
        yield StatusStrip(id="home-status")
        yield Static("", id="home-banner", classes="hidden")
        yield RecentActivity(id="home-recent")
        yield CommandInput(id="home-cmd")
        yield Footer()

    def on_mount(self) -> None:
        self._render_logo()
        self.call_after_refresh(self._check_first_run)

    def _render_logo(self) -> None:
        try:
            from rich_pixels import Pixels
            from tui.assets.logo.generate_logo import ensure_logo
            logo_path = ensure_logo()
            if logo_path.exists():
                self.query_one("#home-logo", Static).update(
                    Pixels.from_image_path(str(logo_path))
                )
                return
        except Exception:
            pass
        try:
            self.query_one("#home-logo", Static).update("A U T O M E D A L")
        except Exception:
            pass

    def _check_first_run(self) -> None:
        """Auto-push setup wizard on first run (no provider configured)."""
        import os
        try:
            from automedal.dispatch import _needs_setup
            if _needs_setup(dict(os.environ)):
                banner = self.query_one("#home-banner", Static)
                banner.update("First run detected — let's set up a provider")
                banner.remove_class("hidden")
                self.app.spawn_command("setup", [])
        except ImportError:
            pass

    def update_state(self, state: AppState) -> None:
        try:
            self.query_one("#home-status", StatusStrip).update_state(state)
        except Exception:
            pass
        try:
            self.query_one("#home-recent", RecentActivity).update_state(state)
        except Exception:
            pass

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

    def action_quit(self) -> None:
        self.app.exit()
