"""Home screen dispatch — quit aliases must never spawn a subprocess.

Regression guard for the bug where typing `q` at the prompt routed to
CommandOutputScreen and cold-launched `python -m automedal q`, which
imports the full advisor + openai + anthropic stack just to print
"Unknown command". That flash of heavy output was the "cryptic UI" the
user was seeing when they tried to quit.
"""

from __future__ import annotations

import pytest

from tui.screens.home import HomeScreen


class _FakeApp:
    def __init__(self):
        self.exited = False
        self.spawned: list[tuple[str, list[str]]] = []
        self.help_shown = False

    def exit(self):
        self.exited = True

    def spawn_command(self, cmd, args):
        self.spawned.append((cmd, args))

    def action_show_help(self):
        self.help_shown = True


def _dispatch(text: str) -> _FakeApp:
    # Bypass Textual's Screen.__init__ so we don't need an App event loop.
    screen = HomeScreen.__new__(HomeScreen)
    app = _FakeApp()
    screen.__dict__["_app"] = app
    # HomeScreen's method accesses self.app — patch with a property stand-in
    type(screen).app = property(lambda s: s.__dict__["_app"])
    try:
        screen._dispatch_text(text)
    finally:
        # Don't leak the monkey-patched descriptor across tests
        del type(screen).app
    return app


@pytest.mark.parametrize("alias", ["q", "Q", "quit", "exit", ":q", ":quit", ":wq"])
def test_quit_aliases_exit_without_spawning(alias):
    app = _dispatch(alias)
    assert app.exited is True
    assert app.spawned == []          # never cold-launch a subprocess
    assert app.help_shown is False


def test_help_is_handled_inline():
    app = _dispatch("help")
    assert app.help_shown is True
    assert app.spawned == []
    assert app.exited is False


def test_other_commands_still_spawn():
    app = _dispatch("doctor")
    assert app.spawned == [("doctor", [])]
    assert app.exited is False


def test_run_passes_args_through():
    app = _dispatch("run 10 --advisor kimi-k2.6")
    assert app.spawned == [("run", ["10", "--advisor", "kimi-k2.6"])]


def test_empty_input_is_noop():
    app = _dispatch("   ")
    assert app.exited is False
    assert app.spawned == []
