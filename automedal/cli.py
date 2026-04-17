"""automedal — single CLI entry point.

  automedal                   → open TUI home screen
  automedal tui [--demo ...]  → open TUI home screen (passes args to __main__)
  automedal run [N]           → open TUI + auto-spawn run
  automedal <cmd> [args]      → open TUI + auto-spawn <cmd>
  automedal dispatch <cmd> …  → headless dispatch (CI / scripts)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else ""

    if not cmd:
        _launch_tui_home()
    elif cmd == "tui":
        _launch_tui_with_args(args[1:])
    elif cmd == "dispatch":
        from automedal.dispatch import dispatch
        sub = args[1] if len(args) > 1 else ""
        if not sub:
            print("Usage: automedal dispatch <cmd> [args...]", file=sys.stderr)
            sys.exit(2)
        sys.exit(dispatch(sub, args[2:]))
    else:
        _launch_tui_home(auto_spawn=(cmd, args[1:]))


def _prepare_env() -> None:
    """Inject Layout paths + pi binary into os.environ before the TUI starts."""
    try:
        from automedal.paths import Layout
        layout = Layout()
        os.environ.update(layout.as_env())
    except Exception:
        pass
    try:
        from automedal.pi_runtime import ensure_pi
        os.environ["AUTOMEDAL_PI_BIN"] = str(ensure_pi())
    except SystemExit:
        pass


def _require_textual() -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        sys.exit(
            "TUI dependencies not installed.\n"
            "Run:  pip install -e .\n"
        )


def _launch_tui_home(auto_spawn: tuple[str, list[str]] | None = None) -> None:
    """Open the TUI directly to HomeScreen, optionally auto-spawning a command."""
    _require_textual()
    _prepare_env()
    _ensure_logo()
    from tui.app import AutoMedalApp
    app = AutoMedalApp(repo_root=Path.cwd(), auto_spawn=auto_spawn)
    app.run()


def _launch_tui_with_args(extra_args: list[str]) -> None:
    """Delegate to tui.__main__ (supports --demo, --log-file, etc.)."""
    _require_textual()
    _prepare_env()
    _ensure_logo()
    sys.argv = ["automedal-tui"] + list(extra_args)
    from tui.__main__ import main as tui_main
    sys.exit(tui_main() or 0)


def _ensure_logo() -> None:
    """Generate the splash PNG before the TUI opens (HomeScreen renders it)."""
    try:
        from tui.assets.logo.generate_logo import ensure_logo
        ensure_logo()
    except Exception:
        pass
