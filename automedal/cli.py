"""automedal — single CLI entry point.

  automedal                   → open TUI home screen
  automedal tui [--demo ...]  → open TUI home screen (passes args to __main__)
  automedal run [N]           → open TUI + auto-spawn run
  automedal <cmd> [args]      → open TUI + auto-spawn <cmd>
  automedal dispatch <cmd> …  → headless dispatch (CI / scripts)
"""

from __future__ import annotations

import os
import shutil
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
    """Inject Layout paths into os.environ before the TUI starts."""
    try:
        from automedal.paths import Layout
        layout = Layout()
        os.environ.update(layout.as_env())
    except Exception:
        pass


def _require_textual() -> None:
    try:
        import textual  # noqa: F401
    except ImportError:
        sys.exit(
            "TUI dependencies not installed.\n"
            "Run:  pip install -e .\n"
        )


def _go_tui_path() -> str | None:
    """Return path to the Go TUI binary, or None if unavailable.

    Checked locations (in order):
      1. $AUTOMEDAL_TUI_GO_BIN  (explicit override)
      2. automedal-tui on $PATH
      3. ./tui-go/automedal-tui  (in-tree development build)

    Returns None if AUTOMEDAL_NO_GO_TUI=1 — keeps an escape hatch for
    debugging the Python fallback path.
    """
    if os.environ.get("AUTOMEDAL_NO_GO_TUI") == "1":
        return None
    override = os.environ.get("AUTOMEDAL_TUI_GO_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    found = shutil.which("automedal-tui")
    if found:
        return found
    local = Path.cwd() / "tui-go" / "automedal-tui"
    if local.is_file() and os.access(str(local), os.X_OK):
        return str(local)
    return None


def _exec_go_tui(path: str, extra_args: list[str]) -> None:
    """Replace the Python process with the Go binary — never returns."""
    _prepare_env()
    argv = [path] + list(extra_args)
    os.execvp(path, argv)


def _launch_tui_home(auto_spawn: tuple[str, list[str]] | None = None) -> None:
    """Open the TUI directly to HomeScreen, optionally auto-spawning a command.

    Prefers the Go TUI if the binary is present; falls back to the Python
    Textual TUI otherwise. The Go binary owns its own first-frame path —
    we just hand off argv.
    """
    # Go shell doesn't yet accept auto-spawn; fall back for that case.
    if auto_spawn is None:
        if go := _go_tui_path():
            _exec_go_tui(go, [])
            return
    _require_textual()
    _prepare_env()
    _ensure_logo()
    from tui.app import AutoMedalApp
    app = AutoMedalApp(repo_root=Path.cwd(), auto_spawn=auto_spawn)
    app.run()


def _launch_tui_with_args(extra_args: list[str]) -> None:
    """Delegate to tui.__main__ (supports --demo, --log-file, etc.).

    If the Go TUI is present AND the caller didn't pass Python-only flags
    (--demo, --log-file), hand off to Go. Otherwise fall through.
    """
    python_only = {"--demo", "--log-file", "--theme"}
    has_py_only = any(a in python_only or a.startswith("--log-file=") for a in extra_args)
    if not has_py_only:
        if go := _go_tui_path():
            _exec_go_tui(go, extra_args)
            return
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
