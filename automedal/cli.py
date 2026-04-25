"""automedal — single CLI entry point.

The user-facing shell is the Go TUI (``tui-go/``). This module only:

  * locates the ``automedal-tui`` binary (override → PATH → in-tree build),
  * execs it with the user's argv, or
  * hands off to ``automedal dispatch`` for headless runs (CI / scripts).

  automedal                   → exec Go TUI (home)
  automedal tui [args...]     → exec Go TUI with args
  automedal dispatch <cmd> …  → headless dispatch
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else ""

    if cmd == "dispatch":
        from automedal.dispatch import dispatch
        sub = args[1] if len(args) > 1 else ""
        if not sub:
            print("Usage: automedal dispatch <cmd> [args...]", file=sys.stderr)
            sys.exit(2)
        sys.exit(dispatch(sub, args[2:]))

    # Everything else: hand off to the Go TUI (no auto-spawn forwarding yet;
    # the Go shell owns its own command palette).
    _exec_go_tui(args[1:] if cmd == "tui" else args)


def _prepare_env() -> None:
    """Inject Layout paths into os.environ before the TUI starts."""
    try:
        from automedal.paths import Layout
        layout = Layout()
        os.environ.update(layout.as_env())
    except Exception:
        pass


def _go_tui_path() -> str | None:
    """Return path to the Go TUI binary, or None if unavailable.

    Checked locations (in order):
      1. $AUTOMEDAL_TUI_GO_BIN  (explicit override)
      2. automedal-tui on $PATH
      3. ./tui-go/automedal-tui  (in-tree development build)
    """
    override = os.environ.get("AUTOMEDAL_TUI_GO_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    found = shutil.which("automedal-tui")
    if found:
        return found
    # In-tree development build locations, in priority order. Phase 1 of the
    # Go-control-plane port moved tui-go/ → internal/ui/ and the Makefile
    # builds binaries into bin/ — keep the legacy tui-go/ path for users who
    # haven't pulled the move yet.
    for rel in ("bin/automedal-tui", "internal/ui/automedal-tui",
                "tui-go/automedal-tui"):
        local = Path.cwd() / rel
        if local.is_file() and os.access(str(local), os.X_OK):
            return str(local)
    return None


def _exec_go_tui(extra_args: list[str]) -> None:
    """Replace the Python process with the Go binary — never returns."""
    path = _go_tui_path()
    if path is None:
        sys.exit(
            "automedal-tui (Go binary) not found.\n"
            "Build it with:\n"
            "  cd tui-go && go build -o automedal-tui .\n"
            "Or install to PATH, or set AUTOMEDAL_TUI_GO_BIN=/abs/path.\n"
            "For headless runs use:  automedal dispatch <cmd> [args...]"
        )
    _prepare_env()
    argv = [path] + list(extra_args)
    os.execvp(path, argv)
