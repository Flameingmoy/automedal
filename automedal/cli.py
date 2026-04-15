"""automedal — single CLI entry point.

  automedal              → open TUI home screen
  automedal tui          → same as above
  automedal <cmd> [args] → headless dispatch (like the old `./am` script)
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else ""

    if not cmd or cmd in ("tui",):
        _launch_tui(args[1:] if cmd == "tui" else args)
    else:
        from automedal.dispatch import dispatch
        sys.exit(dispatch(cmd, args[1:]))


def _launch_tui(extra_args: list[str]) -> None:
    """Open the Textual TUI (home screen or dashboard)."""
    try:
        import textual  # noqa: F401
    except ImportError:
        sys.exit(
            "TUI dependencies not installed.\n"
            "Run:  pip install 'automedal[tui]'\n"
            "  or: uv sync --extra tui"
        )

    from automedal.paths import Layout
    layout = Layout()

    # Inject Layout paths + pi binary into the environment before the TUI starts.
    # Sources and subprocesses inherit this env.
    os.environ.update(layout.as_env())
    try:
        from automedal.pi_runtime import ensure_pi
        os.environ["AUTOMEDAL_PI_BIN"] = str(ensure_pi())
    except SystemExit:
        pass  # pi missing — TUI still usable for attach/demo mode

    # Build sys.argv for tui.__main__.main so its argparse sees the right flags
    sys.argv = ["automedal-tui"] + list(extra_args)

    from tui.__main__ import main as tui_main
    sys.exit(tui_main() or 0)
