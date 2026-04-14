"""Startup-only backfill from git exp/* tags and commit subjects.

Used to seed the leaderboard from history before any journal events roll in. Live
signal comes from the journal source; this is just so the TUI looks right on launch.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from tui.bus import EventBus

COMMIT_SUBJECT_RE = re.compile(
    r"^experiment\s+(\d+)\s+\(([^)]+)\):\s*val_loss\s+([0-9.]+)\s*->\s*([0-9.]+)"
)


async def backfill(bus: EventBus, repo_root: Path) -> None:
    """Run `git log` over exp/* tags and emit synthetic JournalEntry events for history.

    Only emits entries that aren't already covered by the journal source (the journal
    source runs first on startup; this is best-effort when journal files are missing).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_root), "log", "--tags=exp/*", "--no-walk",
            "--format=%H%x1f%s",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except (FileNotFoundError, OSError):
        return
    if proc.returncode != 0 or not out:
        return

    _ = out.decode("utf-8", errors="replace").splitlines()
    # Parse but don't publish — journal parser is authoritative. This function exists
    # as a scaffold so future "gap filling" (missing journal file for a tag) can hook
    # in without touching app wiring.
    return
