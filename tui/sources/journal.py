"""Watch the journal/ directory and emit JournalEntry events.

Each iteration the Experimenter-eval phase writes `journal/NNNN-<slug>.md` with a YAML
frontmatter (see journal/0024-*.md for schema). The frontmatter has:
  id, slug, timestamp, git_tag, queue_entry, status, val_loss, val_accuracy, best_so_far

Status values seen in the wild: "better", "worse", "crash".
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Optional

import yaml

from tui.bus import EventBus
from tui.events import JournalEntry

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
HYPOTHESIS_RE = re.compile(r"##\s+Hypothesis\s*\n(.*?)(?:\n##\s|\Z)", re.DOTALL)


def _parse_journal(path: Path) -> Optional[JournalEntry]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return None
    body = m.group(2)
    hypothesis = ""
    hm = HYPOTHESIS_RE.search(body)
    if hm:
        hypothesis = hm.group(1).strip()

    def _f(k: str) -> Optional[float]:
        v = fm.get(k)
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return JournalEntry(
        id=str(fm.get("id", "") or path.stem.split("-", 1)[0]),
        slug=str(fm.get("slug", "") or path.stem.split("-", 1)[-1]),
        timestamp=str(fm.get("timestamp", "")),
        git_tag=str(fm.get("git_tag", "")),
        status=str(fm.get("status", "")),
        val_loss=_f("val_loss"),
        val_accuracy=_f("val_accuracy"),
        best_so_far=_f("best_so_far"),
        hypothesis=hypothesis,
        path=str(path),
    )


def load_all(journal_dir: Path) -> list[JournalEntry]:
    entries: list[JournalEntry] = []
    if not journal_dir.exists():
        return entries
    for p in sorted(journal_dir.glob("[0-9]*-*.md")):
        e = _parse_journal(p)
        if e is not None:
            entries.append(e)
    return entries


async def run(bus: EventBus, journal_dir: Path, *, poll_interval: float = 1.0) -> None:
    """Emit all existing journals on startup, then watch for new/changed files."""
    journal_dir = Path(journal_dir)
    seen: dict[str, float] = {}  # path -> mtime

    # Startup backfill.
    if journal_dir.exists():
        for p in sorted(journal_dir.glob("[0-9]*-*.md")):
            try:
                mt = p.stat().st_mtime
            except FileNotFoundError:
                continue
            seen[str(p)] = mt
            e = _parse_journal(p)
            if e is not None:
                bus.publish_nowait(e)

    while True:
        try:
            if not journal_dir.exists():
                await asyncio.sleep(poll_interval)
                continue
            for p in sorted(journal_dir.glob("[0-9]*-*.md")):
                key = str(p)
                try:
                    mt = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                prev = seen.get(key)
                if prev is None or mt > prev:
                    seen[key] = mt
                    e = _parse_journal(p)
                    if e is not None:
                        bus.publish_nowait(e)
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(poll_interval)
    _ = time  # noqa
