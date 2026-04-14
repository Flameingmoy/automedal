"""Watch knowledge.md, experiment_queue.md, research_notes.md.

experiment_queue.md drives the "Current hypothesis" widget — we extract the first
`[STATUS: pending]` entry's title and body.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Optional

from tui.bus import EventBus
from tui.events import MemoryTouched, QueueState

QUEUE_ENTRY_RE = re.compile(
    r"^##\s+\d+\.\s+([a-zA-Z0-9_\-]+)\s+.*?\[STATUS:\s*pending\s*\]\s*$",
    re.MULTILINE,
)


def _parse_queue(text: str) -> QueueState:
    pending_count = len(re.findall(r"\[STATUS:\s*pending\s*\]", text, re.IGNORECASE))
    m = QUEUE_ENTRY_RE.search(text)
    if not m:
        return QueueState(pending_count=pending_count)
    slug = m.group(1).strip()
    start = m.end()
    # Hypothesis: the chunk up to the next `## ` (next entry).
    next_m = re.search(r"^##\s+\d+\.\s+", text[start:], re.MULTILINE)
    body = text[start:start + (next_m.start() if next_m else len(text) - start)].strip()
    # Grab the line under "Hypothesis:" if present; else the first non-empty paragraph.
    hyp = ""
    hm = re.search(r"(?:^|\n)\s*[-*]\s*Hypothesis[:\s]+(.+?)(?:\n[-*]|\n\n|\Z)", body, re.DOTALL | re.IGNORECASE)
    if hm:
        hyp = hm.group(1).strip().replace("\n", " ")
    else:
        for para in body.split("\n\n"):
            if para.strip():
                hyp = para.strip().replace("\n", " ")
                break
    return QueueState(current_slug=slug, current_hypothesis=hyp, pending_count=pending_count)


def parse_queue_file(path: Path) -> QueueState:
    try:
        return _parse_queue(path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return QueueState()


async def run(
    bus: EventBus,
    repo_root: Path,
    *,
    poll_interval: float = 2.0,
) -> None:
    paths = {
        "knowledge": repo_root / "knowledge.md",
        "queue": repo_root / "experiment_queue.md",
        "research_notes": repo_root / "research_notes.md",
    }
    mtimes: dict[str, Optional[float]] = {k: None for k in paths}

    # Startup push.
    qs = parse_queue_file(paths["queue"])
    bus.publish_nowait(qs)

    while True:
        try:
            for name, p in paths.items():
                try:
                    mt = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                prev = mtimes[name]
                if prev is None or mt > prev:
                    mtimes[name] = mt
                    if name == "queue":
                        bus.publish_nowait(parse_queue_file(p))
                    else:
                        bus.publish_nowait(MemoryTouched(which=name))
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(poll_interval)
    _ = time  # noqa
