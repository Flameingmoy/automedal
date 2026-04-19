"""Replay a fixture agent_loop.log line-by-line into the bus.

Used by `./am tui --demo`. The fixture includes every harness marker, a few pi JSON
events, and training output so all widgets exercise their render paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tui.bus import EventBus
from tui.sources.log_tail import (
    _classify_marker,
    _strip_ansi,
    ITER_END_RE,
    ITER_START_RE,
    TRAINING_DONE_RE,
    _parse_val_loss,
)
from tui.events import IterationEnd, IterationStart, RawLine, TrainingFinished
import time


async def run(bus: EventBus, fixture_path: Path, *, line_delay: float = 0.04) -> None:
    fixture_path = Path(fixture_path)
    if not fixture_path.exists():
        bus.publish_nowait(RawLine(text=f"[demo] fixture not found: {fixture_path}", ts=time.time()))
        return
    with open(fixture_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = _strip_ansi(line).rstrip("\r\n")
            if not stripped:
                await asyncio.sleep(line_delay)
                continue

            m = ITER_START_RE.match(stripped)
            if m:
                bus.publish_nowait(IterationStart(exp_id=m.group(3), i=int(m.group(1)), total=int(m.group(2)), ts=time.time()))
                bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                await asyncio.sleep(line_delay)
                continue
            m = ITER_END_RE.match(stripped)
            if m:
                bus.publish_nowait(IterationEnd(exp_id=m.group(2), i=int(m.group(1)), ts=time.time()))
                bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                await asyncio.sleep(line_delay)
                continue
            m = TRAINING_DONE_RE.search(stripped)
            if m:
                bus.publish_nowait(
                    TrainingFinished(val_loss=_parse_val_loss(m.group(1)), exit_code=int(m.group(2)), ts=time.time())
                )
                bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                await asyncio.sleep(line_delay)
                continue
            mk = _classify_marker(stripped)
            if mk is not None:
                bus.publish_nowait(mk)
                bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                await asyncio.sleep(line_delay)
                continue
            bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
            await asyncio.sleep(line_delay)
