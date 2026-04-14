"""Tail agent/results.tsv and emit ResultRow events.

Schema (header row):
  timestamp  method  trials  val_loss  val_accuracy  submission  notes

Duplicates the 15-line stdlib CSV reader from harness/check_stagnation.py::_read_val_losses
rather than importing it (harness/ has no __init__.py; keeping tui/ self-contained).
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
from pathlib import Path
from typing import Optional

from tui.bus import EventBus
from tui.events import ResultRow

EXPECTED_COLS = ("timestamp", "method", "trials", "val_loss", "val_accuracy", "submission", "notes")


def _maybe_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _maybe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def read_all(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(_row(r))
    return rows


def _row(r: dict) -> ResultRow:
    return ResultRow(
        timestamp=(r.get("timestamp") or "").strip(),
        method=(r.get("method") or "").strip(),
        trials=_maybe_int(r.get("trials") or ""),
        val_loss=_maybe_float(r.get("val_loss") or ""),
        val_accuracy=_maybe_float(r.get("val_accuracy") or ""),
        submission=(r.get("submission") or "").strip(),
        notes=(r.get("notes") or "").strip(),
    )


async def run(bus: EventBus, tsv_path: Path, *, poll_interval: float = 1.0) -> None:
    tsv_path = Path(tsv_path)
    # Startup backfill.
    initial = read_all(tsv_path)
    header_seen = tsv_path.exists()
    last_count = len(initial)
    for row in initial:
        bus.publish_nowait(row)

    while True:
        try:
            if not tsv_path.exists():
                header_seen = False
                last_count = 0
                await asyncio.sleep(poll_interval)
                continue
            if not header_seen:
                header_seen = True
            # Cheap: read entire file (it's small — ≤50KB typical).
            rows = read_all(tsv_path)
            if len(rows) > last_count:
                for row in rows[last_count:]:
                    bus.publish_nowait(row)
                last_count = len(rows)
            elif len(rows) < last_count:
                # File was reset (e.g. ./am clean). Re-publish everything.
                last_count = 0
                for row in rows:
                    bus.publish_nowait(row)
                last_count = len(rows)
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(poll_interval)
    _ = io, time  # noqa
