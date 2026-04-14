"""Tail agent_loop.log from EOF, emit harness markers / training events / pi JSON events / raw lines.

The log is a mix of:
  - Bash echoes written by run.sh (e.g. `[harness] dispatching Researcher (stagnation)`)
  - Raw `pi --mode json` JSONL lines (one event per line: tool_execution_start/end,
    message_update with text_delta, turn_end, agent_end)
  - Training stdout from `python agent/train.py` (plain text, sometimes ANSI-colored)

We poll (~20Hz) from the current EOF rather than watchdog — append rates can exceed
inotify coalescing under heavy training output, and polling is simpler for rotation.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from tui.bus import EventBus
from tui.events import (
    HarnessMarker,
    IterationEnd,
    IterationStart,
    RawLine,
    TrainingFinished,
)

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

ITER_START_RE = re.compile(
    r"^=+\s*Iteration\s+(\d+)\s*/\s*(\d+)\s+exp=([0-9]+)\s+\[[^\]]+\]\s*=+\s*$"
)
ITER_END_RE = re.compile(
    r"^---\s*Iteration\s+(\d+)\s+complete\s+exp=([0-9]+)\s+\[[^\]]+\]\s*---\s*$"
)
TRAINING_DONE_RE = re.compile(
    r"\[harness\]\s+training\s+done:\s+val_loss=([0-9.nan]+)\s+exit=(\-?\d+)"
)
DISPATCH_RE = re.compile(r"\[harness\]\s+dispatching\s+(Researcher|Strategist|Experimenter)(?:\s+\(([^)]+)\))?")
TRAINING_START_RE = re.compile(r"\[harness\]\s+running\s+training")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _classify_marker(line: str) -> Optional[HarnessMarker]:
    if TRAINING_START_RE.search(line):
        return HarnessMarker(kind="training_start", raw=line, ts=time.time())
    m = DISPATCH_RE.search(line)
    if not m:
        return None
    agent = m.group(1).lower()
    sub = (m.group(2) or "").lower()
    if agent == "researcher":
        return HarnessMarker(kind="researcher", raw=line, ts=time.time())
    if agent == "strategist":
        return HarnessMarker(kind="strategist", raw=line, ts=time.time())
    if agent == "experimenter":
        if "eval" in sub:
            return HarnessMarker(kind="experimenter_eval", raw=line, ts=time.time())
        return HarnessMarker(kind="experimenter_edit", raw=line, ts=time.time())
    return None


def _try_parse_pi_json(line: str) -> Optional[str]:
    """If `line` is a pi `--mode json` event, render a human-readable summary; else None."""
    if not line.startswith("{"):
        return None
    try:
        ev = json.loads(line)
    except Exception:
        return None
    t = ev.get("type", "")
    if t == "tool_execution_start":
        tool = ev.get("toolName", "?")
        args = ev.get("args", {})
        if isinstance(args, dict):
            arg = args.get("command") or args.get("file_path") or args.get("path") or ""
        else:
            arg = str(args)
        arg = str(arg).replace("\n", " ")
        if len(arg) > 120:
            arg = arg[:117] + "..."
        return f"[{tool}] {arg}"
    if t == "tool_execution_end":
        if ev.get("isError"):
            result = str(ev.get("result", ""))[:200].replace("\n", " ")
            return f"[{ev.get('toolName', '?')}] ERROR: {result}"
        return None
    if t == "message_update":
        ame = ev.get("assistantMessageEvent", {})
        if ame.get("type") == "text_delta":
            return ame.get("delta", "")
    return None


def _parse_val_loss(raw: str) -> Optional[float]:
    try:
        v = float(raw)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


async def run(
    bus: EventBus,
    log_path: Path,
    *,
    from_start: bool = False,
    poll_interval: float = 0.05,
) -> None:
    """Tail `log_path` forever, publishing events to `bus`."""
    log_path = Path(log_path)
    # Start at EOF by default so we don't replay an entire multi-GB history.
    inode: Optional[int] = None
    f = None
    pending = ""
    last_total = 1

    def _open():
        nonlocal f, inode, pending
        try:
            stat = log_path.stat()
        except FileNotFoundError:
            f = None
            inode = None
            return
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        f = open(log_path, "r", encoding="utf-8", errors="replace")
        inode = stat.st_ino
        if not from_start:
            f.seek(0, os.SEEK_END)
        pending = ""

    _open()

    while True:
        try:
            if f is None:
                _open()
                if f is None:
                    await asyncio.sleep(0.5)
                    continue

            # Detect log rotation or truncation.
            try:
                stat = log_path.stat()
                if stat.st_ino != inode or (stat.st_size < (f.tell() if f else 0)):
                    _open()
                    if f is None:
                        await asyncio.sleep(0.5)
                        continue
            except FileNotFoundError:
                await asyncio.sleep(0.5)
                continue

            chunk = f.read(65536)
            if not chunk:
                await asyncio.sleep(poll_interval)
                continue

            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                stripped = _strip_ansi(line).rstrip("\r")
                if not stripped:
                    continue

                # Iteration boundaries.
                m = ITER_START_RE.match(stripped)
                if m:
                    i = int(m.group(1))
                    total = int(m.group(2))
                    last_total = total
                    exp_id = m.group(3)
                    bus.publish_nowait(IterationStart(exp_id=exp_id, i=i, total=total, ts=time.time()))
                    bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                    continue
                m = ITER_END_RE.match(stripped)
                if m:
                    i = int(m.group(1))
                    exp_id = m.group(2)
                    bus.publish_nowait(IterationEnd(exp_id=exp_id, i=i, ts=time.time()))
                    bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                    continue

                # Training done.
                m = TRAINING_DONE_RE.search(stripped)
                if m:
                    loss = _parse_val_loss(m.group(1))
                    exit_code = int(m.group(2))
                    bus.publish_nowait(TrainingFinished(val_loss=loss, exit_code=exit_code, ts=time.time()))
                    bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                    continue

                # Harness dispatch / training-start markers.
                marker = _classify_marker(stripped)
                if marker is not None:
                    bus.publish_nowait(marker)
                    bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
                    continue

                # Pi JSON event → extract a human line.
                summary = _try_parse_pi_json(stripped)
                if summary is not None:
                    if summary:
                        bus.publish_nowait(RawLine(text=summary, ts=time.time()))
                    continue

                # Plain text (training output, bash echoes).
                bus.publish_nowait(RawLine(text=stripped, ts=time.time()))
        except asyncio.CancelledError:
            break
        except Exception as e:
            bus.publish_nowait(RawLine(text=f"[tui:log_tail] error: {e!r}", ts=time.time()))
            await asyncio.sleep(0.5)

    if f is not None:
        try:
            f.close()
        except Exception:
            pass

    _ = last_total  # silence linter (kept for future per-iteration budget math)
