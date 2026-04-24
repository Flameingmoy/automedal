"""Tail `agent_loop.events.jsonl` from EOF — emit RawLine + HarnessMarker events.

The bespoke kernel (automedal/agent/events.py:EventSink) writes one JSON record
per line. Kinds: phase_start | phase_end | delta | thinking | tool_start |
tool_end | usage | subagent_start | subagent_end | error.

We translate those into the existing TUI event shapes:

  - `phase_start` → HarnessMarker(kind=<phase>) so the dashboard's phase chip
    advances even before the harness echoes its own dispatch line.
  - Everything else → RawLine with a compact human-readable rendering, so the
    live-stream widget shows tool calls, deltas (clipped), and errors.

Polling at ~20Hz mirrors `log_tail.py`. Rotation/truncation handled the same way.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from tui.bus import EventBus
from tui.events import HarnessMarker, RawLine


_PHASE_TO_MARKER = {
    "researcher": "researcher",
    "strategist": "strategist",
    "experimenter_edit": "experimenter_edit",
    "experimenter_eval": "experimenter_eval",
    "analyzer": "analyzer",
}


def _phase_to_marker_kind(phase: str) -> Optional[str]:
    base = (phase or "").split(">", 1)[0].strip()
    return _PHASE_TO_MARKER.get(base)


def _format_args(args: Any) -> str:
    if not isinstance(args, dict):
        s = str(args)
        return s if len(s) <= 80 else s[:77] + "..."
    parts = []
    for k, v in args.items():
        sv = str(v).replace("\n", " ")
        if len(sv) > 60:
            sv = sv[:57] + "..."
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


def _render(ev: dict) -> Optional[str]:
    """Map a JSONL event dict → human-readable line, or None to drop."""
    kind = ev.get("kind", "")
    phase = ev.get("phase", "")
    depth = int(ev.get("depth", 0) or 0)
    indent = "  " * (depth + 1)
    tag = f"[{phase}]" if phase else ""

    if kind == "phase_start":
        return f"{tag} ── start ──"
    if kind == "phase_end":
        usage = ev.get("usage") or {}
        stop = ev.get("stop", "")
        utxt = (
            f" usage={usage.get('in', '?')}/{usage.get('out', '?')}"
            if usage else ""
        )
        return f"{tag} ── end (stop={stop}{utxt}) ──"
    if kind == "tool_start":
        name = ev.get("name", "?")
        args = _format_args(ev.get("args", {}))
        return f"{indent}[{name}] {args}"
    if kind == "tool_end":
        name = ev.get("name", "?")
        ok = ev.get("ok", True)
        preview = (ev.get("preview") or "").replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:97] + "..."
        if not ok:
            return f"{indent}[{name}] ERROR: {preview}"
        return None  # successful tool ends are noisy; tool_start already said it
    if kind == "delta":
        text = (ev.get("text") or "").replace("\n", " ")
        if not text.strip():
            return None
        if len(text) > 200:
            text = text[:197] + "..."
        return text
    if kind == "thinking":
        n = len(ev.get("text") or "")
        return f"{indent}[thinking] ({n} chars)"
    if kind == "subagent_start":
        label = ev.get("label", "?")
        prompt = (ev.get("prompt") or "").replace("\n", " ")
        if len(prompt) > 100:
            prompt = prompt[:97] + "..."
        return f"{indent}[subagent:{label}] start — {prompt}"
    if kind == "subagent_end":
        label = ev.get("label", "?")
        ok = ev.get("ok", True)
        return f"{indent}[subagent:{label}] end ok={ok}"
    if kind == "advisor_consult":
        purpose = ev.get("purpose", "?")
        model = ev.get("model", "?")
        if ev.get("skipped"):
            reason = ev.get("reason", "") or "no_reason"
            return f"{indent}[advisor:{purpose}] skipped ({reason})"
        preview = (ev.get("preview") or "").replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        in_t = ev.get("in", 0)
        out_t = ev.get("out", 0)
        return f"{indent}[advisor:{purpose}] {model} ({in_t}/{out_t}) — {preview}"
    if kind == "usage":
        return None  # boring; surfaced in phase_end
    if kind == "error":
        where = ev.get("where", "")
        msg = ev.get("msg", "")
        return f"{indent}[error] {where}: {msg}"
    if kind == "notice":
        ntag = ev.get("tag", "notice")
        msg = (ev.get("message") or "").replace("\n", " ")
        if len(msg) > 200:
            msg = msg[:197] + "..."
        return f"{indent}[{ntag}] {msg}"
    return None


async def run(
    bus: EventBus,
    events_path: Path,
    *,
    from_start: bool = False,
    poll_interval: float = 0.05,
) -> None:
    """Tail `events_path` (JSONL) forever, publishing translated events to `bus`."""
    events_path = Path(events_path)
    inode: Optional[int] = None
    f = None
    pending = ""

    def _open():
        nonlocal f, inode, pending
        try:
            stat = events_path.stat()
        except FileNotFoundError:
            f = None
            inode = None
            return
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
        f = open(events_path, "r", encoding="utf-8", errors="replace")
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

            try:
                stat = events_path.stat()
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
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    bus.publish_nowait(RawLine(text=f"[events:bad-json] {line[:120]}", ts=time.time()))
                    continue

                kind = ev.get("kind", "")
                if kind == "phase_start":
                    mk = _phase_to_marker_kind(ev.get("phase", ""))
                    if mk:
                        bus.publish_nowait(HarnessMarker(kind=mk, raw=line, ts=time.time()))

                rendered = _render(ev)
                if rendered:
                    bus.publish_nowait(RawLine(text=rendered, ts=time.time()))
        except asyncio.CancelledError:
            break
        except Exception as e:
            bus.publish_nowait(RawLine(text=f"[tui:events_jsonl] error: {e!r}", ts=time.time()))
            await asyncio.sleep(0.5)

    if f is not None:
        try:
            f.close()
        except Exception:
            pass
