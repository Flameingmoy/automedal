#!/usr/bin/env python3
"""AutoMedal — agent event-stream → terminal output.

Two modes, one output format. Both render lines that match the regexes in
``tui/sources/log_tail.py`` — do not change the ``  [tool] ...`` shape
without updating the TUI parser.

Legacy (pi coding-agent, stdin):
    pi --mode json -p "..." | tee -a log | python3 -u harness/stream_events.py

Modern (deepagents / LangGraph, in-process):
    from harness.stream_events import format_langgraph_event
    for ev in agent.astream_events(..., version="v2"):
        for line, inline in format_langgraph_event(ev):
            ...

``format_langgraph_event`` yields ``(text, inline)`` pairs. ``inline=True``
means the text is a streaming delta (no trailing newline); ``inline=False``
means append a newline. Callers should insert a newline before the first
non-inline line if the previous output was inline.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Iterable, Iterator, Mapping


# ── shared formatters ────────────────────────────────────────────────────────

def _truncate(s: Any, maxlen: int = 120) -> str:
    s = str(s).replace("\n", " ")
    return s[: maxlen - 3] + "..." if len(s) > maxlen else s


def _fmt_tool_args(args: Any) -> str:
    """Render a tool-call args dict as one informative line."""
    if isinstance(args, Mapping):
        for key in ("command", "path", "file_path", "pattern", "query", "arxiv_id"):
            if key in args:
                return _truncate(args[key])
        try:
            return _truncate(json.dumps(args, separators=(",", ":")))
        except (TypeError, ValueError):
            return _truncate(args)
    return _truncate(args)


def _extract_text_delta(content: Any) -> str:
    """Pull plain text out of a chat-model-stream chunk's content block(s)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, Mapping):
                # Skip thinking / reasoning blocks
                if blk.get("type") == "text":
                    parts.append(blk.get("text", ""))
            elif isinstance(blk, str):
                parts.append(blk)
        return "".join(parts)
    return ""


# ── LangGraph v2 events (deepagents path) ────────────────────────────────────

def format_langgraph_event(ev: Mapping[str, Any]) -> Iterator[tuple[str, bool]]:
    """Yield ``(text, inline)`` pairs for one LangGraph astream_events(v2) event.

    ``inline=True`` means "streaming delta — do not add a newline".
    ``inline=False`` means "complete line — append newline".
    """
    et = ev.get("event", "")

    if et == "on_tool_start":
        name = ev.get("name", "?")
        args = (ev.get("data") or {}).get("input", {})
        yield f"  [{name}] {_fmt_tool_args(args)}", False
        return

    if et == "on_tool_end":
        data = ev.get("data") or {}
        out = data.get("output")
        status = getattr(out, "status", None) if out is not None else None
        if status and str(status).lower() != "success":
            name = ev.get("name", "?")
            result = getattr(out, "content", out)
            yield f"  [{name}] ERROR: {_truncate(result, 200)}", False
        return

    if et == "on_chat_model_stream":
        data = ev.get("data") or {}
        chunk = data.get("chunk")
        if chunk is None:
            return
        text = _extract_text_delta(getattr(chunk, "content", ""))
        if text:
            yield text, True


# ── Pi JSON events (legacy stdin mode) ───────────────────────────────────────

def _format_pi_event(event: Mapping[str, Any]) -> Iterable[tuple[str, bool]]:
    """Yield ``(text, inline)`` pairs for one pi coding-agent JSON event."""
    t = event.get("type", "")

    if t == "tool_execution_start":
        tool = event.get("toolName", "?")
        yield f"  [{tool}] {_fmt_tool_args(event.get('args', {}))}", False

    elif t == "tool_execution_end":
        if event.get("isError"):
            tool = event.get("toolName", "?")
            yield f"  [{tool}] ERROR: {_truncate(event.get('result', ''), 200)}", False

    elif t == "message_update":
        ame = event.get("assistantMessageEvent", {})
        if ame.get("type") == "text_delta":
            delta = ame.get("delta", "")
            if delta:
                yield delta, True

    elif t in ("turn_end", "agent_end"):
        yield "", False


def _main_pi_stdin() -> None:
    inline_active = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for text, inline in _format_pi_event(event):
            if inline:
                sys.stdout.write(text)
                sys.stdout.flush()
                inline_active = True
            else:
                if inline_active:
                    sys.stdout.write("\n")
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
                inline_active = False


if __name__ == "__main__":
    try:
        _main_pi_stdin()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
