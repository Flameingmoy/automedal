"""Structured JSONL event sink for the bespoke agent runtime.

One line per event in `agent_loop.events.jsonl`. A parallel human-readable
mirror is appended to `agent_loop.log` (matching the format the existing
`tui/sources/log_tail.py` regexes look for) so the TUI keeps working
during the JSONL-parser rewrite.

Event kinds:
    phase_start  — about to invoke a phase (researcher/strategist/...)
    phase_end    — phase invocation finished (with usage totals)
    delta        — incremental assistant text chunk
    tool_start   — a tool call is about to execute
    tool_end     — tool call finished (ok=bool, preview=str)
    usage        — token usage for one chat turn
    subagent_start / subagent_end — nested kernel run
    advisor_consult — one advisor consultation (Kimi K2.6 by default)
    error        — any unexpected failure
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _preview(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


@dataclass
class EventSink:
    """Writes JSONL events + optional human-readable mirror.

    Use `with EventSink(...) as sink:` or call `close()` explicitly.
    """

    jsonl_path: Path | None = None
    human_path: Path | None = None
    echo: bool = False
    phase: str = ""
    step: int = 0
    depth: int = 0  # subagent nesting

    _jsonl_fh: Any = field(default=None, init=False, repr=False)
    _human_fh: Any = field(default=None, init=False, repr=False)
    _inline_active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")
        if self.human_path is not None:
            self.human_path.parent.mkdir(parents=True, exist_ok=True)
            self._human_fh = open(self.human_path, "a", encoding="utf-8")

    def __enter__(self) -> "EventSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._end_inline()
        for fh_attr in ("_jsonl_fh", "_human_fh"):
            fh = getattr(self, fh_attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, fh_attr, None)

    # ── public API ───────────────────────────────────────────────────────────

    def with_phase(self, phase: str) -> "EventSink":
        """Return a shallow copy of this sink scoped to a new phase name.

        Reuses the open file handles; do not call close() on the child.
        """
        clone = EventSink.__new__(EventSink)
        clone.jsonl_path = self.jsonl_path
        clone.human_path = self.human_path
        clone.echo = self.echo
        clone.phase = phase
        clone.step = 0
        clone.depth = self.depth
        clone._jsonl_fh = self._jsonl_fh
        clone._human_fh = self._human_fh
        clone._inline_active = False
        return clone

    def child_subagent(self, label: str) -> "EventSink":
        """Return a sink for a subagent run (depth+1, fresh step counter)."""
        clone = self.with_phase(f"{self.phase}>{label}")
        clone.depth = self.depth + 1
        return clone

    def phase_start(self, **extra: Any) -> None:
        self._emit("phase_start", extra=extra)
        self._human(f"\n========== phase: {self.phase} ==========")

    def phase_end(self, *, usage: dict | None = None, stop: str = "", **extra: Any) -> None:
        self._end_inline()
        body: dict[str, Any] = {"stop": stop}
        if usage:
            body["usage"] = usage
        body.update(extra)
        self._emit("phase_end", extra=body)
        self._human(f"  [phase_end] stop={stop} usage={usage or {}}")

    def step_advance(self) -> None:
        self.step += 1

    def delta(self, text: str) -> None:
        if not text:
            return
        self._emit("delta", extra={"text": text})
        # Human mirror: stream inline (no leading newline)
        if self._human_fh is not None:
            self._human_fh.write(text)
            self._human_fh.flush()
            self._inline_active = True
        if self.echo:
            sys.stdout.write(text)
            sys.stdout.flush()

    def thinking(self, text: str) -> None:
        if not text:
            return
        self._emit("thinking", extra={"text": text})
        # Human mirror: do not echo full thinking, just length, on its own line
        self._end_inline()
        self._human(f"  [thinking] ({len(text)} chars)")

    def tool_start(self, *, call_id: str, name: str, args: dict) -> None:
        self._end_inline()
        self._emit("tool_start", extra={"call_id": call_id, "name": name, "args": args})
        try:
            args_brief = ", ".join(f"{k}={_preview(str(v), 40)!r}" for k, v in args.items())
        except Exception:
            args_brief = "<unprintable>"
        self._human(f"  [tool] {name}({args_brief})")

    def tool_end(self, *, call_id: str, name: str, ok: bool, result: str) -> None:
        self._emit(
            "tool_end",
            extra={"call_id": call_id, "name": name, "ok": ok, "preview": _preview(result, 200)},
        )
        tag = "ok" if ok else "ERROR"
        self._human(f"  [tool] {name} → {tag}: {_preview(result, 80)}")

    def usage(self, *, in_tokens: int, out_tokens: int) -> None:
        self._emit("usage", extra={"in": in_tokens, "out": out_tokens})

    def subagent_start(self, *, label: str, prompt_preview: str) -> None:
        self._emit("subagent_start", extra={"label": label, "prompt": _preview(prompt_preview, 120)})
        self._human(f"  [subagent:{label}] start — {_preview(prompt_preview, 80)}")

    def subagent_end(self, *, label: str, ok: bool, result_preview: str) -> None:
        self._emit("subagent_end", extra={"label": label, "ok": ok, "preview": _preview(result_preview, 200)})
        self._human(f"  [subagent:{label}] end ok={ok}")

    def advisor_consult(
        self,
        *,
        purpose: str,
        model: str,
        in_tokens: int = 0,
        out_tokens: int = 0,
        skipped: bool = False,
        reason: str = "",
        preview: str = "",
    ) -> None:
        self._end_inline()
        self._emit(
            "advisor_consult",
            extra={
                "purpose": purpose,
                "model": model,
                "in": int(in_tokens),
                "out": int(out_tokens),
                "skipped": bool(skipped),
                "reason": reason,
                "preview": _preview(preview, 280),
            },
        )
        if skipped:
            self._human(f"  [advisor:{purpose}] skipped ({reason or 'no_reason'})")
        else:
            self._human(
                f"  [advisor:{purpose}] {model} ({in_tokens}/{out_tokens}) — {_preview(preview, 120)}"
            )

    def error(self, *, where: str, exc: BaseException) -> None:
        self._end_inline()
        self._emit("error", extra={"where": where, "type": type(exc).__name__, "msg": str(exc)})
        self._human(f"  [error] {where}: {type(exc).__name__}: {exc}")

    def notice(self, *, tag: str, message: str) -> None:
        """Informational log: retry attempts, truncation hints, doom-loop warnings.

        Neutral severity — not an error, not a tool call. Used by
        ``automedal.agent.retry.with_retry`` and the kernel's self-healing
        paths so the JSONL trail captures what the harness did behind the scenes.
        """
        self._end_inline()
        self._emit("notice", extra={"tag": tag, "message": message})
        self._human(f"  [{tag}] {message}")

    # Back-compat alias matching ml-intern's `tool_log`; callers can use either.
    def tool_log(self, *, tool: str, log: str) -> None:
        self.notice(tag=tool, message=log)

    # ── internals ────────────────────────────────────────────────────────────

    def _emit(self, kind: str, *, extra: dict[str, Any]) -> None:
        if self._jsonl_fh is None:
            return
        rec: dict[str, Any] = {
            "t": _utcnow(),
            "phase": self.phase,
            "step": self.step,
            "depth": self.depth,
            "kind": kind,
        }
        rec.update(extra)
        self._jsonl_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._jsonl_fh.flush()

    def _human(self, line: str) -> None:
        if self._human_fh is None:
            return
        self._end_inline()
        self._human_fh.write(line + "\n")
        self._human_fh.flush()
        if self.echo:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _end_inline(self) -> None:
        if self._inline_active and self._human_fh is not None:
            self._human_fh.write("\n")
            self._human_fh.flush()
            self._inline_active = False
            if self.echo:
                sys.stdout.write("\n")
                sys.stdout.flush()
