"""Doom-loop detector — catch tool-call cycles before they burn the step budget.

Scans recent assistant turns for two failure modes ml-intern identified
on their own unbounded runs (ml-intern/agent/core/doom_loop.py):

1. **Identical repetition** — the same tool called ≥ N times in a row
   with identical args. Classic "the model refuses to accept the tool
   said no" pattern.
2. **Short cyclical pattern** — [A, B, A, B] or [A, B, C, A, B, C] over
   the tail of the signature list. Harder to spot by eye; the model
   keeps alternating between two or three calls without progress.

When either fires we inject a corrective user message into the transcript
before the next ``chat_stream`` call and emit a ``notice`` event so the
TUI / JSONL trail captures the intervention.

The module is defensive and cheap — O(N) over a bounded lookback — so it
is safe to call on every kernel step. Gate with ``AUTOMEDAL_DOOM_LOOP=0``
to disable during dogfooding if a false positive is suspected.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


_LOOKBACK = 30                 # most-recent tool_use blocks to consider
_IDENTICAL_THRESHOLD = 3       # N identical calls in a row → doom
_MIN_PATTERN = 2               # smallest cycle length we detect (A,B,A,B)
_MAX_PATTERN = 3               # largest cycle length we detect (A,B,C,A,B,C)
_PATTERN_REPEATS = 2           # pattern must repeat this many times to fire


@dataclass(frozen=True)
class _Sig:
    name: str
    args_hash: str


def _sig(name: str, args: object) -> _Sig:
    """Stable signature for a tool call: name + hash of canonicalized args."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        blob = repr(args)
    h = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return _Sig(name=name, args_hash=h)


def _recent_signatures(messages: list[dict], lookback: int = _LOOKBACK) -> list[_Sig]:
    """Extract the last ``lookback`` tool_use signatures from assistant turns."""
    sigs: list[_Sig] = []
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            sigs.append(_sig(block.get("name", ""), block.get("input", {}) or {}))
            if len(sigs) >= lookback:
                sigs.reverse()
                return sigs
    sigs.reverse()
    return sigs


def _detect_identical(sigs: list[_Sig], threshold: int = _IDENTICAL_THRESHOLD) -> _Sig | None:
    """Return the offending signature if the last ``threshold`` calls are equal."""
    if len(sigs) < threshold:
        return None
    tail = sigs[-threshold:]
    first = tail[0]
    if all(s == first for s in tail):
        return first
    return None


def _detect_cycle(
    sigs: list[_Sig],
    min_len: int = _MIN_PATTERN,
    max_len: int = _MAX_PATTERN,
    repeats: int = _PATTERN_REPEATS,
) -> list[_Sig] | None:
    """Find a repeating cycle of length [min_len..max_len] at the tail.

    Returns the pattern (list of _Sig, length = cycle length) when the tail
    of ``sigs`` is composed of that pattern repeated ``repeats`` times.
    """
    for length in range(min_len, max_len + 1):
        needed = length * repeats
        if len(sigs) < needed:
            continue
        tail = sigs[-needed:]
        pattern = tail[:length]
        # Reject degenerate all-equal patterns — covered by _detect_identical.
        if all(s == pattern[0] for s in pattern):
            continue
        ok = True
        for r in range(1, repeats):
            chunk = tail[r * length : (r + 1) * length]
            if chunk != pattern:
                ok = False
                break
        if ok:
            return pattern
    return None


def check_for_doom_loop(messages: list[dict]) -> str | None:
    """Return a corrective user message if the transcript is stuck, else None."""
    if os.environ.get("AUTOMEDAL_DOOM_LOOP", "1") == "0":
        return None

    sigs = _recent_signatures(messages)
    if not sigs:
        return None

    if (hit := _detect_identical(sigs)) is not None:
        return (
            f"[SYSTEM] You have called tool {hit.name!r} {_IDENTICAL_THRESHOLD} times "
            f"with identical arguments. The response isn't changing — repeating "
            f"won't help. Try a different tool, change the arguments, or produce "
            f"a final assistant message with the conclusion you currently have."
        )

    if (pattern := _detect_cycle(sigs)) is not None:
        names = " → ".join(s.name for s in pattern)
        return (
            f"[SYSTEM] You appear to be cycling between tool calls [{names}] "
            f"without making progress. Break out of the loop: pick one path and "
            f"follow it, or stop calling tools and produce your best final answer."
        )

    return None
