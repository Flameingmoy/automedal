"""Transient-error retry helper for provider / advisor calls.

Wraps an async callable with N attempts and backoff delays. Matches
ml-intern's posture (agent_loop.py L118-L136 / L295-L334):

  • Retry only on known-transient patterns: 5xx, 429, timeout, connection reset.
  • Never retry on auth / credits / model-not-found — those are deterministic.
  • Emit one `tool_log`-style event per retry so the TUI shows what's happening.
  • Max 3 attempts with [5, 15, 30]s backoff — ≤50s total before surfacing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

_MAX_ATTEMPTS = 3
_RETRY_DELAYS: tuple[int, ...] = (5, 15, 30)

_TRANSIENT_PATTERNS = (
    "timeout", "timed out",
    "429", "rate limit", "rate_limit",
    "503", "service unavailable",
    "502", "bad gateway",
    "500", "internal server error",
    "504", "gateway timeout",
    "overloaded", "capacity",
    "connection reset", "connection refused", "connection error",
    "eof", "broken pipe",
    "remote end closed",
)


def is_transient_error(exc: BaseException) -> bool:
    """True if the exception matches a known transient network/provider pattern."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    s = str(exc).lower()
    return any(p in s for p in _TRANSIENT_PATTERNS)


T = TypeVar("T")


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    label: str,
    events: Any = None,
    attempts: int = _MAX_ATTEMPTS,
    delays: tuple[int, ...] = _RETRY_DELAYS,
) -> T:
    """Run `call()`; on transient error, sleep-and-retry up to `attempts` times.

    ``label`` identifies the call in retry events (e.g. 'openai.chat_stream').
    ``events`` is an optional EventSink with a ``tool_log(tool, log)`` method.
    Non-transient exceptions bubble immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not is_transient_error(exc):
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)]
            if events is not None:
                try:
                    events.tool_log(
                        tool="retry",
                        log=(
                            f"{label}: transient error (attempt {attempt}/{attempts}) "
                            f"— retrying in {delay}s: {type(exc).__name__}: {exc}"
                        ),
                    )
                except Exception:
                    pass
            await asyncio.sleep(delay)
    # Unreachable — either we returned or raised — but keep type-checkers happy.
    assert last_exc is not None
    raise last_exc
