"""Per-iteration token budget + junction gating for the advisor.

Process-global state: `reset_iteration_budget()` is called at the top of each
loop iteration; `consume_tokens(n)` is called inside `consult()` after each
successful advisor call. `remaining_tokens()` short-circuits further calls
once the cap is reached.
"""

from __future__ import annotations

import os

_STATE: dict[str, int] = {"used_this_iter": 0}


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def reset_iteration_budget() -> None:
    _STATE["used_this_iter"] = 0


def consume_tokens(n: int) -> None:
    _STATE["used_this_iter"] = int(_STATE["used_this_iter"]) + max(0, int(n))


def remaining_tokens() -> int:
    cap = _env_int("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", 8000)
    return max(0, cap - int(_STATE["used_this_iter"]))


def budget_state() -> dict:
    cap = _env_int("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", 8000)
    used = int(_STATE["used_this_iter"])
    return {"used_this_iter": used, "cap_per_iter": cap, "remaining": max(0, cap - used)}


def _junctions_allowed() -> set[str]:
    raw = os.environ.get("AUTOMEDAL_ADVISOR_JUNCTIONS") or "stagnation,audit,tool"
    return {s.strip() for s in raw.split(",") if s.strip()}


def is_enabled(junction: str | None = None) -> bool:
    """Master flag + (optional) per-junction allowlist."""
    if not _env_bool("AUTOMEDAL_ADVISOR", False):
        return False
    if junction is None:
        return True
    return junction in _junctions_allowed()
