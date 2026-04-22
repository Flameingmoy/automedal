"""Shared parser for `automedal run` argv — used by CLI dispatch + TUI spawn.

Recognized flags:
    --advisor [model]    Enable advisor; optional model id (default kimi-k2.6).
                         Sets AUTOMEDAL_ADVISOR=1 + AUTOMEDAL_ADVISOR_MODEL.

Anything else is left untouched and passed through positionally — the
existing `[N] [fast]` shape is preserved.
"""

from __future__ import annotations

DEFAULT_ADVISOR_MODEL = "kimi-k2.6"


def parse_run_args(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """Strip recognized flags from `args`. Return (remaining_args, env_overrides)."""
    out: list[str] = []
    env: dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--advisor":
            env["AUTOMEDAL_ADVISOR"] = "1"
            # Next token is the model iff it's not another flag and not a digit
            # (digits are run-iteration counts, e.g. `--advisor 10` would be a
            # mistake — leave them alone).
            nxt = args[i + 1] if i + 1 < len(args) else ""
            if nxt and not nxt.startswith("--") and not nxt.isdigit():
                env["AUTOMEDAL_ADVISOR_MODEL"] = nxt
                i += 2
            else:
                env["AUTOMEDAL_ADVISOR_MODEL"] = DEFAULT_ADVISOR_MODEL
                i += 1
            continue
        out.append(tok)
        i += 1
    return out, env
