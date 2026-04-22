"""Advisor LLM client — one non-streaming Chat Completions round-trip.

The advisor is invoked against an OpenAI-compatible endpoint (default
`https://opencode.ai/zen/go/v1`, which also hosts the executor's `minimax-m2.7`
behind the same `OPENCODE_API_KEY`). The default model is `kimi-k2.6`.

Design points:
  - Non-streaming. The advisor returns a short directive; incremental deltas
    aren't useful and streaming adds SDK surface for no payoff here.
  - No tools. Matches Anthropic's advisor-strategy contract — advisor reasons,
    executor acts.
  - Never raises. Every failure returns `AdvisorOpinion(skipped=True, reason=...)`.
  - Emits one `advisor_consult` event per call (even for skipped calls) so the
    JSONL trail captures gating/budget/error paths too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from automedal.advisor.budget import consume_tokens, is_enabled, remaining_tokens

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@dataclass
class AdvisorOpinion:
    text: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    skipped: bool = False
    reason: str = ""


def _short(s: str, n: int = 280) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _get_env(key: str, default: str) -> str:
    val = os.environ.get(key) or ""
    return val if val else default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _emit(events: Any, purpose: str, model: str, op: AdvisorOpinion) -> None:
    if events is None:
        return
    try:
        events.advisor_consult(
            purpose=purpose,
            model=model,
            in_tokens=op.in_tokens,
            out_tokens=op.out_tokens,
            skipped=op.skipped,
            reason=op.reason,
            preview=_short(op.text, 280),
        )
    except Exception:
        # Never let an observability failure bubble up to the agent loop.
        pass


async def consult(
    *,
    purpose: str,
    question: str,
    context: str,
    max_tokens: int | None = None,
    events: Any = None,
) -> AdvisorOpinion:
    """Ask the advisor model. Never raises; returns an `AdvisorOpinion`.

    `purpose` controls (a) the prompt template (`prompts/{purpose}.md.j2`)
    and (b) the junction allowlist gate. Callers should use the canonical
    set: `"stagnation"`, `"audit"`, `"tool"`.
    """
    model = _get_env("AUTOMEDAL_ADVISOR_MODEL", "kimi-k2.6")

    if not is_enabled(purpose):
        op = AdvisorOpinion(skipped=True, reason=f"disabled:{purpose}")
        _emit(events, purpose, model, op)
        return op

    cap_per_consult = _get_int("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT", 2000)
    if max_tokens is None:
        max_tokens = cap_per_consult
    else:
        max_tokens = min(int(max_tokens), cap_per_consult)

    rem = remaining_tokens()
    if rem <= 0:
        op = AdvisorOpinion(skipped=True, reason="budget:iter")
        _emit(events, purpose, model, op)
        return op
    if max_tokens > rem:
        max_tokens = rem

    try:
        prompt = _env.get_template(f"{purpose}.md.j2").render(
            question=question or "(no question)",
            context=context or "(no context)",
        )
    except Exception as exc:
        op = AdvisorOpinion(skipped=True, reason=f"template:{type(exc).__name__}")
        _emit(events, purpose, model, op)
        return op

    base_url = _get_env("AUTOMEDAL_ADVISOR_BASE_URL", "https://opencode.ai/zen/go/v1")
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        op = AdvisorOpinion(skipped=True, reason="no_api_key")
        _emit(events, purpose, model, op)
        return op

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=int(max_tokens),
                messages=[{"role": "user", "content": prompt}],
            )
        finally:
            try:
                await client.close()
            except Exception:
                pass
    except Exception as exc:
        op = AdvisorOpinion(skipped=True, reason=f"error:{type(exc).__name__}")
        _emit(events, purpose, model, op)
        return op

    text = ""
    in_t = 0
    out_t = 0
    try:
        if getattr(resp, "choices", None):
            text = (resp.choices[0].message.content or "").strip()
        if getattr(resp, "usage", None):
            in_t = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            out_t = int(getattr(resp.usage, "completion_tokens", 0) or 0)
    except Exception:
        pass

    consume_tokens(in_t + out_t)

    if not text:
        op = AdvisorOpinion(in_tokens=in_t, out_tokens=out_t, skipped=True, reason="empty")
        _emit(events, purpose, model, op)
        return op

    op = AdvisorOpinion(text=text, in_tokens=in_t, out_tokens=out_t, skipped=False)
    _emit(events, purpose, model, op)
    return op
