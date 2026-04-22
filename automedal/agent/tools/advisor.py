"""`consult_advisor` tool — worker-triggered advisor consult, one per phase.

Included in the tool allowlist for `strategist` and `experimenter_edit` only,
and only when `AUTOMEDAL_ADVISOR=1` and `tool` is in `AUTOMEDAL_ADVISOR_JUNCTIONS`.
The per-phase uses counter lives on the Tool instance (closure), so each call
to `make_advisor_tool()` returns a fresh tool with its own counter — phases
get a new instance from `run_phase` every iteration.
"""

from __future__ import annotations

from typing import Any

from automedal.agent.tools.base import Tool, ToolResult


_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "The specific decision or tradeoff you want adjudicated. "
                "Be concrete — name the model, feature, or hyperparameter."
            ),
        },
        "context_hint": {
            "type": "string",
            "description": (
                "Short summary of the relevant code or state the advisor "
                "should consider. Include the failing/contested fragment verbatim "
                "if small; otherwise paraphrase."
            ),
        },
    },
    "required": ["question", "context_hint"],
}


_DESC = (
    "Ask a frontier model (configured advisor, e.g. Kimi K2.6) for a second "
    "opinion on a hard design decision. Expensive — use only when the choice "
    "materially affects outcome. At most one call per phase."
)


def make_advisor_tool(*, events: Any = None, max_uses: int = 1) -> Tool:
    """Return a fresh `consult_advisor` Tool with its own per-phase uses counter."""
    uses = {"n": 0}

    async def run(*, question: str, context_hint: str) -> ToolResult:
        if uses["n"] >= max_uses:
            return ToolResult(
                text="Budget exhausted for this phase — proceed without advisor.",
                ok=False,
            )
        uses["n"] += 1

        from automedal.advisor import consult

        opinion = await consult(
            purpose="tool",
            question=question,
            context=context_hint,
            events=events,
        )
        if opinion.skipped:
            return ToolResult(
                text=f"Advisor unavailable ({opinion.reason}) — proceed without advisor.",
                ok=False,
            )
        return ToolResult(text=opinion.text, ok=True)

    return Tool(name="consult_advisor", description=_DESC, schema=_SCHEMA, run=run)


__all__ = ["make_advisor_tool"]
