"""Analyzer phase — compress one iteration's signal into knowledge.md."""

from __future__ import annotations

from automedal.agent.events import EventSink
from automedal.agent.kernel import RunReport
from automedal.agent.phases._common import run_phase
from automedal.agent.tools import (
    READ_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL,
)


def _analyzer_tools():
    # No write_file — analyzer must edit knowledge.md in place.
    return [READ_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL]


async def run(
    *,
    provider,
    events: EventSink | None,
    exp_id: str,
    slug: str,
    status: str,
    final_loss: float | str,
    best_val_loss: float | str,
    val_loss_delta: float | str,
    max_steps: int = 20,
) -> RunReport:
    return await run_phase(
        phase="analyzer",
        provider=provider,
        tools=_analyzer_tools(),
        events=events,
        max_steps=max_steps,
        slots={
            "exp_id": exp_id,
            "slug": slug,
            "status": status,
            "final_loss": final_loss,
            "best_val_loss": best_val_loss,
            "val_loss_delta": val_loss_delta,
        },
    )
