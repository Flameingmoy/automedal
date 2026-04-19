"""Strategist phase — curates knowledge.md and writes the next 5 queue entries."""

from __future__ import annotations

from automedal.agent.events import EventSink
from automedal.agent.kernel import RunReport
from automedal.agent.phases._common import run_phase
from automedal.agent.tools import (
    READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL,
)


def _strategist_tools():
    return [READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL]


async def run(
    *,
    provider,
    events: EventSink | None,
    exp_id: str,
    iteration: int,
    max_iters: int,
    stagnating: bool | str,
    best_val_loss: float | str,
    pending: int,
    reflective: str,
    ranked: str,
    max_steps: int = 30,
) -> RunReport:
    return await run_phase(
        phase="strategist",
        provider=provider,
        tools=_strategist_tools(),
        events=events,
        max_steps=max_steps,
        slots={
            "exp_id": exp_id,
            "iteration": iteration,
            "max_iters": max_iters,
            "stagnating": stagnating,
            "best_val_loss": best_val_loss,
            "pending": pending,
            "reflective": reflective,
            "ranked": ranked,
        },
    )
