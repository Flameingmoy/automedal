"""Experimenter-eval phase — parse training result, write journal, commit/revert."""

from __future__ import annotations

from automedal.agent.events import EventSink
from automedal.agent.kernel import RunReport
from automedal.agent.phases._common import run_phase
from automedal.agent.tools import (
    READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RUN_SHELL,
)


def _eval_tools():
    return [READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RUN_SHELL]


async def run(
    *,
    provider,
    events: EventSink | None,
    exp_id: str,
    best_val_loss: float | str,
    train_rc: int,
    final_loss: float | str,
    max_steps: int = 30,
) -> RunReport:
    return await run_phase(
        phase="experimenter_eval",
        provider=provider,
        tools=_eval_tools(),
        events=events,
        max_steps=max_steps,
        slots={
            "exp_id": exp_id,
            "best_val_loss": best_val_loss,
            "train_rc": train_rc,
            "final_loss": final_loss,
        },
    )
