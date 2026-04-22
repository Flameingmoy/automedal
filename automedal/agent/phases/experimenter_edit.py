"""Experimenter-edit phase — pop top queue entry and edit agent/train.py."""

from __future__ import annotations

from automedal import advisor
from automedal.agent.events import EventSink
from automedal.agent.kernel import RunReport
from automedal.agent.phases._common import run_phase
from automedal.agent.tools import (
    EDIT_FILE, GREP, LIST_DIR, READ_FILE, RECALL, RUN_SHELL, WRITE_FILE,
    make_advisor_tool,
)


def _experimenter_tools():
    return [READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL, RUN_SHELL]


async def run(
    *,
    provider,
    events: EventSink | None,
    exp_id: str,
    best_val_loss: float | str,
    retry: bool = False,
    prev_loss: float | str = "",
    max_steps: int = 50,
) -> RunReport:
    def _extra_tools(sink):
        if advisor.is_enabled("tool"):
            return [make_advisor_tool(events=sink)]
        return []

    return await run_phase(
        phase="experimenter",
        provider=provider,
        tools=_experimenter_tools(),
        events=events,
        max_steps=max_steps,
        slots={
            "exp_id": exp_id,
            "best_val_loss": best_val_loss,
            "retry": retry,
            "prev_loss": prev_loss,
        },
        extra_tools_factory=_extra_tools,
    )
