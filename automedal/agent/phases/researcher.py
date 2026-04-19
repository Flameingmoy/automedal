"""Researcher phase — arxiv-driven idea injection into research_notes.md."""

from __future__ import annotations

from typing import Any

from automedal.agent.events import EventSink
from automedal.agent.kernel import RunReport
from automedal.agent.phases._common import run_phase
from automedal.agent.tools import (
    READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP,
    RECALL, ARXIV_SEARCH, make_subagent_tool,
)


def _researcher_tools(provider, events: EventSink | None):
    base = [READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, RECALL, ARXIV_SEARCH]
    sub = make_subagent_tool(provider=provider, parent_tools=base, events=events)
    return base + [sub]


async def run(
    *,
    provider,
    events: EventSink | None,
    exp_id: str,
    trigger: str,
    stagnating: bool | str,
    scheduled_research: bool | int,
    best_val_loss: float | str,
    max_steps: int = 30,
) -> RunReport:
    return await run_phase(
        phase="researcher",
        provider=provider,
        tools=_researcher_tools(provider, events),
        events=events,
        max_steps=max_steps,
        slots={
            "exp_id": exp_id,
            "trigger": trigger,
            "stagnating": stagnating,
            "scheduled_research": int(bool(scheduled_research)),
            "best_val_loss": best_val_loss,
        },
    )
