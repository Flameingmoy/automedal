"""Shared helpers for phase orchestrators."""

from __future__ import annotations

from typing import Any

from automedal.agent.events import EventSink
from automedal.agent.kernel import AgentKernel, RunReport
from automedal.agent.prompts import render_prompt
from automedal.agent.tools import Tool


_PHASE_SYSTEM = (
    "You are an AutoMedal phase agent. Read the phase instructions in the "
    "user message carefully and use the provided tools to accomplish the "
    "task. When you are done, produce a brief final assistant message "
    "summarizing what you did. Do not chat — be terse and factual."
)


async def run_phase(
    *,
    phase: str,
    provider,
    tools: list[Tool],
    events: EventSink | None,
    max_steps: int = 50,
    slots: dict[str, Any],
) -> RunReport:
    """Render the phase prompt + run a fresh kernel. Returns the kernel report."""
    user_message = render_prompt(phase, **slots)

    sink = events.with_phase(phase) if events is not None else None
    if sink is not None:
        sink.phase_start()

    kernel = AgentKernel(
        provider=provider,
        system_prompt=_PHASE_SYSTEM,
        tools=tools,
        events=sink,
        max_steps=max_steps,
    )
    report = await kernel.run(user_message)

    if sink is not None:
        sink.phase_end(
            stop=report.stop,
            usage={"in": report.usage_total.in_tokens, "out": report.usage_total.out_tokens},
        )
    return report
