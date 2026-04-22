"""Orchestrator wiring — `advisor_note` flows into the Strategist prompt.

Rather than drive the full `run_loop._loop` (which spawns subprocesses for
training), we verify the three wiring contracts directly:

1. `phases.strategist.run` forwards its `advisor_note` kwarg into
   `render_prompt` via the `slots` dict.
2. `strategist.md.j2` renders the `## Advisor directive` block when
   `advisor_note` is non-empty, and omits it when empty.
3. The `consult_advisor` tool is added to the strategist tool list only
   when `advisor.is_enabled("tool")` is true.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from automedal.advisor import budget
from automedal.agent.prompts import render_prompt
from automedal.agent.providers.base import ChatTurn, Usage


# ── fake provider that ends immediately ──────────────────────────────────────

@dataclass
class _TerminateProvider:
    model: str = "fake"

    async def chat_stream(self, *, system, messages, tools, events) -> ChatTurn:
        return ChatTurn(
            assistant_blocks=[{"type": "text", "text": "done"}],
            assistant_text="done",
            tool_calls=[],
            usage=Usage(in_tokens=1, out_tokens=1),
            stop_reason="end_turn",
        )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    budget.reset_iteration_budget()
    for k in ("AUTOMEDAL_ADVISOR", "AUTOMEDAL_ADVISOR_JUNCTIONS"):
        monkeypatch.delenv(k, raising=False)
    yield
    budget.reset_iteration_budget()


# ── prompt-rendering contract ────────────────────────────────────────────────

def _render_with_note(note: str) -> str:
    return render_prompt(
        "strategist",
        exp_id="0042",
        iteration=3,
        max_iters=10,
        stagnating=False,
        best_val_loss=0.0512,
        pending=0,
        reflective="(none)",
        ranked="(none)",
        advisor_note=note,
    )


def test_strategist_prompt_includes_advisor_block_when_note_given():
    text = _render_with_note("1. Switch to LightGBM\n2. Raise n_estimators")
    assert "Advisor directive" in text
    assert "Switch to LightGBM" in text
    assert "Raise n_estimators" in text


def test_strategist_prompt_omits_advisor_block_when_note_empty():
    text = _render_with_note("")
    assert "Advisor directive" not in text


# ── phases.strategist.run wiring ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_strategist_run_forwards_advisor_note(monkeypatch):
    captured: dict = {}

    original = render_prompt

    def spy(phase, **slots):
        captured["phase"] = phase
        captured["slots"] = slots
        return original(phase, **slots)

    # _common.py imports render_prompt at module scope; patch there.
    monkeypatch.setattr("automedal.agent.phases._common.render_prompt", spy)

    from automedal.agent.phases import strategist as p_strategist

    await p_strategist.run(
        provider=_TerminateProvider(),
        events=None,
        exp_id="0099",
        iteration=5,
        max_iters=20,
        stagnating=False,
        best_val_loss=0.04,
        pending=0,
        reflective="(none)",
        ranked="(none)",
        advisor_note="ADVISOR_SAYS_X",
        max_steps=1,
    )

    assert captured["phase"] == "strategist"
    slots = captured["slots"]
    assert slots["advisor_note"] == "ADVISOR_SAYS_X"
    assert slots["exp_id"] == "0099"


@pytest.mark.asyncio
async def test_strategist_run_default_advisor_note_is_empty(monkeypatch):
    captured: dict = {}
    original = render_prompt

    def spy(phase, **slots):
        captured["slots"] = slots
        return original(phase, **slots)

    monkeypatch.setattr("automedal.agent.phases._common.render_prompt", spy)

    from automedal.agent.phases import strategist as p_strategist

    await p_strategist.run(
        provider=_TerminateProvider(),
        events=None,
        exp_id="0100",
        iteration=1,
        max_iters=10,
        stagnating=False,
        best_val_loss=0.1,
        pending=5,
        reflective="(none)",
        ranked="(none)",
        max_steps=1,
    )
    assert captured["slots"]["advisor_note"] == ""


# ── consult_advisor tool gating via _extra_tools_factory ─────────────────────

@pytest.mark.asyncio
async def test_consult_advisor_tool_added_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "1")
    # Capture the tool list the kernel is given.
    captured_tools: list = []

    class _KernelSpy:
        def __init__(self, *, provider, system_prompt, tools, events, max_steps):
            captured_tools.extend(tools)

        async def run(self, user_message):
            from automedal.agent.kernel import RunReport
            return RunReport(
                stop="assistant_done",
                final_text="",
                messages=[],
                steps=0,
                usage_total=Usage(),
                error=None,
            )

    monkeypatch.setattr("automedal.agent.phases._common.AgentKernel", _KernelSpy)

    from automedal.agent.phases import strategist as p_strategist

    await p_strategist.run(
        provider=_TerminateProvider(),
        events=None,
        exp_id="0101",
        iteration=1,
        max_iters=10,
        stagnating=False,
        best_val_loss=0.1,
        pending=5,
        reflective="(none)",
        ranked="(none)",
        advisor_note="",
        max_steps=1,
    )
    names = [t.name for t in captured_tools]
    assert "consult_advisor" in names


@pytest.mark.asyncio
async def test_consult_advisor_tool_absent_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "0")
    captured_tools: list = []

    class _KernelSpy:
        def __init__(self, *, provider, system_prompt, tools, events, max_steps):
            captured_tools.extend(tools)

        async def run(self, user_message):
            from automedal.agent.kernel import RunReport
            return RunReport(
                stop="assistant_done",
                final_text="",
                messages=[],
                steps=0,
                usage_total=Usage(),
                error=None,
            )

    monkeypatch.setattr("automedal.agent.phases._common.AgentKernel", _KernelSpy)

    from automedal.agent.phases import strategist as p_strategist

    await p_strategist.run(
        provider=_TerminateProvider(),
        events=None,
        exp_id="0102",
        iteration=1,
        max_iters=10,
        stagnating=False,
        best_val_loss=0.1,
        pending=0,
        reflective="(none)",
        ranked="(none)",
        max_steps=1,
    )
    names = [t.name for t in captured_tools]
    assert "consult_advisor" not in names
