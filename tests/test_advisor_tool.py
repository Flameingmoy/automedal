"""`consult_advisor` worker tool — max_uses + skipped-from-advisor handling."""

from __future__ import annotations

import pytest

from automedal.advisor import budget
from automedal.advisor.client import AdvisorOpinion
from automedal.agent.tools import make_advisor_tool


@pytest.fixture(autouse=True)
def _reset_state():
    budget.reset_iteration_budget()
    yield
    budget.reset_iteration_budget()


@pytest.mark.asyncio
async def test_first_call_succeeds_returns_advisor_text(monkeypatch):
    async def fake_consult(*, purpose, question, context, events=None, **_):
        assert purpose == "tool"
        return AdvisorOpinion(text="USE LIGHTGBM", in_tokens=10, out_tokens=5)

    monkeypatch.setattr("automedal.advisor.consult", fake_consult)

    tool = make_advisor_tool(events=None)
    result = await tool(question="lgbm vs xgb?", context_hint="snippet")
    assert result.ok is True
    assert result.text == "USE LIGHTGBM"


@pytest.mark.asyncio
async def test_second_call_in_same_phase_returns_budget_error(monkeypatch):
    calls = {"n": 0}

    async def fake_consult(**_):
        calls["n"] += 1
        return AdvisorOpinion(text="ok")

    monkeypatch.setattr("automedal.advisor.consult", fake_consult)

    tool = make_advisor_tool(events=None, max_uses=1)
    r1 = await tool(question="q1", context_hint="c")
    r2 = await tool(question="q2", context_hint="c")
    assert r1.ok is True
    assert r2.ok is False
    assert "Budget exhausted" in r2.text
    assert calls["n"] == 1  # second call never reached the advisor


@pytest.mark.asyncio
async def test_uses_counter_is_per_instance(monkeypatch):
    async def fake_consult(**_):
        return AdvisorOpinion(text="ok")

    monkeypatch.setattr("automedal.advisor.consult", fake_consult)

    tool_a = make_advisor_tool(events=None, max_uses=1)
    tool_b = make_advisor_tool(events=None, max_uses=1)
    assert (await tool_a(question="q", context_hint="c")).ok is True
    # Fresh instance has its own counter
    assert (await tool_b(question="q", context_hint="c")).ok is True


@pytest.mark.asyncio
async def test_skipped_advisor_yields_unavailable_string(monkeypatch):
    async def fake_consult(**_):
        return AdvisorOpinion(skipped=True, reason="budget:iter")

    monkeypatch.setattr("automedal.advisor.consult", fake_consult)

    tool = make_advisor_tool(events=None)
    result = await tool(question="q", context_hint="c")
    assert result.ok is False
    assert "Advisor unavailable" in result.text
    assert "budget:iter" in result.text


@pytest.mark.asyncio
async def test_tool_schema_requires_question_and_context_hint():
    tool = make_advisor_tool(events=None)
    assert tool.name == "consult_advisor"
    props = tool.schema["properties"]
    assert "question" in props and "context_hint" in props
    assert set(tool.schema["required"]) == {"question", "context_hint"}


@pytest.mark.asyncio
async def test_max_uses_two_allows_two_calls(monkeypatch):
    async def fake_consult(**_):
        return AdvisorOpinion(text="ok")

    monkeypatch.setattr("automedal.advisor.consult", fake_consult)

    tool = make_advisor_tool(events=None, max_uses=2)
    assert (await tool(question="q", context_hint="c")).ok is True
    assert (await tool(question="q", context_hint="c")).ok is True
    assert (await tool(question="q", context_hint="c")).ok is False
