"""Advisor client — gating, success path, error paths, event emission."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from automedal.advisor import budget, client


# ── fake openai SDK ──────────────────────────────────────────────────────────

@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list
    usage: _FakeUsage


class _FakeChatCompletions:
    def __init__(self, *, content="ADVICE", in_t=42, out_t=7, raise_exc=None):
        self.content = content
        self.in_t = in_t
        self.out_t = out_t
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return _FakeResponse(
            choices=[_FakeChoice(message=_FakeMessage(content=self.content))],
            usage=_FakeUsage(prompt_tokens=self.in_t, completion_tokens=self.out_t),
        )


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeAsyncOpenAI:
    last_init_kwargs: dict = {}
    completions_factory = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        comps = type(self).completions_factory() if type(self).completions_factory else _FakeChatCompletions()
        self.chat = _FakeChat(comps)

    async def close(self):
        return None


def _install_fake_openai(monkeypatch, *, completions=None):
    """Install a fake `openai` module so client.consult uses our stubs."""
    if completions is None:
        completions = _FakeChatCompletions()

    _FakeAsyncOpenAI.completions_factory = lambda: completions  # type: ignore[assignment]
    fake_mod = types.ModuleType("openai")
    fake_mod.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    return completions


# ── recording event sink ─────────────────────────────────────────────────────

class _Recorder:
    def __init__(self):
        self.events: list[dict] = []

    def advisor_consult(self, **kw):
        self.events.append(kw)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    budget.reset_iteration_budget()
    for k in (
        "AUTOMEDAL_ADVISOR",
        "AUTOMEDAL_ADVISOR_MODEL",
        "AUTOMEDAL_ADVISOR_BASE_URL",
        "AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER",
        "AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT",
        "AUTOMEDAL_ADVISOR_JUNCTIONS",
        "OPENCODE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "1")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")
    yield
    budget.reset_iteration_budget()


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_consult_skipped_when_master_disabled(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "0")
    rec = _Recorder()
    op = await client.consult(purpose="stagnation", question="q", context="c", events=rec)
    assert op.skipped is True
    assert op.reason.startswith("disabled:")
    assert op.text == ""
    assert rec.events and rec.events[0]["skipped"] is True


@pytest.mark.asyncio
async def test_consult_skipped_when_junction_not_allowed(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_JUNCTIONS", "audit")
    op = await client.consult(purpose="stagnation", question="q", context="c")
    assert op.skipped is True and op.reason == "disabled:stagnation"


@pytest.mark.asyncio
async def test_consult_skipped_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    op = await client.consult(purpose="tool", question="q", context="c")
    assert op.skipped is True and op.reason == "no_api_key"


@pytest.mark.asyncio
async def test_consult_skipped_when_budget_exhausted():
    budget.consume_tokens(99999)
    op = await client.consult(purpose="tool", question="q", context="c")
    assert op.skipped is True and op.reason == "budget:iter"


@pytest.mark.asyncio
async def test_consult_success_path_emits_event_and_consumes_budget(monkeypatch):
    comps = _install_fake_openai(monkeypatch)
    rec = _Recorder()
    op = await client.consult(
        purpose="tool", question="hello?", context="ctx", events=rec
    )
    assert op.skipped is False
    assert op.text == "ADVICE"
    assert op.in_tokens == 42 and op.out_tokens == 7
    # budget consumption tracked
    assert budget.budget_state()["used_this_iter"] == 49
    # event emitted
    assert len(rec.events) == 1
    e = rec.events[0]
    assert e["purpose"] == "tool" and e["skipped"] is False
    assert "ADVICE" in e["preview"]
    # one chat completion call with kimi-k2.6 default model
    assert comps.calls and comps.calls[0]["model"] == "kimi-k2.6"


@pytest.mark.asyncio
async def test_consult_uses_advisor_model_env_override(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_MODEL", "kimi-custom-7b")
    comps = _install_fake_openai(monkeypatch)
    op = await client.consult(purpose="tool", question="q", context="c")
    assert op.skipped is False
    assert comps.calls[0]["model"] == "kimi-custom-7b"


@pytest.mark.asyncio
async def test_consult_caps_max_tokens_to_remaining(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", "100")
    comps = _install_fake_openai(monkeypatch)
    await client.consult(
        purpose="tool", question="q", context="c", max_tokens=500
    )
    # 100 < per_consult cap (2000) and < requested (500), so it wins
    assert comps.calls[0]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_consult_handles_provider_exception(monkeypatch):
    completions = _FakeChatCompletions(raise_exc=RuntimeError("kapow"))
    _install_fake_openai(monkeypatch, completions=completions)
    rec = _Recorder()
    op = await client.consult(purpose="tool", question="q", context="c", events=rec)
    assert op.skipped is True
    assert op.reason.startswith("error:")
    assert rec.events and rec.events[0]["skipped"] is True


@pytest.mark.asyncio
async def test_consult_handles_empty_response(monkeypatch):
    completions = _FakeChatCompletions(content="", in_t=10, out_t=0)
    _install_fake_openai(monkeypatch, completions=completions)
    op = await client.consult(purpose="tool", question="q", context="c")
    assert op.skipped is True and op.reason == "empty"
    # we still consumed input tokens
    assert budget.budget_state()["used_this_iter"] == 10


@pytest.mark.asyncio
async def test_consult_unknown_purpose_skipped_template_error(monkeypatch):
    # Allow the junction so we get past the gate and hit the template loader.
    monkeypatch.setenv(
        "AUTOMEDAL_ADVISOR_JUNCTIONS", "stagnation,audit,tool,not_a_real_purpose"
    )
    _install_fake_openai(monkeypatch)
    op = await client.consult(
        purpose="not_a_real_purpose", question="q", context="c"
    )
    assert op.skipped is True and op.reason.startswith("template:")
