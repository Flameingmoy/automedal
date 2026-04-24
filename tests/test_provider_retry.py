"""Retry wrapper — success after flakes, non-retry on permanent errors."""

from __future__ import annotations

import asyncio

import pytest

from automedal.agent.retry import is_transient_error, with_retry


class _RecordingEvents:
    def __init__(self) -> None:
        self.notices: list[tuple[str, str]] = []

    def tool_log(self, *, tool: str, log: str) -> None:
        self.notices.append((tool, log))


@pytest.mark.parametrize("msg", [
    "503 Service Unavailable",
    "HTTP 429: rate limit exceeded",
    "Connection reset by peer",
    "Bad gateway",
    "Timed out",
    "remote end closed connection without response",
])
def test_transient_patterns_matched(msg):
    assert is_transient_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "401 Unauthorized",
    "400 Invalid request",
    "model_not_found",
    "insufficient_credits",
])
def test_non_transient_patterns(msg):
    assert is_transient_error(RuntimeError(msg)) is False


def test_timeout_error_type_matches():
    assert is_transient_error(asyncio.TimeoutError()) is True
    assert is_transient_error(TimeoutError()) is True


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_succeeds_on_first_attempt():
    events = _RecordingEvents()

    async def call():
        return "ok"

    result = _run(with_retry(call, label="test", events=events))
    assert result == "ok"
    assert events.notices == []


def test_retries_on_transient_then_succeeds(monkeypatch):
    events = _RecordingEvents()
    sleeps: list[int] = []

    async def fake_sleep(n):
        sleeps.append(n)

    monkeypatch.setattr("automedal.agent.retry.asyncio.sleep", fake_sleep)

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 service unavailable")
        return "ok"

    result = _run(with_retry(flaky, label="flaky.op", events=events))
    assert result == "ok"
    assert calls["n"] == 3
    assert sleeps == [5, 15]           # two retries before the 3rd succeeds
    assert len(events.notices) == 2
    assert all(n[0] == "retry" for n in events.notices)
    assert "flaky.op" in events.notices[0][1]


def test_does_not_retry_on_permanent_error(monkeypatch):
    events = _RecordingEvents()
    slept = []
    monkeypatch.setattr("automedal.agent.retry.asyncio.sleep",
                        lambda n: (slept.append(n), asyncio.sleep(0))[1])

    calls = {"n": 0}

    async def auth_fails():
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")

    with pytest.raises(RuntimeError, match="401"):
        _run(with_retry(auth_fails, label="test", events=events))
    assert calls["n"] == 1
    assert slept == []
    assert events.notices == []


def test_raises_after_max_attempts(monkeypatch):
    events = _RecordingEvents()
    async def _noop_sleep(n):
        return None
    monkeypatch.setattr("automedal.agent.retry.asyncio.sleep", _noop_sleep)

    calls = {"n": 0}

    async def always_503():
        calls["n"] += 1
        raise RuntimeError("503 bad gateway")

    with pytest.raises(RuntimeError, match="503"):
        _run(with_retry(always_503, label="test", events=events))
    assert calls["n"] == 3
    assert len(events.notices) == 2    # one per retry (N-1 retries for N attempts)


def test_events_optional(monkeypatch):
    async def _noop_sleep(n):
        return None
    monkeypatch.setattr("automedal.agent.retry.asyncio.sleep", _noop_sleep)

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("timeout")
        return "ok"

    # Should not explode when events=None
    assert _run(with_retry(flaky, label="test", events=None)) == "ok"


def test_broken_events_swallowed(monkeypatch):
    async def _noop_sleep(n):
        return None
    monkeypatch.setattr("automedal.agent.retry.asyncio.sleep", _noop_sleep)

    class Broken:
        def tool_log(self, **kw):
            raise Exception("event sink broken")

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("502")
        return "ok"

    # Retry should still work even if the event sink raises
    assert _run(with_retry(flaky, label="test", events=Broken())) == "ok"
