"""Coverage for the friendly error mapper — each pattern + pass-through."""

from __future__ import annotations

import pytest

from automedal.agent.errors import format_error, friendly_error


@pytest.mark.parametrize("msg", [
    "401 Unauthorized",
    "Invalid x-api-key",
    "invalid api key",
    "Error code: 401 - {'error': 'unauthorized'}",
])
def test_auth_patterns_matched(msg):
    out = friendly_error(RuntimeError(msg))
    assert out is not None
    assert "OPENCODE_API_KEY" in out
    assert "ANTHROPIC_API_KEY" in out


@pytest.mark.parametrize("msg", [
    "Error: insufficient credits on your account",
    "insufficient_quota",
    "HTTP 402: payment required",
])
def test_credits_patterns_matched(msg):
    out = friendly_error(RuntimeError(msg))
    assert out is not None
    assert "credits" in out.lower()


@pytest.mark.parametrize("msg", [
    "model_not_found",
    "The model 'gpt-fake' does not exist",
    "Model claude-opus-9 not found",
])
def test_model_not_found(msg):
    out = friendly_error(RuntimeError(msg))
    assert out is not None
    assert "automedal models" in out


def test_provider_mismatch():
    out = friendly_error(RuntimeError("not supported by provider: cerebras"))
    assert out is not None
    assert "provider" in out.lower()


def test_context_exceeded():
    out = friendly_error(RuntimeError("context_length_exceeded: too many tokens"))
    assert out is not None
    assert "compaction" in out.lower() or "context" in out.lower()


def test_unknown_error_returns_none():
    assert friendly_error(RuntimeError("some totally unknown failure mode")) is None


def test_format_error_includes_raw_and_friendly():
    msg = format_error(RuntimeError("401 Unauthorized"))
    assert "OPENCODE_API_KEY" in msg      # friendly
    assert "[raw]" in msg                  # raw prefix
    assert "RuntimeError" in msg           # exception type


def test_format_error_unknown_falls_back_to_raw():
    msg = format_error(ValueError("something weird"))
    assert msg == "ValueError: something weird"
