"""Per-iteration budget + junction gating for the advisor."""

from __future__ import annotations

import pytest

from automedal.advisor import budget


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with a clean budget + a known env."""
    budget.reset_iteration_budget()
    # Default env for each test — individual tests may override via monkeypatch.
    for k in (
        "AUTOMEDAL_ADVISOR",
        "AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER",
        "AUTOMEDAL_ADVISOR_JUNCTIONS",
    ):
        monkeypatch.delenv(k, raising=False)
    yield
    budget.reset_iteration_budget()


def test_remaining_defaults_to_cap_when_unused():
    assert budget.remaining_tokens() == 8000


def test_consume_decrements_remaining():
    budget.consume_tokens(3000)
    assert budget.remaining_tokens() == 5000
    state = budget.budget_state()
    assert state == {"used_this_iter": 3000, "cap_per_iter": 8000, "remaining": 5000}


def test_consume_negative_treated_as_zero():
    budget.consume_tokens(-50)
    assert budget.remaining_tokens() == 8000


def test_remaining_floors_at_zero_when_overspent():
    budget.consume_tokens(20000)
    assert budget.remaining_tokens() == 0


def test_reset_clears_used():
    budget.consume_tokens(1000)
    budget.reset_iteration_budget()
    assert budget.remaining_tokens() == 8000


def test_cap_respects_env(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", "200")
    budget.consume_tokens(150)
    assert budget.remaining_tokens() == 50


def test_is_enabled_false_when_master_off(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "0")
    assert budget.is_enabled("stagnation") is False
    assert budget.is_enabled("tool") is False
    assert budget.is_enabled() is False


def test_is_enabled_true_when_master_on_no_junction(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "1")
    assert budget.is_enabled() is True


def test_junction_allowlist(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "1")
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_JUNCTIONS", "stagnation,audit")
    assert budget.is_enabled("stagnation") is True
    assert budget.is_enabled("audit") is True
    assert budget.is_enabled("tool") is False


def test_default_junctions_include_all_three(monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_ADVISOR", "1")
    for j in ("stagnation", "audit", "tool"):
        assert budget.is_enabled(j) is True
