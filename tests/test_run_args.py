"""Shared `automedal run` argv parser — handles the --advisor flag."""

from __future__ import annotations

from automedal.run_args import DEFAULT_ADVISOR_MODEL, parse_run_args


def test_no_flags_returns_args_unchanged():
    args, env = parse_run_args(["10"])
    assert args == ["10"]
    assert env == {}


def test_advisor_with_explicit_model():
    args, env = parse_run_args(["10", "--advisor", "kimi-k2.6"])
    assert args == ["10"]
    assert env == {
        "AUTOMEDAL_ADVISOR": "1",
        "AUTOMEDAL_ADVISOR_MODEL": "kimi-k2.6",
    }


def test_advisor_without_model_uses_default():
    args, env = parse_run_args(["10", "--advisor"])
    assert args == ["10"]
    assert env["AUTOMEDAL_ADVISOR"] == "1"
    assert env["AUTOMEDAL_ADVISOR_MODEL"] == DEFAULT_ADVISOR_MODEL


def test_advisor_followed_by_digit_does_not_consume_digit():
    # `--advisor 10` is a user mistake — leave 10 as a positional arg.
    args, env = parse_run_args(["--advisor", "10"])
    assert args == ["10"]
    assert env["AUTOMEDAL_ADVISOR_MODEL"] == DEFAULT_ADVISOR_MODEL


def test_advisor_followed_by_other_flag_does_not_consume_it():
    args, env = parse_run_args(["10", "--advisor", "--fast"])
    assert args == ["10", "--fast"]
    assert env["AUTOMEDAL_ADVISOR_MODEL"] == DEFAULT_ADVISOR_MODEL


def test_flag_position_independent():
    args, env = parse_run_args(["--advisor", "kimi-k2.6", "10", "fast"])
    assert args == ["10", "fast"]
    assert env["AUTOMEDAL_ADVISOR_MODEL"] == "kimi-k2.6"
