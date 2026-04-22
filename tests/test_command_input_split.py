"""TUI command-input parser used to detect the `--advisor <prefix>` autocomplete window."""

from __future__ import annotations

from tui.widgets.command_input import _split_for_advisor


def test_no_advisor_flag_falls_through():
    is_adv, prefix, head = _split_for_advisor("run 10")
    assert is_adv is False
    assert head == "run 10"


def test_advisor_with_no_prefix_yet():
    is_adv, prefix, head = _split_for_advisor("run 10 --advisor")
    assert is_adv is True
    assert prefix == ""
    assert head == "run 10 --advisor "


def test_advisor_with_partial_prefix():
    is_adv, prefix, head = _split_for_advisor("run 10 --advisor ki")
    assert is_adv is True
    assert prefix == "ki"
    assert head == "run 10 --advisor "


def test_advisor_with_completed_model_and_trailing_token_falls_through():
    # If the user has already typed a model and continued, don't fight them.
    is_adv, prefix, head = _split_for_advisor("run 10 --advisor kimi-k2.6 fast")
    assert is_adv is False
