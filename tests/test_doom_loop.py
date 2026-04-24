"""Coverage for automedal/agent/doom_loop.py."""

from __future__ import annotations

import os

import pytest

from automedal.agent.doom_loop import check_for_doom_loop


def _tool_use(name: str, input_: dict, id_: str = "t") -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


def _assistant(*blocks) -> dict:
    return {"role": "assistant", "content": list(blocks)}


def _tool_turns(names_and_args: list[tuple[str, dict]]) -> list[dict]:
    """Build a transcript where each tool call lives in its own assistant turn,
    each followed by a ``role=tool`` result — matching what the kernel writes."""
    msgs: list[dict] = [{"role": "user", "content": "task"}]
    for i, (name, args) in enumerate(names_and_args):
        call_id = f"t{i}"
        msgs.append(_assistant(_tool_use(name, args, call_id)))
        msgs.append({"role": "tool", "tool_use_id": call_id, "content": "ok"})
    return msgs


def test_identical_three_triggers():
    msgs = _tool_turns([("bash", {"cmd": "echo hi"})] * 3)
    out = check_for_doom_loop(msgs)
    assert out is not None
    assert "bash" in out
    assert "identical" in out.lower() or "3 times" in out.lower()


def test_different_args_does_not_trigger():
    msgs = _tool_turns([
        ("bash", {"cmd": "echo 1"}),
        ("bash", {"cmd": "echo 2"}),
        ("bash", {"cmd": "echo 3"}),
    ])
    assert check_for_doom_loop(msgs) is None


def test_ab_ab_cycle_triggers():
    msgs = _tool_turns([
        ("read_file", {"path": "a"}),
        ("write_file", {"path": "b", "text": "x"}),
        ("read_file", {"path": "a"}),
        ("write_file", {"path": "b", "text": "x"}),
    ])
    out = check_for_doom_loop(msgs)
    assert out is not None
    assert "cycle" in out.lower() or "cycling" in out.lower()


def test_abc_abc_cycle_triggers():
    msgs = _tool_turns([
        ("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 3}),
        ("a", {"x": 1}), ("b", {"y": 2}), ("c", {"z": 3}),
    ])
    assert check_for_doom_loop(msgs) is not None


def test_varied_sequence_does_not_trigger():
    msgs = _tool_turns([
        ("a", {"n": 1}),
        ("b", {"n": 2}),
        ("c", {"n": 3}),
        ("d", {"n": 4}),
    ])
    assert check_for_doom_loop(msgs) is None


def test_empty_transcript_is_none():
    assert check_for_doom_loop([]) is None
    assert check_for_doom_loop([{"role": "user", "content": "hi"}]) is None


def test_env_kill_switch(monkeypatch: pytest.MonkeyPatch):
    msgs = _tool_turns([("bash", {"cmd": "echo hi"})] * 5)
    monkeypatch.setenv("AUTOMEDAL_DOOM_LOOP", "0")
    assert check_for_doom_loop(msgs) is None
    monkeypatch.setenv("AUTOMEDAL_DOOM_LOOP", "1")
    assert check_for_doom_loop(msgs) is not None


def test_two_identical_does_not_trigger():
    """Threshold is 3; two identical calls should not fire."""
    msgs = _tool_turns([("bash", {"cmd": "echo hi"})] * 2)
    assert check_for_doom_loop(msgs) is None
