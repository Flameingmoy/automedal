"""EventSink JSONL + human-mirror tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automedal.agent.events import EventSink


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_jsonl_records_phase_events(tmp_path):
    j = tmp_path / "events.jsonl"
    h = tmp_path / "log.txt"
    with EventSink(jsonl_path=j, human_path=h, phase="researcher") as sink:
        sink.phase_start()
        sink.step_advance()
        sink.delta("hello ")
        sink.delta("world")
        sink.phase_end(usage={"in": 100, "out": 50}, stop="end_turn")

    events = _read_jsonl(j)
    kinds = [e["kind"] for e in events]
    assert kinds == ["phase_start", "delta", "delta", "phase_end"]
    assert all(e["phase"] == "researcher" for e in events)
    assert events[1]["text"] == "hello "
    assert events[3]["usage"] == {"in": 100, "out": 50}


def test_human_mirror_writes_legacy_format(tmp_path):
    j = tmp_path / "events.jsonl"
    h = tmp_path / "log.txt"
    with EventSink(jsonl_path=j, human_path=h, phase="strategist") as sink:
        sink.phase_start()
        sink.tool_start(call_id="x1", name="read_file", args={"path": "knowledge.md"})
        sink.tool_end(call_id="x1", name="read_file", ok=True, result="contents…")
        sink.phase_end(stop="end_turn")

    log = h.read_text()
    assert "========== phase: strategist ==========" in log
    assert "[tool] read_file(path='knowledge.md')" in log
    assert "[tool] read_file → ok" in log


def test_inline_delta_followed_by_tool_event_terminates_line(tmp_path):
    j = tmp_path / "events.jsonl"
    h = tmp_path / "log.txt"
    with EventSink(jsonl_path=j, human_path=h, phase="researcher") as sink:
        sink.delta("partial assistant text")
        sink.tool_start(call_id="t", name="grep", args={"pattern": "x"})

    log = h.read_text()
    # The streaming text must end with a newline before the tool line
    assert "partial assistant text\n  [tool] grep" in log


def test_with_phase_returns_sibling_sink_sharing_handles(tmp_path):
    j = tmp_path / "events.jsonl"
    parent = EventSink(jsonl_path=j, phase="parent")
    try:
        child = parent.with_phase("child")
        child.phase_start()
        parent.phase_start()
    finally:
        parent.close()

    rows = _read_jsonl(j)
    phases = [r["phase"] for r in rows]
    assert "child" in phases and "parent" in phases


def test_subagent_event_pair(tmp_path):
    j = tmp_path / "events.jsonl"
    with EventSink(jsonl_path=j, phase="researcher") as sink:
        sink.subagent_start(label="arxiv-1", prompt_preview="search 'tabnet'")
        sink.subagent_end(label="arxiv-1", ok=True, result_preview="3 papers")

    rows = _read_jsonl(j)
    assert [r["kind"] for r in rows] == ["subagent_start", "subagent_end"]
    assert rows[0]["label"] == "arxiv-1"
    assert rows[1]["ok"] is True
