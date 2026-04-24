"""Coverage for the plan tool (automedal/agent/tools/plan.py)."""

from __future__ import annotations

import asyncio

import pytest

from automedal.agent.tools.plan import make_plan_tool


class _FakeEvents:
    def __init__(self) -> None:
        self.notices: list[tuple[str, str]] = []

    def notice(self, *, tag: str, message: str) -> None:
        self.notices.append((tag, message))


def _call(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def test_valid_plan_updates_session_and_emits_event():
    session: dict = {}
    events = _FakeEvents()
    tool = make_plan_tool(session=session, events=events)

    res = _call(tool, items=[
        {"id": "a", "content": "read the task prompt", "status": "completed"},
        {"id": "b", "content": "scan knowledge.md", "status": "in_progress"},
        {"id": "c", "content": "write next 5 queue entries", "status": "pending"},
    ])
    assert res.ok, res.text
    assert len(session["plan"]) == 3
    assert session["plan"][1]["status"] == "in_progress"
    assert len(events.notices) == 1
    assert events.notices[0][0] == "plan_update"


def test_replacement_semantics():
    session: dict = {}
    tool = make_plan_tool(session=session, events=None)
    _call(tool, items=[{"id": "x", "content": "one", "status": "pending"}])
    _call(tool, items=[{"id": "y", "content": "two", "status": "completed"}])
    assert len(session["plan"]) == 1
    assert session["plan"][0]["id"] == "y"


@pytest.mark.parametrize("bad", [
    [],
    "not-a-list",
    [{"id": "", "content": "c", "status": "pending"}],
    [{"id": "a", "content": "", "status": "pending"}],
    [{"id": "a", "content": "c", "status": "invalid"}],
    [{"id": "a", "content": "c", "status": "pending"},
     {"id": "a", "content": "d", "status": "pending"}],   # dup id
    [{"id": "a", "content": "c", "status": "in_progress"},
     {"id": "b", "content": "d", "status": "in_progress"}],  # two in-progress
])
def test_invalid_plan_returns_error(bad):
    session: dict = {}
    tool = make_plan_tool(session=session, events=None)
    res = _call(tool, items=bad)
    assert not res.ok
    assert "error" in res.text.lower()
    assert "plan" not in session


def test_content_strip_and_len_cap():
    session: dict = {}
    tool = make_plan_tool(session=session, events=None)
    # Whitespace gets stripped
    _call(tool, items=[{"id": "a", "content": "  trimmed  ", "status": "pending"}])
    assert session["plan"][0]["content"] == "trimmed"
    # Overlong content rejected
    long = "x" * 281
    res = _call(tool, items=[{"id": "a", "content": long, "status": "pending"}])
    assert not res.ok


def test_tool_surface():
    session: dict = {}
    tool = make_plan_tool(session=session, events=None)
    assert tool.name == "plan"
    assert "pending" in tool.schema["properties"]["items"]["items"]["properties"]["status"]["enum"]
    assert "completed" in tool.schema["properties"]["items"]["items"]["properties"]["status"]["enum"]
