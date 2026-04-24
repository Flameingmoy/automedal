"""Coverage for patch_dangling_tool_calls in automedal/agent/messages.py."""

from __future__ import annotations

from automedal.agent.messages import patch_dangling_tool_calls


def _assistant_tool_use(*blocks):
    return {"role": "assistant", "content": list(blocks)}


def _tool(id_: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_use_id": id_, "content": content}


def test_clean_transcript_is_noop():
    msgs = [
        {"role": "user", "content": "do a thing"},
        _assistant_tool_use({"type": "tool_use", "id": "t1", "name": "bash", "input": {}}),
        _tool("t1"),
        _assistant_tool_use({"type": "text", "text": "done"}),
    ]
    assert patch_dangling_tool_calls(msgs) == 0
    assert len(msgs) == 4


def test_single_dangling_tool_call_gets_stub():
    msgs = [
        {"role": "user", "content": "do a thing"},
        _assistant_tool_use({"type": "tool_use", "id": "t1", "name": "bash", "input": {}}),
    ]
    appended = patch_dangling_tool_calls(msgs)
    assert appended == 1
    assert msgs[-1]["role"] == "tool"
    assert msgs[-1]["tool_use_id"] == "t1"
    assert msgs[-1].get("is_error") is True


def test_multiple_dangling_calls_all_stubbed():
    msgs = [
        {"role": "user", "content": "x"},
        _assistant_tool_use(
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            {"type": "tool_use", "id": "t3", "name": "c", "input": {}},
        ),
    ]
    appended = patch_dangling_tool_calls(msgs)
    assert appended == 3
    ids = [m["tool_use_id"] for m in msgs if m.get("role") == "tool"]
    assert ids == ["t1", "t2", "t3"]


def test_partially_answered_only_patches_missing():
    msgs = [
        {"role": "user", "content": "x"},
        _assistant_tool_use(
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
        ),
        _tool("t1"),
    ]
    appended = patch_dangling_tool_calls(msgs)
    assert appended == 1
    # t2 stub should follow t1's real result
    assert msgs[-1]["tool_use_id"] == "t2"


def test_user_shaped_tool_results_count_as_answered():
    """Anthropic provider flattens tool messages into user role with tool_result blocks."""
    msgs = [
        {"role": "user", "content": "x"},
        _assistant_tool_use({"type": "tool_use", "id": "t1", "name": "a", "input": {}}),
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]},
    ]
    assert patch_dangling_tool_calls(msgs) == 0


def test_idempotent():
    msgs = [
        {"role": "user", "content": "x"},
        _assistant_tool_use({"type": "tool_use", "id": "t1", "name": "a", "input": {}}),
    ]
    assert patch_dangling_tool_calls(msgs) == 1
    assert patch_dangling_tool_calls(msgs) == 0  # nothing left to patch


def test_empty_or_text_only_transcripts():
    assert patch_dangling_tool_calls([]) == 0
    assert patch_dangling_tool_calls([{"role": "user", "content": "hi"}]) == 0
    assert patch_dangling_tool_calls([
        _assistant_tool_use({"type": "text", "text": "no tools here"}),
    ]) == 0
