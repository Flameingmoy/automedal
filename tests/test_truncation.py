"""Coverage for the kernel's truncation handler (finish_reason=length + tool_calls)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from automedal.agent.kernel import AgentKernel, _is_length_stop
from automedal.agent.providers.base import ChatTurn, ToolCall, Usage
from automedal.agent.tools import Tool, ToolResult


def test_is_length_stop_recognizes_provider_variants():
    assert _is_length_stop("length")          # OpenAI
    assert _is_length_stop("max_tokens")      # Anthropic
    assert _is_length_stop("LENGTH")          # case-insensitive
    assert _is_length_stop("max_output_tokens")
    assert not _is_length_stop("end_turn")
    assert not _is_length_stop("tool_use")
    assert not _is_length_stop("")


@dataclass
class _ScriptedProvider:
    """Provider stub that returns queued ChatTurns in order."""
    model: str = "test-model"
    turns: list = None

    async def chat_stream(self, *, system, messages, tools, events):
        if not self.turns:
            raise RuntimeError("scripted provider exhausted")
        return self.turns.pop(0)


def _truncated_turn():
    # Assistant started a tool call but ran out of output budget.
    return ChatTurn(
        assistant_blocks=[
            {"type": "text", "text": "writing the file now"},
            {"type": "tool_use", "id": "t1", "name": "write_file", "input": {"_raw": "{incomplete"}},
        ],
        assistant_text="writing the file now",
        tool_calls=[ToolCall(id="t1", name="write_file", args={"_raw": "{incomplete"})],
        usage=Usage(in_tokens=10, out_tokens=4096),
        stop_reason="length",
    )


def _final_turn():
    return ChatTurn(
        assistant_blocks=[{"type": "text", "text": "done"}],
        assistant_text="done",
        tool_calls=[],
        usage=Usage(in_tokens=5, out_tokens=1),
        stop_reason="end_turn",
    )


def test_truncation_drops_tool_calls_and_injects_hint():
    provider = _ScriptedProvider(turns=[_truncated_turn(), _final_turn()])
    kernel = AgentKernel(
        provider=provider,
        system_prompt="sys",
        tools=[],
        events=None,
        max_steps=5,
    )

    report = asyncio.run(kernel.run("go"))
    assert report.stop == "assistant_done"
    # After truncation we expect: user → assistant(text only) → user(hint) → assistant(done)
    roles = [m["role"] for m in report.messages]
    assert roles[:2] == ["user", "assistant"]
    # Find the injected hint
    hint = next((m["content"] for m in report.messages
                 if m["role"] == "user" and isinstance(m["content"], str)
                 and "truncated" in m["content"].lower()), None)
    assert hint is not None
    assert "write_file" in hint
    # The broken tool_use block should NOT appear in the echoed assistant turn.
    trunc_asst = report.messages[1]
    assert isinstance(trunc_asst["content"], list)
    assert all(b.get("type") == "text" for b in trunc_asst["content"])
