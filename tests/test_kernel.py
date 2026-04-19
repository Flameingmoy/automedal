"""AgentKernel tests via a fake provider yielding canned ChatTurn sequences.

We exercise:
    - assistant_done early termination
    - tool_call → tool_result round-trip
    - parallel batched tool_calls
    - max_steps cutoff
    - provider error capture
    - tool exception → ok=False propagation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from automedal.agent.kernel import AgentKernel, RunReport
from automedal.agent.providers.base import ChatTurn, ToolCall, Usage
from automedal.agent.tools.base import Tool, ToolResult, make_tool


# ── fake provider ────────────────────────────────────────────────────────────

@dataclass
class FakeProvider:
    """Yields a pre-canned list of ChatTurns, one per chat_stream call."""
    model: str
    turns: list[ChatTurn]
    raise_on: int | None = None
    _call: int = 0

    async def chat_stream(self, *, system, messages, tools, events) -> ChatTurn:
        idx = self._call
        self._call += 1
        if self.raise_on is not None and idx == self.raise_on:
            raise RuntimeError("simulated provider blowup")
        if idx >= len(self.turns):
            # Default: empty assistant turn ends the loop
            return ChatTurn(
                assistant_blocks=[{"type": "text", "text": ""}],
                assistant_text="", tool_calls=[], usage=Usage(), stop_reason="end_turn",
            )
        return self.turns[idx]


# ── helpers ──────────────────────────────────────────────────────────────────

def _txt(text: str, **u) -> ChatTurn:
    return ChatTurn(
        assistant_blocks=[{"type": "text", "text": text}],
        assistant_text=text,
        tool_calls=[],
        usage=Usage(in_tokens=u.get("in", 5), out_tokens=u.get("out", 7)),
        stop_reason="end_turn",
    )


def _call(text: str, calls: list[ToolCall], **u) -> ChatTurn:
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for c in calls:
        blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
    return ChatTurn(
        assistant_blocks=blocks,
        assistant_text=text,
        tool_calls=calls,
        usage=Usage(in_tokens=u.get("in", 10), out_tokens=u.get("out", 4)),
        stop_reason="tool_use",
    )


def _echo_tool(name: str = "echo") -> Tool:
    def fn(value: str) -> ToolResult:
        return ToolResult(text=f"echo:{value}")
    return make_tool(
        name=name,
        description="echo",
        schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        fn=fn,
    )


def _boom_tool() -> Tool:
    def fn() -> ToolResult:
        raise ValueError("kaboom")
    return make_tool(
        name="boom",
        description="raises",
        schema={"type": "object", "properties": {}, "required": []},
        fn=fn,
    )


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_assistant_done_terminates_immediately():
    prov = FakeProvider(model="m", turns=[_txt("hi user")])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[], events=None)
    rep = await k.run("hello")
    assert rep.stop == "assistant_done"
    assert rep.final_text == "hi user"
    assert rep.steps == 1
    assert rep.usage_total.in_tokens == 5
    assert rep.usage_total.out_tokens == 7


@pytest.mark.asyncio
async def test_tool_call_roundtrip_appends_tool_message():
    call = ToolCall(id="t1", name="echo", args={"value": "x"})
    prov = FakeProvider(model="m", turns=[_call("calling", [call]), _txt("done")])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_echo_tool()], events=None)
    rep = await k.run("go")
    assert rep.stop == "assistant_done"
    assert rep.final_text == "done"
    assert rep.steps == 2
    # transcript: user, assistant(tool_use), tool, assistant(text)
    roles = [m["role"] for m in rep.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert rep.messages[2]["tool_use_id"] == "t1"
    assert rep.messages[2]["content"] == "echo:x"


@pytest.mark.asyncio
async def test_parallel_batched_tool_calls_run_concurrently():
    calls = [
        ToolCall(id="a", name="echo", args={"value": "1"}),
        ToolCall(id="b", name="echo", args={"value": "2"}),
        ToolCall(id="c", name="echo", args={"value": "3"}),
    ]
    prov = FakeProvider(model="m", turns=[_call("", calls), _txt("done")])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_echo_tool()], events=None)
    rep = await k.run("go")
    # Three tool messages between the two assistant turns
    tool_msgs = [m for m in rep.messages if m["role"] == "tool"]
    assert [m["tool_use_id"] for m in tool_msgs] == ["a", "b", "c"]
    assert [m["content"] for m in tool_msgs] == ["echo:1", "echo:2", "echo:3"]


@pytest.mark.asyncio
async def test_unknown_tool_call_returns_error_to_model():
    call = ToolCall(id="t", name="nonexistent", args={})
    prov = FakeProvider(model="m", turns=[_call("", [call]), _txt("recovered")])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_echo_tool()], events=None)
    rep = await k.run("go")
    tool_msg = next(m for m in rep.messages if m["role"] == "tool")
    assert tool_msg.get("is_error") is True
    assert "unknown tool" in tool_msg["content"]


@pytest.mark.asyncio
async def test_tool_exception_marks_is_error_true():
    call = ToolCall(id="t", name="boom", args={})
    prov = FakeProvider(model="m", turns=[_call("", [call]), _txt("ok")])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_boom_tool()], events=None)
    rep = await k.run("go")
    tool_msg = next(m for m in rep.messages if m["role"] == "tool")
    assert tool_msg.get("is_error") is True
    assert "kaboom" in tool_msg["content"]


@pytest.mark.asyncio
async def test_max_steps_cutoff_when_model_loops_on_tools():
    call = ToolCall(id="t", name="echo", args={"value": "loop"})
    # provider returns tool calls forever
    prov = FakeProvider(model="m", turns=[_call("", [call])] * 10)
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_echo_tool()],
                    events=None, max_steps=3)
    rep = await k.run("go")
    assert rep.stop == "max_steps"
    assert rep.steps == 3


@pytest.mark.asyncio
async def test_provider_error_captured_in_run_report():
    prov = FakeProvider(model="m", turns=[], raise_on=0)
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[], events=None)
    rep = await k.run("go")
    assert rep.stop == "provider_error"
    assert rep.error and "simulated provider blowup" in rep.error


@pytest.mark.asyncio
async def test_usage_accumulates_across_steps():
    call = ToolCall(id="t", name="echo", args={"value": "v"})
    prov = FakeProvider(model="m", turns=[
        _call("a", [call], **{"in": 100, "out": 20}),
        _txt("done", **{"in": 50, "out": 10}),
    ])
    k = AgentKernel(provider=prov, system_prompt="sys", tools=[_echo_tool()], events=None)
    rep = await k.run("go")
    assert rep.usage_total.in_tokens == 150
    assert rep.usage_total.out_tokens == 30
