"""AgentKernel — the bespoke async tool-call loop.

This is the entire agent runtime: ~150 lines of pure orchestration.
No middleware, no graph, no framework. Just:

    while step < max_steps:
        turn = await provider.chat_stream(...)
        if not turn.tool_calls:
            return    # assistant produced final text
        for call in turn.tool_calls:
            result = await tool(**call.args)
            messages.append(tool_result for call.id)

The kernel is provider-agnostic: it speaks the internal Anthropic-flavored
message shape defined in `providers/base.py` and lets each provider adapter
translate at the wire. Tools are plain `Tool` dataclasses (see `tools/base.py`).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from automedal.agent.doom_loop import check_for_doom_loop
from automedal.agent.errors import format_error
from automedal.agent.messages import patch_dangling_tool_calls
from automedal.agent.providers.base import ChatProvider, ChatTurn, ToolCall, Usage
from automedal.agent.tools import Tool, ToolResult


_LENGTH_STOP_REASONS = {"length", "max_tokens", "max_output_tokens"}


def _is_length_stop(stop_reason: str) -> bool:
    """Providers vary ('length' for OpenAI, 'max_tokens' for Anthropic)."""
    return (stop_reason or "").lower() in _LENGTH_STOP_REASONS


@dataclass
class RunReport:
    """Summary of one kernel run."""
    stop: str                           # "assistant_done" | "max_steps" | "provider_error"
    final_text: str                     # last assistant text (or empty)
    messages: list[dict]                # full transcript including tool turns
    steps: int
    usage_total: Usage
    error: str | None = None


@dataclass
class AgentKernel:
    """One kernel = one phase invocation. Build a fresh kernel per phase."""

    provider: ChatProvider
    system_prompt: str
    tools: list[Tool]
    events: Any                                       # EventSink or None
    max_steps: int = 50
    parallel_tool_calls: bool = True                  # honor the model's batched tool_use blocks

    _tools_by_name: dict[str, Tool] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._tools_by_name = {t.name: t for t in self.tools}

    async def run(self, user_message: str) -> RunReport:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        usage_total = Usage()

        for step in range(1, self.max_steps + 1):
            if self.events is not None:
                self.events.step_advance()

            # Self-healing: patch any unanswered tool_use blocks from a
            # prior interrupted step so the provider doesn't 400 on us.
            stubs = patch_dangling_tool_calls(messages)
            if stubs and self.events is not None:
                try:
                    self.events.notice(
                        tag="self_heal",
                        message=f"patched {stubs} dangling tool_use block(s)",
                    )
                except Exception:
                    pass

            # Doom-loop guard: detect repetition / cycles in the recent tool
            # calls and inject a corrective user message. Cheap; runs every step.
            if (doom := check_for_doom_loop(messages)) is not None:
                messages.append({"role": "user", "content": doom})
                if self.events is not None:
                    try:
                        self.events.notice(tag="doom_loop", message=doom)
                    except Exception:
                        pass

            try:
                turn: ChatTurn = await self.provider.chat_stream(
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.tools,
                    events=self.events,
                )
            except Exception as exc:
                if self.events is not None:
                    self.events.error(where=f"provider.chat_stream step={step}", exc=exc)
                return RunReport(
                    stop="provider_error",
                    final_text="",
                    messages=messages,
                    steps=step,
                    usage_total=usage_total,
                    error=format_error(exc),
                )

            usage_total.in_tokens += turn.usage.in_tokens
            usage_total.out_tokens += turn.usage.out_tokens
            if self.events is not None and (turn.usage.in_tokens or turn.usage.out_tokens):
                self.events.usage(in_tokens=turn.usage.in_tokens, out_tokens=turn.usage.out_tokens)

            # Truncation handler: when the model ran out of output budget
            # mid-tool-call the JSON arguments are garbage. Drop the calls,
            # keep any text prefix, and inject a hint to retry with smaller
            # content (heredocs / split edits).
            if _is_length_stop(turn.stop_reason) and turn.tool_calls:
                dropped_names = [tc.name for tc in turn.tool_calls]
                text_blocks = [b for b in turn.assistant_blocks if b.get("type") == "text"]
                messages.append({"role": "assistant", "content": text_blocks})
                hint = (
                    "Your previous response was truncated by the output token limit, so "
                    f"the following tool calls were dropped: {dropped_names}. Do NOT "
                    "retry with the same large content. For 'write_file' use bash with "
                    "cat<<'HEREDOC', or split into multiple smaller edit_file calls."
                )
                messages.append({"role": "user", "content": hint})
                if self.events is not None:
                    try:
                        self.events.notice(
                            tag="truncation",
                            message=f"stop=length; dropped {dropped_names}",
                        )
                    except Exception:
                        pass
                continue

            # Echo assistant blocks back into the transcript verbatim
            messages.append({"role": "assistant", "content": turn.assistant_blocks})

            if not turn.tool_calls:
                return RunReport(
                    stop="assistant_done",
                    final_text=turn.assistant_text,
                    messages=messages,
                    steps=step,
                    usage_total=usage_total,
                )

            # Execute tool calls — in parallel when the model batched several
            results = await self._execute_tools(turn.tool_calls)

            # Append a tool_result for every call, in the same order
            for call, result in zip(turn.tool_calls, results):
                messages.append({
                    "role": "tool",
                    "tool_use_id": call.id,
                    "content": result.text,
                    **({"is_error": True} if not result.ok else {}),
                })

        return RunReport(
            stop="max_steps",
            final_text="",
            messages=messages,
            steps=self.max_steps,
            usage_total=usage_total,
        )

    # ── tool execution ───────────────────────────────────────────────────────

    async def _execute_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        async def _one(call: ToolCall) -> ToolResult:
            tool = self._tools_by_name.get(call.name)
            if self.events is not None:
                self.events.tool_start(call_id=call.id, name=call.name, args=call.args)
            if tool is None:
                res = ToolResult(text=f"error: unknown tool {call.name!r}", ok=False)
            else:
                res = await tool(**call.args)
            if self.events is not None:
                self.events.tool_end(call_id=call.id, name=call.name, ok=res.ok, result=res.text)
            return res

        if self.parallel_tool_calls and len(calls) > 1:
            return await asyncio.gather(*(_one(c) for c in calls))
        return [await _one(c) for c in calls]
