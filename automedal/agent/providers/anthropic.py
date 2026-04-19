"""Anthropic provider — wraps the official `anthropic` SDK.

Handles three sources behind the same code path:
    base_url=None                            → Anthropic direct
    base_url="https://opencode.ai/zen/go"    → opencode-go (minimax-m2.7 etc)
    base_url=<custom>                        → any Anthropic-compatible gateway

Streaming is via `client.messages.stream(...)`. Per-token text deltas are
forwarded to the EventSink. Thinking blocks (returned by minimax-m2.7
through opencode-go) are surfaced via `events.thinking()` and echoed back
verbatim on the next turn so the upstream "thinking continuity" contract
holds.

Tool messages from our internal shape (role="tool") are repacked into
Anthropic's user-role tool_result blocks before sending.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from automedal.agent.providers.base import ChatProvider, ChatTurn, ToolCall, Usage


@dataclass
class AnthropicProvider:
    model: str
    api_key: str
    base_url: str | None = None
    max_tokens: int = 4096
    timeout: int = 120

    def _client(self):
        from anthropic import AsyncAnthropic
        kw: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kw["base_url"] = self.base_url
        return AsyncAnthropic(**kw)

    # ── outbound serialization ───────────────────────────────────────────────

    @staticmethod
    def _to_anthropic_messages(internal: list[dict]) -> list[dict]:
        """Translate our internal messages into Anthropic's wire format.

        Tool-role messages collapse into the *previous* (or next) user message
        as `tool_result` content blocks per Anthropic's API convention.
        """
        out: list[dict] = []
        pending_tool_results: list[dict] = []

        def _flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results:
                out.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

        for msg in internal:
            role = msg.get("role")
            if role == "tool":
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg["tool_use_id"],
                    "content": msg.get("content", ""),
                    **({"is_error": True} if msg.get("is_error") else {}),
                })
                continue

            # Any non-tool message ends the pending tool_result run
            _flush_tool_results()

            if role == "user":
                content = msg["content"]
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                else:
                    out.append({"role": "user", "content": content})
            elif role == "assistant":
                out.append({"role": "assistant", "content": msg["content"]})
            else:
                raise ValueError(f"unknown message role: {role!r}")

        _flush_tool_results()
        return out

    @staticmethod
    def _tool_specs(tools: list) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools]

    # ── chat ─────────────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list,
        events: Any,
    ) -> ChatTurn:
        client = self._client()
        api_messages = self._to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = self._tool_specs(tools)

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for ev in stream:
                    self._handle_stream_event(ev, events)
                final = await stream.get_final_message()
        finally:
            try:
                await client.close()
            except Exception:
                pass

        return self._final_to_turn(final)

    @staticmethod
    def _handle_stream_event(ev: Any, events: Any) -> None:
        if events is None:
            return
        # `text` events fire for each text-delta in a TextBlock
        kind = getattr(ev, "type", "")
        if kind == "text":
            txt = getattr(ev, "text", "")
            if txt:
                events.delta(txt)
        elif kind == "content_block_start":
            block = getattr(ev, "content_block", None)
            if block is not None and getattr(block, "type", "") == "tool_use":
                # Surface tool_start lazily on actual call execution; nothing here.
                pass
        # Streaming thinking deltas come as input_json_delta inside ThinkingBlock —
        # we can't easily tell them apart from tool-input deltas mid-stream, so
        # we leave thinking surfacing to the final-message pass below.

    def _final_to_turn(self, final: Any) -> ChatTurn:
        """Convert SDK final Message → our ChatTurn."""
        blocks_raw = getattr(final, "content", []) or []
        # Serialize each block to a dict so we can echo it back verbatim
        assistant_blocks: list[dict] = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for b in blocks_raw:
            btype = getattr(b, "type", "")
            if btype == "text":
                t = getattr(b, "text", "")
                assistant_blocks.append({"type": "text", "text": t})
                if t:
                    text_parts.append(t)
            elif btype == "thinking":
                # Echo thinking blocks back unchanged so "extended thinking"
                # continuity holds across turns when supported.
                assistant_blocks.append({
                    "type": "thinking",
                    "thinking": getattr(b, "thinking", ""),
                    **({"signature": b.signature} if getattr(b, "signature", None) else {}),
                })
            elif btype == "redacted_thinking":
                assistant_blocks.append({
                    "type": "redacted_thinking",
                    "data": getattr(b, "data", ""),
                })
            elif btype == "tool_use":
                tc = ToolCall(
                    id=getattr(b, "id", "") or "",
                    name=getattr(b, "name", "") or "",
                    args=getattr(b, "input", {}) or {},
                )
                tool_calls.append(tc)
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                })
            else:
                # Unknown block type — try a best-effort serialization
                try:
                    assistant_blocks.append(json.loads(b.model_dump_json()))  # pydantic
                except Exception:
                    assistant_blocks.append({"type": btype or "unknown"})

        usage_obj = getattr(final, "usage", None)
        usage = Usage(
            in_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            out_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        )

        return ChatTurn(
            assistant_blocks=assistant_blocks,
            assistant_text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=getattr(final, "stop_reason", "") or "",
        )
