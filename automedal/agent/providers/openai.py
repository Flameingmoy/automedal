"""OpenAI provider — wraps the official `openai` SDK.

Used for OpenAI direct, Ollama (`/v1`), OpenRouter, Groq, and any other
OpenAI-compatible endpoint via `base_url`.

Translates between our internal Anthropic-flavored message shape and
OpenAI's chat-completions schema:

    internal "tool" role  → OpenAI "tool" role with tool_call_id
    internal assistant tool_use blocks → assistant message with `tool_calls`
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from automedal.agent.providers.base import ChatProvider, ChatTurn, ToolCall, Usage
from automedal.agent.retry import with_retry


@dataclass
class OpenAIProvider:
    model: str
    api_key: str
    base_url: str | None = None
    timeout: int = 120

    def _client(self):
        from openai import AsyncOpenAI
        kw: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kw["base_url"] = self.base_url
        return AsyncOpenAI(**kw)

    # ── outbound serialization ───────────────────────────────────────────────

    @staticmethod
    def _to_openai_messages(system: str, internal: list[dict]) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in internal:
            role = msg.get("role")

            if role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"],
                    "content": msg.get("content", ""),
                })
                continue

            if role == "user":
                c = msg["content"]
                if isinstance(c, str):
                    out.append({"role": "user", "content": c})
                else:
                    # Anthropic-shape tool_result blocks coming through
                    # the internal channel — flatten each as a tool message.
                    flushed = False
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            out.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block.get("content", "") if isinstance(block.get("content"), str)
                                           else json.dumps(block.get("content", "")),
                            })
                            flushed = True
                    if not flushed:
                        # User content given as raw block list (rare). Concatenate text.
                        text = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                        out.append({"role": "user", "content": text})
                continue

            if role == "assistant":
                blocks = msg.get("content", [])
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for b in blocks:
                    btype = b.get("type")
                    if btype == "text":
                        text_parts.append(b.get("text", ""))
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": b["id"],
                            "type": "function",
                            "function": {
                                "name": b["name"],
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        })
                    # ignore thinking / redacted_thinking — OpenAI has no analog
                m: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    m["content"] = "".join(text_parts)
                if tool_calls:
                    m["tool_calls"] = tool_calls
                if "content" not in m:
                    m["content"] = None
                out.append(m)
                continue

            raise ValueError(f"unknown message role: {role!r}")

        return out

    @staticmethod
    def _tool_specs(tools: list) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            }} for t in tools
        ]

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
        api_messages = self._to_openai_messages(system, messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = self._tool_specs(tools)

        text_acc: list[str] = []
        # tool_calls accumulator — OpenAI streams them as deltas keyed by index
        tc_acc: dict[int, dict] = {}
        usage = Usage()
        stop_reason = ""

        try:
            stream = await with_retry(
                lambda: client.chat.completions.create(**kwargs),
                label=f"openai.chat_stream model={self.model}",
                events=events,
            )
            async for chunk in stream:
                # Usage may arrive on its own with empty choices
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    usage.in_tokens = getattr(u, "prompt_tokens", 0) or 0
                    usage.out_tokens = getattr(u, "completion_tokens", 0) or 0

                if not chunk.choices:
                    continue
                ch0 = chunk.choices[0]

                if getattr(ch0, "finish_reason", None):
                    stop_reason = ch0.finish_reason

                delta = getattr(ch0, "delta", None)
                if delta is None:
                    continue

                txt = getattr(delta, "content", None)
                if txt:
                    text_acc.append(txt)
                    if events is not None:
                        events.delta(txt)

                tcs = getattr(delta, "tool_calls", None) or []
                for tc_delta in tcs:
                    idx = getattr(tc_delta, "index", 0) or 0
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args_str": ""})
                    if getattr(tc_delta, "id", None):
                        slot["id"] = tc_delta.id
                    fn = getattr(tc_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["args_str"] += fn.arguments
        finally:
            try:
                await client.close()
            except Exception:
                pass

        # Build assistant_blocks + tool_calls
        assistant_blocks: list[dict] = []
        text = "".join(text_acc)
        if text:
            assistant_blocks.append({"type": "text", "text": text})

        tool_calls: list[ToolCall] = []
        for idx in sorted(tc_acc):
            slot = tc_acc[idx]
            args: dict = {}
            try:
                args = json.loads(slot["args_str"]) if slot["args_str"] else {}
            except json.JSONDecodeError:
                args = {"_raw": slot["args_str"]}
            tc_id = slot["id"] or f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(ToolCall(id=tc_id, name=slot["name"], args=args))
            assistant_blocks.append({
                "type": "tool_use", "id": tc_id, "name": slot["name"], "input": args,
            })

        return ChatTurn(
            assistant_blocks=assistant_blocks,
            assistant_text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
        )
