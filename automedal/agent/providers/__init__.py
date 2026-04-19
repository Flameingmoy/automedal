"""LLM provider adapters.

Public surface:
    build_provider(name, model, **kw) -> ChatProvider
    smoke(name, model)               -> tuple[bool, str]

Supported providers:
    opencode-go  — Anthropic SDK against base_url="https://opencode.ai/zen/go"
    anthropic    — Anthropic SDK direct
    openai       — OpenAI SDK direct
    ollama       — OpenAI SDK against http://localhost:11434/v1
    openrouter   — OpenAI SDK against https://openrouter.ai/api/v1
    groq         — OpenAI SDK against https://api.groq.com/openai/v1

`model` may be the bare slug (`minimax-m2.7`) or the namespaced form
(`opencode-go/minimax-m2.7`); the latter is split for convenience.
"""

from __future__ import annotations

import os
from typing import Any

from automedal.agent.providers.base import ChatProvider, ChatTurn, ToolCall, Usage


_OPENCODE_BASE_URL = "https://opencode.ai/zen/go"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def parse_slug(slug: str) -> tuple[str, str]:
    """'opencode-go/minimax-m2.7' → ('opencode-go', 'minimax-m2.7'). Raises if no '/'."""
    if "/" not in slug:
        raise ValueError(f"expected 'provider/model', got {slug!r}")
    p, m = slug.split("/", 1)
    return p, m


def _normalize_model(provider: str, model: str) -> str:
    if "/" in model and model.split("/", 1)[0] == provider:
        return model.split("/", 1)[1]
    return model


def build_provider(name: str, model: str, **kw: Any) -> ChatProvider:
    """Return a ChatProvider for `name` bound to `model`."""
    model = _normalize_model(name, model)

    if name == "opencode-go":
        from automedal.agent.providers.anthropic import AnthropicProvider
        api_key = os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            raise RuntimeError("OPENCODE_API_KEY not set (run `automedal setup`)")
        return AnthropicProvider(
            model=model, api_key=api_key, base_url=_OPENCODE_BASE_URL,
            max_tokens=kw.get("max_tokens", 4096), timeout=kw.get("timeout", 120),
        )

    if name == "anthropic":
        from automedal.agent.providers.anthropic import AnthropicProvider
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicProvider(
            model=model, api_key=api_key, base_url=None,
            max_tokens=kw.get("max_tokens", 4096), timeout=kw.get("timeout", 120),
        )

    if name == "openai":
        from automedal.agent.providers.openai import OpenAIProvider
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIProvider(
            model=model, api_key=api_key, base_url=None,
            timeout=kw.get("timeout", 120),
        )

    if name == "ollama":
        from automedal.agent.providers.openai import OpenAIProvider
        base = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return OpenAIProvider(
            model=model, api_key="ollama", base_url=base,
            timeout=kw.get("timeout", 120),
        )

    if name == "openrouter":
        from automedal.agent.providers.openai import OpenAIProvider
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return OpenAIProvider(
            model=model, api_key=api_key, base_url=_OPENROUTER_BASE_URL,
            timeout=kw.get("timeout", 120),
        )

    if name == "groq":
        from automedal.agent.providers.openai import OpenAIProvider
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        return OpenAIProvider(
            model=model, api_key=api_key, base_url=_GROQ_BASE_URL,
            timeout=kw.get("timeout", 120),
        )

    raise ValueError(f"unknown provider: {name}")


def smoke(name: str, model: str, *, timeout: int = 30) -> tuple[bool, str]:
    """Synchronous smoke test — invokes provider.chat_stream once asking for 'READY'."""
    import asyncio

    try:
        prov = build_provider(name, model, timeout=timeout, max_tokens=64)
    except Exception as exc:
        return False, f"init error: {type(exc).__name__}: {exc}"

    async def _run() -> tuple[bool, str]:
        try:
            turn = await prov.chat_stream(
                system="Reply with one word.",
                messages=[{"role": "user", "content": "Say READY and nothing else."}],
                tools=[],
                events=None,
            )
        except Exception as exc:
            return False, f"invoke error: {type(exc).__name__}: {exc}"
        text = (turn.assistant_text or "").strip()
        if "ready" in text.lower():
            return True, f"ok ({text[:60]!r})"
        return False, f"unexpected response: {text!r}"

    try:
        return asyncio.run(_run())
    except RuntimeError:
        # Already in an event loop — fall back to sync via new loop in thread
        import threading
        out: list[tuple[bool, str]] = []
        def _t() -> None:
            out.append(asyncio.run(_run()))
        th = threading.Thread(target=_t)
        th.start()
        th.join()
        return out[0]


__all__ = [
    "ChatProvider", "ChatTurn", "ToolCall", "Usage",
    "build_provider", "parse_slug", "smoke",
]
