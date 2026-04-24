"""Friendly error mapper — turn provider exceptions into actionable fixes.

Patterns are matched against the lowercased ``str(exc)`` so they catch
errors from both `openai` and `anthropic` SDKs plus litellm-style
passthroughs. Returns ``None`` for anything we don't recognize — callers
should fall back to the raw traceback in that case.

Inspired by ml-intern/agent/core/agent_loop.py:_friendly_error_message,
adapted for our provider set (opencode-go / anthropic direct).
"""

from __future__ import annotations


def friendly_error(exc: Exception) -> str | None:
    """Return a user-facing fix for known error patterns, else None."""
    s = str(exc).lower()

    if any(p in s for p in ("unauthorized", "invalid x-api-key", "invalid api key", "401")):
        return (
            "Authentication failed — your API key is missing or invalid.\n\n"
            "Fix:\n"
            "  • opencode-go:  export OPENCODE_API_KEY=...\n"
            "  • anthropic:    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  • openai:       export OPENAI_API_KEY=sk-...\n\n"
            "Add it to ~/.automedal/.env or a project .env file if you want it persistent."
        )

    if ("insufficient" in s and "credit" in s) or "insufficient_quota" in s or "402" in s:
        return (
            "Out of credits at the provider. Check your balance at\n"
            "  • opencode.ai dashboard (for OPENCODE_API_KEY)\n"
            "  • console.anthropic.com (for ANTHROPIC_API_KEY)"
        )

    if "model_not_found" in s or ("model" in s and ("not found" in s or "does not exist" in s)):
        return (
            "Model id not recognized by the provider.\n"
            "  • `automedal models` lists available models cached from opencode-go.\n"
            "  • For anthropic, use a current id (e.g. claude-opus-4-7, claude-sonnet-4-6)."
        )

    if "not supported by provider" in s or "no provider supports" in s:
        return (
            "This model isn't served by the provider you pinned.\n"
            "Drop any `:provider` suffix to let routing pick automatically."
        )

    if "context" in s and ("exceed" in s or "too long" in s or "too many tokens" in s):
        return (
            "Context window exceeded. The conversation is longer than the model accepts.\n"
            "Tier 2 context compaction will fix this automatically. For now, start a fresh run."
        )

    return None


def format_error(exc: Exception) -> str:
    """Compose a user-facing message: friendly explanation + raw error line.

    Always returns a non-empty string so callers can print it unconditionally.
    """
    raw = f"{type(exc).__name__}: {exc}"
    friendly = friendly_error(exc)
    if friendly is None:
        return raw
    return f"{friendly}\n\n[raw] {raw}"
