"""Provider contract — uniform ChatProvider Protocol returning a ChatTurn.

Internal message shape (Anthropic-flavored, used for both providers):

    {"role": "user",      "content": "<string OR list of blocks>"}
    {"role": "assistant", "content": [<text block>, <thinking block>, <tool_use block>...]}
    {"role": "tool",      "tool_use_id": "<id>", "content": "<string>"}

The kernel speaks this shape; OpenAIProvider translates at its edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    args: dict


@dataclass
class Usage:
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class ChatTurn:
    """One assistant turn from the LLM.

    `assistant_blocks` is the raw provider-native content (Anthropic block
    list, or our normalized list for OpenAI) that the kernel must echo
    verbatim back to the model on the next turn so tool_use ids line up.
    """
    assistant_blocks: list[dict]
    assistant_text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str = ""


@runtime_checkable
class ChatProvider(Protocol):
    model: str

    async def chat_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list,
        events: Any,
    ) -> ChatTurn:
        ...
