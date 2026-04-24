"""Transcript self-healing helpers — run before every provider.chat_stream.

These operate on the internal message list (Anthropic-flavored shape from
``providers/base.py``). They never talk to the model; they only massage the
transcript so the next turn is well-formed.

Why this exists:
- If a tool raises or the kernel is interrupted between ``tool_start`` and
  the matching ``tool_result`` append, the transcript ends up with an
  assistant ``tool_use`` block that has no answering ``tool`` message.
  Both providers then 400 with "tool_use_id has no matching tool_result".
  ``patch_dangling_tool_calls`` stuffs a stub result so the phase self-heals.
"""

from __future__ import annotations


def _answered_ids(messages: list[dict]) -> set[str]:
    """All tool_use_ids that already have a matching ``role=tool`` result."""
    ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_use_id"):
            ids.add(m["tool_use_id"])
            continue
        # User messages may carry Anthropic-shape tool_result blocks.
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                    ids.add(b["tool_use_id"])
    return ids


def patch_dangling_tool_calls(messages: list[dict]) -> int:
    """Append stub tool_result entries for any unanswered tool_use blocks.

    Scans backwards for the last assistant turn with tool_use blocks and
    appends a synthetic ``role=tool`` reply (``is_error=True``) for every
    block id not already answered. Stops at the first user message or at
    the last assistant-with-tools turn we touched — providers only 400 on
    the most recent dangling batch.

    Returns the number of stubs appended (0 if the transcript was clean).
    Safe to call on any transcript; a no-op when there are no danglers.
    """
    answered = _answered_ids(messages)
    appended = 0

    # Walk backwards to find the most recent assistant turn carrying tool_use.
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            return appended
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            bid = block.get("id")
            if not bid or bid in answered:
                continue
            messages.append({
                "role": "tool",
                "tool_use_id": bid,
                "content": "Tool was not executed (interrupted or error).",
                "is_error": True,
            })
            answered.add(bid)
            appended += 1
        # Only patch the most recent assistant turn; earlier turns would
        # already have been patched on previous steps if they were dangling.
        return appended

    return appended
