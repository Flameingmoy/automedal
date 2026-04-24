"""Plan tool — lets the strategist maintain a structured todo list.

Each call replaces the whole plan. Emits a ``plan_update`` event (via the
EventSink) so the TUI can render the current plan in a side-panel. State
lives in ``session`` (a simple dict passed by the caller) so a strategist
invocation that spans multiple kernel steps sees a consistent plan.

Status vocabulary matches ml-intern/agent/tools/plan_tool.py:
    pending | in_progress | completed

Validation rules:
- ``id`` must be a non-empty string, unique within the plan.
- ``content`` must be a non-empty string (max ~280 chars).
- ``status`` must be one of the vocabulary above.
- At most one item may be ``in_progress`` at a time (mirrors ml-intern).
"""

from __future__ import annotations

from typing import Any

from automedal.agent.tools.base import Tool, ToolResult

_VALID_STATUS = {"pending", "in_progress", "completed"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Full replacement plan — every call overwrites the previous one.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique short identifier."},
                    "content": {"type": "string", "description": "One-line task description."},
                    "status": {"type": "string", "enum": sorted(_VALID_STATUS)},
                },
                "required": ["id", "content", "status"],
            },
        },
    },
    "required": ["items"],
}

_DESCRIPTION = (
    "Maintain a structured plan of what you intend to do. Each call replaces "
    "the entire plan. Use status='in_progress' for the single item you are "
    "currently working on (at most one), 'pending' for future items, and "
    "'completed' for finished items. Keep content short — one line each."
)


def _validate(items: list[dict]) -> str | None:
    if not isinstance(items, list) or not items:
        return "items must be a non-empty array"
    seen_ids: set[str] = set()
    in_progress_count = 0
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            return f"items[{idx}] is not an object"
        iid = it.get("id")
        content = it.get("content")
        status = it.get("status")
        if not isinstance(iid, str) or not iid.strip():
            return f"items[{idx}].id must be a non-empty string"
        if iid in seen_ids:
            return f"duplicate id {iid!r}"
        seen_ids.add(iid)
        if not isinstance(content, str) or not content.strip():
            return f"items[{idx}].content must be a non-empty string"
        if len(content) > 280:
            return f"items[{idx}].content exceeds 280 chars"
        if status not in _VALID_STATUS:
            return f"items[{idx}].status must be one of {sorted(_VALID_STATUS)}"
        if status == "in_progress":
            in_progress_count += 1
    if in_progress_count > 1:
        return "at most one item may be status='in_progress'"
    return None


def make_plan_tool(*, session: dict[str, Any], events: Any = None) -> Tool:
    """Return a Tool that mutates ``session['plan']`` and emits plan_update.

    ``session`` is any dict-like the caller owns for the current phase; we
    do not hold state in the tool itself so concurrent phases don't collide.
    """

    async def _run(**kwargs: Any) -> ToolResult:
        items = kwargs.get("items")
        err = _validate(items if isinstance(items, list) else [])
        if err is not None:
            return ToolResult(text=f"error: {err}", ok=False)

        # Normalize: keep only known fields, strip whitespace.
        normalized = [
            {
                "id": it["id"].strip(),
                "content": it["content"].strip(),
                "status": it["status"],
            }
            for it in items
        ]
        session["plan"] = normalized

        if events is not None:
            try:
                # Reuse the generic `notice` event so we don't need a new
                # kind — TUI formatters can match on tag='plan_update'.
                summary = " | ".join(
                    f"[{it['status'][0]}] {it['content']}" for it in normalized
                )
                events.notice(
                    tag="plan_update",
                    message=f"{len(normalized)} items: {summary}",
                )
            except Exception:
                pass

        counts = {s: sum(1 for it in normalized if it["status"] == s) for s in _VALID_STATUS}
        return ToolResult(
            text=(
                f"plan updated ({len(normalized)} items: "
                f"{counts['pending']} pending, {counts['in_progress']} in_progress, "
                f"{counts['completed']} completed)"
            ),
            ok=True,
        )

    return Tool(
        name="plan",
        description=_DESCRIPTION,
        schema=_SCHEMA,
        run=_run,
    )
