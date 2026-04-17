"""Queue summary widget — lists the next 5 pending queue entries from experiment_queue.md."""

from __future__ import annotations

import re
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tui.state import AppState


_ENTRY_RE = re.compile(
    r"^##\s+\d+\.\s+(?P<slug>[a-z0-9-]+)\s+\[axis:\s*(?P<axis>[a-zA-Z-]+)\]"
    r"\s+\[STATUS:\s*(?P<status>[a-zA-Z]+)\]",
    re.IGNORECASE,
)


def _parse_pending(text: str, limit: int = 5) -> list[dict]:
    entries = []
    for line in text.splitlines():
        m = _ENTRY_RE.match(line.strip())
        if m and m.group("status").lower() == "pending":
            entries.append({"slug": m.group("slug"), "axis": m.group("axis")})
            if len(entries) >= limit:
                break
    return entries


class QueueSummary(Vertical):
    DEFAULT_CSS = """
    QueueSummary { height: 8; border: round #6272A4; padding: 0 1; background: #0F111A; }
    QueueSummary > #qs-title { height: 1; color: #8BE9FD; }
    QueueSummary > #qs-body  { height: 6; }
    """

    def __init__(self, queue_md: Path | None = None, **kw) -> None:
        super().__init__(**kw)
        self._queue_md = queue_md

    def compose(self) -> ComposeResult:
        yield Static("experiment queue (pending)", id="qs-title")
        yield Static("(no pending experiments)", id="qs-body")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self._queue_md is None or not self._queue_md.exists():
            return
        try:
            text = self._queue_md.read_text(encoding="utf-8")
            entries = _parse_pending(text)
            if entries:
                lines = [f"  {e['slug']:<32}  [{e['axis']}]" for e in entries]
                body = "\n".join(lines)
            else:
                body = "  (no pending entries)"
            self.query_one("#qs-body", Static).update(body)
        except Exception:
            pass

    def update_state(self, state: AppState) -> None:
        self._refresh()
