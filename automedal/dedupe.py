"""Motivation-similarity dedupe for the experiment queue.

After the Strategist writes a fresh queue, scan each pending entry's
**Hypothesis** field against the journal's recent diff_summaries. If
BM25 similarity exceeds a configurable threshold, mark the queue entry
as `[STATUS: skipped-duplicate]` and carry the duplicate-citation as a
comment so the next Strategist pass can read why.

Bypass: include the literal token `[force]` anywhere in a queue entry
to skip dedupe for that entry.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from automedal.agent.tools.cognition import bm25_score_pairs


# 1-indexed entry header: "## 3. catboost-native-cats [axis: HPO] [STATUS: pending]"
_ENTRY_RE = re.compile(
    r"^## (?P<idx>\d+)\. (?P<slug>[^\s\[]+)[^\n]*?\[STATUS:\s*(?P<status>[^\]]+)\]",
    re.MULTILINE,
)
_HYPOTHESIS_RE = re.compile(r"\*\*Hypothesis:\*\*\s*(?P<text>.+?)(?:\n\*\*|\nsuccess_criteria|\Z)", re.DOTALL)


def _split_entries(queue_text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, body)] for each ## section. End is exclusive."""
    headers = list(_ENTRY_RE.finditer(queue_text))
    out: list[tuple[int, int, str]] = []
    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(queue_text)
        out.append((start, end, queue_text[start:end]))
    return out


def _extract_hypothesis(entry: str) -> str:
    m = _HYPOTHESIS_RE.search(entry)
    return (m.group("text").strip() if m else "").splitlines()[0] if m else ""


def _journal_diffs(journal_dir: Path, n: int = 30) -> list[tuple[str, str]]:
    """Return [(slug_or_id, diff_summary)] for the most recent N journal entries."""
    if not journal_dir.is_dir():
        return []
    entries = sorted(journal_dir.glob("*.md"))[-n:]
    out: list[tuple[str, str]] = []
    for p in entries:
        try:
            txt = p.read_text(encoding="utf-8")
        except OSError:
            continue
        m = re.search(r"^diff_summary:\s*(.+)$", txt, re.MULTILINE)
        if m:
            out.append((p.stem, m.group(1).strip()))
    return out


def _mark_skipped(entry: str, reason: str) -> str:
    """Replace `[STATUS: pending]` with skipped-duplicate and append a note line."""
    new = re.sub(r"\[STATUS:\s*pending\]", "[STATUS: skipped-duplicate]", entry, count=1)
    if "skipped-duplicate" not in new:
        return entry
    note = f"\n_dedupe note: {reason}_\n"
    # Insert note before the next entry boundary (just append; entry is one block).
    return new.rstrip() + note


def apply(
    *,
    queue_path: str | Path,
    journal_path: str | Path,
    threshold: float | None = None,
) -> dict:
    """Walk pending entries; mark duplicates against recent journal diffs.

    Returns a summary dict: {scanned, marked, threshold, journal_n}.
    """
    if threshold is None:
        try:
            threshold = float(os.environ.get("AUTOMEDAL_DEDUPE_THRESHOLD", "5.0"))
        except ValueError:
            threshold = 5.0

    qp = Path(queue_path)
    if not qp.exists():
        return {"scanned": 0, "marked": 0, "threshold": threshold, "journal_n": 0}

    text = qp.read_text(encoding="utf-8")
    entries = _split_entries(text)
    diffs = _journal_diffs(Path(journal_path))
    diff_bodies = [d for _, d in diffs]
    diff_labels = [s for s, _ in diffs]

    scanned = 0
    marked = 0
    new_text = text

    # Process in reverse so absolute offsets stay valid as we mutate.
    for start, end, body in reversed(entries):
        m = _ENTRY_RE.search(body)
        if not m or m.group("status").strip().lower() != "pending":
            continue
        if "[force]" in body:
            continue
        hyp = _extract_hypothesis(body)
        if not hyp:
            continue
        scanned += 1
        if not diff_bodies:
            continue
        scores = bm25_score_pairs(hyp, diff_bodies)
        if not scores:
            continue
        peak_idx = max(range(len(scores)), key=lambda i: scores[i])
        peak = scores[peak_idx]
        if peak < threshold:
            continue
        reason = (
            f"matches journal entry {diff_labels[peak_idx]!r} "
            f"(BM25={peak:.2f} ≥ {threshold:.2f})"
        )
        new_body = _mark_skipped(body, reason)
        if new_body != body:
            new_text = new_text[:start] + new_body + new_text[end:]
            marked += 1

    if new_text != text:
        qp.write_text(new_text, encoding="utf-8")

    return {
        "scanned": scanned,
        "marked": marked,
        "threshold": threshold,
        "journal_n": len(diffs),
    }
