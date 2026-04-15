"""
AutoMedal — Learning-Value Journal Ranker
==========================================
Reads the last M journal entries and returns the top K ranked by a
lightweight learning-value score (no LLM calls).

Score per entry:
  +2  if status == "better" (or "improved")
  -1  if status == "worse"
  +1  if |val_loss_delta| > 0.5 * rolling_stddev of all deltas
  +1  if axis not in the last-5-experimented axes (diversity bonus)

Output: compact markdown summary of top-K entries.

Usage (called by run.sh for Strategist context):
    python harness/rank_journals.py [--m 30] [--k 10] [--journal-dir journal/]
"""

import argparse
import math
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_frontmatter(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm: dict = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        if ":" in lines[i]:
            k, _, v = lines[i].partition(":")
            fm[k.strip()] = v.strip()
        i += 1
    return fm


def _extract_section(text: str, section: str) -> str:
    lines = text.splitlines()
    in_section = False
    body: list[str] = []
    for line in lines:
        if line.strip().lower().startswith(f"## {section.lower()}"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            body.append(line)
    return "\n".join(body).strip()


def _safe_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s.lstrip("+"))
    except ValueError:
        return None


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def rank_journals(journal_dir: str, m: int = 30, k: int = 10) -> str:
    """Return top-K journal summary sorted by learning value."""
    if not os.path.isdir(journal_dir):
        return "(no journal entries yet)"

    files = sorted(
        [f for f in os.listdir(journal_dir) if f.endswith(".md")],
        reverse=True,
    )[:m]

    if not files:
        return "(no journal entries yet)"

    entries = []
    for fname in files:
        path = os.path.join(journal_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        fm = _read_frontmatter(text)
        learned = _extract_section(text, "What I learned")
        entries.append({
            "fname":    fname,
            "id":       fm.get("id", "?"),
            "slug":     fm.get("slug", fname.replace(".md", "")),
            "status":   fm.get("status", "").lower(),
            "val_loss": _safe_float(fm.get("val_loss")),
            "delta":    _safe_float(fm.get("val_loss_delta")),
            "axis":     fm.get("axis", ""),
            "learned":  learned,
        })

    # Compute rolling stddev of |deltas| for threshold
    deltas = [abs(e["delta"]) for e in entries if e["delta"] is not None]
    delta_std = _stddev(deltas)

    # Determine last-5 axes for diversity bonus
    recent_axes = [e["axis"] for e in entries[:5] if e["axis"]]

    # Score
    for e in entries:
        score = 0
        status = e["status"]
        if status in ("better", "improved", "kept"):
            score += 2
        elif status in ("worse", "reverted"):
            score -= 1
        if e["delta"] is not None and delta_std > 0:
            if abs(e["delta"]) > 0.5 * delta_std:
                score += 1
        if e["axis"] and e["axis"] not in recent_axes:
            score += 1
        e["score"] = score

    ranked = sorted(entries, key=lambda x: x["score"], reverse=True)[:k]

    lines = [f"## Top-{k} experiments by learning value (out of last {len(entries)})\n"]
    for e in ranked:
        delta_str = f"  Δ{e['delta']:+.4f}" if e["delta"] is not None else ""
        loss_str  = f"val_loss={e['val_loss']:.4f}" if e["val_loss"] is not None else ""
        lines.append(
            f"### exp {e['id']} — {e['slug']}  "
            f"[score={e['score']}  {e['status']}  {loss_str}{delta_str}]"
        )
        if e["learned"]:
            for bullet in e["learned"].splitlines()[:3]:
                if bullet.strip():
                    lines.append(f"  {bullet.strip()}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank journals by learning value")
    parser.add_argument("--m", type=int, default=30, help="Max journals to read (default 30)")
    parser.add_argument("--k", type=int, default=10, help="Top-K to output (default 10)")
    parser.add_argument("--journal-dir", default=None, help="Path to journal/ directory")
    args = parser.parse_args()

    journal_dir = args.journal_dir or os.path.join(PROJECT_ROOT, "journal")
    print(rank_journals(journal_dir, args.m, args.k))


if __name__ == "__main__":
    main()
