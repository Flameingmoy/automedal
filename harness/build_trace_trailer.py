"""
AutoMedal — Reflective Trace Builder
======================================
Reads the last N journal entries and renders a compact markdown block:
  - frontmatter fields (id, status, val_loss, val_loss_delta, diff_summary)
  - "## What I learned" body

Output is injected into the Strategist prompt as {{reflective_trace}}.

Usage (called by run.sh):
    python harness/build_trace_trailer.py [--n 3] [--journal-dir journal/]

Exit 0 and prints the block; caller substitutes into prompt.
"""

import argparse
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
    """Return the body of the first H2 section matching `section`."""
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


def build_trace(journal_dir: str, n: int = 3) -> str:
    """Return a markdown trace block of the last n journal entries."""
    if not os.path.isdir(journal_dir):
        return "(no journal entries yet)"

    files = sorted(
        [f for f in os.listdir(journal_dir) if f.endswith(".md")],
        reverse=True,
    )[:n]

    if not files:
        return "(no journal entries yet)"

    blocks: list[str] = []
    for fname in reversed(files):  # oldest first so context reads chronologically
        path = os.path.join(journal_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        fm = _read_frontmatter(text)
        learned = _extract_section(text, "What I learned")

        exp_id      = fm.get("id", "?")
        slug        = fm.get("slug", fname.replace(".md", ""))
        status      = fm.get("status", "?")
        val_loss    = fm.get("val_loss", "?")
        delta       = fm.get("val_loss_delta", "")
        diff_summ   = fm.get("diff_summary", "")

        delta_str   = f"  delta={delta}" if delta else ""
        diff_str    = f"\n  diff: {diff_summ}" if diff_summ else ""

        block = (
            f"### exp {exp_id} — {slug}\n"
            f"status={status}  val_loss={val_loss}{delta_str}{diff_str}\n"
        )
        if learned:
            block += f"\n**What I learned:**\n{learned}\n"

        blocks.append(block)

    return "\n---\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reflective trace for Strategist context")
    parser.add_argument("--n", type=int, default=3, help="Number of recent journal entries (default 3)")
    parser.add_argument("--journal-dir", default=None, help="Path to journal/ directory")
    args = parser.parse_args()

    journal_dir = args.journal_dir or os.path.join(PROJECT_ROOT, "journal")
    print(build_trace(journal_dir, args.n))


if __name__ == "__main__":
    main()
