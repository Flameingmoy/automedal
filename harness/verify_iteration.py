"""
AutoMedal — Iteration Invariant Checker
==========================================
Post-phase verification. Called by run.sh after each phase with:

    python harness/verify_iteration.py --phase strategist
    python harness/verify_iteration.py --phase researcher
    python harness/verify_iteration.py --phase experimenter --exp-id 0013

Prints WARN: lines to stderr for any violation and exits 1. Exits 0 on success.
Soft enforcement — run.sh logs warnings but does not abort.

Standard library only.
"""

import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_PATH = os.path.join(PROJECT_ROOT, "knowledge.md")
QUEUE_PATH = os.path.join(PROJECT_ROOT, "experiment_queue.md")
RESEARCH_PATH = os.path.join(PROJECT_ROOT, "research_notes.md")
JOURNAL_DIR = os.path.join(PROJECT_ROOT, "journal")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "agent", "results.tsv")

VALID_AXES = {
    "preprocessing",
    "feature-eng",
    "HPO",
    "new-model",
    "ensembling",
    "pseudo-label",
    "architecture",
}
VALID_STATUSES = {"pending", "running", "done"}
KNOWLEDGE_BULLET_CAP = 80
QUEUE_ENTRY_COUNT = 5
QUEUE_MAX_PER_AXIS = 2


# ─── parsing helpers ─────────────────────────────────────────────────────

_file_cache = {}


def _read_file(path):
    if path in _file_cache:
        return _file_cache[path]
    if not os.path.exists(path):
        _file_cache[path] = None
        return None
    with open(path, encoding="utf-8") as f:
        content = f.read()
    _file_cache[path] = content
    return content


def _parse_knowledge_bullets(text):
    """Return list of (section_title, bullet_text) tuples from knowledge.md.

    A bullet is any line starting with '- ' (dash + space). Section titles
    are H2 headers (## ...). The 'Last curated' italic line is not a bullet.
    """
    bullets = []
    current_section = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            continue
        if stripped.startswith("- "):
            bullets.append((current_section, stripped[2:].strip()))
    return bullets


def _knowledge_sections(text):
    """Return a dict of section_title → list[bullet]."""
    sections = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections.setdefault(current, [])
        elif stripped.startswith("- ") and current is not None:
            sections[current].append(stripped[2:].strip())
    return sections


QUEUE_ENTRY_RE = re.compile(
    r"^##\s+\d+\.\s+(?P<slug>[a-z0-9-]+)\s+"
    r"\[axis:\s*(?P<axis>[a-zA-Z-]+)\]\s+"
    r"\[STATUS:\s*(?P<status>[a-zA-Z]+)\]",
    re.IGNORECASE,
)


def _parse_queue_entries(text):
    """Return list of dicts describing each queue entry."""
    entries = []
    current = None
    for line in text.splitlines():
        match = QUEUE_ENTRY_RE.match(line.strip())
        if match:
            if current is not None:
                entries.append(current)
            current = {
                "slug": match.group("slug"),
                "axis": match.group("axis"),
                "status": match.group("status").lower(),
                "has_hypothesis": False,
                "has_sketch": False,
                "has_expected": False,
            }
            continue
        if current is not None:
            lower = line.strip().lower()
            if lower.startswith("**hypothesis:**"):
                current["has_hypothesis"] = True
            elif lower.startswith("**sketch:**"):
                current["has_sketch"] = True
            elif lower.startswith("**expected:**"):
                current["has_expected"] = True
    if current is not None:
        entries.append(current)
    return entries


def _parse_journal_frontmatter(text):
    """Return a dict of frontmatter keys, or None if the file has no frontmatter."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    fm = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
        i += 1
    return fm


def _journal_sections(text):
    """Return a dict of H2 section title → body text."""
    sections = {}
    current = None
    body = []
    for line in text.splitlines():
        if line.strip().startswith("## "):
            if current is not None:
                sections[current] = "\n".join(body).strip()
            current = line.strip()[3:].strip()
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        sections[current] = "\n".join(body).strip()
    return sections


# ─── phase checks ────────────────────────────────────────────────────────

def check_researcher(warnings):
    text = _read_file(RESEARCH_PATH)
    if text is None:
        warnings.append("research_notes.md does not exist")
        return

    entries = re.split(r"(?m)^##\s+exp\s+\d+", text)
    # First split chunk is the header preamble; real entries are the rest
    num_entries = max(len(entries) - 1, 0)
    if num_entries == 0:
        warnings.append("research_notes.md has no entries after header")
        return

    # Validate the most recent entry
    last_entry = entries[-1]
    paper_bullets = re.findall(r"(?m)^-\s+Paper:", last_entry)
    if not (2 <= len(paper_bullets) <= 3):
        warnings.append(
            f"last research_notes.md entry has {len(paper_bullets)} papers (expected 2-3)"
        )

    if "query:" not in last_entry.lower():
        warnings.append("last research_notes.md entry missing 'query:' header marker")


def check_strategist(warnings):
    # ─── knowledge.md ────
    kb_text = _read_file(KNOWLEDGE_PATH)
    if kb_text is None:
        warnings.append("knowledge.md does not exist")
    else:
        bullets = _parse_knowledge_bullets(kb_text)
        if len(bullets) > KNOWLEDGE_BULLET_CAP:
            warnings.append(
                f"knowledge.md has {len(bullets)} bullets (cap is {KNOWLEDGE_BULLET_CAP})"
            )

        sections = _knowledge_sections(kb_text)
        if "Open questions" not in sections or not sections["Open questions"]:
            warnings.append("knowledge.md missing non-empty 'Open questions' section")

        # Bullets outside 'Open questions' must cite at least one exp ID
        exp_cite_re = re.compile(r"\bexps?\s*0*\d+", re.IGNORECASE)
        for section, bullet in bullets:
            if section == "Open questions":
                continue
            if not exp_cite_re.search(bullet):
                warnings.append(
                    f"knowledge.md bullet missing exp citation: {bullet[:60]!r}"
                )
                break  # one warn is enough; don't spam

    # ─── experiment_queue.md ────
    q_text = _read_file(QUEUE_PATH)
    if q_text is None:
        warnings.append("experiment_queue.md does not exist")
        return
    entries = _parse_queue_entries(q_text)
    if len(entries) != QUEUE_ENTRY_COUNT:
        warnings.append(
            f"experiment_queue.md has {len(entries)} entries (expected {QUEUE_ENTRY_COUNT})"
        )

    axis_counts = {}
    for entry in entries:
        if entry["axis"] not in VALID_AXES:
            warnings.append(
                f"queue entry '{entry['slug']}' has invalid axis: {entry['axis']}"
            )
        if entry["status"] not in VALID_STATUSES:
            warnings.append(
                f"queue entry '{entry['slug']}' has invalid status: {entry['status']}"
            )
        if not (entry["has_hypothesis"] and entry["has_sketch"] and entry["has_expected"]):
            warnings.append(
                f"queue entry '{entry['slug']}' missing Hypothesis/Sketch/Expected"
            )
        axis_counts[entry["axis"]] = axis_counts.get(entry["axis"], 0) + 1

    for axis, count in axis_counts.items():
        if count > QUEUE_MAX_PER_AXIS:
            warnings.append(
                f"queue has {count} entries on axis '{axis}' (max {QUEUE_MAX_PER_AXIS})"
            )


def check_experimenter(warnings, exp_id):
    if not exp_id:
        warnings.append("experimenter check requires --exp-id")
        return

    # ─── journal entry ────
    journal_files = []
    if os.path.isdir(JOURNAL_DIR):
        for name in os.listdir(JOURNAL_DIR):
            if name.startswith(f"{exp_id}-") and name.endswith(".md"):
                journal_files.append(name)

    if not journal_files:
        warnings.append(f"no journal entry found for exp {exp_id}")
        return
    if len(journal_files) > 1:
        warnings.append(f"multiple journal entries for exp {exp_id}: {journal_files}")

    journal_path = os.path.join(JOURNAL_DIR, journal_files[0])
    text = _read_file(journal_path)
    fm = _parse_journal_frontmatter(text)
    if fm is None:
        warnings.append(f"journal {journal_files[0]} missing frontmatter")
        return

    required_keys = {
        "id",
        "slug",
        "timestamp",
        "git_tag",
        "queue_entry",
        "status",
        "val_loss",
        "val_accuracy",
        "best_so_far",
    }
    missing = required_keys - set(fm.keys())
    if missing:
        warnings.append(f"journal {journal_files[0]} missing frontmatter keys: {sorted(missing)}")

    valid_statuses = {"improved", "no_change", "worse", "crashed"}
    if fm.get("status") not in valid_statuses:
        warnings.append(
            f"journal {journal_files[0]} has invalid status: {fm.get('status')!r}"
        )

    sections = _journal_sections(text)
    required_sections = {"Hypothesis", "What I changed", "Result", "What I learned"}
    for name in required_sections:
        if name not in sections or not sections[name]:
            warnings.append(f"journal {journal_files[0]} missing section: {name!r}")

    # KB-consulted section: required unless knowledge.md is empty (bootstrap case)
    kb_text = _read_file(KNOWLEDGE_PATH) or ""
    kb_has_content = bool(_parse_knowledge_bullets(kb_text))
    if kb_has_content:
        if "KB entries consulted" not in sections or not sections["KB entries consulted"]:
            warnings.append(
                f"journal {journal_files[0]} missing 'KB entries consulted' (KB is non-empty)"
            )

    # ─── results.tsv grew by at least one row for crashed runs it's optional ────
    if not os.path.exists(RESULTS_PATH):
        warnings.append("results.tsv missing")
    # We don't diff row counts here — the training script appends on success.
    # A crashed run with status=crashed intentionally skips the append.


# ─── entrypoint ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoMedal iteration invariant checker")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["strategist", "researcher", "experimenter"],
        help="Which phase just finished",
    )
    parser.add_argument("--exp-id", help="Experiment ID (required for experimenter)")
    args = parser.parse_args()

    warnings = []
    if args.phase == "researcher":
        check_researcher(warnings)
    elif args.phase == "strategist":
        check_strategist(warnings)
    elif args.phase == "experimenter":
        check_experimenter(warnings, args.exp_id)

    if warnings:
        for w in warnings:
            print(f"WARN: {w}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {args.phase} phase invariants satisfied")
    sys.exit(0)


if __name__ == "__main__":
    main()
