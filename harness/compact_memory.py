"""
AutoMedal — Memory Compaction
================================
Checks whether research_notes.md (or knowledge.md) exceeds a token-budget
threshold and, if so, invokes the compactor.md prompt via pi to summarise it.

The original file is preserved as <file>.archive.md. The compacted version
replaces the original.

Usage:
    python harness/compact_memory.py [--target research_notes.md] [--threshold-kb 40]
    automedal compact

Not auto-scheduled in run.sh — the user opts in via `automedal compact`.
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_THRESHOLD_KB = 40  # ~40 KB before compaction kicks in
_PROMPTS_DIR = os.environ.get("AUTOMEDAL_PROMPTS_DIR", os.path.join(PROJECT_ROOT, "prompts"))


def needs_compaction(path: str, threshold_kb: int = DEFAULT_THRESHOLD_KB) -> bool:
    """Return True if file exceeds threshold."""
    if not os.path.exists(path):
        return False
    size_kb = os.path.getsize(path) / 1024
    return size_kb > threshold_kb


def compact_file(path: str, pi: str, model: str, dry_run: bool = False) -> int:
    """Compact a memory file.  Returns 0 on success, 1 on failure."""
    compactor_prompt = os.path.join(_PROMPTS_DIR, "compactor.md")
    if not os.path.exists(compactor_prompt):
        print(f"ERROR: compactor.md not found at {compactor_prompt}")
        return 1

    archive_path = path.replace(".md", ".archive.md")

    # Sanity check: don't overwrite an existing archive without confirmation
    if os.path.exists(archive_path):
        ans = input(f"Archive {archive_path} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    size_kb = os.path.getsize(path) / 1024
    print(f"  {os.path.basename(path)}: {size_kb:.1f} KB — compacting…")

    if dry_run:
        print(f"  [dry-run] Would compact {path} → {archive_path}")
        return 0

    # Read file
    with open(path, encoding="utf-8") as f:
        content = f.read()

    prompt_body = open(compactor_prompt, encoding="utf-8").read()
    full_prompt = f"{prompt_body}\n\n---\n# File to compact: {os.path.basename(path)}\n\n{content}"

    # Call pi
    result = subprocess.run(
        [pi, "--no-session", "--model", model, "-p", full_prompt],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        print(f"  ERROR: pi exited {result.returncode}")
        print(result.stderr[:500])
        return 1

    compacted = result.stdout.strip()
    if not compacted:
        print("  ERROR: pi returned empty output")
        return 1

    # Archive original, write compacted
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(path, "w", encoding="utf-8") as f:
        f.write(compacted)

    new_kb = len(compacted.encode()) / 1024
    print(f"  ✓ Compacted {size_kb:.1f} KB → {new_kb:.1f} KB")
    print(f"  Archive saved to {archive_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact memory files when they exceed a threshold")
    parser.add_argument("--target", default=None,
                        help="File to compact (default: research_notes.md)")
    parser.add_argument("--threshold-kb", type=int, default=DEFAULT_THRESHOLD_KB,
                        help=f"Size threshold in KB (default: {DEFAULT_THRESHOLD_KB})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check threshold but don't actually compact")
    args = parser.parse_args()

    target = args.target or os.path.join(PROJECT_ROOT, "research_notes.md")
    pi = os.environ.get("AUTOMEDAL_PI_BIN", "pi")
    model = os.environ.get("MODEL", "opencode-go/minimax-m2.7")

    if not os.path.exists(target):
        print(f"File not found: {target}")
        sys.exit(1)

    size_kb = os.path.getsize(target) / 1024
    if not needs_compaction(target, args.threshold_kb):
        print(f"  {os.path.basename(target)}: {size_kb:.1f} KB — below threshold ({args.threshold_kb} KB), no action needed")
        sys.exit(0)

    rc = compact_file(target, pi, model, dry_run=args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
