"""
AutoMedal — Next Experiment ID
================================
Scans journal/ for the highest existing NNNN experiment ID and prints the
next one zero-padded to 4 digits. Prints '0001' if the directory is empty.

Usage:
    python harness/next_exp_id.py
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_DIR = os.path.join(PROJECT_ROOT, "journal")
JOURNAL_PATTERN = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")


def _scan_journal(journal_dir):
    """Fallback: directory scan for highest experiment ID."""
    highest = 0
    for name in os.listdir(journal_dir):
        match = JOURNAL_PATTERN.match(name)
        if match:
            n = int(match.group(1))
            if n > highest:
                highest = n
    return highest


def next_id(journal_dir=JOURNAL_DIR):
    sentinel = os.path.join(journal_dir, ".last_exp_id")

    # Fast path: read sentinel
    if os.path.exists(sentinel):
        try:
            with open(sentinel) as f:
                highest = int(f.read().strip())
        except (ValueError, OSError):
            highest = _scan_journal(journal_dir)
    elif os.path.isdir(journal_dir):
        highest = _scan_journal(journal_dir)
    else:
        os.makedirs(journal_dir, exist_ok=True)
        highest = 0

    next_num = highest + 1
    # Atomically update sentinel
    tmp = sentinel + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(next_num))
    os.replace(tmp, sentinel)

    return f"{next_num:04d}"


def main():
    print(next_id())


if __name__ == "__main__":
    main()
