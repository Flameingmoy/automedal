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


def next_id(journal_dir=JOURNAL_DIR):
    if not os.path.isdir(journal_dir):
        return "0001"
    highest = 0
    for name in os.listdir(journal_dir):
        match = JOURNAL_PATTERN.match(name)
        if match:
            n = int(match.group(1))
            if n > highest:
                highest = n
    return f"{highest + 1:04d}"


def main():
    print(next_id())


if __name__ == "__main__":
    main()
