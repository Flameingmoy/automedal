"""
AutoMedal — Stagnation Detector
==================================
Deterministic check over results.tsv. Returns 1 if the rolling best val_loss
has not improved in the last K runs, else 0.

Usage:
    python harness/check_stagnation.py --k 3          # prints 0 or 1
    python harness/check_stagnation.py --print-best   # prints best val_loss

Rules:
  - "Improvement" = strictly less than previous best (ties don't count).
  - Fewer than K rows of data = no stagnation (returns 0). Covers the
    freshly-bootstrapped competition case where results.tsv is header-only.
  - Standard library only — no pandas dependency for the harness.
"""

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(PROJECT_ROOT, "agent", "results.tsv")
VAL_LOSS_COL = "val_loss"


def _read_val_losses(path):
    """Parse results.tsv and return a list of val_loss floats in row order."""
    if not os.path.exists(path):
        return []
    losses = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if VAL_LOSS_COL not in (reader.fieldnames or []):
            return []
        for row in reader:
            raw = row.get(VAL_LOSS_COL, "").strip()
            if not raw:
                continue
            try:
                losses.append(float(raw))
            except ValueError:
                continue
    return losses


def is_stagnating(k, losses=None):
    """Return True if the best val_loss has not improved in the last k runs."""
    if losses is None:
        losses = _read_val_losses(RESULTS_FILE)
    if len(losses) < k + 1:
        return False
    best_before_window = min(losses[:-k])
    best_in_window = min(losses[-k:])
    return best_in_window >= best_before_window


def best_val_loss(losses=None):
    """Return the best (minimum) val_loss seen so far, or float('inf')."""
    if losses is None:
        losses = _read_val_losses(RESULTS_FILE)
    return min(losses) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser(description="AutoMedal stagnation detector")
    parser.add_argument("--k", type=int, default=3, help="Stagnation window size")
    parser.add_argument(
        "--print-best",
        action="store_true",
        help="Print the current best val_loss instead of the stagnation flag",
    )
    args = parser.parse_args()

    losses = _read_val_losses(RESULTS_FILE)

    if args.print_best:
        best = best_val_loss(losses)
        print(f"{best:.6f}" if best != float("inf") else "inf")
        return

    print("1" if is_stagnating(args.k, losses) else "0")


if __name__ == "__main__":
    main()
