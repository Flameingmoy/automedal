"""
AutoMedal — Competition Selector
===================================
Phase 2: Interactive terminal picker for competition selection.

Loads the ranked shortlist from scout/outputs/competition_candidates.json,
displays it, and lets the user pick a competition by number or slug.
Optionally triggers bootstrap immediately after selection.

Usage:
    python scout/select.py
    python -m scout.select
"""

import os
import sys
import json

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CANDIDATES_JSON = os.path.join(PROJECT_ROOT, "scout", "outputs", "competition_candidates.json")


def load_candidates(candidates_json=None):
    """Load competition candidates from JSON."""
    path = candidates_json or CANDIDATES_JSON
    if not os.path.exists(path):
        print("ERROR: No candidates file found.")
        print(f"  Expected: {path}")
        print(f"  Run 'automedal discover' first.")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    candidates = data.get("candidates", [])
    generated = data.get("generated_at", "unknown")
    print(f"  Loaded {len(candidates)} candidates (generated: {generated})")
    return candidates


def display_candidates(candidates):
    """Display ranked candidates in terminal."""
    if not candidates:
        print("  No candidates to display.")
        return

    # Header
    print()
    print(f"  {'#':>3}  {'Score':>5}  {'Category':<15}  {'Metric':<12}  {'Teams':>6}  {'Slug'}")
    print(f"  {'─' * 3}  {'─' * 5}  {'─' * 15}  {'─' * 12}  {'─' * 6}  {'─' * 40}")

    for i, candidate in enumerate(candidates, 1):
        comp = candidate["competition"]
        slug = comp.get("ref", "?")
        category = comp.get("category", "?")[:15]
        metric = (comp.get("evaluationMetric", "?") or "?")[:12]
        teams = comp.get("teamCount", 0)
        score = candidate["final_score"]

        print(f"  {i:>3}  {score:>5}  {category:<15}  {metric:<12}  {teams:>6}  {slug}")

    print()


def select_competition(candidates):
    """Interactive selection — user picks by number or slug.

    Returns:
        Selected competition slug, or None if cancelled.
    """
    while True:
        choice = input("  Select competition (number, slug, or 'q' to quit): ").strip()

        if choice.lower() in ("q", "quit", "exit"):
            return None

        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                slug = candidates[idx]["competition"]["ref"]
                print(f"\n  Selected: {slug}")
                return slug
            else:
                print(f"  Invalid number. Choose 1-{len(candidates)}.")
                continue
        except ValueError:
            pass

        # Try as slug
        for candidate in candidates:
            if candidate["competition"].get("ref", "") == choice:
                print(f"\n  Selected: {choice}")
                return choice

        print(f"  '{choice}' not found. Try a number or exact slug.")


def pick_and_bootstrap(slug: str, yes: bool = False) -> int:
    """Non-interactive: bootstrap a competition by slug.

    Used by the TUI SelectCompetitionScreen after the user picks a row.
    Returns subprocess exit code.

    Args:
        slug: Competition slug to bootstrap.
        yes: Pass --yes to bootstrap (skip low-confidence warning prompt).
    """
    from scout.bootstrap import bootstrap
    extra_args = ["--yes"] if yes else []
    return bootstrap(slug, extra_args=extra_args)


def main():
    print("=" * 60)
    print("AutoMedal — Competition Selector")
    print("=" * 60)

    candidates = load_candidates()
    if not candidates:
        sys.exit(1)

    display_candidates(candidates)

    slug = select_competition(candidates)
    if not slug:
        print("  Selection cancelled.")
        sys.exit(0)

    # Ask whether to bootstrap
    print()
    response = input("  Bootstrap this competition now? [Y/n] ").strip().lower()
    if response in ("", "y", "yes"):
        pick_and_bootstrap(slug)
    else:
        print(f"\n  To bootstrap later, run:")
        print(f"    automedal init {slug}")


if __name__ == "__main__":
    main()
