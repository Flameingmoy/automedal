"""
AutoMedal — Competition Discovery
====================================
Phase 1: Discover and rank Kaggle competitions for AutoMedal compatibility.

Uses the Kaggle Python API (not kagglehub, not subprocess CLI) to:
1. List all active competitions with pagination
2. Score each with Stage 1 (metadata-only) heuristics
3. Score top-N with Stage 2 (file listing) heuristics
4. Output ranked shortlist to scout/outputs/

Usage:
    python scout/discover.py
    python -m scout.discover
"""

import os
import sys
import json
import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scout.scoring import score_stage1, score_stage2, compute_final_score

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "scout", "outputs")
STAGE2_TOP_N = 20
MIN_FINAL_SCORE = 30


def _check_kaggle_auth():
    """Verify Kaggle API credentials are available."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    # Also check Windows-style path
    kaggle_json_win = os.path.join(os.environ.get("USERPROFILE", ""), ".kaggle", "kaggle.json")

    if not os.path.exists(kaggle_json) and not os.path.exists(kaggle_json_win):
        # Check env vars as fallback
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            print("ERROR: Kaggle API credentials not found.")
            print()
            print("Set up credentials using ONE of these methods:")
            print("  1. Place kaggle.json in ~/.kaggle/kaggle.json")
            print("  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
            print()
            print("Get your API token from: https://www.kaggle.com/settings")
            return False
    return True


def _competition_to_dict(comp):
    """Convert a Kaggle API competition object to a serializable dict."""
    fields = [
        "ref", "title", "description", "url", "category",
        "reward", "deadline", "teamCount", "isKernelsSubmissionsOnly",
        "evaluationMetric", "maxDailySubmissions", "maxTeamSize",
        "enabledDate",
    ]
    result = {}
    for field in fields:
        val = getattr(comp, field, None)
        if isinstance(val, datetime.datetime):
            val = val.isoformat()
        result[field] = val

    # Handle tags
    tags = getattr(comp, "tags", []) or []
    result["tags"] = []
    for tag in tags:
        if isinstance(tag, str):
            result["tags"].append(tag)
        elif hasattr(tag, "name"):
            result["tags"].append(tag.name)
        elif hasattr(tag, "ref"):
            result["tags"].append(tag.ref)

    return result


def _file_info_to_dict(f):
    """Convert a Kaggle API file object to a serializable dict."""
    return {
        "name": getattr(f, "name", str(f)),
        "totalBytes": getattr(f, "totalBytes", 0) or 0,
    }


def discover_competitions():
    """Discover and rank Kaggle competitions.

    Returns:
        List of candidate dicts sorted by final_score descending.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # ─── Fetch all active competitions with pagination ───
    print("  Fetching active competitions...")
    all_competitions = []
    page = 1
    while True:
        batch = api.competitions_list(page=page)
        if not batch:
            break
        all_competitions.extend(batch)
        print(f"    Page {page}: {len(batch)} competitions")
        page += 1
        # Safety limit to avoid runaway pagination
        if page > 20:
            break

    print(f"  Total competitions found: {len(all_competitions)}")

    # ─── Stage 1: Metadata scoring ───
    print("\n  Stage 1: Metadata scoring...")
    candidates = []
    for comp in all_competitions:
        comp_dict = _competition_to_dict(comp)
        s1_score, s1_reasons, disqualified = score_stage1(comp_dict)

        if disqualified:
            continue

        candidates.append({
            "competition": comp_dict,
            "stage1_score": s1_score,
            "stage1_reasons": s1_reasons,
            "stage2_score": 0,
            "stage2_reasons": [],
            "final_score": s1_score,
        })

    # Sort by stage1 score for stage2 selection
    candidates.sort(key=lambda x: x["stage1_score"], reverse=True)
    print(f"  Stage 1: {len(candidates)} candidates after filtering")

    # ─── Stage 2: File listing check (top N only) ───
    top_n = candidates[:STAGE2_TOP_N]
    if top_n:
        print(f"\n  Stage 2: Checking file listings for top {len(top_n)} candidates...")
        for candidate in top_n:
            slug = candidate["competition"]["ref"]
            try:
                files = api.competition_list_files(slug)
                file_dicts = [_file_info_to_dict(f) for f in files]
                candidate["competition"]["files"] = file_dicts
                s2_score, s2_reasons = score_stage2(file_dicts)
                candidate["stage2_score"] = s2_score
                candidate["stage2_reasons"] = s2_reasons
                candidate["final_score"] = compute_final_score(
                    candidate["stage1_score"], s2_score
                )
                print(f"    {slug}: +{s2_score} (files)")
            except Exception as e:
                candidate["stage2_reasons"] = [f"Error checking files: {e}"]
                print(f"    {slug}: [ERROR] {e}")

    # Re-sort by final score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Filter out low scores
    candidates = [c for c in candidates if c["final_score"] >= MIN_FINAL_SCORE]
    print(f"\n  Final candidates: {len(candidates)} (score >= {MIN_FINAL_SCORE})")

    return candidates


def write_json_output(candidates, path):
    """Write full structured JSON output."""
    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "min_score_threshold": MIN_FINAL_SCORE,
        "candidates": candidates,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  JSON: {path}")


def write_markdown_output(candidates, path):
    """Write human-readable markdown output."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# AutoMedal — Competition Candidates",
        f"",
        f"Generated: {now}  ",
        f"Candidates: {len(candidates)} (score >= {MIN_FINAL_SCORE})",
        "",
        "---",
        "",
    ]

    for i, candidate in enumerate(candidates, 1):
        comp = candidate["competition"]
        slug = comp.get("ref", "?")
        title = comp.get("title", "?")
        category = comp.get("category", "?")
        metric = comp.get("evaluationMetric", "?")
        teams = comp.get("teamCount", 0)
        deadline = comp.get("deadline", "?")
        final_score = candidate["final_score"]
        s1 = candidate["stage1_score"]
        s2 = candidate["stage2_score"]

        lines.append(f"## {i}. {title}")
        lines.append(f"")
        lines.append(f"- **Slug:** `{slug}`")
        lines.append(f"- **Score:** {final_score}/100 (stage1={s1}, stage2={s2})")
        lines.append(f"- **Category:** {category}")
        lines.append(f"- **Metric:** {metric}")
        lines.append(f"- **Teams:** {teams}")
        lines.append(f"- **Deadline:** {deadline}")
        lines.append(f"- **URL:** https://www.kaggle.com/competitions/{slug}")
        lines.append(f"")

        # Score breakdown
        all_reasons = candidate["stage1_reasons"] + candidate["stage2_reasons"]
        if all_reasons:
            lines.append(f"<details><summary>Score breakdown</summary>")
            lines.append(f"")
            for reason in all_reasons:
                lines.append(f"- {reason}")
            lines.append(f"")
            lines.append(f"</details>")
            lines.append(f"")

        lines.append("---")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Markdown: {path}")


def main():
    print("=" * 60)
    print("AutoMedal — Competition Discovery")
    print("=" * 60)

    if not _check_kaggle_auth():
        sys.exit(1)

    candidates = discover_competitions()

    # Write outputs
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUTS_DIR, "competition_candidates.json")
    md_path = os.path.join(OUTPUTS_DIR, "competition_candidates.md")

    print("\n  Writing outputs...")
    write_json_output(candidates, json_path)
    write_markdown_output(candidates, md_path)

    # Quick summary
    if candidates:
        top = candidates[0]
        print(f"\n  Top candidate: {top['competition']['ref']} "
              f"(score={top['final_score']})")
    else:
        print("\n  No compatible competitions found.")

    print("\nDone. Review scout/outputs/competition_candidates.md")


if __name__ == "__main__":
    main()
