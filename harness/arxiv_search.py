"""
AutoMedal — Arxiv Search Helper
==================================
CLI tool for the Researcher phase. Replaces inline Python arxiv queries
with a single standardised command.

Usage:
    python harness/arxiv_search.py --query "gradient boosting ensemble diversity"
    python harness/arxiv_search.py --query "tabular classification calibration" --max-results 3
    python harness/arxiv_search.py --id "2405.03389,2507.20048"  # fetch specific papers
"""

import argparse
import datetime
import sys


def _import_arxiv():
    try:
        import arxiv
        return arxiv
    except ImportError:
        print(
            "ERROR: arxiv package not installed. "
            "Run: uv sync --extra research",
            file=sys.stderr,
        )
        sys.exit(1)


def search_by_query(query, max_results=5, max_age_days=1095):
    arxiv = _import_arxiv()
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,  # fetch extra to filter by date
        sort_by=arxiv.SortCriterion.Relevance,
    )

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=max_age_days
    )
    results = []
    for paper in client.results(search):
        if paper.published.replace(tzinfo=datetime.timezone.utc) < cutoff:
            continue
        results.append(paper)
        if len(results) >= max_results:
            break
    return results


def search_by_ids(id_list):
    arxiv = _import_arxiv()
    client = arxiv.Client()
    search = arxiv.Search(id_list=id_list)
    return list(client.results(search))


def format_paper(paper, index, full_abstract=False):
    published = paper.published.strftime("%Y-%m-%d")
    age_days = (
        datetime.datetime.now(datetime.timezone.utc)
        - paper.published.replace(tzinfo=datetime.timezone.utc)
    ).days
    abstract = paper.summary.replace("\n", " ").strip()
    if not full_abstract and len(abstract) > 300:
        abstract = abstract[:297] + "..."

    lines = [
        f"=== Paper {index} ===",
        f"Title: {paper.title}",
        f"ArXiv ID: {paper.entry_id.split('/')[-1].split('v')[0]}",
        f"Date: {published} ({age_days}d ago)",
        f"Abstract: {abstract}",
        "===",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AutoMedal arxiv search helper for the Researcher phase"
    )
    parser.add_argument(
        "--query",
        help="Search query (3-6 keywords, no quotes needed)",
    )
    parser.add_argument(
        "--id",
        help="Comma-separated arxiv IDs to fetch directly (e.g. 2405.03389,2507.20048)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=1095,
        help="Skip papers older than this many days (default: 1095 = 3 years)",
    )
    args = parser.parse_args()

    if not args.query and not args.id:
        parser.error("Either --query or --id is required")

    if args.id:
        id_list = [s.strip() for s in args.id.split(",") if s.strip()]
        results = search_by_ids(id_list)
        if not results:
            print("No papers found for the given IDs.")
            return
        for i, paper in enumerate(results, 1):
            print(format_paper(paper, i, full_abstract=True))
            print()
    else:
        results = search_by_query(
            args.query,
            max_results=args.max_results,
            max_age_days=args.max_age_days,
        )
        if not results:
            print(f"No results found for query: {args.query}")
            return
        for i, paper in enumerate(results, 1):
            print(format_paper(paper, i, full_abstract=False))
            print()


if __name__ == "__main__":
    main()
