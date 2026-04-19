"""arxiv_search tool — researcher-only.

Wraps the `arxiv` Python package. Returns a fixed-format text block the
Researcher prompt already knows how to consume. Skips papers older than
`max_age_days` (default 3 years) to bias toward recent work.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from automedal.agent.tools.base import Tool, ToolResult, make_tool


def _format_paper(paper: Any, idx: int, full_abstract: bool) -> str:
    published = paper.published.strftime("%Y-%m-%d")
    age_days = (
        _dt.datetime.now(_dt.timezone.utc)
        - paper.published.replace(tzinfo=_dt.timezone.utc)
    ).days
    abstract = (paper.summary or "").replace("\n", " ").strip()
    if not full_abstract and len(abstract) > 300:
        abstract = abstract[:297] + "..."
    arxiv_id = paper.entry_id.split("/")[-1].split("v")[0]
    return (
        f"=== Paper {idx} ===\n"
        f"Title: {paper.title}\n"
        f"ArXiv ID: {arxiv_id}\n"
        f"Date: {published} ({age_days}d ago)\n"
        f"Abstract: {abstract}\n==="
    )


def _arxiv_search(
    query: str | None = None,
    ids: str | None = None,
    max_results: int = 5,
    max_age_days: int = 1095,
) -> ToolResult:
    if not query and not ids:
        return ToolResult("error: provide either `query` or `ids`", ok=False)
    try:
        import arxiv  # type: ignore
    except ImportError:
        return ToolResult(
            "error: the `arxiv` package is not installed. "
            "Install it with `pip install arxiv` (or `automedal[research]`).",
            ok=False,
        )

    client = arxiv.Client()
    if ids:
        id_list = [s.strip() for s in ids.split(",") if s.strip()]
        results = list(client.results(arxiv.Search(id_list=id_list)))
        if not results:
            return ToolResult(f"(no papers found for IDs {id_list})")
        return ToolResult("\n\n".join(_format_paper(p, i, True) for i, p in enumerate(results, 1)))

    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=max_age_days)
    keep = []
    for paper in client.results(search):
        if paper.published.replace(tzinfo=_dt.timezone.utc) < cutoff:
            continue
        keep.append(paper)
        if len(keep) >= max_results:
            break
    if not keep:
        return ToolResult(f"(no recent results for query {query!r})")
    return ToolResult("\n\n".join(_format_paper(p, i, False) for i, p in enumerate(keep, 1)))


ARXIV_SEARCH = make_tool(
    name="arxiv_search",
    description=(
        "Search arxiv by `query` (3-6 keywords) or fetch full abstracts by "
        "comma-separated `ids`. Filters out papers older than 3 years by "
        "default. Researcher-only."
    ),
    schema={
        "type": "object",
        "properties": {
            "query":        {"type": "string"},
            "ids":          {"type": "string", "description": "Comma-separated arxiv IDs"},
            "max_results":  {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "max_age_days": {"type": "integer", "default": 1095, "minimum": 30},
        },
        "required": [],
    },
    fn=_arxiv_search,
)
