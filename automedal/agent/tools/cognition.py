"""BM25 cognition tool — `recall(query, k)` over knowledge.md + research_notes.

Lexical retrieval (rank-bm25) is good enough for AutoMedal's technical
text where the relevant vocabulary (model names, metric names, feature
names) matches between query and chunks. Avoids the dep weight of
sentence-transformers + faiss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from automedal.agent.tools.base import REPO_ROOT, Tool, ToolResult, _safe, make_tool


_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _chunk_by_heading(text: str, source: str) -> list[tuple[str, str]]:
    """Split a markdown doc on `##`-level headings; return (heading, body) pairs.

    Lines before the first `##` form a single chunk under the file's name.
    """
    chunks: list[tuple[str, str]] = []
    current_head = source
    current_body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                chunks.append((current_head, "\n".join(current_body).strip()))
            current_head = line[3:].strip() or source
            current_body = [line]
        else:
            current_body.append(line)
    if current_body:
        chunks.append((current_head, "\n".join(current_body).strip()))
    return [(h, b) for h, b in chunks if b]


def _tail_research_notes(text: str, n: int) -> list[tuple[str, str]]:
    """Return the last `n` `##` blocks of research_notes.md as (heading, body) pairs."""
    blocks = _chunk_by_heading(text, "research_notes.md")
    return blocks[-n:] if len(blocks) > n else blocks


@dataclass
class _BM25Index:
    """Wraps a rank_bm25.BM25Okapi over a list of (label, body) chunks."""
    sources: dict[str, float] = field(default_factory=dict)  # path → mtime at index time
    labels: list[str] = field(default_factory=list)
    bodies: list[str] = field(default_factory=list)
    _bm25: object | None = None

    def stale(self, paths: Iterable[Path]) -> bool:
        seen: dict[str, float] = {}
        for p in paths:
            try:
                seen[str(p)] = p.stat().st_mtime
            except OSError:
                seen[str(p)] = 0.0
        if seen.keys() != self.sources.keys():
            return True
        return any(seen[k] > self.sources[k] for k in seen)

    def rebuild(self, paths: Iterable[Path], notes_tail: int = 20) -> None:
        from rank_bm25 import BM25Okapi

        labels: list[str] = []
        bodies: list[str] = []
        sources: dict[str, float] = {}
        for p in paths:
            try:
                txt = p.read_text(encoding="utf-8")
                sources[str(p)] = p.stat().st_mtime
            except OSError:
                sources[str(p)] = 0.0
                continue
            if p.name == "research_notes.md":
                blocks = _tail_research_notes(txt, notes_tail)
            else:
                blocks = _chunk_by_heading(txt, p.name)
            for head, body in blocks:
                labels.append(f"{p.name} — {head}")
                bodies.append(body)
        self.sources = sources
        self.labels = labels
        self.bodies = bodies
        if bodies:
            self._bm25 = BM25Okapi([_tokenize(b) for b in bodies])
        else:
            self._bm25 = None

    def query(self, q: str, k: int) -> list[tuple[float, str, str]]:
        if not self._bm25 or not self.bodies:
            return []
        scores = self._bm25.get_scores(_tokenize(q))  # type: ignore[attr-defined]
        ranked = sorted(zip(scores, self.labels, self.bodies), key=lambda t: -t[0])
        return [(s, l, b) for s, l, b in ranked[:k] if s > 0]


_INDEX = _BM25Index()


def _index_paths() -> list[Path]:
    return [REPO_ROOT / "knowledge.md", REPO_ROOT / "research_notes.md"]


def _ensure_fresh() -> None:
    paths = [p for p in _index_paths() if p.exists()]
    if _INDEX.stale(paths) or _INDEX._bm25 is None:
        _INDEX.rebuild(paths)


def _recall(query: str, k: int = 5) -> ToolResult:
    _ensure_fresh()
    hits = _INDEX.query(query, max(1, min(k, 10)))
    if not hits:
        return ToolResult("(no chunks matched — knowledge base may be empty)", ok=True)
    lines = []
    for score, label, body in hits:
        snippet = body if len(body) <= 600 else body[:597] + "..."
        lines.append(f"### {label}  (score={score:.2f})\n{snippet}")
    return ToolResult("\n\n".join(lines))


RECALL = make_tool(
    name="recall",
    description=(
        "BM25-search the curated knowledge base (knowledge.md + recent "
        "research_notes.md) for chunks relevant to `query`. Returns up to "
        "`k` ranked snippets."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text query"},
            "k":     {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
    fn=_recall,
)


# Re-exported for dedupe.py — same scoring on arbitrary text pairs.
def bm25_score_pairs(query: str, candidates: list[str]) -> list[float]:
    """BM25 score `query` against each candidate body. Returns same-order scores."""
    from rank_bm25 import BM25Okapi
    if not candidates:
        return []
    bm = BM25Okapi([_tokenize(c) for c in candidates])
    return list(bm.get_scores(_tokenize(query)))


COGNITION_TOOLS: list[Tool] = [RECALL]
