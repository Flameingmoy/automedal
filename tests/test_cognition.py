"""BM25 cognition tests — recall + dedupe scoring."""

from __future__ import annotations

import importlib

import pytest


def _set_repo_root(tmp_path, monkeypatch):
    """Repoint REPO_ROOT at tmp_path without reloading `base` (which would
    swap the ToolResult class identity and break other test modules)."""
    monkeypatch.setenv("AUTOMEDAL_CWD", str(tmp_path))
    import automedal.agent.tools.base as base
    import automedal.agent.tools.cognition as cog
    monkeypatch.setattr(base, "REPO_ROOT", tmp_path.resolve())
    monkeypatch.setattr(cog, "REPO_ROOT", tmp_path.resolve())
    return cog


@pytest.mark.asyncio
async def test_recall_returns_relevant_chunks(tmp_path, monkeypatch):
    cog = _set_repo_root(tmp_path, monkeypatch)
    (tmp_path / "knowledge.md").write_text(
        "# KB\n\n"
        "## Models\n- xgboost early stopping rounds=50 was optimal (exp 0007)\n"
        "## Features\n- target encoding smoothing was noise (exp 0004)\n"
        "## HPO\n- Optuna trials below 25 collapse the metric (exps 0010, 0011)\n"
    )
    res = await cog.RECALL(query="xgboost early stopping", k=3)
    assert res.ok
    assert "xgboost" in res.text.lower()
    assert "early stopping" in res.text.lower()


@pytest.mark.asyncio
async def test_recall_handles_empty_knowledge_base(tmp_path, monkeypatch):
    cog = _set_repo_root(tmp_path, monkeypatch)
    res = await cog.RECALL(query="anything", k=5)
    assert res.ok
    assert "no chunks" in res.text.lower() or "empty" in res.text.lower()


@pytest.mark.asyncio
async def test_recall_uses_research_notes_tail(tmp_path, monkeypatch):
    cog = _set_repo_root(tmp_path, monkeypatch)
    notes = "\n".join(
        f"## exp {i:04d} · scheduled · query: \"x\"\n- Paper: \"old paper {i}\" lorem ipsum"
        for i in range(1, 25)
    ) + "\n## exp 0099 · stagnation · query: \"catboost depth\"\n- Paper: \"depth tuning catboost\" lorem"
    (tmp_path / "research_notes.md").write_text(notes)
    res = await cog.RECALL(query="catboost depth", k=2)
    assert res.ok
    assert "catboost" in res.text.lower()


def test_bm25_score_pairs_ranks_overlapping_higher(tmp_path, monkeypatch):
    cog = _set_repo_root(tmp_path, monkeypatch)
    candidates = [
        "switched LGBM to dart boosting, n_estimators 800→1200",
        "added one-hot encoding to categorical_cols",
        "changed XGB max_depth to 10",
    ]
    scores = cog.bm25_score_pairs("LGBM dart boosting tweak", candidates)
    assert len(scores) == 3
    assert scores[0] == max(scores)
