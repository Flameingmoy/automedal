"""Tests for automedal.agent_runtime — path-guard tools + model factory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from automedal import agent_runtime as ar


# ── path guard ───────────────────────────────────────────────────────────────

def test_safe_rejects_parent_escape():
    with pytest.raises(PermissionError):
        ar._safe("../../etc/passwd")


def test_safe_rejects_absolute_outside_repo(tmp_path):
    with pytest.raises(PermissionError):
        ar._safe("/tmp/attack")


def test_safe_allows_repo_relative():
    p = ar._safe("prompts/researcher.md")
    assert p.is_absolute()
    assert ar.REPO_ROOT in p.parents


def test_safe_allows_repo_root():
    p = ar._safe(".")
    assert p == ar.REPO_ROOT


# ── read/write/edit roundtrip (via @tool .invoke interface) ──────────────────

def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    # Redirect REPO_ROOT to a throwaway dir so we don't pollute the repo
    monkeypatch.setattr(ar, "REPO_ROOT", tmp_path)

    ar.write_file.invoke({"path": "scratch.txt", "content": "hello\nworld\n"})
    out = ar.read_file.invoke({"path": "scratch.txt"})
    assert out == "hello\nworld\n"
    assert (tmp_path / "scratch.txt").read_text() == "hello\nworld\n"


def test_edit_file_exact_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "REPO_ROOT", tmp_path)
    ar.write_file.invoke({"path": "a.txt", "content": "alpha beta gamma"})

    ar.edit_file.invoke({"path": "a.txt", "old": "beta", "new": "BETA"})
    assert (tmp_path / "a.txt").read_text() == "alpha BETA gamma"


def test_edit_file_ambiguous_match_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "REPO_ROOT", tmp_path)
    ar.write_file.invoke({"path": "a.txt", "content": "x x x"})

    result = ar.edit_file.invoke({"path": "a.txt", "old": "x", "new": "y"})
    assert "error" in result.lower()
    # File must be unchanged
    assert (tmp_path / "a.txt").read_text() == "x x x"


def test_list_dir_returns_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "REPO_ROOT", tmp_path)
    (tmp_path / "one.txt").write_text("a")
    (tmp_path / "two.txt").write_text("b")
    (tmp_path / "subdir").mkdir()

    out = ar.list_dir.invoke({"path": "."})
    assert "one.txt" in out
    assert "two.txt" in out
    assert "subdir" in out


# ── slug parsing ─────────────────────────────────────────────────────────────

def test_parse_slug_splits_on_first_slash():
    assert ar.parse_slug("opencode-go/minimax-m2.7") == ("opencode-go", "minimax-m2.7")
    assert ar.parse_slug("openrouter/openai/gpt-4o-mini") == ("openrouter", "openai/gpt-4o-mini")


def test_parse_slug_rejects_bare_name():
    with pytest.raises(ValueError):
        ar.parse_slug("just-a-model")


# ── prompt loading ───────────────────────────────────────────────────────────

def test_load_prompt_has_all_phases():
    # All four phases must have prompt files present in the repo
    for phase in ("researcher", "strategist", "experimenter_edit", "experimenter_eval"):
        text = ar._load_prompt(phase)
        assert len(text) > 100, f"{phase} prompt suspiciously short"


def test_phase_tools_non_empty():
    for phase, tools in ar.PHASE_TOOLS.items():
        assert tools, f"{phase} has no tools registered"
        # Researcher gets arxiv_search; others don't need it
        names = {t.name for t in tools}
        if phase == "researcher":
            assert "arxiv_search" in names
