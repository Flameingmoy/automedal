"""Tests for automedal.paths.Layout in both dev and user modes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from automedal.paths import Layout


# ── dev mode (this repo) ──────────────────────────────────────────────────────

def test_dev_mode_detected_from_repo(tmp_path):
    """A directory with pyproject.toml + automedal/run_loop.py is dev mode."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'automedal'\n")
    (tmp_path / "automedal").mkdir()
    (tmp_path / "automedal" / "run_loop.py").write_text("# stub\n")
    layout = Layout(cwd=tmp_path)
    assert layout.mode == "dev"


def test_user_mode_detected_from_bare_dir(tmp_path):
    """An empty directory is user mode."""
    layout = Layout(cwd=tmp_path)
    assert layout.mode == "user"


def test_env_var_forces_dev(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMEDAL_DEV", "1")
    layout = Layout(cwd=tmp_path)
    assert layout.mode == "dev"


# ── visible paths are always at cwd root ─────────────────────────────────────

@pytest.mark.parametrize("mode", ["dev", "user"])
def test_visible_paths_at_cwd(tmp_path, mode):
    layout = Layout(cwd=tmp_path, mode=mode)
    assert layout.data_dir        == tmp_path / "data"
    assert layout.submissions_dir == tmp_path / "submissions"
    assert layout.journal_dir     == tmp_path / "journal"
    assert layout.knowledge_md    == tmp_path / "knowledge.md"
    assert layout.queue_md        == tmp_path / "experiment_queue.md"
    assert layout.research_md     == tmp_path / "research_notes.md"


# ── hidden paths differ by mode ───────────────────────────────────────────────

def test_dev_hidden_root_is_cwd(tmp_path):
    layout = Layout(cwd=tmp_path, mode="dev")
    assert layout.hidden_root == tmp_path
    assert layout.agent_dir   == tmp_path / "agent"
    assert layout.train_py    == tmp_path / "agent" / "train.py"
    assert layout.config_yaml == tmp_path / "configs" / "competition.yaml"
    assert layout.log_file    == tmp_path / "agent_loop.log"


def test_user_hidden_root_is_automedal_dir(tmp_path):
    layout = Layout(cwd=tmp_path, mode="user")
    assert layout.hidden_root == tmp_path / ".automedal"
    assert layout.agent_dir   == tmp_path / ".automedal" / "agent"
    assert layout.train_py    == tmp_path / ".automedal" / "agent" / "train.py"
    assert layout.config_yaml == tmp_path / ".automedal" / "configs" / "competition.yaml"
    assert layout.log_file    == tmp_path / ".automedal" / "logs" / "agent_loop.log"


def test_dev_results_tsv(tmp_path):
    layout = Layout(cwd=tmp_path, mode="dev")
    assert layout.results_tsv == tmp_path / "agent" / "results.tsv"


def test_user_results_tsv(tmp_path):
    layout = Layout(cwd=tmp_path, mode="user")
    assert layout.results_tsv == tmp_path / "results.tsv"


# ── LOG_FILE env var respected in dev mode ────────────────────────────────────

def test_dev_log_file_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_FILE", "/tmp/custom.log")
    layout = Layout(cwd=tmp_path, mode="dev")
    assert layout.log_file == Path("/tmp/custom.log")


def test_user_log_file_ignores_legacy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_FILE", "/tmp/custom.log")
    layout = Layout(cwd=tmp_path, mode="user")
    # User mode always uses .automedal/logs/
    assert layout.log_file == tmp_path / ".automedal" / "logs" / "agent_loop.log"


# ── as_env() returns strings ──────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["dev", "user"])
def test_as_env_all_strings(tmp_path, mode):
    layout = Layout(cwd=tmp_path, mode=mode)
    env = layout.as_env()
    assert all(isinstance(k, str) for k in env)
    assert all(isinstance(v, str) for v in env.values())


def test_as_env_contains_log_file(tmp_path):
    layout = Layout(cwd=tmp_path, mode="dev")
    env = layout.as_env()
    assert "LOG_FILE" in env
    assert "AUTOMEDAL_LOG_FILE" in env
    assert env["LOG_FILE"] == env["AUTOMEDAL_LOG_FILE"]
