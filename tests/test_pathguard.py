"""Path-guard regression tests for the bespoke fs tools.

Ports the relevant cases from the older `tests/test_agent_runtime.py`
which targeted the deepagents-based runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _set_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force REPO_ROOT to a sandbox before importing tools.

    The tools module reads AUTOMEDAL_CWD at import time, so we have to
    reload after monkeypatching for the override to take effect.
    """
    monkeypatch.setenv("AUTOMEDAL_CWD", str(tmp_path))
    import importlib
    import automedal.agent.tools.base as base
    importlib.reload(base)
    import automedal.agent.tools.fs as fs_mod
    importlib.reload(fs_mod)
    return base, fs_mod


def test_safe_resolves_relative_path(tmp_path, monkeypatch):
    base, _ = _set_repo_root(tmp_path, monkeypatch)
    p = base._safe("foo/bar.txt")
    assert p == (tmp_path / "foo" / "bar.txt").resolve()


def test_safe_rejects_parent_traversal(tmp_path, monkeypatch):
    base, _ = _set_repo_root(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        base._safe("../../etc/passwd")


def test_safe_rejects_absolute_outside(tmp_path, monkeypatch):
    base, _ = _set_repo_root(tmp_path, monkeypatch)
    with pytest.raises(PermissionError):
        base._safe("/etc/passwd")


def test_safe_accepts_absolute_inside(tmp_path, monkeypatch):
    base, _ = _set_repo_root(tmp_path, monkeypatch)
    inside = tmp_path / "deep" / "nested.txt"
    p = base._safe(str(inside))
    assert p == inside.resolve()


@pytest.mark.asyncio
async def test_read_write_edit_roundtrip(tmp_path, monkeypatch):
    _, fs = _set_repo_root(tmp_path, monkeypatch)
    r = await fs.WRITE_FILE(path="hello.md", content="hi\nworld")
    assert r.ok and "wrote hello.md" in r.text

    r = await fs.READ_FILE(path="hello.md")
    assert r.ok and r.text == "hi\nworld"

    r = await fs.EDIT_FILE(path="hello.md", old="world", new="planet")
    assert r.ok
    assert (tmp_path / "hello.md").read_text() == "hi\nplanet"


@pytest.mark.asyncio
async def test_edit_file_rejects_ambiguous_match(tmp_path, monkeypatch):
    _, fs = _set_repo_root(tmp_path, monkeypatch)
    (tmp_path / "x.md").write_text("aa aa")
    r = await fs.EDIT_FILE(path="x.md", old="aa", new="bb")
    assert not r.ok and "appears 2 times" in r.text


@pytest.mark.asyncio
async def test_read_outside_repo_returns_error(tmp_path, monkeypatch):
    _, fs = _set_repo_root(tmp_path, monkeypatch)
    r = await fs.READ_FILE(path="../../etc/passwd")
    assert not r.ok and "path escapes repo" in r.text


@pytest.mark.asyncio
async def test_grep_finds_pattern(tmp_path, monkeypatch):
    _, fs = _set_repo_root(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("def hello():\n    return 42\n")
    (tmp_path / "b.py").write_text("HELLO = 1\n")
    r = await fs.GREP(pattern=r"def\s+hello", path=".", glob="*.py")
    assert r.ok and "a.py:1" in r.text
    assert "b.py" not in r.text


@pytest.mark.asyncio
async def test_list_dir_orders_dirs_first(tmp_path, monkeypatch):
    _, fs = _set_repo_root(tmp_path, monkeypatch)
    (tmp_path / "afile.txt").write_text("x")
    (tmp_path / "bdir").mkdir()
    r = await fs.LIST_DIR(path=".")
    lines = r.text.splitlines()
    # dirs first, then files
    assert lines[0].startswith("d")
    assert "bdir" in lines[0]
    assert lines[1].startswith("f") and "afile.txt" in lines[1]
