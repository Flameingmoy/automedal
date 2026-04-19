"""Filesystem tools — read_file, write_file, edit_file, list_dir, grep.

All paths are resolved relative to REPO_ROOT and rejected if they escape it.
Tool bodies are ported from the prior automedal/agent_runtime.py so existing
prompts (which were written against pi's identical surface) port unchanged.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable

from automedal.agent.tools.base import REPO_ROOT, Tool, ToolResult, _safe, make_tool


# ── implementations (sync — kernel wraps in a coroutine) ─────────────────────

def _read_file(path: str) -> ToolResult:
    return ToolResult(_safe(path).read_text(encoding="utf-8"))


def _write_file(path: str, content: str) -> ToolResult:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ToolResult(f"wrote {path} ({len(content)} chars)")


def _edit_file(path: str, old: str, new: str) -> ToolResult:
    p = _safe(path)
    txt = p.read_text(encoding="utf-8")
    n = txt.count(old)
    if n != 1:
        return ToolResult(f"error: old string appears {n} times (needs exactly 1)", ok=False)
    p.write_text(txt.replace(old, new), encoding="utf-8")
    return ToolResult(f"edited {path}")


def _list_dir(path: str = ".") -> ToolResult:
    p = _safe(path)
    if not p.is_dir():
        return ToolResult(f"error: {path} is not a directory", ok=False)
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    return ToolResult("\n".join(f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries))


def _grep(pattern: str, path: str = ".", glob: str = "*") -> ToolResult:
    root = _safe(path)
    rx = re.compile(pattern)
    hits: list[str] = []
    walker: Iterable[Path]
    walker = [root] if root.is_file() else root.rglob("*")
    for f in walker:
        if not f.is_file() or not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    rel = f.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(hits) >= 80:
                        return ToolResult("\n".join(hits) + "\n... (truncated at 80 matches)")
        except (UnicodeDecodeError, PermissionError):
            continue
    return ToolResult("\n".join(hits) if hits else "(no matches)")


# ── Tool objects ─────────────────────────────────────────────────────────────

READ_FILE = make_tool(
    name="read_file",
    description="Read a UTF-8 file. Path is resolved relative to the repo root.",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative path"}},
        "required": ["path"],
    },
    fn=_read_file,
)

WRITE_FILE = make_tool(
    name="write_file",
    description="Create or overwrite a UTF-8 file relative to the repo root.",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    fn=_write_file,
)

EDIT_FILE = make_tool(
    name="edit_file",
    description=(
        "Replace `old` with `new` in `path`. `old` must appear EXACTLY once "
        "or the call fails."
    ),
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old":  {"type": "string"},
            "new":  {"type": "string"},
        },
        "required": ["path", "old", "new"],
    },
    fn=_edit_file,
)

LIST_DIR = make_tool(
    name="list_dir",
    description="List immediate children of a directory relative to the repo root.",
    schema={
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
        "required": [],
    },
    fn=_list_dir,
)

GREP = make_tool(
    name="grep",
    description=(
        "Search for `pattern` (regex) in files matching `glob` under `path`. "
        "Returns up to 80 'relpath:lineno: line' matches."
    ),
    schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path":    {"type": "string", "default": "."},
            "glob":    {"type": "string", "default": "*"},
        },
        "required": ["pattern"],
    },
    fn=_grep,
)


FS_TOOLS: list[Tool] = [READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP]
