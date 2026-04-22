"""Tool surface exposed to the bespoke agent kernel.

Each tool is a `Tool` dataclass with:
    name        — string identifier the model uses to call it
    description — natural-language summary the model sees
    schema      — JSON-Schema input shape (Anthropic/OpenAI tool spec)
    run         — async callable that returns ToolResult

All filesystem and shell tools are path-guarded against `REPO_ROOT`,
which is resolved from the `AUTOMEDAL_CWD` env var (with cwd fallback).
"""

from automedal.agent.tools.base import Tool, ToolResult, REPO_ROOT, _safe
from automedal.agent.tools.fs import (
    READ_FILE, WRITE_FILE, EDIT_FILE, LIST_DIR, GREP, FS_TOOLS,
)
from automedal.agent.tools.shell import RUN_SHELL
from automedal.agent.tools.cognition import RECALL, COGNITION_TOOLS, bm25_score_pairs
from automedal.agent.tools.arxiv import ARXIV_SEARCH
from automedal.agent.tools.subagent import make_subagent_tool
from automedal.agent.tools.advisor import make_advisor_tool

__all__ = [
    "Tool", "ToolResult", "REPO_ROOT", "_safe",
    "READ_FILE", "WRITE_FILE", "EDIT_FILE", "LIST_DIR", "GREP", "RUN_SHELL",
    "RECALL", "ARXIV_SEARCH",
    "FS_TOOLS", "COGNITION_TOOLS",
    "bm25_score_pairs", "make_subagent_tool", "make_advisor_tool",
]
