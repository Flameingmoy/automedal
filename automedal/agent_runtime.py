"""LangChain deepagents runtime — replaces the pi coding agent.

Public surface:
    build_model(provider, model)    -> BaseChatModel
    build_phase_agent(phase, model) -> CompiledStateGraph (deep agent)
    invoke_phase(agent, context, log_path) [async] -> exit code
    smoke_test(provider, model)     -> bool
    REPO_ROOT                       -> pathlib.Path (process-wide, from AUTOMEDAL_CWD)

Tool surface exposed to agents (all path-guarded to REPO_ROOT):
    read_file, write_file, edit_file, list_dir, grep, run_shell,
    arxiv_search (Researcher only — wraps harness/arxiv_search.py)
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

# deepagents + LangGraph
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend


# ── Repo root (every tool call is path-guarded against this) ─────────────────

def _resolve_repo_root() -> Path:
    """AUTOMEDAL_CWD takes precedence; fall back to the current working dir."""
    env_cwd = os.environ.get("AUTOMEDAL_CWD")
    if env_cwd:
        return Path(env_cwd).resolve()
    return Path.cwd().resolve()


REPO_ROOT: Path = _resolve_repo_root()


def _safe(p: str | os.PathLike) -> Path:
    """Resolve `p` relative to REPO_ROOT, rejecting paths that escape it."""
    raw = Path(p)
    q = (REPO_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        q.relative_to(REPO_ROOT)
    except ValueError:
        raise PermissionError(f"path escapes repo: {p}")
    return q


# ── Tool shims exposed to agents ─────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read a UTF-8 file. Path is resolved relative to the repo root."""
    return _safe(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 file relative to the repo root."""
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {path} ({len(content)} chars)"


@tool
def edit_file(path: str, old: str, new: str) -> str:
    """Replace `old` with `new` in `path`. `old` must appear exactly once.

    Matches pi's `edit_file` semantics so existing prompts port unchanged.
    """
    p = _safe(path)
    txt = p.read_text(encoding="utf-8")
    n = txt.count(old)
    if n != 1:
        return f"error: old string appears {n} times (needs exactly 1)"
    p.write_text(txt.replace(old, new), encoding="utf-8")
    return f"edited {path}"


@tool
def list_dir(path: str = ".") -> str:
    """List immediate children of a directory relative to the repo root."""
    p = _safe(path)
    if not p.is_dir():
        return f"error: {path} is not a directory"
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    return "\n".join(f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries)


@tool
def grep(pattern: str, path: str = ".", glob: str = "*") -> str:
    """Search for `pattern` in files matching `glob` under `path`.

    Returns up to 80 matches as 'relpath:lineno: line'.
    """
    root = _safe(path)
    rx = re.compile(pattern)
    hits: list[str] = []
    walker: Iterable[Path]
    if root.is_file():
        walker = [root]
    else:
        walker = root.rglob("*")
    for f in walker:
        if not f.is_file() or not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    rel = f.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(hits) >= 80:
                        return "\n".join(hits) + "\n... (truncated at 80 matches)"
        except (UnicodeDecodeError, PermissionError):
            continue
    return "\n".join(hits) if hits else "(no matches)"


@tool
def run_shell(command: str, timeout: int = 120) -> str:
    """Run a shell command from the repo root. Combined stdout+stderr, capped at 8KB."""
    try:
        r = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    out = (r.stdout or "") + (r.stderr or "")
    return out[:8000] if len(out) <= 8000 else out[:8000] + "\n... (truncated)"


@tool
def arxiv_search(query: str = "", arxiv_id: str = "") -> str:
    """Thin wrapper over harness/arxiv_search.py (Researcher-only tool).

    Use `query` for keyword search (returns top 5) or `arxiv_id` (comma-separated) to
    fetch specific papers.
    """
    script = REPO_ROOT / "harness" / "arxiv_search.py"
    if not script.exists():
        return "error: harness/arxiv_search.py not found"
    args = [sys.executable, str(script)]
    if arxiv_id:
        args += ["--id", arxiv_id]
    elif query:
        args += ["--query", query]
    else:
        return "error: provide `query` or `arxiv_id`"
    try:
        r = subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "error: arxiv_search timed out"
    return (r.stdout + r.stderr)[:8000]


# Phase → tool whitelist (matches run.sh's phase semantics)
PHASE_TOOLS: dict[str, list] = {
    "researcher":        [read_file, list_dir, grep, arxiv_search, run_shell],
    "strategist":        [read_file, write_file, edit_file, list_dir, grep, run_shell],
    "experimenter_edit": [read_file, write_file, edit_file, list_dir, grep, run_shell],
    "experimenter_eval": [read_file, write_file, edit_file, list_dir, grep, run_shell],
}

# Phase → prompt filename (relative to REPO_ROOT/prompts/)
PHASE_PROMPT: dict[str, str] = {
    "researcher":        "researcher.md",
    "strategist":        "strategist.md",
    "experimenter_edit": "experimenter.md",
    "experimenter_eval": "experimenter_eval.md",
}


# ── Model factory ────────────────────────────────────────────────────────────

_OPENCODE_BASE_URL = "https://opencode.ai/zen/go"


def build_model(provider: str, model: str, **kw: Any) -> BaseChatModel:
    """Return a LangChain chat model for (provider, model).

    `model` is the short id (e.g. `minimax-m2.7`) OR the combined slug
    (`opencode-go/minimax-m2.7`); the slug form is accepted for convenience.
    """
    # Accept both "opencode-go/minimax-m2.7" and "minimax-m2.7"
    if "/" in model and model.split("/", 1)[0] == provider:
        model = model.split("/", 1)[1]

    if provider == "opencode-go":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("OPENCODE_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENCODE_API_KEY not set (run `automedal setup`)")
        return ChatAnthropic(
            base_url=_OPENCODE_BASE_URL,
            api_key=api_key,
            model=model,
            max_tokens=kw.get("max_tokens", 4096),
            timeout=kw.get("timeout", 120),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=kw.get("max_tokens", 4096),
            timeout=kw.get("timeout", 120),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=kw.get("timeout", 120),
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=kw.get("timeout", 120),
        )

    if provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            timeout=kw.get("timeout", 120),
        )

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint
        from langchain_openai import ChatOpenAI
        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1"
        return ChatOpenAI(
            model=model,
            api_key="ollama",
            base_url=base,
            timeout=kw.get("timeout", 120),
        )

    # Generic OpenAI-compatible fallback via init_chat_model
    from langchain.chat_models import init_chat_model
    return init_chat_model(f"{provider}:{model}")


def parse_slug(slug: str) -> tuple[str, str]:
    """'opencode-go/minimax-m2.7' -> ('opencode-go', 'minimax-m2.7')."""
    if "/" not in slug:
        raise ValueError(f"expected 'provider/model', got {slug!r}")
    provider, model = slug.split("/", 1)
    return provider, model


# ── Phase agent factory ──────────────────────────────────────────────────────

def _load_prompt(phase: str) -> str:
    fname = PHASE_PROMPT[phase]
    path = REPO_ROOT / "prompts" / fname
    return path.read_text(encoding="utf-8")


def build_phase_agent(phase: str, model: BaseChatModel):
    """Return a compiled deepagents state graph for `phase`."""
    if phase not in PHASE_TOOLS:
        raise ValueError(f"unknown phase: {phase}")
    return create_deep_agent(
        model=model,
        tools=PHASE_TOOLS[phase],
        system_prompt=_load_prompt(phase),
        backend=LocalShellBackend(root_dir=str(REPO_ROOT), virtual_mode=True, timeout=120),
    )


# ── Event streaming ──────────────────────────────────────────────────────────
# Event → text formatting is shared with the pi stdin path in
# harness/stream_events.py (format_langgraph_event). Single source of truth
# keeps tui/sources/log_tail.py working for both runtimes.


class _EventWriter:
    """Streams LangGraph astream_events(v2) output to log file + optional tty."""

    def __init__(self, log_path: Path | None = None, echo: bool = True) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.echo = echo
        self._fh = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.log_path, "a", encoding="utf-8")
        self._inline_active = False

    def close(self) -> None:
        if self._inline_active:
            self._write_raw("\n")
            self._inline_active = False
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def _write_raw(self, s: str) -> None:
        if self._fh is not None:
            self._fh.write(s)
            self._fh.flush()
        if self.echo:
            sys.stdout.write(s)
            sys.stdout.flush()

    def handle(self, ev: dict) -> None:
        from harness.stream_events import format_langgraph_event
        for text, inline in format_langgraph_event(ev):
            if inline:
                self._write_raw(text)
                self._inline_active = True
            else:
                if self._inline_active:
                    self._write_raw("\n")
                self._write_raw(text + "\n")
                self._inline_active = False


async def invoke_phase(
    agent,
    runtime_context: str,
    log_path: Path | None = None,
    *,
    echo: bool = True,
    recursion_limit: int = 50,
) -> int:
    """Stream an agent invocation. Returns 0 on success, 1 on failure."""
    writer = _EventWriter(log_path=log_path, echo=echo)
    try:
        async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": runtime_context}]},
            version="v2",
            config={"recursion_limit": recursion_limit},
        ):
            writer.handle(ev)
        writer.close()
        return 0
    except Exception as exc:
        writer.close()
        msg = f"  [agent] ERROR: {type(exc).__name__}: {exc}"
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except Exception:
                pass
        if echo:
            sys.stderr.write(msg + "\n")
        return 1


# ── Smoke test (called by dispatch.py doctor + setup wizard) ─────────────────

def smoke_test(provider: str, model: str, timeout: int = 30) -> tuple[bool, str]:
    """Return (passed, detail). True iff the model responds with READY."""
    try:
        m = build_model(provider, model, timeout=timeout, max_tokens=64)
    except Exception as exc:
        return False, f"init error: {type(exc).__name__}: {exc}"
    try:
        r = m.invoke("Say READY and nothing else.")
    except Exception as exc:
        return False, f"invoke error: {type(exc).__name__}: {exc}"
    from harness.stream_events import _extract_text_delta
    text = _extract_text_delta(getattr(r, "content", ""))
    if "ready" in text.strip().lower():
        return True, f"ok ({text.strip()[:60]!r})"
    return False, f"unexpected response: {text!r}"


__all__ = [
    "REPO_ROOT",
    "PHASE_PROMPT",
    "PHASE_TOOLS",
    "build_model",
    "build_phase_agent",
    "invoke_phase",
    "parse_slug",
    "smoke_test",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "run_shell",
    "arxiv_search",
]
