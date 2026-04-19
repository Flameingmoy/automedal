"""Tool primitives shared by all tool modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


# ── Repo-root resolution (process-wide) ──────────────────────────────────────

def _resolve_repo_root() -> Path:
    """AUTOMEDAL_CWD takes precedence; fall back to the current working dir."""
    env_cwd = os.environ.get("AUTOMEDAL_CWD")
    if env_cwd:
        return Path(env_cwd).resolve()
    return Path.cwd().resolve()


REPO_ROOT: Path = _resolve_repo_root()


def _safe(p: str | os.PathLike) -> Path:
    """Resolve `p` relative to REPO_ROOT; reject paths that escape it.

    Ported verbatim from the prior automedal/agent_runtime.py path guard.
    """
    raw = Path(p)
    q = (REPO_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        q.relative_to(REPO_ROOT)
    except ValueError:
        raise PermissionError(f"path escapes repo: {p}")
    return q


# ── Tool dataclass ───────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a tool invocation. `text` is fed back to the model verbatim."""
    text: str
    ok: bool = True


@dataclass
class Tool:
    """A single tool the agent can invoke.

    `run` is an async callable taking keyword args matching `schema.properties`
    and returning a `ToolResult`. We always wrap the user's callable so we can
    catch and serialize exceptions back to the model as `ok=False`.
    """
    name: str
    description: str
    schema: dict
    run: Callable[..., Awaitable[ToolResult]]

    async def __call__(self, **kwargs: Any) -> ToolResult:
        try:
            res = await self.run(**kwargs)
        except PermissionError as exc:
            return ToolResult(text=f"error: {exc}", ok=False)
        except TypeError as exc:
            return ToolResult(text=f"error: bad arguments: {exc}", ok=False)
        except Exception as exc:
            return ToolResult(text=f"error: {type(exc).__name__}: {exc}", ok=False)
        if isinstance(res, ToolResult):
            return res
        # Tolerate plain-string returns from simpler tools
        return ToolResult(text=str(res), ok=True)


def make_tool(
    name: str,
    description: str,
    schema: dict,
    fn: Callable[..., Any],
) -> Tool:
    """Wrap a sync or async function into a Tool.

    If `fn` is sync we run it directly (the kernel awaits the wrapper).
    """
    import asyncio
    import inspect

    if inspect.iscoroutinefunction(fn):
        async def run(**kw: Any) -> ToolResult:
            r = await fn(**kw)
            return r if isinstance(r, ToolResult) else ToolResult(text=str(r))
    else:
        async def run(**kw: Any) -> ToolResult:  # type: ignore[no-redef]
            r = fn(**kw)
            return r if isinstance(r, ToolResult) else ToolResult(text=str(r))

    return Tool(name=name, description=description, schema=schema, run=run)
