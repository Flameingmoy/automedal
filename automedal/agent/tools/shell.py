"""Shell tool — `run_shell` executes a bash command bound to REPO_ROOT.

Combined stdout+stderr capped at 8KB. Defense-in-depth, not a security
boundary: the model already has direct file-write access via fs tools.
"""

from __future__ import annotations

import asyncio

from automedal.agent.tools.base import REPO_ROOT, Tool, ToolResult, make_tool


async def _run_shell(command: str, timeout: int = 120) -> ToolResult:
    proc = await asyncio.create_subprocess_exec(
        "bash", "-lc", command,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return ToolResult(f"error: command timed out after {timeout}s", ok=False)
    out = (stdout or b"").decode(errors="replace")
    truncated = len(out) > 8000
    out = out[:8000] + ("\n... (truncated)" if truncated else "")
    ok = proc.returncode == 0
    return ToolResult(out, ok=ok)


RUN_SHELL: Tool = make_tool(
    name="run_shell",
    description=(
        "Run a shell command via bash -lc, anchored at the repo root. "
        "Returns combined stdout+stderr (capped at 8KB)."
    ),
    schema={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "default": 120, "minimum": 1, "maximum": 600},
        },
        "required": ["command"],
    },
    fn=_run_shell,
)
