"""spawn_subagent tool — runs a focused sub-task in a fresh kernel.

Primary use: Researcher fanning out 3 arxiv queries in parallel via
asyncio.gather. The subagent inherits the parent's provider but gets a
fresh message history and a tool allowlist intersected against the
parent's available tools (subagent cannot escalate).
"""

from __future__ import annotations

from typing import Iterable

from automedal.agent.tools.base import Tool, ToolResult, make_tool


_DEFAULT_ALLOWLIST = ("read_file", "grep", "list_dir", "recall", "arxiv_search")


def make_subagent_tool(
    *,
    provider,
    parent_tools: list[Tool],
    events,
    depth: int = 0,
    max_depth: int = 2,
    default_max_steps: int = 20,
    default_allowlist: Iterable[str] = _DEFAULT_ALLOWLIST,
) -> Tool:
    """Build a `spawn_subagent` tool bound to the given provider + parent toolset.

    The returned tool, when called, builds a fresh AgentKernel with the
    subset of `parent_tools` named in `tools` (or the default allowlist),
    runs it on the supplied `prompt`, and returns the assistant's final text.
    """
    by_name = {t.name: t for t in parent_tools}
    default_set = tuple(default_allowlist)

    async def _spawn(
        prompt: str,
        tools: list[str] | None = None,
        max_steps: int = default_max_steps,
        label: str = "subagent",
    ) -> ToolResult:
        if depth >= max_depth:
            return ToolResult(
                f"error: subagent depth cap reached (depth={depth}, max={max_depth})",
                ok=False,
            )
        wanted = tuple(tools) if tools else default_set
        allowed = [by_name[n] for n in wanted if n in by_name]
        if not allowed:
            return ToolResult(
                f"error: none of the requested tools {wanted!r} are available "
                f"to the parent (parent has {sorted(by_name)})",
                ok=False,
            )

        # Local import to avoid a circular dep (kernel imports tools, tools
        # don't normally know about the kernel).
        from automedal.agent.kernel import AgentKernel

        sink = events.child_subagent(label=label) if events is not None else None
        if sink is not None:
            sink.subagent_start(label=label, prompt_preview=prompt[:160])

        sub_tools = list(allowed)
        # Allow nested subagents up to max_depth
        sub_tools.append(make_subagent_tool(
            provider=provider, parent_tools=allowed, events=sink,
            depth=depth + 1, max_depth=max_depth,
            default_max_steps=default_max_steps, default_allowlist=default_set,
        ))

        kernel = AgentKernel(
            provider=provider,
            system_prompt=(
                "You are a focused sub-agent. Complete the requested sub-task "
                "with the tools available, then summarize the result in your "
                "final assistant message. Do not chat — be terse and factual."
            ),
            tools=sub_tools,
            events=sink,
            max_steps=max_steps,
        )
        report = await kernel.run(prompt)

        if sink is not None:
            sink.subagent_end(label=label, ok=(report.stop == "assistant_done"),
                              result_preview=(report.final_text or "")[:160])
            # Do NOT call sink.close() — child sinks share file handles with the parent.

        if report.stop == "assistant_done":
            return ToolResult(report.final_text or "(empty)", ok=True)
        return ToolResult(
            f"subagent stopped with {report.stop}: {report.error or ''}",
            ok=False,
        )

    return make_tool(
        name="spawn_subagent",
        description=(
            "Spawn a focused sub-agent on `prompt`. The sub-agent gets a "
            "restricted tool allowlist (intersected against the parent's). "
            "Use for parallel fanout (e.g., 3 arxiv queries in parallel)."
        ),
        schema={
            "type": "object",
            "properties": {
                "prompt":    {"type": "string"},
                "tools":     {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool name allowlist; subset of the parent's tools.",
                },
                "max_steps": {"type": "integer", "default": default_max_steps,
                              "minimum": 1, "maximum": 50},
                "label":     {"type": "string", "default": "subagent"},
            },
            "required": ["prompt"],
        },
        fn=_spawn,
    )
