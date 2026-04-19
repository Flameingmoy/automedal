"""Jinja-templated phase prompts.

Each phase has a `<name>.md.j2` file in this directory. The body of
the original prompt is preserved verbatim; the runtime-context block
at the bottom uses named slots that the orchestrator fills in.

Usage:
    from automedal.agent.prompts import render_prompt
    text = render_prompt("strategist", exp_id="0042", iteration=3, ...)
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

PROMPTS_DIR = Path(__file__).parent

PHASES = (
    "researcher",
    "strategist",
    "experimenter",
    "experimenter_eval",
    "analyzer",
)

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2", "md"), default=False),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)


def render_prompt(phase: str, **slots) -> str:
    """Render `<phase>.md.j2` with the given slot values.

    Raises `ValueError` if `phase` is unknown,
    `jinja2.TemplateNotFound` if the file is missing, and
    `jinja2.UndefinedError` if a required slot was omitted.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    return _env.get_template(f"{phase}.md.j2").render(**slots)


def available_phases() -> tuple[str, ...]:
    return PHASES


__all__ = ["render_prompt", "available_phases", "PHASES", "PROMPTS_DIR"]
