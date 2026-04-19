"""Per-phase orchestrators.

Each module exposes a `run(provider, events, **slots)` coroutine that
builds a fresh AgentKernel with the right tool allowlist and renders
the phase's jinja prompt with the supplied runtime-context slots.
"""

from automedal.agent.phases import (
    researcher,
    strategist,
    experimenter_edit,
    experimenter_eval,
    analyzer,
)

__all__ = [
    "researcher",
    "strategist",
    "experimenter_edit",
    "experimenter_eval",
    "analyzer",
]
