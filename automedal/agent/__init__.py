"""AutoMedal bespoke agent runtime.

A small, hand-written async agent kernel purpose-built for AutoMedal's
four-phase ML loop (Researcher / Strategist / Experimenter / Analyzer).
Replaces the prior pi (Node) and deepagents (LangChain) runtimes.

Public surface:
    AgentKernel          — the tool-call loop
    build_provider(name) — provider factory (anthropic, openai, ollama, opencode-go)
    EventSink            — JSONL event emitter
    REPO_ROOT            — process-wide repo root resolved from AUTOMEDAL_CWD

See plans/stateful-dancing-peacock.md for the design rationale.
"""

from automedal.agent.kernel import AgentKernel, RunReport
from automedal.agent.events import EventSink
from automedal.agent.providers import build_provider
from automedal.agent.tools import REPO_ROOT

__all__ = ["AgentKernel", "RunReport", "EventSink", "build_provider", "REPO_ROOT"]
