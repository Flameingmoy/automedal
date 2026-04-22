"""Advisor loop — consult a frontier model (Kimi K2.6 by default) at key junctions.

Inspired by Anthropic's advisor-strategy post. The advisor never calls tools;
it returns a short directive the executor (or orchestrator) consumes.

Public surface:
    consult(purpose, question, context, *, max_tokens, events) -> AdvisorOpinion
    is_enabled(junction) -> bool
    reset_iteration_budget() -> None
    budget_state() -> dict
"""

from automedal.advisor.client import AdvisorOpinion, consult
from automedal.advisor.budget import (
    budget_state,
    consume_tokens,
    is_enabled,
    remaining_tokens,
    reset_iteration_budget,
)

__all__ = [
    "AdvisorOpinion",
    "consult",
    "is_enabled",
    "reset_iteration_budget",
    "remaining_tokens",
    "consume_tokens",
    "budget_state",
]
