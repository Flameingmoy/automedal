"""Pure state reducer for the AutoMedal TUI.

PhaseMachine.reduce((state, event)) -> state. Kept pure so it's trivial to unit-test
against synthetic event sequences. See tests/test_phase_machine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from tui.events import (
    CompetitionInfo,
    Event,
    GpuSample,
    HarnessMarker,
    HeartBeat,
    IterationEnd,
    IterationStart,
    JournalEntry,
    MemoryTouched,
    Phase,
    QueueState,
    RawLine,
    ResultRow,
    TrainingFinished,
)

# Seconds of silence in EXPERIMENT/CODING before switching to FROZEN.
FREEZE_SECS = 120.0


@dataclass
class ExperimentSummary:
    exp_id: str
    slug: str = ""
    status: str = "running"     # running | kept | reverted | crash
    val_loss: Optional[float] = None
    best_so_far: Optional[float] = None
    git_tag: str = ""
    hypothesis: str = ""


@dataclass
class AppState:
    phase: Phase = Phase.IDLE
    phase_since_ts: float = 0.0
    last_event_ts: float = 0.0

    iteration: int = 0
    total_iterations: int = 0
    current_exp_id: str = ""

    # Training budget (minutes) — default mirrors run.sh TRAIN_BUDGET_MINUTES.
    train_budget_minutes: float = 10.0
    training_started_ts: float = 0.0

    val_losses: list[tuple[str, float]] = field(default_factory=list)   # (exp_id, loss), ordered
    best_val_loss: float = float("inf")

    experiments: dict[str, ExperimentSummary] = field(default_factory=dict)
    experiment_order: list[str] = field(default_factory=list)           # insertion order for display

    queue: QueueState = field(default_factory=QueueState)

    gpu: Optional[GpuSample] = None
    competition: CompetitionInfo = field(default_factory=CompetitionInfo)

    # Transient UI flags.
    new_best_toast_for: str = ""                                        # exp_id that just took gold
    frozen_reason: str = ""

    def top_n(self, n: int = 5) -> list[ExperimentSummary]:
        candidates = [e for e in self.experiments.values() if e.val_loss is not None and e.status != "crash"]
        return sorted(candidates, key=lambda e: e.val_loss)[:n]  # type: ignore[arg-type,return-value]


def _ensure_exp(state: AppState, exp_id: str) -> ExperimentSummary:
    if exp_id not in state.experiments:
        state.experiments[exp_id] = ExperimentSummary(exp_id=exp_id)
        state.experiment_order.append(exp_id)
    return state.experiments[exp_id]


def _normalize_status(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r in ("better", "improved", "kept", "success"):
        return "kept"
    if r in ("worse", "reverted", "revert"):
        return "reverted"
    if r in ("crash", "error", "failed"):
        return "crash"
    return r or "running"


class PhaseMachine:
    """Stateless reducer. Call PhaseMachine.reduce(state, event) -> new state."""

    @staticmethod
    def reduce(state: AppState, event: Event) -> AppState:
        ts = getattr(event, "ts", 0.0) or state.last_event_ts
        st = replace(state)
        # HeartBeat is a pure "silence detector" tick — it must NOT bump last_event_ts,
        # otherwise the freeze check compares the tick against itself and never fires.
        if ts and not isinstance(event, HeartBeat):
            st.last_event_ts = ts

        if isinstance(event, HarnessMarker):
            return _on_marker(st, event, ts)
        if isinstance(event, IterationStart):
            st.iteration = event.i
            if event.total:
                st.total_iterations = event.total
            st.current_exp_id = event.exp_id
            _ensure_exp(st, event.exp_id)
            return _set_phase(st, Phase.IDLE, ts)
        if isinstance(event, IterationEnd):
            return _set_phase(st, Phase.IDLE, ts)
        if isinstance(event, TrainingFinished):
            if st.current_exp_id:
                exp = _ensure_exp(st, st.current_exp_id)
                if event.val_loss is not None and event.exit_code == 0:
                    exp.val_loss = event.val_loss
            return _set_phase(st, Phase.SUBMITTING, ts)
        if isinstance(event, ResultRow):
            if event.val_loss is not None:
                # Key by current_exp_id (results.tsv has no exp id column, so this is our best bet).
                key = st.current_exp_id or f"row{len(st.val_losses)+1:04d}"
                st.val_losses.append((key, event.val_loss))
                st.best_val_loss = min(st.best_val_loss, event.val_loss)
            return st
        if isinstance(event, JournalEntry):
            return _on_journal(st, event)
        if isinstance(event, QueueState):
            st.queue = event
            if st.current_exp_id:
                exp = _ensure_exp(st, st.current_exp_id)
                if event.current_hypothesis and not exp.hypothesis:
                    exp.hypothesis = event.current_hypothesis
            return st
        if isinstance(event, MemoryTouched):
            # Research touch implies RESEARCH phase even absent a harness marker.
            if event.which == "research_notes" and st.phase == Phase.IDLE:
                return _set_phase(st, Phase.RESEARCH, ts)
            return st
        if isinstance(event, GpuSample):
            st.gpu = event
            return st
        if isinstance(event, HeartBeat):
            return _maybe_freeze(st, event.ts)
        if isinstance(event, CompetitionInfo):
            st.competition = event
            return st
        if isinstance(event, RawLine):
            return st
        return st


def _on_marker(st: AppState, ev: HarnessMarker, ts: float) -> AppState:
    k = ev.kind
    if k == "researcher":
        return _set_phase(st, Phase.RESEARCH, ts)
    if k in ("strategist", "experimenter_edit"):
        return _set_phase(st, Phase.CODING, ts)
    if k == "training_start":
        st.training_started_ts = ts
        return _set_phase(st, Phase.EXPERIMENT, ts)
    if k == "experimenter_eval":
        return _set_phase(st, Phase.SUBMITTING, ts)
    return st


def _on_journal(st: AppState, ev: JournalEntry) -> AppState:
    exp = _ensure_exp(st, ev.id)
    exp.slug = ev.slug or exp.slug
    exp.status = _normalize_status(ev.status)
    if ev.val_loss is not None:
        exp.val_loss = ev.val_loss
        # Feed the chart — avoid duplicates from re-parsed journals
        if not any(eid == ev.id for eid, _ in st.val_losses):
            st.val_losses.append((ev.id, ev.val_loss))
    if ev.best_so_far is not None:
        exp.best_so_far = ev.best_so_far
    if ev.git_tag:
        exp.git_tag = ev.git_tag
    if ev.hypothesis and not exp.hypothesis:
        exp.hypothesis = ev.hypothesis

    if exp.status == "kept" and exp.val_loss is not None:
        # Promote to best if lower than anything we've seen.
        prev_best = st.best_val_loss
        if exp.val_loss < prev_best:
            st.best_val_loss = exp.val_loss
            st.new_best_toast_for = exp.exp_id
    return st


def _set_phase(st: AppState, phase: Phase, ts: float) -> AppState:
    if phase != st.phase:
        st.phase = phase
        st.phase_since_ts = ts or st.last_event_ts
        st.frozen_reason = ""
    return st


def _maybe_freeze(st: AppState, ts: float) -> AppState:
    # Only "working" phases are considered for freeze detection.
    if st.phase in (Phase.EXPERIMENT, Phase.CODING, Phase.RESEARCH):
        if st.last_event_ts and (ts - st.last_event_ts) > FREEZE_SECS:
            st.phase = Phase.FROZEN
            st.frozen_reason = f"no log activity for {int(ts - st.last_event_ts)}s"
    return st
