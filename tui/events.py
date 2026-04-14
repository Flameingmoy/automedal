"""Event types emitted by tui/sources/* and consumed by tui/state.PhaseMachine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(str, Enum):
    RESEARCH = "research"
    CODING = "coding"
    EXPERIMENT = "experiment"
    SUBMITTING = "submitting"
    IDLE = "idle"
    FROZEN = "frozen"


@dataclass
class HarnessMarker:
    """A `[harness] dispatching ...` / `[harness] running training ...` line from run.sh."""
    kind: str  # "researcher" | "strategist" | "experimenter_edit" | "experimenter_eval" | "training_start"
    raw: str = ""
    ts: float = 0.0


@dataclass
class IterationStart:
    exp_id: str
    i: int
    total: int
    ts: float = 0.0


@dataclass
class IterationEnd:
    exp_id: str
    i: int
    ts: float = 0.0


@dataclass
class TrainingFinished:
    val_loss: Optional[float]
    exit_code: int
    ts: float = 0.0


@dataclass
class ResultRow:
    timestamp: str
    method: str
    trials: Optional[int]
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    submission: str
    notes: str


@dataclass
class JournalEntry:
    id: str                 # "0024"
    slug: str               # "irredundant-kfold-hpo"
    timestamp: str
    git_tag: str
    status: str             # "better" | "worse" | "crash" | ... (free-form; normalized in state)
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    best_so_far: Optional[float]
    hypothesis: str = ""
    path: str = ""


@dataclass
class QueueState:
    """Snapshot of experiment_queue.md — current pending hypothesis."""
    current_slug: str = ""
    current_hypothesis: str = ""
    pending_count: int = 0


@dataclass
class MemoryTouched:
    """knowledge.md / research_notes.md mtime tick."""
    which: str  # "knowledge" | "research_notes"


@dataclass
class GpuSample:
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    temp_c: float
    ts: float = 0.0


@dataclass
class RawLine:
    """A line from agent_loop.log for the live-stream widget (phase-colored downstream)."""
    text: str
    ts: float = 0.0


@dataclass
class HeartBeat:
    """Emitted ~1Hz by the app so the PhaseMachine can detect FROZEN via silence."""
    ts: float


@dataclass
class CompetitionInfo:
    slug: str = ""
    title: str = ""


Event = (
    HarnessMarker
    | IterationStart
    | IterationEnd
    | TrainingFinished
    | ResultRow
    | JournalEntry
    | QueueState
    | MemoryTouched
    | GpuSample
    | RawLine
    | HeartBeat
    | CompetitionInfo
)
