"""Unit tests for tui.state.PhaseMachine — verifies the pure (state, event) -> state reducer."""

from __future__ import annotations

from tui.events import (
    GpuSample,
    HarnessMarker,
    HeartBeat,
    IterationEnd,
    IterationStart,
    JournalEntry,
    Phase,
    QueueState,
    ResultRow,
    TrainingFinished,
)
from tui.state import AppState, PhaseMachine, FREEZE_SECS


def reduce_seq(events):
    st = AppState()
    for e in events:
        st = PhaseMachine.reduce(st, e)
    return st


def test_starts_idle():
    assert AppState().phase == Phase.IDLE


def test_researcher_marker_switches_phase():
    st = reduce_seq([HarnessMarker(kind="researcher", ts=1.0)])
    assert st.phase == Phase.RESEARCH


def test_strategist_marker_is_coding():
    st = reduce_seq([HarnessMarker(kind="strategist", ts=1.0)])
    assert st.phase == Phase.CODING


def test_experimenter_edit_is_coding():
    st = reduce_seq([HarnessMarker(kind="experimenter_edit", ts=1.0)])
    assert st.phase == Phase.CODING


def test_training_start_is_experiment_and_sets_ts():
    st = reduce_seq([HarnessMarker(kind="training_start", ts=42.0)])
    assert st.phase == Phase.EXPERIMENT
    assert st.training_started_ts == 42.0


def test_experimenter_eval_is_submitting():
    st = reduce_seq([HarnessMarker(kind="experimenter_eval", ts=1.0)])
    assert st.phase == Phase.SUBMITTING


def test_iteration_start_tracks_exp_and_iter():
    st = reduce_seq([IterationStart(exp_id="0007", i=7, total=50, ts=1.0)])
    assert st.current_exp_id == "0007"
    assert st.iteration == 7
    assert st.total_iterations == 50
    assert "0007" in st.experiments


def test_iteration_end_returns_to_idle():
    st = reduce_seq([
        IterationStart(exp_id="0007", i=7, total=50, ts=1.0),
        HarnessMarker(kind="training_start", ts=2.0),
        IterationEnd(exp_id="0007", i=7, ts=3.0),
    ])
    assert st.phase == Phase.IDLE


def test_training_finished_records_val_loss():
    st = reduce_seq([
        IterationStart(exp_id="0007", i=7, total=50, ts=1.0),
        HarnessMarker(kind="training_start", ts=2.0),
        TrainingFinished(val_loss=0.1234, exit_code=0, ts=3.0),
    ])
    assert st.phase == Phase.SUBMITTING
    assert st.experiments["0007"].val_loss == 0.1234


def test_training_failed_does_not_record_loss():
    st = reduce_seq([
        IterationStart(exp_id="0007", i=7, total=50, ts=1.0),
        TrainingFinished(val_loss=None, exit_code=1, ts=2.0),
    ])
    assert st.experiments["0007"].val_loss is None


def test_journal_better_promotes_best():
    st = reduce_seq([
        IterationStart(exp_id="0007", i=7, total=50, ts=1.0),
        JournalEntry(id="0007", slug="x", timestamp="", git_tag="exp/0007",
                     status="better", val_loss=0.05, val_accuracy=None, best_so_far=0.05),
    ])
    assert st.experiments["0007"].status == "kept"
    assert st.best_val_loss == 0.05
    assert st.new_best_toast_for == "0007"


def test_journal_worse_marks_reverted():
    st = reduce_seq([
        JournalEntry(id="0008", slug="y", timestamp="", git_tag="exp/0008",
                     status="worse", val_loss=0.2, val_accuracy=None, best_so_far=0.05),
    ])
    assert st.experiments["0008"].status == "reverted"
    assert st.best_val_loss == float("inf")  # not promoted


def test_journal_crash_normalizes():
    st = reduce_seq([
        JournalEntry(id="0009", slug="z", timestamp="", git_tag="",
                     status="crash", val_loss=None, val_accuracy=None, best_so_far=None),
    ])
    assert st.experiments["0009"].status == "crash"


def test_result_row_appends_to_val_losses():
    st = reduce_seq([
        IterationStart(exp_id="0001", i=1, total=10, ts=1.0),
        ResultRow(timestamp="", method="xgb", trials=100, val_loss=0.1,
                  val_accuracy=None, submission="", notes=""),
        ResultRow(timestamp="", method="xgb", trials=200, val_loss=0.08,
                  val_accuracy=None, submission="", notes=""),
    ])
    assert len(st.val_losses) == 2
    assert st.best_val_loss == 0.08


def test_heartbeat_triggers_frozen_after_timeout():
    st = AppState()
    st = PhaseMachine.reduce(st, HarnessMarker(kind="training_start", ts=100.0))
    assert st.phase == Phase.EXPERIMENT
    # Heartbeat long after last event — set last_event_ts via another event below would update it.
    # Directly advance heartbeat without interim activity:
    st = PhaseMachine.reduce(st, HeartBeat(ts=100.0 + FREEZE_SECS + 5))
    assert st.phase == Phase.FROZEN
    assert "no log activity" in st.frozen_reason


def test_heartbeat_not_frozen_during_idle():
    st = AppState()
    st.phase = Phase.IDLE
    st.last_event_ts = 0.0
    st2 = PhaseMachine.reduce(st, HeartBeat(ts=10000.0))
    assert st2.phase == Phase.IDLE


def test_queue_state_sets_hypothesis_for_current_exp():
    st = reduce_seq([
        IterationStart(exp_id="0010", i=10, total=50, ts=1.0),
        QueueState(current_slug="abc", current_hypothesis="try X", pending_count=3),
    ])
    assert st.experiments["0010"].hypothesis == "try X"
    assert st.queue.pending_count == 3


def test_gpu_sample_stored():
    st = reduce_seq([GpuSample(util_pct=80.0, mem_used_mb=4000, mem_total_mb=8000, temp_c=60.0, ts=1.0)])
    assert st.gpu is not None
    assert st.gpu.util_pct == 80.0


def test_full_pipeline_sequence():
    events = [
        IterationStart(exp_id="0001", i=1, total=50, ts=1.0),
        HarnessMarker(kind="researcher", ts=2.0),
        HarnessMarker(kind="strategist", ts=3.0),
        HarnessMarker(kind="experimenter_edit", ts=4.0),
        HarnessMarker(kind="training_start", ts=5.0),
        TrainingFinished(val_loss=0.3, exit_code=0, ts=60.0),
        HarnessMarker(kind="experimenter_eval", ts=61.0),
        JournalEntry(id="0001", slug="baseline", timestamp="", git_tag="exp/0001",
                     status="better", val_loss=0.3, val_accuracy=None, best_so_far=0.3),
        IterationEnd(exp_id="0001", i=1, ts=62.0),
    ]
    st = reduce_seq(events)
    assert st.phase == Phase.IDLE
    assert st.experiments["0001"].status == "kept"
    assert st.best_val_loss == 0.3


def test_top_n_excludes_crashes():
    st = reduce_seq([
        JournalEntry(id="01", slug="a", timestamp="", git_tag="", status="better",
                     val_loss=0.1, val_accuracy=None, best_so_far=0.1),
        JournalEntry(id="02", slug="b", timestamp="", git_tag="", status="crash",
                     val_loss=0.05, val_accuracy=None, best_so_far=0.1),
    ])
    top = st.top_n(5)
    ids = [e.exp_id for e in top]
    assert "01" in ids
    assert "02" not in ids
