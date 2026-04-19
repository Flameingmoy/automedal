"""Phase prompt template rendering tests."""

from __future__ import annotations

import pytest

from automedal.agent.prompts import PHASES, render_prompt


def test_all_phase_templates_exist():
    for phase in PHASES:
        # smoke-render with throwaway slots; we just want TemplateNotFound to fire if missing
        try:
            render_prompt(phase, **_dummy_slots(phase))
        except Exception as exc:  # pragma: no cover — debugging aid
            raise AssertionError(f"phase {phase!r} failed to render: {exc}")


def test_researcher_includes_runtime_context():
    out = render_prompt(
        "researcher",
        exp_id="0042",
        trigger="stagnation",
        stagnating=True,
        scheduled_research=False,
        best_val_loss=0.0507,
    )
    assert "Triggering experiment: 0042" in out
    assert "Trigger type: stagnation" in out
    assert "## Runtime context" in out


def test_strategist_renders_reflective_trace():
    out = render_prompt(
        "strategist",
        exp_id="0042",
        iteration=7,
        max_iters=20,
        stagnating=False,
        best_val_loss=0.0507,
        pending=2,
        reflective="(trace body)",
        ranked="(ranked body)",
    )
    assert "Upcoming experiment: 0042" in out
    assert "Current iteration: 7 / 20" in out
    assert "(trace body)" in out
    assert "(ranked body)" in out


def test_experimenter_retry_block_is_conditional():
    base = render_prompt("experimenter", exp_id="0042", best_val_loss=0.0507, retry=False, prev_loss="")
    assert "RETRY:" not in base

    retry = render_prompt(
        "experimenter",
        exp_id="0042",
        best_val_loss=0.0507,
        retry=True,
        prev_loss=0.0511,
    )
    assert "RETRY: Previous attempt val_loss=0.0511" in retry


def test_experimenter_eval_includes_train_outcome():
    out = render_prompt(
        "experimenter_eval",
        exp_id="0042",
        best_val_loss=0.0507,
        train_rc=0,
        final_loss=0.0501,
    )
    assert "Training exit code: 0" in out
    assert "Training val_loss: 0.0501" in out


def test_analyzer_includes_iteration_outcome():
    out = render_prompt(
        "analyzer",
        exp_id="0042",
        slug="catboost-depth-tune",
        status="improved",
        final_loss=0.0501,
        best_val_loss=0.0507,
        val_loss_delta="-0.0006",
    )
    assert "Experiment ID: 0042" in out
    assert "catboost-depth-tune" in out
    assert "Status: improved" in out


def test_unknown_phase_raises():
    with pytest.raises(ValueError):
        render_prompt("nonexistent")


def test_missing_slot_raises():
    # StrictUndefined → omitting a required slot must blow up
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        render_prompt("researcher", exp_id="0042")  # missing trigger/stagnating/etc.


def _dummy_slots(phase: str) -> dict:
    base = {
        "exp_id": "0001",
        "best_val_loss": 0.05,
    }
    if phase == "researcher":
        base.update(trigger="scheduled", stagnating=False, scheduled_research=True)
    elif phase == "strategist":
        base.update(iteration=1, max_iters=10, stagnating=False, pending=0,
                    reflective="(empty)", ranked="(empty)")
    elif phase == "experimenter":
        base.update(retry=False, prev_loss="")
    elif phase == "experimenter_eval":
        base.update(train_rc=0, final_loss=0.05)
    elif phase == "analyzer":
        base.update(slug="x", status="no_change", final_loss=0.05, val_loss_delta="0.0")
    return base
