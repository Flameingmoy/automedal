"""Motivation-similarity dedupe tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from automedal import dedupe


_QUEUE = """\
# Experiment Queue
_Planned: exp 0042 | Runs 0042-0046_

## 1. lgb-dart-boost [axis: HPO] [STATUS: pending]
**Hypothesis:** switching LGBM to dart boosting with n_estimators 800 then 1200 will reduce overfitting on the validation fold and beat the current best by 0.001
**Sketch:** in agent/train.py change boosting_type to dart, bump n_estimators
**Expected:** -0.001 val_loss
success_criteria: val_loss <= 0.0500

## 2. xgb-monotonic [axis: feature-eng] [STATUS: pending]
**Hypothesis:** adding monotonic constraints to the XGB feature set will help generalization on tail customers
**Sketch:** in agent/train.py wire monotone_constraints=
**Expected:** -0.0005 val_loss
success_criteria: val_loss <= 0.0505

## 3. forced-dup [axis: HPO] [STATUS: pending] [force]
**Hypothesis:** switching LGBM to dart boosting again — this is forced and should NOT be deduped
**Sketch:** trivial
**Expected:** information value
success_criteria: val_loss <= 0.0510
"""


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    queue_md = tmp_path / "experiment_queue.md"
    queue_md.write_text(_QUEUE)
    journal = tmp_path / "journal"
    journal.mkdir()
    # BM25 needs corpus diversity — provide several historical diffs.
    diffs = {
        "0035-xgb-tune.md":      "tuned XGB max_depth and learning_rate via Optuna 50 trials",
        "0036-cat-encoding.md":  "target encoding smoothing 0.3 swap on categorical features",
        "0037-stacking.md":      "added a stacking meta-learner on top of XGB and LGBM bases",
        "0038-pseudo-label.md":  "added pseudo-labelling with confidence threshold 0.92 on test set",
        "0039-feature-eng.md":   "added 12 ratio features and 4 frequency-encoding features for prepare.py",
        "0040-lgb-dart-tune.md": "switched LGBM to dart boosting, n_estimators 800 to 1200, reduced overfitting attempt",
        "0041-monotonic-xgb.md": "removed monotonic constraints; XGB monotone_constraints had no effect",
    }
    for name, summary in diffs.items():
        (journal / name).write_text(
            f"---\nid: {name[:4]}\nslug: {name[5:-3]}\nstatus: worse\n"
            f"diff_summary: {summary}\n---\n"
        )
    return queue_md, journal


def test_dedupe_marks_duplicate_against_recent_journal(tmp_path):
    queue_md, journal = _setup(tmp_path)
    summary = dedupe.apply(queue_path=queue_md, journal_path=journal, threshold=2.0)
    text = queue_md.read_text()
    assert summary["scanned"] >= 1
    assert summary["marked"] >= 1
    # entry 1 should be marked
    head_lines = [ln for ln in text.splitlines() if ln.startswith("## 1.")]
    assert any("skipped-duplicate" in ln for ln in head_lines)


def test_dedupe_respects_force_tag(tmp_path):
    queue_md, journal = _setup(tmp_path)
    dedupe.apply(queue_path=queue_md, journal_path=journal, threshold=0.0)
    text = queue_md.read_text()
    head3 = [ln for ln in text.splitlines() if ln.startswith("## 3.")]
    assert head3, "entry 3 missing"
    assert "[STATUS: pending]" in head3[0]
    assert "skipped-duplicate" not in head3[0]


def test_dedupe_no_journal_means_no_marks(tmp_path):
    queue_md = tmp_path / "experiment_queue.md"
    queue_md.write_text(_QUEUE)
    journal = tmp_path / "journal"
    journal.mkdir()
    summary = dedupe.apply(queue_path=queue_md, journal_path=journal, threshold=0.5)
    assert summary["marked"] == 0
    assert "skipped-duplicate" not in queue_md.read_text()


def test_dedupe_threshold_too_high_marks_nothing(tmp_path):
    queue_md, journal = _setup(tmp_path)
    summary = dedupe.apply(queue_path=queue_md, journal_path=journal, threshold=1e9)
    assert summary["marked"] == 0
