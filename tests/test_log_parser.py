"""Tests for tui/sources/log_tail.py parsing helpers."""

from __future__ import annotations

from tui.sources.log_tail import (
    ITER_END_RE,
    ITER_START_RE,
    TRAINING_DONE_RE,
    _classify_marker,
    _parse_val_loss,
    _strip_ansi,
)


def test_strip_ansi():
    s = "\x1b[31mred\x1b[0m plain"
    assert _strip_ansi(s) == "red plain"


def test_iter_start_regex():
    line = "========== Iteration 7 / 50  exp=0007  [Wed Apr 10 22:00:00 UTC 2026] =========="
    m = ITER_START_RE.match(line)
    assert m is not None
    assert m.group(1) == "7"
    assert m.group(2) == "50"
    assert m.group(3) == "0007"


def test_iter_end_regex():
    line = "--- Iteration 7 complete  exp=0007  [timestamp] ---"
    m = ITER_END_RE.match(line)
    assert m is not None
    assert m.group(1) == "7"
    assert m.group(2) == "0007"


def test_training_done_regex():
    line = "[harness] training done: val_loss=0.0505 exit=0"
    m = TRAINING_DONE_RE.search(line)
    assert m is not None
    assert m.group(1) == "0.0505"
    assert m.group(2) == "0"


def test_training_done_failure():
    line = "[harness] training done: val_loss=nan exit=1"
    m = TRAINING_DONE_RE.search(line)
    assert m is not None
    assert m.group(2) == "1"


def test_classify_researcher():
    m = _classify_marker("[harness] dispatching Researcher (stagnation)")
    assert m is not None and m.kind == "researcher"


def test_classify_strategist():
    m = _classify_marker("[harness] dispatching Strategist")
    assert m is not None and m.kind == "strategist"


def test_classify_experimenter_edit():
    m = _classify_marker("[harness] dispatching Experimenter (edit)")
    assert m is not None and m.kind == "experimenter_edit"


def test_classify_experimenter_eval():
    m = _classify_marker("[harness] dispatching Experimenter (eval)")
    assert m is not None and m.kind == "experimenter_eval"


def test_classify_training_start():
    m = _classify_marker("[harness] running training (budget=10m)")
    assert m is not None and m.kind == "training_start"


def test_classify_non_marker():
    assert _classify_marker("some random output") is None


def test_parse_val_loss_ok():
    assert _parse_val_loss("0.1234") == 0.1234


def test_parse_val_loss_nan():
    assert _parse_val_loss("nan") is None


def test_parse_val_loss_junk():
    assert _parse_val_loss("oops") is None


def test_classify_analyzer():
    m = _classify_marker("[harness] dispatching Analyzer (status=better)")
    assert m is not None and m.kind == "analyzer"
