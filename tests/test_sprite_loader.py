"""Tests for tui/sprite_loader.py."""

from __future__ import annotations

from tui.events import Phase
from tui.sprite_loader import load_sprite, load_medal
from tui.assets.sprites.text_fallback import TEXT_SPRITES, get_text_sprite


def test_all_phases_have_text_fallback():
    for p in Phase:
        assert p in TEXT_SPRITES
        assert TEXT_SPRITES[p].strip() != ""


def test_get_text_sprite_idle_default():
    assert get_text_sprite(Phase.IDLE) == TEXT_SPRITES[Phase.IDLE]


def test_load_sprite_returns_something_per_phase():
    for p in Phase:
        s = load_sprite(p, size=24, frame=0)
        assert s is not None


def test_load_sprite_snaps_size():
    # size=20 should map to nearest (16). Should not error.
    for p in Phase:
        assert load_sprite(p, size=20, frame=0) is not None


def test_load_sprite_frame_wraps():
    assert load_sprite(Phase.RESEARCH, size=24, frame=7) is not None


def test_load_medal_all_ranks():
    for rank in ("gold", "silver", "bronze"):
        assert load_medal(rank, size=16) is not None
