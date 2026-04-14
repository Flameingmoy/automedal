"""ASCII/Unicode sprites. Used when rich-pixels or Pillow is missing, or terminal
can't render half-block truecolor. Each entry is 5 lines tall, ~8 cols wide."""

from __future__ import annotations

from tui.events import Phase


TEXT_SPRITES: dict[Phase, str] = {
    Phase.RESEARCH: (
        "  ____  \n"
        " /    \\ \n"
        "|  []  |\n"
        " \\____/ \n"
        " research"
    ),
    Phase.CODING: (
        "  ___   \n"
        " /_*_\\  \n"
        " |*|*| \n"
        " \\___/  \n"
        "  coding"
    ),
    Phase.EXPERIMENT: (
        "  _|_   \n"
        "  | |   \n"
        "  | |   \n"
        " /~~~\\  \n"
        " exp    "
    ),
    Phase.SUBMITTING: (
        "   /\\   \n"
        "  /  \\  \n"
        " / /\\ \\ \n"
        "/_/  \\_\\\n"
        " submit "
    ),
    Phase.IDLE: (
        "  ___   \n"
        " /   \\  \n"
        "(  z  ) \n"
        " \\___/  \n"
        "  idle  "
    ),
    Phase.FROZEN: (
        " \\ | /  \n"
        "  \\|/   \n"
        " --+--  \n"
        "  /|\\   \n"
        " FROZEN "
    ),
}


MEDALS: dict[str, str] = {
    "gold":   " _  \n(1) \n/ \\ ",
    "silver": " _  \n(2) \n/ \\ ",
    "bronze": " _  \n(3) \n/ \\ ",
}


def get_text_sprite(phase: Phase) -> str:
    return TEXT_SPRITES.get(phase, TEXT_SPRITES[Phase.IDLE])
