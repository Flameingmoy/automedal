"""Theme palettes. Keep simple — colors used by widgets + the sprite tinter."""

from __future__ import annotations

from dataclasses import dataclass, field

from tui.events import Phase


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    fg: str
    accent: str
    dim: str
    phase_colors: dict[Phase, str] = field(default_factory=dict)
    medal_gold: str = "#FFD700"
    medal_silver: str = "#C0C0C0"
    medal_bronze: str = "#CD7F32"


DARK = Theme(
    name="dark",
    bg="#0F111A",
    fg="#E1E4E8",
    accent="#8BE9FD",
    dim="#6272A4",
    phase_colors={
        Phase.RESEARCH: "#BD93F9",     # violet
        Phase.CODING: "#8BE9FD",       # cyan
        Phase.EXPERIMENT: "#FFB86C",   # amber
        Phase.SUBMITTING: "#FFD700",   # gold
        Phase.IDLE: "#6272A4",         # dim gray
        Phase.FROZEN: "#FF5555",       # red
    },
)


THEMES: dict[str, Theme] = {
    "dark": DARK,
}

DEFAULT_THEME: Theme = DARK
