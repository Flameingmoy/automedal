"""Theme loader. Currently ships `dark`; new themes drop in as `<name>.tcss` + a palette entry."""

from __future__ import annotations

from pathlib import Path

from tui.themes.palette import DEFAULT_THEME, THEMES, Theme


def available_themes() -> list[str]:
    return list(THEMES.keys())


def get_theme(name: str) -> Theme:
    return THEMES.get(name, DEFAULT_THEME)


def tcss_path(name: str) -> Path:
    here = Path(__file__).parent
    p = here / f"{name}.tcss"
    if not p.exists():
        p = here / "dark.tcss"
    return p
