"""Sprite loader: Pillow-generated PNGs cached to ~/.automedal/sprites/, with
text-fallback when Pillow/rich-pixels unavailable.

API:
    load_sprite(phase, size=24, frame=0, theme=DEFAULT_THEME) -> Pixels | Text

Shape per phase (simple geometric glyphs, color from theme.phase_colors):
    RESEARCH    open book     (two rectangles joined at spine)
    CODING      cog           (disk with square teeth)
    EXPERIMENT  flask         (neck + flared base)
    SUBMITTING  upward arrow  (triangle + stem)
    IDLE        moon          (disk with notch)
    FROZEN      snowflake     (six radial lines)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Union

from tui.events import Phase
from tui.themes.palette import Theme, DEFAULT_THEME
from tui.assets.sprites.text_fallback import TEXT_SPRITES, MEDALS

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from rich_pixels import Pixels
    _HAS_PIXELS = True
except ImportError:
    _HAS_PIXELS = False

from rich.text import Text


SIZES = (16, 24, 32)
FRAMES = 2

# Package-bundled default sprites (committed to repo / installed with the package)
_ASSETS_DIR = Path(__file__).parent / "assets" / "sprites"


def cache_dir() -> Path:
    return Path(os.path.expanduser("~/.automedal/sprites"))


def _user_sprite_path(phase: Phase, size: int, frame: int, theme_name: str) -> Path:
    """Path under ~/.automedal/sprites/ (user override — highest priority)."""
    return cache_dir() / theme_name / phase.value / f"{size}" / f"frame_{frame:02d}.png"


def _bundled_sprite_path(phase: Phase, size: int, frame: int, theme_name: str) -> Path:
    """Path inside tui/assets/sprites/ (ships with the package)."""
    return _ASSETS_DIR / theme_name / phase.value / f"{size}" / f"frame_{frame:02d}.png"


def _sprite_path(phase: Phase, size: int, frame: int, theme_name: str) -> Path:
    """Resolve sprite path: user override → bundled asset → generate."""
    user = _user_sprite_path(phase, size, frame, theme_name)
    if user.exists():
        return user
    bundled = _bundled_sprite_path(phase, size, frame, theme_name)
    if bundled.exists():
        return bundled
    # Fall through to generate into cache_dir
    return user


def _user_medal_path(rank: str, size: int, theme_name: str) -> Path:
    return cache_dir() / theme_name / "medal" / f"{rank}_{size}.png"


def _bundled_medal_path(rank: str, size: int, theme_name: str) -> Path:
    return _ASSETS_DIR / theme_name / "medal" / f"{rank}_{size}.png"


def _medal_path(rank: str, theme_name: str) -> Path:
    return cache_dir() / theme_name / "medal" / f"{rank}.png"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_phase(draw: "ImageDraw.ImageDraw", phase: Phase, size: int, color: tuple[int, int, int], frame: int) -> None:
    pad = max(1, size // 8)
    cx, cy = size // 2, size // 2
    r = size // 2 - pad

    if phase is Phase.RESEARCH:
        draw.rectangle((pad, pad + size // 6, cx - 1, size - pad), outline=color, width=2)
        draw.rectangle((cx + 1, pad + size // 6, size - pad, size - pad), outline=color, width=2)
        draw.line((cx, pad + size // 6, cx, size - pad), fill=color)
    elif phase is Phase.CODING:
        draw.ellipse((pad, pad, size - pad, size - pad), outline=color, width=2)
        for angle in range(0, 360, 60):
            import math
            a = math.radians(angle + frame * 15)
            x1 = cx + int((r - 2) * math.cos(a))
            y1 = cy + int((r - 2) * math.sin(a))
            x2 = cx + int((r + 2) * math.cos(a))
            y2 = cy + int((r + 2) * math.sin(a))
            draw.line((x1, y1, x2, y2), fill=color, width=2)
        draw.ellipse((cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3), outline=color, width=1)
    elif phase is Phase.EXPERIMENT:
        neck_w = size // 5
        draw.rectangle((cx - neck_w, pad, cx + neck_w, cy), outline=color, width=2)
        draw.polygon(
            [(cx - neck_w, cy), (pad, size - pad), (size - pad, size - pad), (cx + neck_w, cy)],
            outline=color,
        )
        bubble_y = cy + (frame * size // 6)
        draw.ellipse((cx - 2, bubble_y, cx + 2, bubble_y + 4), outline=color)
    elif phase is Phase.SUBMITTING:
        draw.polygon([(cx, pad), (pad, cy), (size - pad, cy)], outline=color)
        draw.rectangle((cx - size // 8, cy, cx + size // 8, size - pad), outline=color, width=2)
    elif phase is Phase.IDLE:
        draw.ellipse((pad, pad, size - pad, size - pad), outline=color, width=2)
        bg = (15, 17, 26)
        off = size // 4
        draw.ellipse((pad + off, pad, size - pad + off, size - pad), fill=bg, outline=bg)
    elif phase is Phase.FROZEN:
        import math
        for angle in range(0, 360, 60):
            a = math.radians(angle + frame * 10)
            x = cx + int(r * math.cos(a))
            y = cy + int(r * math.sin(a))
            draw.line((cx, cy, x, y), fill=color, width=2)


def _draw_medal(draw: "ImageDraw.ImageDraw", rank: str, size: int, color: tuple[int, int, int]) -> None:
    pad = max(1, size // 8)
    draw.ellipse((pad, pad, size - pad, size - pad), outline=color, width=2)
    glyph = {"gold": "1", "silver": "2", "bronze": "3"}[rank]
    draw.text((size // 2 - 3, size // 2 - 4), glyph, fill=color)


def _generate_phase_sprite(path: Path, phase: Phase, size: int, frame: int, color: tuple[int, int, int]) -> None:
    if not _HAS_PIL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_phase(draw, phase, size, color, frame)
    img.save(path)


def _generate_medal_sprite(path: Path, rank: str, size: int, color: tuple[int, int, int]) -> None:
    if not _HAS_PIL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_medal(draw, rank, size, color)
    img.save(path)


def generate_default_sprites(theme: Theme = DEFAULT_THEME, *, force: bool = False) -> bool:
    """Generate all phase × size × frame PNGs + medal PNGs for the given theme.
    Returns True if Pillow was used; False if unavailable (text fallback).
    Idempotent — skips files that already exist unless `force=True`.
    """
    if not _HAS_PIL:
        return False
    for phase in Phase:
        color_hex = theme.phase_colors.get(phase, theme.fg)
        color = _hex_to_rgb(color_hex)
        for size in SIZES:
            for frame in range(FRAMES):
                p = _sprite_path(phase, size, frame, theme.name)
                if force or not p.exists():
                    _generate_phase_sprite(p, phase, size, frame, color)
    for rank, hex_col in (("gold", theme.medal_gold), ("silver", theme.medal_silver), ("bronze", theme.medal_bronze)):
        for size in SIZES:
            p = cache_dir() / theme.name / "medal" / f"{rank}_{size}.png"
            if force or not p.exists():
                _generate_medal_sprite(p, rank, size, _hex_to_rgb(hex_col))
    return True


@lru_cache(maxsize=256)
def _load_pixels(path_str: str) -> Union["Pixels", None]:
    if not _HAS_PIXELS:
        return None
    try:
        return Pixels.from_image_path(path_str)
    except (FileNotFoundError, OSError, ValueError):
        return None


def load_sprite(
    phase: Phase,
    size: int = 24,
    frame: int = 0,
    theme: Theme = DEFAULT_THEME,
) -> Union["Pixels", Text]:
    """Return a Pixels sprite for the phase, or a Text fallback."""
    size = min(SIZES, key=lambda s: abs(s - size))
    frame = frame % FRAMES

    if _HAS_PIL and _HAS_PIXELS:
        p = _sprite_path(phase, size, frame, theme.name)
        if not p.exists():
            generate_default_sprites(theme)
        pixels = _load_pixels(str(p))
        if pixels is not None:
            return pixels

    color = theme.phase_colors.get(phase, theme.fg)
    return Text(TEXT_SPRITES.get(phase, TEXT_SPRITES[Phase.IDLE]), style=color)


def load_medal(rank: str, size: int = 16, theme: Theme = DEFAULT_THEME) -> Union["Pixels", Text]:
    size = min(SIZES, key=lambda s: abs(s - size))
    if _HAS_PIL and _HAS_PIXELS:
        user = _user_medal_path(rank, size, theme.name)
        bundled = _bundled_medal_path(rank, size, theme.name)
        if user.exists():
            p = user
        elif bundled.exists():
            p = bundled
        else:
            generate_default_sprites(theme)
            p = user
        pixels = _load_pixels(str(p))
        if pixels is not None:
            return pixels
    color = {"gold": theme.medal_gold, "silver": theme.medal_silver, "bronze": theme.medal_bronze}.get(rank, theme.fg)
    return Text(MEDALS.get(rank, ""), style=color)
