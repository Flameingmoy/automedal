"""Generates the AUTOMEDAL pixel-art splash PNG (idempotent).

Run directly to regenerate:  python -m tui.assets.logo.generate_logo
Called from automedal.cli._print_splash via ensure_logo().
"""

from __future__ import annotations

from pathlib import Path

_LOGO_DIR = Path(__file__).parent
_LOGO_PATH = _LOGO_DIR / "automedal.png"

# Dracula palette
_BG = (15, 17, 26, 255)       # #0F111A screen bg
_FG = (255, 215, 0, 255)      # #FFD700 gold (submitting-phase color)
_ACCENT = (139, 233, 253, 255)  # #8BE9FD cyan — tagline

# 5x7 bitmap font for AUTOMEDAL letters
_FONT = {
    "A": [
        ".XXX.",
        "X...X",
        "X...X",
        "XXXXX",
        "X...X",
        "X...X",
        "X...X",
    ],
    "U": [
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    "T": [
        "XXXXX",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
        "..X..",
    ],
    "O": [
        ".XXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        ".XXX.",
    ],
    "M": [
        "X...X",
        "XX.XX",
        "X.X.X",
        "X.X.X",
        "X...X",
        "X...X",
        "X...X",
    ],
    "E": [
        "XXXXX",
        "X....",
        "X....",
        "XXXX.",
        "X....",
        "X....",
        "XXXXX",
    ],
    "D": [
        "XXXX.",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "X...X",
        "XXXX.",
    ],
    "L": [
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "X....",
        "XXXXX",
    ],
    " ": [
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ],
}

_TEXT = "AUTOMEDAL"
_SCALE = 2
_PAD_X = 2
_PAD_Y = 1


def _render(text: str, color: tuple[int, int, int, int]):
    from PIL import Image

    letter_w, letter_h = 5, 7
    gap = 1
    text_w = len(text) * letter_w + max(0, len(text) - 1) * gap
    w = text_w + _PAD_X * 2
    h = letter_h + _PAD_Y * 2
    img = Image.new("RGBA", (w, h), _BG)
    px = img.load()
    x_off = _PAD_X
    for ch in text:
        glyph = _FONT.get(ch.upper())
        if glyph is None:
            x_off += letter_w + gap
            continue
        for y, row in enumerate(glyph):
            for x, c in enumerate(row):
                if c == "X":
                    px[x_off + x, _PAD_Y + y] = color
        x_off += letter_w + gap

    if _SCALE > 1:
        img = img.resize((img.width * _SCALE, img.height * _SCALE), Image.NEAREST)
    return img


def ensure_logo(force: bool = False) -> Path:
    """Return the path to automedal.png; generate it if missing or force=True."""
    if _LOGO_PATH.exists() and not force:
        return _LOGO_PATH
    try:
        img = _render(_TEXT, _FG)
        _LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(_LOGO_PATH, "PNG")
    except Exception:
        pass
    return _LOGO_PATH


if __name__ == "__main__":
    p = ensure_logo(force=True)
    print(f"wrote {p}")
