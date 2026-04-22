"""Command input box with Tab autocomplete and live hints — sits at the bottom of HomeScreen."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Static

COMMANDS = [
    "run", "init", "discover", "select", "doctor", "status",
    "clean", "prepare", "render", "setup", "models", "help", "quit",
]


def _models_for_autocomplete() -> list[str]:
    """Cached advisor models for the `--advisor <TAB>` completer.

    Returns [] on any failure — autocomplete just shows nothing in that case.
    Cheap on the hot path: hits ~/.automedal/models_cache.json, no network.
    """
    try:
        from automedal.advisor import list_models
        return list_models()
    except Exception:
        return []


def _split_for_advisor(text: str) -> tuple[bool, str, str]:
    """If `text` ends mid-`--advisor <prefix>`, return (True, prefix, head).

    `head` is the text up to and including `--advisor `, so the caller can
    splice the completion: head + completion.
    Otherwise returns (False, "", text).
    """
    parts = text.split(" ")
    if "--advisor" not in parts:
        return (False, "", text)
    # Find the last --advisor token; complete the token immediately after it.
    idx = len(parts) - 1 - list(reversed(parts)).index("--advisor")
    if idx == len(parts) - 1:
        # Cursor sits right after `--advisor ` with no prefix typed yet
        head = " ".join(parts[: idx + 1]) + " "
        return (True, "", head)
    prefix = parts[idx + 1]
    head = " ".join(parts[: idx + 1]) + " "
    # Only treat as model-completion if the prefix has no spaces past it (it's
    # still the trailing token).
    if idx + 1 == len(parts) - 1:
        return (True, prefix, head)
    return (False, "", text)

_TRAILING_DIGITS = re.compile(r"^([a-zA-Z]+)(\d+)$")


def normalize(text: str) -> str:
    """Forgiving parser: 'run30' → 'run 30'; leaves 'run 30' untouched."""
    text = text.strip()
    if not text:
        return text
    first, _, rest = text.partition(" ")
    m = _TRAILING_DIGITS.match(first)
    if m and m.group(1).lower() in COMMANDS:
        fixed = f"{m.group(1)} {m.group(2)}"
        return f"{fixed} {rest}".strip() if rest else fixed
    return text


class CommandInput(Vertical):
    """An `>` prompt with inline suggester, live hints row, and a Submitted message."""

    DEFAULT_CSS = """
    CommandInput {
        height: 4;
        border: round #6272A4;
        padding: 0 1;
        background: #0F111A;
    }
    CommandInput > #ci-row {
        height: 1;
    }
    CommandInput > #ci-row > #ci-prompt { width: 2; color: #50FA7B; }
    CommandInput > #ci-row > #ci-input  { width: 1fr; border: none; background: transparent; }
    CommandInput > #ci-hints { height: 1; color: #6272A4; padding: 0 0 0 2; }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        with Horizontal(id="ci-row"):
            yield Static(">", id="ci-prompt")
            yield Input(
                placeholder="type a command (e.g. run 30)",
                id="ci-input",
            )
        yield Static(f"  {'  '.join(COMMANDS[:6])}", id="ci-hints")

    def on_mount(self) -> None:
        self.query_one("#ci-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        text = normalize(event.value or "")
        is_adv, prefix, _head = _split_for_advisor(text)
        if is_adv:
            models = _models_for_autocomplete()
            matches = [m for m in models if m.startswith(prefix)] if prefix else models
            if not models:
                hint = "(no models cached — run 'automedal models refresh')"
            elif not matches:
                hint = f"(no model starts with {prefix!r})"
            else:
                hint = "  ".join(matches[:6])
                if len(matches) > 6:
                    hint += f"  +{len(matches) - 6}"
        else:
            word = (text.split() or [""])[0].lower()
            if not word:
                hint = "  ".join(COMMANDS[:6])
            else:
                matches = [c for c in COMMANDS if c.startswith(word)]
                hint = "  ".join(matches[:6]) if matches else "(no matches)"
        try:
            self.query_one("#ci-hints", Static).update(f"  {hint}")
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = normalize(event.value)
        if value:
            self.post_message(self.Submitted(value))
            inp = self.query_one("#ci-input", Input)
            inp.value = ""
            try:
                self.query_one("#ci-hints", Static).update(f"  {'  '.join(COMMANDS[:6])}")
            except Exception:
                pass

    def on_key(self, event) -> None:
        if event.key == "tab":
            event.stop()
            self._autocomplete()

    def _autocomplete(self) -> None:
        inp = self.query_one("#ci-input", Input)
        text = normalize(inp.value)
        if not text:
            return

        # Model autocomplete after `--advisor `
        is_adv, prefix, head = _split_for_advisor(text)
        if is_adv:
            models = _models_for_autocomplete()
            matches = [m for m in models if m.startswith(prefix)]
            if len(matches) == 1:
                inp.value = head + matches[0]
                inp.cursor_position = len(inp.value)
            elif len(matches) > 1:
                # Complete to the longest common prefix so a second Tab narrows.
                lcp = matches[0]
                for m in matches[1:]:
                    while not m.startswith(lcp):
                        lcp = lcp[:-1]
                        if not lcp:
                            break
                    if not lcp:
                        break
                if lcp and lcp != prefix:
                    inp.value = head + lcp
                    inp.cursor_position = len(inp.value)
            return

        # Command autocomplete on the first word. On a single match, append
        # a trailing space so the user can start typing args immediately —
        # without this, the cursor sits right after the completed word and
        # typing space feels like it "did nothing" (the hint row re-renders
        # the same command).
        first, _, rest = text.partition(" ")
        word = first.lower()
        matches = [c for c in COMMANDS if c.startswith(word)]
        if len(matches) == 1:
            completed = matches[0]
            if rest:
                inp.value = f"{completed} {rest}"
            else:
                inp.value = f"{completed} "
            inp.cursor_position = len(inp.value)
