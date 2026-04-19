"""Doctor screen — diagnostic checklist with per-row background workers."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static

from tui.state import AppState


_CHECK_PENDING = "⏳"
_CHECK_OK      = "✓"
_CHECK_FAIL    = "✗"


class DoctorScreen(Screen):
    DEFAULT_CSS = """
    DoctorScreen {
        layout: vertical;
        background: #0F111A;
    }
    DoctorScreen > #dr-title {
        height: 1;
        background: #0F111A;
        padding: 0 1;
        color: #8BE9FD;
    }
    DoctorScreen > #dr-checks {
        height: 1fr;
        padding: 1 2;
        layout: vertical;
    }
    DoctorScreen > #dr-footer-hint {
        height: 1;
        color: #6272A4;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("r",      "refresh", "Re-run"),
        ("q",      "back",    "Back"),
        ("escape", "back",    "Back"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []
        self._rows: list[Static] = []

    _CHECKS = [
        ("sdk",      "anthropic / openai SDKs"),
        ("env_file", "~/.automedal/.env"),
        ("provider", "active provider configured"),
        ("smoke",    "provider smoke test"),
        ("layout",   "Layout paths"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("AutoMedal Doctor", id="dr-title")
        with Vertical(id="dr-checks"):
            for key, label in self._CHECKS:
                w = Static(f"  {_CHECK_PENDING}  {label:<30}  checking…", id=f"dr-{key}")
                self._rows.append(w)
                yield w
        yield Static("  r=re-run  q=back", id="dr-footer-hint")
        yield Footer()

    def on_mount(self) -> None:
        asyncio.get_event_loop().create_task(self._run_checks())

    async def _run_checks(self) -> None:
        import os

        # SDKs
        try:
            from importlib.metadata import version as _v
            self._set("sdk", True, f"anthropic {_v('anthropic')} / openai {_v('openai')}")
        except Exception as exc:
            self._set("sdk", False, f"missing: {exc}")

        # env file
        from automedal.auth import ENV_FILE, configured_providers
        self._set("env_file", ENV_FILE.exists(),
                  str(ENV_FILE) + (" (present)" if ENV_FILE.exists() else " (missing)"))

        # active provider
        provider = os.environ.get("AUTOMEDAL_PROVIDER", "opencode-go")
        model = os.environ.get("AUTOMEDAL_MODEL")
        if not model:
            slug = os.environ.get("MODEL", "opencode-go/minimax-m2.7")
            if "/" in slug:
                provider, model = slug.split("/", 1)
            else:
                model = slug
        active = configured_providers()
        self._set("provider", provider in active or provider == "ollama",
                  f"{provider}/{model} (configured: {', '.join(active) or 'none'})")

        # smoke test (offload to thread pool)
        try:
            from automedal.agent.providers import smoke
            ok, detail = await asyncio.get_event_loop().run_in_executor(
                None, smoke, provider, model
            )
            self._set("smoke", ok, detail)
        except Exception as exc:
            self._set("smoke", False, str(exc))

        # layout paths
        if self._layout is not None:
            missing = [str(p) for p in [self._layout.data_dir] if not p.parent.exists()]
            self._set("layout", not missing, "OK" if not missing else f"missing: {missing[0]}")
        else:
            self._set("layout", None, "layout unavailable (standalone mode)")

    def _set(self, key: str, ok: bool | None, detail: str) -> None:
        glyph = _CHECK_OK if ok else (_CHECK_FAIL if ok is False else "—")
        label = dict(self._CHECKS).get(key, key)
        try:
            self.query_one(f"#dr-{key}", Static).update(
                f"  {glyph}  {label:<30}  {detail}"
            )
        except Exception:
            pass

    def action_refresh(self) -> None:
        for key, label in self._CHECKS:
            try:
                self.query_one(f"#dr-{key}", Static).update(
                    f"  {_CHECK_PENDING}  {label:<30}  checking…"
                )
            except Exception:
                pass
        asyncio.get_event_loop().create_task(self._run_checks())

    def action_back(self) -> None:
        self.app.pop_screen()

    def update_state(self, state: AppState) -> None:
        pass
