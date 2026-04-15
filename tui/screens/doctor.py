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
        background: $background;
    }
    DoctorScreen > #dr-title {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $accent;
    }
    DoctorScreen > #dr-checks {
        height: 1fr;
        padding: 1 2;
        layout: vertical;
    }
    DoctorScreen > #dr-footer-hint {
        height: 1;
        color: $text-muted;
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
        ("node",     "Node.js on PATH"),
        ("npm",      "npm on PATH"),
        ("pi_bin",   "pi binary resolved"),
        ("pi_ver",   "pi version"),
        ("auth",     "~/.pi/agent/auth.json"),
        ("smoke",    "pi smoke test"),
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
        pi_bin = None

        # node
        ok = shutil.which("node") is not None
        self._set("node", ok, shutil.which("node") or "not found")

        # npm
        ok = shutil.which("npm") is not None
        self._set("npm", ok, shutil.which("npm") or "not found")

        # pi_bin
        try:
            from automedal.pi_runtime import ensure_pi
            pi_path = ensure_pi()
            pi_bin = str(pi_path)
            self._set("pi_bin", True, pi_bin)
        except SystemExit as exc:
            self._set("pi_bin", False, str(exc))
        except Exception as exc:
            self._set("pi_bin", False, str(exc))

        # pi_ver
        if pi_bin:
            try:
                from automedal.pi_runtime import pi_version
                ver = pi_version()
                self._set("pi_ver", True, ver)
            except Exception as exc:
                self._set("pi_ver", False, str(exc))
        else:
            self._set("pi_ver", False, "pi not found")

        # auth
        auth = Path.home() / ".pi" / "agent" / "auth.json"
        if auth.exists():
            try:
                data = json.loads(auth.read_text())
                providers = ", ".join(data.keys()) if data else "(empty)"
                self._set("auth", bool(data), providers)
            except Exception as exc:
                self._set("auth", False, f"parse error: {exc}")
        else:
            self._set("auth", False, "not found — run 'automedal setup'")

        # smoke test (offload to thread pool)
        if pi_bin:
            from automedal.dispatch import _smoke_test
            env_key = {}
            try:
                if self._layout:
                    env_key = self._layout.as_env()
            except Exception:
                pass
            import os
            model = os.environ.get("MODEL", "opencode-go/minimax-m2.7")
            try:
                passed = await asyncio.get_event_loop().run_in_executor(
                    None, _smoke_test, pi_bin, model
                )
                self._set("smoke", passed, "READY" if passed else f"no READY (model: {model})")
            except Exception as exc:
                self._set("smoke", False, str(exc))
        else:
            self._set("smoke", False, "skipped — pi not found")

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
