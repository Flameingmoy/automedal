"""Setup wizard screen — Textual-native provider config with hidden API key input.

Replaces the input()/getpass() flow in dispatch.py::_cmd_setup when launched
from the TUI. Shell `automedal setup` still uses the old interactive path.
"""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, RadioButton, RadioSet, Static

from tui.state import AppState


_PROVIDERS = [
    ("opencode-go",  "OPENCODE_API_KEY",   "opencode-go/minimax-m2.7",      True),
    ("openrouter",   "OPENROUTER_API_KEY",  "openrouter/openai/gpt-4o-mini", True),
    ("ollama",       "",                    "ollama/llama3.2",                False),
    ("anthropic",    "ANTHROPIC_API_KEY",   "anthropic/claude-sonnet-4-5",    True),
    ("openai",       "OPENAI_API_KEY",      "openai/gpt-4o",                  True),
    ("groq",         "GROQ_API_KEY",        "groq/llama-3.3-70b-versatile",   True),
    ("mistral",      "MISTRAL_API_KEY",     "mistral/mistral-large-latest",   True),
    ("gemini",       "GEMINI_API_KEY",      "gemini/gemini-2.0-flash-exp",    True),
]


class SetupWizardScreen(Screen):
    DEFAULT_CSS = """
    SetupWizardScreen {
        layout: vertical;
        background: #0F111A;
    }
    SetupWizardScreen > #sw-title {
        height: 1;
        background: #0F111A;
        padding: 0 1;
        color: #8BE9FD;
    }
    SetupWizardScreen > #sw-body {
        height: 1fr;
        padding: 1 2;
        layout: vertical;
    }
    SetupWizardScreen RadioSet {
        height: auto;
        margin: 0 0 1 0;
    }
    SetupWizardScreen #sw-key-label { height: 1; margin: 1 0 0 0; }
    SetupWizardScreen #sw-key { height: 3; }
    SetupWizardScreen #sw-submit { margin: 1 0; }
    SetupWizardScreen #sw-status { height: 3; color: #8BE9FD; }
    """

    BINDINGS = [
        ("q",      "back", "Back"),
        ("escape", "back", "Back"),
        ("ctrl+s", "submit", "Submit"),
    ]

    def __init__(self, layout=None, args: list[str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self._layout = layout
        self._args = args or []

    def compose(self) -> ComposeResult:
        yield Static("AutoMedal Setup — Choose a provider", id="sw-title")
        with self.prevent():
            pass
        from textual.containers import Vertical
        with Vertical(id="sw-body"):
            yield Label("Provider:")
            with RadioSet(id="sw-providers"):
                for i, (name, _, _, _) in enumerate(_PROVIDERS):
                    suffix = " (default)" if i == 0 else ""
                    yield RadioButton(f"{name}{suffix}", value=(i == 0), id=f"sw-p{i}")
            yield Label("API key (leave blank for Ollama):", id="sw-key-label")
            yield Input(placeholder="sk-…", password=True, id="sw-key")
            yield Button("Configure & test", id="sw-submit", variant="primary")
            yield Static("", id="sw-status")
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_submit(self) -> None:
        self.query_one("#sw-submit", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sw-submit":
            asyncio.get_event_loop().create_task(self._do_setup())

    async def _do_setup(self) -> None:
        status = self.query_one("#sw-status", Static)
        status.update("  Configuring…")

        # Determine selected provider
        radio_set = self.query_one("#sw-providers", RadioSet)
        selected_idx = 0
        for i in range(len(_PROVIDERS)):
            rb = self.query_one(f"#sw-p{i}", RadioButton)
            if rb.value:
                selected_idx = i
                break

        provider, env_var, default_model, needs_key = _PROVIDERS[selected_idx]
        apikey = self.query_one("#sw-key", Input).value.strip()

        if needs_key and not apikey:
            status.update("  [red]API key required for this provider[/]")
            return

        # Persist the provider credential via the new ~/.automedal/.env store
        if needs_key and apikey:
            try:
                from automedal.auth import save_key
                path = await asyncio.get_event_loop().run_in_executor(
                    None, save_key, provider, apikey
                )
                status.update(f"  ✓ Saved {provider} key to {path}\n  Running smoke test…")
            except Exception as exc:
                status.update(f"  [red]Error saving auth: {exc}[/]")
                return
        elif not needs_key:
            status.update(f"  (no key needed for {provider})\n  Running smoke test…")

        # Smoke test via the deepagents runtime — works for both agent modes
        from automedal import agent_runtime as ar
        try:
            prov, short = ar.parse_slug(default_model)
        except ValueError:
            prov, short = provider, default_model
        ok, detail = await asyncio.get_event_loop().run_in_executor(
            None, ar.smoke_test, prov, short
        )

        if ok:
            status.update(
                f"  [green]✓ Setup complete[/]\n"
                f"  Provider: {provider}\n"
                f"  Smoke test: {detail}"
            )
        else:
            status.update(
                f"  [yellow]⚠  Setup saved but smoke test failed[/]\n"
                f"  Provider: {provider}\n"
                f"  {detail}\n"
                f"  Run 'automedal doctor' for details."
            )

    def update_state(self, state: AppState) -> None:
        pass
