"""Provider credential store.

Replaces the pi-coding-agent's `~/.pi/agent/auth.json` with a
standard dotenv file at `~/.automedal/.env` (mode 0600). Loaded at
process start via `load_env()` so `os.environ[<PROVIDER>_API_KEY]` is
populated before anything inspects it.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from dotenv import load_dotenv, set_key


ENV_FILE = Path.home() / ".automedal" / ".env"

# Canonical provider → env-var map. Order matters — the wizard and doctor
# display providers in this order.
PROVIDER_ENV: dict[str, str] = {
    "opencode-go":  "OPENCODE_API_KEY",
    "anthropic":    "ANTHROPIC_API_KEY",
    "openai":       "OPENAI_API_KEY",
    "openrouter":   "OPENROUTER_API_KEY",
    "groq":         "GROQ_API_KEY",
    "mistral":      "MISTRAL_API_KEY",
    "gemini":       "GEMINI_API_KEY",
    "xai":          "XAI_API_KEY",
    "cerebras":     "CEREBRAS_API_KEY",
    "zai":          "ZAI_API_KEY",
    "ollama":       "",  # no key — local endpoint
}

# Default model slug per provider (used by setup wizard + doctor smoke test)
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "opencode-go":  "opencode-go/minimax-m2.7",
    "anthropic":    "anthropic/claude-sonnet-4-5",
    "openai":       "openai/gpt-4o",
    "openrouter":   "openrouter/openai/gpt-4o-mini",
    "groq":         "groq/llama-3.3-70b-versatile",
    "mistral":      "mistral/mistral-large-latest",
    "gemini":       "gemini/gemini-2.0-flash-exp",
    "xai":          "xai/grok-2",
    "cerebras":     "cerebras/llama-3.3-70b",
    "zai":          "zai/glm-4.6",
    "ollama":       "ollama/llama3.2",
}

# Legacy pi auth file — imported on first run then left in place.
_PI_AUTH = Path.home() / ".pi" / "agent" / "auth.json"


def load_env(env_file: Path | None = None) -> bool:
    """Load ~/.automedal/.env into os.environ. Returns True if file existed."""
    f = env_file or ENV_FILE
    if not f.exists():
        return False
    load_dotenv(f, override=False)
    return True


def save_key(provider: str, key: str, env_file: Path | None = None) -> Path:
    """Save a provider API key to ~/.automedal/.env (mode 0600)."""
    if provider not in PROVIDER_ENV:
        raise ValueError(f"unknown provider: {provider}")
    var = PROVIDER_ENV[provider]
    if not var:
        raise ValueError(f"provider {provider!r} does not take an API key")

    f = env_file or ENV_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.touch()
    f.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600

    set_key(str(f), var, key, quote_mode="never")
    # Update current process env too — callers may smoke-test immediately
    os.environ[var] = key
    return f


def configured_providers(env: dict | None = None) -> list[str]:
    """Return the list of provider names with credentials present in env."""
    env = env if env is not None else os.environ
    out: list[str] = []
    for provider, var in PROVIDER_ENV.items():
        if not var:
            # Ollama counts as always-available if a base URL is set
            if provider == "ollama" and env.get("OLLAMA_HOST"):
                out.append(provider)
            continue
        if env.get(var):
            out.append(provider)
    return out


def needs_setup(env: dict | None = None) -> bool:
    """True iff no provider credentials are available anywhere.

    Checks (a) os.environ, (b) ~/.automedal/.env (auto-loads), (c) legacy
    ~/.pi/agent/auth.json. False as soon as any provider has a key.
    """
    env = env if env is not None else os.environ
    if configured_providers(env):
        return False
    # .env may not be loaded yet
    if load_env():
        if configured_providers(env):
            return False
    # Legacy pi auth — treat as configured if any provider has a key
    try:
        data = json.loads(_PI_AUTH.read_text(encoding="utf-8"))
        return not any(isinstance(v, dict) and v.get("key") for v in data.values())
    except (OSError, json.JSONDecodeError):
        return True


def import_pi_auth(env_file: Path | None = None) -> list[str]:
    """One-time import of ~/.pi/agent/auth.json into ~/.automedal/.env.

    Returns the list of provider names imported. Idempotent — re-running
    overwrites existing keys if the pi auth file is newer (dotenv set_key
    replaces in place).
    """
    try:
        data = json.loads(_PI_AUTH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    imported: list[str] = []
    for provider, entry in data.items():
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not key:
            continue
        if provider not in PROVIDER_ENV or not PROVIDER_ENV[provider]:
            continue
        save_key(provider, key, env_file=env_file)
        imported.append(provider)
    return imported
