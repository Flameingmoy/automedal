"""Headless command dispatch — Python replacement for the `am` bash wrapper.

Each subcommand is a function _cmd_<name>(args, layout, env) -> int.
Exit code 0 = success.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

from automedal.auth import load_env

# Populate os.environ from ~/.automedal/.env before any command inspects it.
# Safe no-op when the file doesn't exist (first run).
load_env()


# ── public entry point ────────────────────────────────────────────────────────

def dispatch(cmd: str, args: list[str]) -> int:
    from automedal.paths import Layout

    layout = Layout()
    env = {**os.environ, **layout.as_env()}

    # Agent runtime selector (pi = legacy Node agent, deepagents = LangChain)
    agent_mode = env.get("AUTOMEDAL_AGENT", "pi")

    # pi binary is only needed for the legacy agent path
    if agent_mode == "pi":
        try:
            from automedal.pi_runtime import ensure_pi
            env["AUTOMEDAL_PI_BIN"] = str(ensure_pi())
        except SystemExit as exc:
            # pi missing — only fail for commands that actually need it
            if cmd in ("run",):
                print(str(exc), file=sys.stderr)
                return 1

    # Commands that don't require prior setup
    ungated = {"setup", "help", "--help", "-h", "doctor", "version", "--version", "compact", "status"}
    if cmd not in ungated and _needs_setup(env):
        print("AutoMedal isn't configured yet.\nRun:  automedal setup")
        return 1

    handlers = {
        "setup":     _cmd_setup,
        "doctor":    _cmd_doctor,
        "discover":  _cmd_discover,
        "select":    _cmd_select,
        "init":      _cmd_init,
        "bootstrap": _cmd_init,   # backward-compat alias
        "prepare":   _cmd_prepare,
        "render":    _cmd_render,
        "run":       _cmd_run,
        "status":    _cmd_status,
        "clean":     _cmd_clean,
        "compact":   _cmd_compact,
        "help":      _cmd_help,
        "--help":    _cmd_help,
        "-h":        _cmd_help,
        "version":   _cmd_version,
        "--version": _cmd_version,
    }

    handler = handlers.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}. Run 'automedal help' for usage.")
        return 1

    return handler(args, layout, env)


# ── helpers ───────────────────────────────────────────────────────────────────

def _needs_setup(env: dict) -> bool:
    from automedal.auth import needs_setup
    return needs_setup(env)


def _run_python(script: Path, extra_args: list[str], env: dict) -> int:
    """Run a Python script with the project's interpreter."""
    return subprocess.call([sys.executable, str(script)] + extra_args, env=env)


def _run_sh(script: Path, extra_args: list[str], env: dict) -> int:
    return subprocess.call(["bash", str(script)] + extra_args, env=env)


# ── command implementations ───────────────────────────────────────────────────

def _cmd_setup(args, layout, env) -> int:
    from automedal.auth import (
        ENV_FILE, PROVIDER_DEFAULT_MODEL, PROVIDER_ENV,
        configured_providers, import_pi_auth, save_key,
    )

    print("AutoMedal first-run setup")
    print("─────────────────────────\n")

    # Legacy pi auth migration (one-shot). If the user already has an
    # ~/.pi/agent/auth.json from v1.0, offer to import it so they don't
    # need to re-paste keys.
    pi_auth = Path.home() / ".pi" / "agent" / "auth.json"
    if pi_auth.exists() and not configured_providers():
        ans = input(
            f"Found legacy {pi_auth}.\n"
            f"Import into {ENV_FILE}? [Y/n] "
        ).strip().lower() or "y"
        if ans.startswith("y"):
            imported = import_pi_auth()
            if imported:
                print(f"✓ Imported: {', '.join(imported)}\n")

    print("Default provider: OpenCode Go (one sk- key unlocks GLM/Kimi/MiMo/MiniMax)\n")
    use_default = input("Use OpenCode Go? [Y/switch] ").strip().lower() or "y"

    providers_menu = [
        ("opencode-go",  "OpenCode Go  (default — MiniMax-M2.7, GLM, Kimi, MiMo)"),
        ("openrouter",   "OpenRouter   (free-tier models available)"),
        ("ollama",       "Ollama       (local, no key needed)"),
        ("anthropic",    "Anthropic    (direct Claude)"),
        ("openai",       "OpenAI       (direct GPT)"),
        ("groq",         "Groq         (fast Llama / Mixtral)"),
        ("mistral",      "Mistral      (mistral.ai)"),
        ("gemini",       "Gemini       (Google AI Studio)"),
    ]

    if use_default.startswith("s"):
        print()
        for i, (_p, label) in enumerate(providers_menu, 1):
            print(f"  {i}) {label}")
        print()
        raw = input(f"Choice [1-{len(providers_menu)}, default 1]: ").strip() or "1"
        try:
            idx = max(1, min(len(providers_menu), int(raw))) - 1
        except ValueError:
            idx = 0
    else:
        idx = 0

    provider = providers_menu[idx][0]
    env_var = PROVIDER_ENV.get(provider, "")
    default_model = PROVIDER_DEFAULT_MODEL.get(provider, f"{provider}/<model>")

    if env_var:
        apikey = getpass.getpass(f"Paste your {provider} API key (input hidden): ")
        if not apikey:
            print("Empty key, aborting.")
            return 1
        path = save_key(provider, apikey)
        print(f"✓ Saved to {path} (mode 0600)")
    else:
        if provider == "ollama":
            host = input("Ollama host URL [http://localhost:11434]: ").strip() or "http://localhost:11434"
            os.environ["OLLAMA_HOST"] = host
            # Persist via save_key for consistency — we reuse the dotenv writer
            from dotenv import set_key as _set
            ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
            ENV_FILE.touch(exist_ok=True); ENV_FILE.chmod(0o600)
            _set(str(ENV_FILE), "OLLAMA_HOST", host, quote_mode="never")
            print(f"✓ Saved OLLAMA_HOST to {ENV_FILE}")
        else:
            print(f"(no key needed for {provider})")

    print(f"\nSetup complete.\n  Provider:      {provider}\n  Default model: {default_model}")
    print(f"  Agent runtime: {env.get('AUTOMEDAL_AGENT', 'pi')} "
          f"(set AUTOMEDAL_AGENT=deepagents to use the Python runtime)\n")

    print("Running smoke test…")
    from automedal import agent_runtime as ar
    try:
        prov, short = ar.parse_slug(default_model)
    except ValueError:
        prov, short = provider, default_model
    ok, detail = ar.smoke_test(prov, short)
    if ok:
        print(f"✓ Smoke test passed — {detail}")
    else:
        print(f"⚠  Smoke test failed — {detail}")
        print("   Run 'automedal doctor' for more details.")
    return 0


def _cmd_doctor(args, layout, env) -> int:
    from automedal.auth import ENV_FILE, PROVIDER_ENV, configured_providers

    agent_mode = env.get("AUTOMEDAL_AGENT", "pi")
    print("── agent runtime ──")
    print(f"  AUTOMEDAL_AGENT = {agent_mode}")
    if agent_mode == "pi":
        try:
            from automedal.pi_runtime import pi_version
            print(f"  pi: {pi_version()}")
        except Exception as exc:
            print(f"  pi: unavailable ({exc})")
    else:
        try:
            import deepagents, langchain  # noqa: F401
            from importlib.metadata import version as _v
            print(f"  deepagents {_v('deepagents')} / langchain {_v('langchain')}")
        except Exception as exc:
            print(f"  deepagents: unavailable ({exc})")
    print()

    print("── credentials ──")
    print(f"  env file: {ENV_FILE} {'(present)' if ENV_FILE.exists() else '(missing)'}")
    active = configured_providers(env)
    if active:
        print(f"  configured providers: {', '.join(active)}")
    else:
        print("  (no provider credentials found — run 'automedal setup')")
    pi_auth = Path.home() / ".pi" / "agent" / "auth.json"
    if pi_auth.exists():
        print(f"  legacy: {pi_auth} exists (importable via 'automedal setup')")
    print()

    print("── smoke test ──")
    model = env.get("MODEL", "opencode-go/minimax-m2.7")
    print(f"  model: {model}")
    from automedal import agent_runtime as ar
    try:
        prov, short = ar.parse_slug(model)
    except ValueError:
        print(f"  ⚠  invalid MODEL slug (expected 'provider/model'): {model}")
        return 1
    ok, detail = ar.smoke_test(prov, short)
    if ok:
        print(f"  ✓ {detail}")
        return 0
    print(f"  ⚠  {detail}")
    return 1


def _cmd_discover(args, layout, env) -> int:
    return _run_python(layout.scout_dir / "discover.py", args, env)


def _cmd_select(args, layout, env) -> int:
    return _run_python(layout.scout_dir / "select.py", args, env)


def _cmd_init(args, layout, env) -> int:
    if not args:
        print("Usage: automedal init <slug>")
        print("Example: automedal init playground-series-s6e4")
        return 1
    # bootstrap.py still uses the legacy name internally
    return _run_python(layout.scout_dir / "bootstrap.py", args, env)


def _cmd_prepare(args, layout, env) -> int:
    return _run_python(layout.prepare_py, args, env)


def _cmd_render(args, layout, env) -> int:
    return _run_python(layout.scout_dir / "render.py", args, env)


def _cmd_run(args, layout, env) -> int:
    n = args[0] if args else "50"
    fast = args[1] if len(args) > 1 else ""
    call_args = [n, fast] if fast else [n]
    if env.get("AUTOMEDAL_AGENT", "pi") == "deepagents":
        return subprocess.call(
            [sys.executable, "-m", "automedal.run_loop"] + call_args, env=env
        )
    return _run_sh(layout.run_sh, call_args, env)


def _cmd_status(args, layout, env) -> int:
    print("── knowledge.md (head) ──")
    if layout.knowledge_md.exists():
        lines = layout.knowledge_md.read_text().splitlines()[:20]
        print("\n".join(lines))
    else:
        print("(no knowledge.md — run 'automedal init' first)")
    print()
    print("── results.tsv (tail) ──")
    if layout.results_tsv.exists():
        lines = layout.results_tsv.read_text().splitlines()[-5:]
        print("\n".join(lines))
    else:
        print("(no results.tsv yet)")
    print()
    print("── latest exp tags ──")
    result = subprocess.run(
        ["git", "tag", "-l", "exp/*"], capture_output=True, text=True,
        cwd=str(layout.cwd),
    )
    tags = result.stdout.strip().splitlines()[-5:] if result.returncode == 0 else []
    print("\n".join(tags) if tags else "(no experiment tags)")
    return 0


def _cmd_clean(args, layout, env) -> int:
    yes = "--yes" in args or "-y" in args
    if not yes:
        ans = input("Wipe memory files and results.tsv? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0
    rc = _run_python(layout.harness_dir / "init_memory.py", ["--force"], env)
    if layout.results_tsv.exists():
        layout.results_tsv.unlink()
    print("✓ Memory reset")
    return rc


def _cmd_compact(args, layout, env) -> int:
    """Compact a memory file when it exceeds the token-budget threshold."""
    target = args[0] if args else str(layout.research_md)
    return _run_python(layout.harness_dir / "compact_memory.py", ["--target", target], env)


def _cmd_version(args, layout, env) -> int:
    from automedal import __version__
    print(f"automedal {__version__}")
    return 0


def _cmd_help(args, layout, env) -> int:
    print("""automedal — autonomous Kaggle ML research agent

One-time:
  automedal setup                configure a model provider (first-run)
  automedal doctor               diagnose pi/provider/env state

Competition setup:
  automedal discover             list active Kaggle competitions
  automedal select               pick one interactively
  automedal init <slug>          download + wire up a competition
  automedal prepare              regenerate .npy arrays from data/
  automedal render               re-render AGENTS.md from template

Loop:
  automedal run [N]              start the three-phase loop (default 50)
  automedal status               quick health check (knowledge + last results)
  automedal clean                wipe memory files + results.tsv (confirms first)
  automedal compact [file]       condense research_notes.md (or another file) when it grows large

Monitor:
  automedal                      open TUI home screen (command palette)
  automedal tui [--demo]         same as above

Env vars honored by 'automedal run':
  MODEL              agent model slug, default opencode-go/minimax-m2.7
  STAGNATION_K       consecutive non-improving runs before research (default 3)
  RESEARCH_EVERY     scheduled research cadence (default 10, 0 disables)
  LOG_FILE           combined loop log path (default agent_loop.log)
""")
    return 0
