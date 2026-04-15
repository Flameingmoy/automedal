"""Headless command dispatch — Python replacement for the `am` bash wrapper.

Each subcommand is a function _cmd_<name>(args, layout, env) -> int.
Exit code 0 = success.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from pathlib import Path


# ── public entry point ────────────────────────────────────────────────────────

def dispatch(cmd: str, args: list[str]) -> int:
    from automedal.paths import Layout

    layout = Layout()
    env = {**os.environ, **layout.as_env()}

    # Resolve pi path early so subprocesses inherit it
    try:
        from automedal.pi_runtime import ensure_pi
        env["AUTOMEDAL_PI_BIN"] = str(ensure_pi())
    except SystemExit as exc:
        # pi missing — only fail for commands that actually need it
        if cmd in ("run", "setup", "doctor"):
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
    provider_vars = [
        "OPENCODE_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
        "MISTRAL_API_KEY", "CEREBRAS_API_KEY", "XAI_API_KEY",
        "ZAI_API_KEY", "AZURE_OPENAI_API_KEY", "AI_GATEWAY_API_KEY",
    ]
    if any(env.get(v) for v in provider_vars):
        return False
    auth = Path.home() / ".pi" / "agent" / "auth.json"
    try:
        return '"key"' not in auth.read_text()
    except OSError:
        return True


def _write_auth_json(provider: str, apikey: str) -> Path:
    """Write provider API key to ~/.pi/agent/auth.json. Returns the path."""
    auth_path = Path.home() / ".pi" / "agent" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(auth_path.read_text()) if auth_path.exists() else {}
    data[provider] = {"type": "api_key", "key": apikey}
    auth_path.write_text(json.dumps(data, indent=2))
    auth_path.chmod(0o600)
    return auth_path


def _smoke_test(pi: str, model: str) -> bool:
    """Run a quick smoke test. Returns True if pi responds with READY."""
    try:
        result = subprocess.run(
            [pi, "--no-session", "--model", model, "-p", "Say READY and nothing else."],
            capture_output=True, text=True, timeout=60,
        )
        output = (result.stdout + result.stderr).lower()
        return "ready" in output
    except Exception:
        return False


def _run_python(script: Path, extra_args: list[str], env: dict) -> int:
    """Run a Python script with the project's interpreter."""
    return subprocess.call([sys.executable, str(script)] + extra_args, env=env)


def _run_sh(script: Path, extra_args: list[str], env: dict) -> int:
    return subprocess.call(["bash", str(script)] + extra_args, env=env)


# ── command implementations ───────────────────────────────────────────────────

def _cmd_setup(args, layout, env) -> int:
    print("AutoMedal first-run setup")
    print("─────────────────────────\n")
    print("Default provider: OpenCode Go (one sk- key unlocks GLM/Kimi/MiMo/MiniMax)\n")

    use_default = input("Use OpenCode Go? [Y/switch] ").strip().lower() or "y"

    providers = {
        "1": ("opencode-go",  "OPENCODE_API_KEY",   "opencode-go/minimax-m2.7"),
        "2": ("openrouter",   "OPENROUTER_API_KEY",  "openrouter/<your-model>"),
        "3": ("ollama",       "",                    "ollama/<local-model>"),
        "4": ("anthropic",    "ANTHROPIC_API_KEY",   "anthropic/claude-sonnet-4-5"),
        "5": ("openai",       "OPENAI_API_KEY",      "openai/gpt-4o"),
    }

    if use_default.startswith("s"):
        print("\n  1) OpenCode Go  (default)")
        print("  2) OpenRouter   (free-tier models available)")
        print("  3) Ollama       (local, no key needed)")
        print("  4) Anthropic    (direct Claude)")
        print("  5) OpenAI       (direct GPT)\n")
        choice = input("Choice [1]: ").strip() or "1"
    else:
        choice = "1"

    provider, env_var, default_model = providers.get(choice, providers["1"])

    if env_var:
        apikey = getpass.getpass(f"Paste your {provider} API key (input hidden): ")
        if not apikey:
            print("Empty key, aborting.")
            return 1
        auth_path = _write_auth_json(provider, apikey)
        print(f"✓ Saved to {auth_path}")
    else:
        print(f"(no key needed for {provider})")

    print(f"\nSetup complete.\n  Provider: {provider}\n  Default model: {default_model}\n")
    print("Running smoke test…")
    pi = env.get("AUTOMEDAL_PI_BIN", "pi")
    if _smoke_test(pi, default_model):
        print("✓ Smoke test passed")
    else:
        print("⚠  Smoke test did not return READY — run 'automedal doctor' for details")
    return 0


def _cmd_doctor(args, layout, env) -> int:
    from automedal.pi_runtime import pi_version
    print("── pi version ──")
    print(f"  {pi_version()}")
    print()
    print("── active provider env vars ──")
    provider_vars = [
        "OPENCODE_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
    ]
    found = [v for v in provider_vars if env.get(v)]
    if found:
        for v in found:
            print(f"  {v} is set")
    else:
        print("  (none exported in current shell)")
    print()
    print("── pi auth.json ──")
    auth = Path.home() / ".pi" / "agent" / "auth.json"
    if auth.exists():
        try:
            data = json.loads(auth.read_text())
            providers = ", ".join(data.keys()) if data else "(empty)"
            print(f"  providers configured: {providers}")
        except Exception as exc:
            print(f"  (parse error: {exc})")
    else:
        print("  (no ~/.pi/agent/auth.json)")
    print()
    print("── smoke test ──")
    model = env.get("MODEL", "opencode-go/minimax-m2.7")
    print(f"  model: {model}")
    pi = env.get("AUTOMEDAL_PI_BIN", "pi")
    subprocess.call([pi, "--no-session", "--model", model, "-p", "Say READY and nothing else."])
    return 0


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
    return _run_sh(layout.run_sh, [n, fast] if fast else [n], env)


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
