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

# Cap CPU thread pools so local training doesn't thrash the host.
# LLM work is remote; XGBoost/LightGBM/sklearn otherwise default to all cores.
for _k, _v in (("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
               ("OPENBLAS_NUM_THREADS", "4"), ("NUMEXPR_NUM_THREADS", "4"),
               ("VECLIB_MAXIMUM_THREADS", "4")):
    os.environ.setdefault(_k, _v)

# Ensure ~/.automedal/ exists on first run — keys, sprites, and user-mode
# logs all live under it. Lazy creation during `setup` is fine for .env, but
# having the root in place up-front lets the TUI, sprite loader, and doctor
# write to it without each checking for parents.
(Path.home() / ".automedal").mkdir(parents=True, exist_ok=True)

# Populate os.environ from ~/.automedal/.env before any command inspects it.
# Safe no-op when the file doesn't exist (first run).
load_env()


# ── public entry point ────────────────────────────────────────────────────────

def dispatch(cmd: str, args: list[str]) -> int:
    from automedal.paths import Layout

    layout = Layout()
    env = {**os.environ, **layout.as_env()}

    # Commands that don't require prior setup
    ungated = {"setup", "help", "--help", "-h", "doctor", "version", "--version", "status"}
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
        "models":    _cmd_models,
        "status":    _cmd_status,
        "clean":     _cmd_clean,
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

    print(f"\nSetup complete.\n  Provider:      {provider}\n  Default model: {default_model}\n")

    print("Running smoke test…")
    from automedal.agent.providers import smoke, parse_slug
    try:
        prov, short = parse_slug(default_model)
    except ValueError:
        prov, short = provider, default_model
    ok, detail = smoke(prov, short)
    if ok:
        print(f"✓ Smoke test passed — {detail}")
    else:
        print(f"⚠  Smoke test failed — {detail}")
        print("   Run 'automedal doctor' for more details.")
    return 0


def _cmd_doctor(args, layout, env) -> int:
    from automedal.auth import ENV_FILE, configured_providers

    print("── agent runtime ──")
    print("  bespoke kernel (automedal.agent)")
    try:
        from importlib.metadata import version as _v
        print(f"  anthropic {_v('anthropic')} / openai {_v('openai')}")
    except Exception as exc:
        print(f"  SDK lookup failed: {exc}")
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
    provider = env.get("AUTOMEDAL_PROVIDER", "opencode-go")
    model = env.get("AUTOMEDAL_MODEL")
    if not model:
        slug = env.get("MODEL", "opencode-go/minimax-m2.7")
        if "/" in slug:
            provider, model = slug.split("/", 1)
        else:
            model = slug
    print(f"  provider: {provider}\n  model:    {model}")
    from automedal.agent.providers import smoke
    ok, detail = smoke(provider, model)
    if ok:
        print(f"  ✓ {detail}")
        rc = 0
    else:
        print(f"  ⚠  {detail}")
        rc = 1

    # ── advisor smoke (only when AUTOMEDAL_ADVISOR=1) ─────────────────────
    if env.get("AUTOMEDAL_ADVISOR", "").strip().lower() in ("1", "true", "yes", "on"):
        print()
        print("── advisor (Kimi K2.6) ──")
        adv_model = env.get("AUTOMEDAL_ADVISOR_MODEL", "kimi-k2.6")
        adv_base = env.get("AUTOMEDAL_ADVISOR_BASE_URL", "https://opencode.ai/zen/go/v1")
        adv_max_consult = env.get("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT", "2000")
        adv_max_iter = env.get("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", "8000")
        junctions = env.get("AUTOMEDAL_ADVISOR_JUNCTIONS", "stagnation,audit,tool")
        print(f"  model:     {adv_model}")
        print(f"  base_url:  {adv_base}")
        print(f"  junctions: {junctions}")
        print(f"  budget:    {adv_max_consult} tok/consult, {adv_max_iter} tok/iter")
        if not env.get("OPENCODE_API_KEY"):
            print("  ⚠  OPENCODE_API_KEY missing — advisor will skip every call")
        else:
            import asyncio
            from automedal import advisor as _adv

            async def _ping():
                # Use purpose="tool" since the tool prompt template is the
                # cheapest/shortest. Strict junction allowlist still applies.
                return await _adv.consult(
                    purpose="tool",
                    question="Reply with the single word READY.",
                    context="Smoke test from automedal doctor.",
                    max_tokens=16,
                )
            try:
                op = asyncio.run(_ping())
            except Exception as exc:
                print(f"  ⚠  smoke failed: {type(exc).__name__}: {exc}")
                return rc or 1
            if op.skipped:
                print(f"  ⚠  skipped: {op.reason}")
                if op.reason.startswith("disabled:"):
                    print(f"      (junction not in {junctions})")
            else:
                preview = (op.text or "").replace("\n", " ").strip()[:80]
                print(f"  ✓ {adv_model} ({op.in_tokens}/{op.out_tokens} tok) — {preview}")
    return rc


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
    from automedal.run_args import parse_run_args

    args, env_overrides = parse_run_args(args)
    if env_overrides:
        env = {**env, **env_overrides}

    n = args[0] if args else "50"
    fast = args[1] if len(args) > 1 else ""
    call_args = [n, fast] if fast else [n]
    return subprocess.call(
        [sys.executable, "-m", "automedal.run_loop"] + call_args, env=env
    )


def _cmd_models(args, layout, env) -> int:
    """List or refresh the cached advisor-model autocomplete list."""
    from automedal import advisor

    sub = args[0] if args else "list"

    if sub in ("refresh", "--refresh", "-r"):
        n, where = advisor.refresh_models()
        if n == 0:
            print(f"⚠  refresh failed: {where}")
            return 1
        print(f"✓ {n} models cached from {where}")
        print(f"  ({advisor.models_cache_path()})")
        return 0

    models = advisor.list_models()
    if not models:
        print("(no models cached — run 'automedal models refresh')")
        return 1
    print(f"# {len(models)} models in {advisor.models_cache_path()}")
    for m in models:
        print(f"  {m}")
    return 0


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


def _cmd_version(args, layout, env) -> int:
    from automedal import __version__
    print(f"automedal {__version__}")
    return 0


def _cmd_help(args, layout, env) -> int:
    print("""automedal — autonomous Kaggle ML research agent

One-time:
  automedal setup                configure a model provider (first-run)
  automedal doctor               diagnose provider/env state + smoke-test the LLM

Competition setup:
  automedal discover             list active Kaggle competitions
  automedal select               pick one interactively
  automedal init <slug>          download + wire up a competition
  automedal prepare              regenerate .npy arrays from data/
  automedal render               re-render AGENTS.md from template

Loop:
  automedal run [N] [--advisor [model]]
                                 start the loop (default 50). --advisor enables
                                 the Kimi K2.6 second-opinion loop; pass a model
                                 id to override (autocompletes in the TUI).
  automedal status               quick health check (knowledge + last results)
  automedal clean                wipe memory files + results.tsv (confirms first)
  automedal models [refresh]     list cached advisor models (--refresh re-fetches)

Monitor:
  automedal                      open TUI home screen (command palette)
  automedal tui [--demo]         same as above

Env vars honored by 'automedal run':
  AUTOMEDAL_PROVIDER     opencode-go | anthropic | openai | ollama | openrouter | groq
  AUTOMEDAL_MODEL        model id for that provider (default: minimax-m2.7)
  MODEL                  back-compat slug (provider/model) — split into the two above
  AUTOMEDAL_ANALYZER     1=on (default), 0=off
  AUTOMEDAL_QUICK_REJECT 0=off (default), 1=on (30s smoke-train guard)
  AUTOMEDAL_DEDUPE       1=on (default), 0=off
  AUTOMEDAL_DEDUPE_THRESHOLD  BM25 score, default 5.0 (higher = stricter)
  STAGNATION_K           consecutive non-improving runs before research (default 3)
  RESEARCH_EVERY         scheduled research cadence (default 10, 0 disables)
  LOG_FILE               human log path (default agent_loop.log)
  AUTOMEDAL_EVENTS_FILE  JSONL event sink (default agent_loop.events.jsonl)

Advisor (Kimi K2.6 second-opinion loop, off by default — see README):
  AUTOMEDAL_ADVISOR                        1=on, 0=off (default 0)
  AUTOMEDAL_ADVISOR_MODEL                  default kimi-k2.6
  AUTOMEDAL_ADVISOR_BASE_URL               default https://opencode.ai/zen/go/v1
  AUTOMEDAL_ADVISOR_JUNCTIONS              stagnation,audit,tool (any subset)
  AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT default 2000
  AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER    default 8000 (hard ceiling)
  AUTOMEDAL_ADVISOR_AUDIT_EVERY            knowledge-audit cadence (default 5)
  AUTOMEDAL_ADVISOR_STAGNATION_EVERY       periodic stagnation gate (default 5)
""")
    return 0
