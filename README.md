<p align="center">
  <h1 align="center">AutoMedal</h1>
  <p align="center">
    Autonomous ML research agent for Kaggle competitions
    <br />
    Point it at a competition. Wake up to a leaderboard-climbing submission.
  </p>
</p>

<p align="center">
  <a href="#install">Install</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#tui--command-centre">TUI</a> &bull;
  <a href="#providers">Providers</a> &bull;
  <a href="#configuration">Configuration</a> &bull;
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

AutoMedal is an autonomous experiment loop for tabular ML competitions. A small coding agent tries different models, features, hyperparameters, ensembles, and literature-inspired ideas — keeping only what improves the score.

**One static Go binary.** The control plane (run-loop, agent kernel, providers, advisor, harness, scout, TUI) is a single ~30 MB binary. Python only runs in two places: (1) a tiny `sniff` shim used once per `automedal init` for pandas-backed CSV schema inference, and (2) the agent's own ML pipeline at `agent/{train,prepare}.py`. No Python in the iteration hot path.

Talks to any OpenAI- or Anthropic-shape provider (opencode-go, Anthropic, OpenAI, OpenRouter, Groq, Ollama, …). All state lives in git-tracked markdown files — the agent itself is stateless.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (same edit → train → check → keep/revert philosophy), extended for Kaggle-style tabular ML.

## Install

Two binaries get installed: **`automedal`** (the headless CLI / control plane) and **`automedal-tui`** (the v2 Bubbletea UI — Hermes-style banner, spring nav, drill-down event stream). They run independently — the TUI tails the same `agent_loop.events.jsonl` the CLI writes.

### One-liner (recommended)

```bash
curl -LsSf https://raw.githubusercontent.com/Flameingmoy/automedal/main/install.sh | bash
```

The installer verifies Go ≥ 1.24, builds **both binaries** into `~/.local/bin`, installs the optional `sniff` Python shim (via `pipx`, falling back to `pip --user`), and creates `~/.automedal/` (mode 0700) for credentials and logs.

Upgrade later (from a checkout — the TUI lives in `./internal/ui`, which `go install` can't target):

```bash
git pull
go install ./cmd/automedal                          # → $GOBIN/automedal
go build  -o ~/.local/bin/automedal-tui ./internal/ui
pipx upgrade automedal-sniff
```

### From source (development)

```bash
git clone https://github.com/Flameingmoy/automedal automedal && cd automedal
go build -o ~/.local/bin/automedal     ./cmd/automedal
go build -o ~/.local/bin/automedal-tui ./internal/ui
pipx install ./py-shim/sniff                        # or: pip install --user ./py-shim/sniff
automedal version          # → automedal 2.0.0-go
automedal-tui --version    # → automedal-tui v2.0.0
```

> **Heads up:** typing `automedal` with no arguments now prints a usage error rather than launching the UI — the TUI is its own binary. Run **`automedal-tui`** to see the live dashboard, or `automedal-tui --screen dashboard` to land directly on the live-events screen.

### Requirements

- **Go** ≥ 1.24 (build-time only; the binary is statically linked).
- **Python** ≥ 3.10 with pandas + numpy (only used by `automedal init` for CSV schema inference). Skip if you only need `automedal run` against an already-bootstrapped competition.
- **NVIDIA GPU** with CUDA (tested on an RTX 4070 Ti Super, 16 GB). CPU-only works but tabular GBMs will be slow.
- **API key** for any [supported provider](#providers) — OpenCode Go recommended (one key unlocks GLM / Kimi / MiMo / MiniMax).
- **Kaggle credentials** at `~/.kaggle/kaggle.json` ([get one here](https://www.kaggle.com/settings)).

### First-run side effects

On first invocation AutoMedal creates:

| Path | Mode | Purpose |
|------|------|---------|
| `~/.automedal/` | 0700 | Per-user root |
| `~/.automedal/.env` | 0600 | Provider API keys (written by `automedal setup`) |

Per-competition artifacts are created by `automedal init <slug>` (see [user project layout](#users-project-directory)).

## Quick Start

```bash
automedal setup                           # 1. paste a provider API key (hidden input)
automedal discover                        # 2. browse ranked active Kaggle competitions
automedal init playground-series-s6e4     # 3. download data + wire up the project
automedal run 50                          # 4. run 50 iterations of the loop
```

That's it. Each iteration runs the four phases (Researcher → Strategist → Experimenter-edit → train → Experimenter-eval → Analyzer), verifies invariants, tags `exp/NNNN`, and — whenever `val_loss` improves — writes a Kaggle-ready CSV to `submissions/`.

Open the [TUI command centre](#tui--command-centre) at any time with **`automedal-tui`** to watch a run live.

## How It Works

Each iteration is a sequence of short, **stateless** LLM calls. No single call lives long enough to hit context limits — that's how the system scales to 100+ experiments without compaction tricks.

```
        ┌──────────────────────────────────────────────────┐
        │    automedal.run_loop  ·  4-phase orchestrator   │
        │    stagnation · dedupe · quick-reject · verify   │
        └──┬──────────┬──────────┬──────────┬──────────┬───┘
           │          │          │          │          │
           ↓          ↓          ↓          ↓          ↓
      ┌────────┐┌──────────┐┌────────────┐┌──────┐┌──────────┐
      │Research││Strategist││Experimenter││Train ││ Analyzer │
      └────────┘└──────────┘└────────────┘└──────┘└──────────┘
           │         │            │                    │
           ↓         ↓            ↓                    ↓
      ┌──────────────────────────────────────────────────────┐
      │           File-based memory  (git-tracked)           │
      │  knowledge.md        — curated KB                    │
      │  experiment_queue.md — next 5 experiments            │
      │  research_notes.md   — arxiv findings                │
      │  journal/NNNN-*.md   — per-experiment record         │
      │  agent_loop.events.jsonl — structured tool-call log  │
      └──────────────────────────────────────────────────────┘
```

| Phase | Trigger | What it does |
|-------|---------|--------------|
| **Researcher** | Stagnation (K non-improving runs) or scheduled cadence | Searches arxiv via a sub-agent fan-out, reads 2–3 abstracts, appends candidate ideas to `research_notes.md` |
| **Strategist** | Empty queue or stagnation | Rewrites `knowledge.md` (capped at 80 cited bullets), plans the next 5 experiments into `experiment_queue.md` with axis-diversity enforcement. Receives a reflective trace of the last 3 experiments (diff + delta) and a learning-value-ranked top-10 journal summary |
| **Experimenter (edit)** | Every iteration | Pops the top pending queue entry, edits `agent/train.py` / `agent/prepare.py`, commits the change |
| **Training** | Every iteration | Fixed wall-clock budget (default 10 min). Runs in a subprocess so the agent can't interfere |
| **Experimenter (eval)** | Every iteration | Parses training output, writes a journal entry with `diff_summary` + `val_loss_delta`, commits or reverts |
| **Analyzer** | Every iteration (default ON) | Compresses the iteration into a one-paragraph lesson and appends to `knowledge.md` |

### Guardrails

| Feature | Default | Env var | Purpose |
|---------|---------|---------|---------|
| **BM25 dedupe** | on | `AUTOMEDAL_DEDUPE` | Skips queue entries whose motivation matches a past journal; bypass with `[force]` in the hypothesis |
| **Quick-reject** | off | `AUTOMEDAL_QUICK_REJECT` | 30-second smoke-train guard aborts clearly-broken configs before burning the full budget |
| **Regression gate** | warn | `AUTOMEDAL_REGRESSION_GATE` | `strict` reverts git tags when val_loss regresses >1% |
| **Analyzer** | on | `AUTOMEDAL_ANALYZER` | Per-iteration knowledge compression |
| **Success criteria** | always | — | Each queue entry carries a measurable target; near-misses (≤1%) trigger one free retry |

### Advisor (Kimi K2.6 second-opinion loop)

Inspired by Anthropic's [advisor strategy](https://claude.com/blog/the-advisor-strategy): the cheap executor (`minimax-m2.7`) drives the loop; a frontier model (`kimi-k2.6` via opencode-go, same `OPENCODE_API_KEY`) is consulted at three junctions only — **stagnation gate** before the Strategist, **knowledge audit** every Nth Analyzer pass, and an opt-in **`consult_advisor` tool** the worker can call (Strategist + Experimenter-edit only, max 1 use per phase). The advisor never calls tools — it returns a short directive the executor weighs.

Off by default. Turn on with the `--advisor` flag (preferred) or the env var:

```bash
automedal run 10 --advisor                 # uses default model (kimi-k2.6)
automedal run 10 --advisor claude-sonnet-4-5   # override model
AUTOMEDAL_ADVISOR=1 automedal run 10       # equivalent env-var form
```

The flag works in the TUI too — type `run 10 --advisor <Tab>` and it autocompletes from the live model list at `<base_url>/models`. Refresh manually with `automedal models refresh`.

| Env var | Default | Purpose |
|---------|---------|---------|
| `AUTOMEDAL_ADVISOR` | `0` | Master on/off |
| `AUTOMEDAL_ADVISOR_MODEL` | `kimi-k2.6` | Advisor model id |
| `AUTOMEDAL_ADVISOR_BASE_URL` | `https://opencode.ai/zen/go/v1` | OpenAI-compatible endpoint |
| `AUTOMEDAL_ADVISOR_JUNCTIONS` | `stagnation,audit,tool` | Allowlist (any subset) |
| `AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT` | `2000` | Per-call output cap |
| `AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER` | `8000` | Hard ceiling per iteration |
| `AUTOMEDAL_ADVISOR_AUDIT_EVERY` | `5` | Knowledge-audit cadence (iterations) |
| `AUTOMEDAL_ADVISOR_STAGNATION_EVERY` | `5` | Periodic stagnation check (iterations) |

Every consult emits an `advisor_consult` JSONL event (`purpose`, `model`, `in_tokens`, `out_tokens`, `skipped`, `preview`) and renders in the TUI event stream. Verify the wiring with `AUTOMEDAL_ADVISOR=1 automedal doctor`.

## TUI — Command Centre

The v2 TUI is its own static binary at `./internal/ui` (built as `automedal-tui`). True-black surface, jade primary, per-phase palette (researcher / strategist / experimenter / analyzer / advisor), spring-physics nav underline, Hermes-style chunky pixel banner with a blue → cyan → jade gradient. Tails `agent_loop.events.jsonl` directly, no subprocess hops, no Python re-imports.

### Run it

```bash
automedal-tui                          # opens the Home screen
automedal-tui --screen dashboard       # land on the live dashboard
automedal-tui --screen timeline        # land on the progression chart
automedal-tui --screen config          # land on the env-var status table
automedal-tui --version                # → automedal-tui v2.0.0
```

If the binary isn't on `$PATH`, build it from the repo root with `go build -o ~/.local/bin/automedal-tui ./internal/ui`. The TUI reads the working-directory's `agent/results.tsv` + `agent_loop.events.jsonl`, so launch it from inside a competition project to see live data; outside one it shows empty panels but still navigates.

### Screens

Every screen carries the same chrome: top spring-nav strip + status bar + footer hint row. Switch screens with the single-letter key in the footer (or click the tab if your terminal supports mouse).

| Screen | What's on it |
|--------|--------------|
| **Home** | Chunky pixel-art `AUTOMEDAL` banner with a blue→cyan→jade column gradient, neofetch-style info table on the right (competition · phase · iter · best_loss · advisor · recent), phase-colour swatches, jade-bordered command palette with Tab autocomplete (incl. `--advisor <Tab>` against the live model cache) |
| **Dashboard** | Stat-card row (phase · experiments · best loss · Δ total · GPU · advisor) → compact running-best sparkline → leaderboard ⏐ live-events split → GPU bar. **Drill-down:** `[` / `]` move focus between events, `space` / `enter` toggles the focused event's expanded body (tool args pretty-printed, advisor consults show purpose + model + tokens, phase transitions show stop reason), `G` jumps to newest + re-enables follow-mode. Stream auto-pins to the newest event until you scroll or focus an older one |
| **Run** | Phase-coloured streaming subprocess log on the left + 30-col side metrics panel (phase / experiments / GPU bar / line count); `q` / `Esc` sends SIGTERM and returns Home |
| **Timeline** | Full-page running-best progression sparkline, four summary cards (baseline · current best · total Δ · success rate), ranked experiment table marking best ★ |
| **Config** | `~/.automedal/.env` + environment status table — set & overridden first (jade dot), defaults next (dim dot), unset last (red ✗). API keys masked to `sk-●●●●●●●●●●●●` |
| **Knowledge** | `knowledge.md` rendered via Glamour, scrollable viewport |
| **Help** | Four-section keymap grid grouped by screen |

### Commands & keybinds

The home command palette accepts: `run [N] [--advisor [model]]`, `init <slug>`, `discover`, `select`, `doctor`, `status`, `clean`, `setup`, `models`, plus `dashboard` / `dash` / `watch`, `timeline` / `tl`, `config` / `cfg`, `knowledge` / `k`, `help`. Quit aliases (`q`, `quit`, `exit`, `:q`, `:quit`, `:wq`) call `tea.Quit` directly — they never spawn a subprocess. Anything else is forwarded to `automedal <cmd>` as a streamed child process.

| Key | Where | Action |
|-----|-------|--------|
| `tab` | Home palette | Autocomplete command (or model after `--advisor`) |
| `enter` | Home palette | Run the typed command (or open screen) |
| `[` / `]` | Dashboard | Move focus prev / next event |
| `space` | Dashboard | Toggle the focused event's expanded body |
| `G` / `end` | Dashboard | Jump to newest event + re-enable follow |
| `↑↓` / PgUp / PgDn | Dashboard / Run | Scroll the viewport |
| `q` / `esc` | Any non-Home screen | Return Home |
| `ctrl+c` | Anywhere | Quit |

## CLI Reference

| Command | Description |
|---------|-------------|
| `automedal-tui` | Open the TUI (Home screen by default; `--screen dashboard\|timeline\|config` to land elsewhere) |
| `automedal setup` | Configure a provider + API key (first-run) |
| `automedal doctor` | Smoke-test the provider + SDK versions + env state |
| `automedal discover` | List and rank active Kaggle competitions |
| `automedal select` | Pick a competition from the ranked list |
| `automedal init <slug>` | Download data, infer schema (via the `sniff` shim), wire up the project |
| `automedal prepare` | Regenerate `.npy` arrays from `data/` |
| `automedal run [N] [--advisor [model]]` | Start the loop (default 50 iterations) |
| `automedal status` | Quick health: knowledge head, last 5 results, latest tags |
| `automedal models [refresh]` | List or refresh the cached advisor model catalogue |
| `automedal clean` | Wipe memory files + `results.tsv` (confirms first) |
| `automedal version` | Print installed version |
| `automedal help` | Print the command list |

`automedal` with no arguments is **not** the TUI anymore — it prints `automedal: no command — try \`automedal help\`` and exits 1. Use the dedicated `automedal-tui` binary instead.

## Providers

One env var picks the provider; one more picks the model. Keys live in `~/.automedal/.env` (mode 0600).

| Provider | Env var (key) | Example `AUTOMEDAL_MODEL` | Notes |
|----------|---------------|---------------------------|-------|
| **OpenCode Go** (default) | `OPENCODE_API_KEY` | `minimax-m2.7` | One key unlocks GLM, Kimi, MiMo, MiniMax. Routes through the Anthropic-shape endpoint |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` | Direct Claude |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` | Direct GPT |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | Free-tier models available; aggregates many providers |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Fast Llama / Mixtral |
| Ollama (local) | — (set `OLLAMA_HOST`) | `llama3.2` | Runs on your own GPU via `http://localhost:11434/v1` |

```bash
AUTOMEDAL_PROVIDER=anthropic AUTOMEDAL_MODEL=claude-sonnet-4-5 automedal run 50
AUTOMEDAL_PROVIDER=ollama    AUTOMEDAL_MODEL=llama3.2          automedal run 10
```

Back-compat: a legacy `MODEL=provider/model-id` slug is still honored and split into the two vars above.

## Configuration

All env vars honored by `automedal run`:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOMEDAL_PROVIDER` | `opencode-go` | Provider name from the table above |
| `AUTOMEDAL_MODEL` | `minimax-m2.7` | Model id for that provider |
| `AUTOMEDAL_ANALYZER` | `1` | `0` disables per-iteration knowledge compression |
| `AUTOMEDAL_QUICK_REJECT` | `0` | `1` enables the 30s smoke-train guard |
| `AUTOMEDAL_DEDUPE` | `1` | `0` disables BM25 motivation dedupe |
| `AUTOMEDAL_DEDUPE_THRESHOLD` | `5.0` | BM25 score threshold (higher = stricter) |
| `STAGNATION_K` | `3` | Trigger Researcher + Strategist after K non-improving runs |
| `RESEARCH_EVERY` | `10` | Scheduled Researcher cadence (`0` disables) |
| `COOLDOWN_SECS` | `1` | Seconds to pause between iterations |
| `TRAIN_BUDGET_MINUTES` | `10` | Training wall-clock limit per experiment |
| `LOG_FILE` | `agent_loop.log` | Human-readable log path |
| `AUTOMEDAL_EVENTS_FILE` | `agent_loop.events.jsonl` | Structured JSONL event sink (TUI reads this) |
| `AUTOMEDAL_REGRESSION_GATE` | `warn` | `strict` reverts experiments that regress >1% |

```bash
STAGNATION_K=5 RESEARCH_EVERY=0 AUTOMEDAL_QUICK_REJECT=1 automedal run 100
```

## Project Structure

Single Go module. Python only inside `py-shim/sniff/` (one-shot CSV schema inference, called by `automedal init`) and inside the user's `agent/` directory (the agent's editable ML pipeline).

```
cmd/
└── automedal/                    `automedal` CLI entry — routes to dispatch
                                  plus local `harness` / `debug` dev verbs

internal/
├── dispatch/                     One function per subcommand (setup, doctor,
│                                 discover, init, run, status, models, …)
├── runloop/                      4-phase orchestrator + dedupe + quickreject + args
├── agent/                        Bespoke agent kernel
│   ├── kernel.go                 Goroutine-based tool-call loop
│   ├── events.go                 JSONL EventSink (TUI tails this)
│   ├── retry.go · errors.go · messages.go · doomloop.go
│   ├── providers/
│   │   ├── anthropic.go          anthropic-sdk-go (Anthropic + opencode-go)
│   │   └── openai.go             openai-go (OpenAI + Ollama + OpenRouter + Groq)
│   ├── tools/
│   │   ├── fs.go                 read/write/edit/list/grep (path-guarded)
│   │   ├── shell.go              run_shell (cwd-bound, timeout, SIGTERM-safe)
│   │   ├── cognition.go          Pure-Go BM25Okapi recall
│   │   ├── arxiv.go              Researcher-only paper search
│   │   ├── subagent.go           spawn_subagent — concurrent via goroutines
│   │   ├── advisor.go            consult_advisor wrapper
│   │   └── plan.go               plan tool
│   ├── phases/                   researcher · strategist · experimenter_* · analyzer
│   └── prompts/*.tmpl            text/template phase prompts (was Jinja2)
├── advisor/                      Kimi-K2.6 second-opinion client + budget + orchestrator
├── harness/                      Deterministic automation (no LLM)
│   ├── stagnation.go · expid.go · memory.go
│   ├── verify.go                 Post-phase invariants + regression + success_criteria
│   ├── trailer.go · rank.go
├── scout/                        Competition discovery + bootstrap
│   ├── discover.go · select.go · scoring.go · bootstrap.go · render.go
│   └── sniffshim.go              Tiny Go wrapper over `python -m sniff`
├── auth/env.go                   ~/.automedal/.env store (godotenv)
├── paths/layout.go               Layout — dev vs user mode resolution
├── config/config.go              Central env-var schema
└── ui/                           v2 Bubbletea TUI (= `automedal-tui` binary)
    ├── main.go                   tea.Program root + spring nav router
    ├── theme/                    Jade palette + per-phase colours + gradient helper
    ├── models/                   home · dashboard · run · timeline · config · help · knowledge
    ├── components/               banner (Hermes pixel font), statusbar, navbar (harmonica),
    │                             phasechip, statcard, eventitem, footer, leaderboard, gpu, …
    ├── events/                   fsnotify-based JSONL tailer + Reduce
    └── proc/                     exec.CommandContext subprocess streamer

py-shim/sniff/                    Pandas-backed CSV schema inference (only Python in the repo)
└── __main__.py · sniff.py        Called once per `automedal init` via `python -m sniff <csv>`
```

### User's project directory

After `automedal init <slug>`, your project looks like this:

```
my-kaggle-project/
├── data/                         raw CSVs + .npy arrays (untracked)
├── submissions/                  Kaggle-ready CSVs (auto-generated)
├── journal/                      NNNN-slug.md per experiment
├── knowledge.md                  Curated KB
├── experiment_queue.md           Next 5 planned experiments
├── research_notes.md             Arxiv findings
├── results.tsv                   Flat experiment log
├── agent_loop.log                Human-readable log
├── agent_loop.events.jsonl       Structured event stream (TUI source)
└── .automedal/                   Hidden harness files
    ├── agent/train.py            Agent-editable model code
    ├── agent/prepare.py          Agent-editable feature pipeline
    ├── configs/competition.yaml
    └── logs/                     (user-mode only)
```

## Available Libraries

Pre-installed for the agent:

| Category | Libraries |
|----------|-----------|
| Gradient Boosting | XGBoost, LightGBM, CatBoost (all GPU-accelerated) |
| Hyperparameter Optimization | Optuna |
| AutoML | FLAML (built-in), AutoGluon (`pipx inject automedal autogluon.tabular`) |
| Deep Learning | PyTorch, TabNet |
| Feature Engineering | category_encoders, scikit-learn |
| Data Augmentation | imbalanced-learn (SMOTE, ADASYN) |
| Research | arxiv (core dep) |

## Harness Invariants

`harness/verify_iteration.py` runs after every phase. Enforcement is soft by default — violations log `WARN:` lines but don't abort. The regression gate and success_criteria retry are opt-in.

| Phase | Invariants |
|-------|------------|
| **Strategist** | `knowledge.md` ≤ 80 bullets; every bullet cites an experiment ID; queue has exactly 5 entries; no axis appears more than twice; every entry has Hypothesis/Sketch/Expected/success_criteria |
| **Researcher** | `research_notes.md` grew by one entry with 2–3 paper bullets + query header |
| **Experimenter** | Journal exists with complete frontmatter; `diff_summary` + `val_loss_delta` present; valid status; KB entries consulted non-empty when KB non-empty; optional regression gate + near-miss retry |

## Running Without Scout

If you already have the data:

```bash
# Place train.csv and test.csv in data/
# Edit .automedal/configs/competition.yaml by hand
automedal render          # regenerate AGENTS.md from the template
automedal prepare         # generate .npy arrays
automedal run 10
```

## Switching Competitions

```bash
automedal init spaceship-titanic
```

Wipes `data/` of the old competition's files, pulls the new data, resets memory, re-renders `AGENTS.md`. Your code in `.automedal/agent/train.py` and `prepare.py` is preserved unless you delete it first. Git history keeps both competitions' progress.

## Design Decisions

- **Two editable files.** The agent edits `agent/train.py` (models, HPO, ensembling) and `agent/prepare.py` (features, encoding, augmentation). Full ML pipeline control, manageable scope.
- **File-based memory over conversational memory.** Every artifact is a git-tracked markdown file. Auto-compaction can't erase `knowledge.md`.
- **Stateless agent calls.** Each phase is a fresh kernel invocation with a short focused prompt. No single call accumulates enough context to degrade.
- **Bespoke kernel, not a framework.** ~300 LOC of async Python + official provider SDKs. No LangChain, no LangGraph, no Node runtime.
- **Deterministic harness, LLM-driven phases.** Stagnation detection, experiment IDs, invariant verification, dedupe — all Python. Planning, curation, research synthesis — all LLM.
- **Fixed time budget.** Each experiment runs for at most 10 minutes, making results directly comparable.
- **GPU-first.** XGBoost `device="cuda"`, LightGBM `device="gpu"`, CatBoost `task_type="GPU"`.
- **Automatic submissions.** Every time `val_loss` improves, a Kaggle-ready CSV is written to `submissions/`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `automedal setup` smoke test fails with "unauthorized" | Key didn't persist. Run `automedal doctor` to check `~/.automedal/.env`, or `export OPENCODE_API_KEY=sk-...` as a fallback |
| `automedal` says "not configured yet" but env var is set | Make sure you exported it in the *same* shell session |
| `scout/bootstrap.py` reports low schema sniff confidence | TUI will prompt; in shell, pass `--yes` to continue or `--abort-on-warning` to abort |
| Strategist queues 5 entries on the same axis | `verify_iteration.py` will warn; fix `experiment_queue.md` by hand or delete it |
| `final_val_loss=` line missing from train.py output | Revert `.automedal/agent/train.py`; the next Experimenter will re-add it |
| Regression gate is reverting good experiments | Set `AUTOMEDAL_REGRESSION_GATE=warn` (default) or check if `best_before` is being read correctly |
| `automedal` (no args) errors out instead of opening the UI | The TUI is a separate binary now — run **`automedal-tui`** |
| TUI shows empty panels | The TUI reads `agent/results.tsv` + `agent_loop.events.jsonl` from the cwd. Launch it from inside a competition project (`cd ~/my-comp && automedal-tui`) |
| TUI shows stale events | `tail -f agent_loop.events.jsonl` to confirm the loop is writing; restart `automedal-tui` if the file was rotated |
| `automedal-tui: command not found` | Build it: `go build -o ~/.local/bin/automedal-tui ./internal/ui` (from a checkout) — the one-liner installer also drops it in `~/.local/bin` |
| TUI banner renders as `â` mojibake | Your terminal isn't using a Unicode/Nerd font; switch to one (e.g. JetBrains Mono, Iosevka, Berkeley Mono) and ensure `LANG=*.UTF-8` |
| `pipx install` fails on git install | Upgrade pipx: `python3 -m pip install --user --upgrade pipx` |

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The four-phase loop, dedupe, quick-reject, bespoke kernel, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT. See [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for attribution of bundled third-party software — Anthropic + OpenAI Go SDKs, the Charmbracelet stack (`bubbletea`, `bubbles`, `lipgloss`, `glamour`, `harmonica`), `fsnotify`, `go-colorful`, `godotenv`, `yaml.v3`, plus `pandas` + `numpy` inside the `py-shim/sniff` shim.
