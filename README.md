<p align="center">
  <h1 align="center">AutoMedal</h1>
  <p align="center">
    Autonomous ML research agent for Kaggle competitions
    <br />
    Point it at a competition. Wake up to a leaderboard-climbing submission.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#tui--command-centre">TUI</a> &bull;
  <a href="#providers">Providers</a> &bull;
  <a href="#configuration">Configuration</a> &bull;
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

AutoMedal is an autonomous experiment loop for tabular ML competitions. It uses a coding agent to try different models, feature engineering, hyperparameters, ensembles, and literature-inspired ideas — keeping only what improves the score.

The harness is **agent-agnostic**: the default stack is [pi coding agent](https://github.com/badlogic/pi-mono) driving MiniMax M2.7 through an [OpenCode Go](https://opencode.ai) API key, but any provider pi supports (OpenRouter, Ollama, Anthropic, OpenAI, Groq, Gemini) works with zero code changes.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — same philosophy (edit, train, check, keep/revert), but targeting tabular ML competitions instead of LLM pretraining.

## How It Works

Each iteration of the loop dispatches up to three agent phases. Each phase is a **separate, stateless agent call** with a small focused prompt — no single call lives long enough to hit context limits, which is how the system scales to 100+ experiments.

```
             ┌──────────────────────────────────────────────┐
             │              run.sh  (loop)                  │
             │   stagnation · tag · log · dispatch phases   │
             └──────────┬──────────┬───────────┬────────────┘
                        │          │           │
             on empty   │ on       │ every     │
             queue or   │ stagnation│ iteration│
             stagnation ↓          ↓           ↓
                  ┌─────────┐┌──────────┐┌──────────────┐
                  │Strategist││Researcher││ Experimenter │
                  └─────────┘└──────────┘└──────────────┘
                       │         │              │
        reads & writes │ reads   │ reads        │ writes
                       ↓         ↓              ↓
         ┌─────────────────────────────────────────────┐
         │         File-based memory  (git-tracked)    │
         │  knowledge.md       — curated KB            │
         │  experiment_queue.md — next 5 experiments   │
         │  research_notes.md  — arxiv findings        │
         │  journal/NNNN-*.md  — per-experiment record │
         │  results.tsv        — flat log (untracked)  │
         └─────────────────────────────────────────────┘
```

| Phase | Trigger | What it does |
|-------|---------|--------------|
| **Researcher** | Stagnation (K non-improving runs) or scheduled cadence | Queries arxiv, reads 2-3 abstracts, appends candidate ideas to `research_notes.md` |
| **Strategist** | Empty queue or stagnation | Rewrites `knowledge.md` (capped at 80 cited bullets), plans the next 5 experiments into `experiment_queue.md` with axis-diversity enforcement. Receives a **reflective trace** of the last 3 experiments (diff + delta) and a **learning-value ranked** top-10 journal summary — so it reads the most informative past experiments first, not just the most recent |
| **Experimenter** | Every iteration | Pops the top pending queue entry, edits `agent/train.py` / `agent/prepare.py`, runs training, writes a journal entry with `diff_summary` + `val_loss_delta`, commits or reverts. If a `success_criteria` target was set and the result misses it by ≤ 1%, one targeted retry edit is attempted automatically |

State lives in **git-tracked markdown files**, not in conversation memory. The agent is stateless; the files are state.

### Self-improvement features (Phase 3)

| Feature | Where |
|---------|-------|
| **Reflective trace** — last 3 journal diffs + deltas injected into Strategist context | `harness/build_trace_trailer.py` |
| **Learning-value ranking** — journals scored by outcome + signal strength + axis diversity; top-10 sent to Strategist | `harness/rank_journals.py` |
| **`success_criteria`** — each queue entry carries a measurable target; near-miss triggers one free retry | `harness/verify_iteration.py` |
| **Regression gate** — `AUTOMEDAL_REGRESSION_GATE=strict` reverts git tags when val_loss regresses >1% | `harness/verify_iteration.py` |
| **Memory compaction** — `automedal compact` distils `research_notes.md` when it grows large | `harness/compact_memory.py` |

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4070 Ti Super, 16 GB VRAM)
- **Python**: 3.10 - 3.12
- **Node.js**: ≥ 22 (for the bundled pi coding agent — auto-installed on first run)
- **[uv](https://docs.astral.sh/uv/)**: Python dependency management
- **API key** for any [supported provider](#providers) (OpenCode Go recommended)
- **Kaggle credentials** at `~/.kaggle/kaggle.json` ([get one here](https://www.kaggle.com/settings))

```bash
pip install automedal        # or: pipx install automedal
# pi is auto-installed from npm on first run — no separate npm install needed
```

For development (running from this repo):

```bash
pip install -e .             # editable install — `automedal` becomes available globally
# or with uv:
uv sync
# Optional extras:
uv sync --extra research   # arxiv lookups for the Researcher phase
uv sync --extra automl     # AutoGluon for the Experimenter
```

`./am` is a convenience wrapper equivalent to the `automedal` command — use either when developing from source.

## Quick Start

### 1. First-run setup

```bash
automedal setup
```

Interactive TUI wizard: choose a provider, paste your API key (hidden input), stores it in pi's auth config at `~/.pi/agent/auth.json`, and runs a smoke test. Any `automedal` command besides `setup`, `help`, and `doctor` is gated behind this step.

If you prefer to manage keys yourself, export the env var directly (e.g. `export OPENCODE_API_KEY=sk-...`) and skip `automedal setup` — env vars take precedence.

### 2. Find and bootstrap a competition

```bash
automedal discover                          # browse ranked active competitions in TUI
automedal select                            # pick one from the TUI DataTable
automedal init playground-series-s6e4       # or pass a slug directly
```

### 3. Run the loop

```bash
automedal run 50      # 50 iterations of the three-phase loop
```

That's it. Each iteration: check stagnation → maybe Researcher → maybe Strategist → always Experimenter → verify invariants → tag `exp/NNNN`. Every improvement auto-generates a Kaggle-ready submission CSV in `submissions/`.

## TUI — Command Centre

`automedal` with no arguments opens the TUI home screen. You can also type any command directly into its command palette instead of using the shell.

```
┌─ AutoMedal · playground-s6e4 · val_loss 0.0503 · iter 24/50 ─────┐
│                                                                    │
│  ● recent activity                                                 │
│    #24 irredundant-kfold-hpo   ✓  0.0503  (-0.0001)               │
│    #23 catboost-depth-tune     ✗  0.0508                          │
│    #22 lgbm-bagging            ✓  0.0504  (-0.0014)               │
│                                                                    │
│  [r] run 50  [d] discover  [i] init  [s] status  [q] quit         │
│                                                                    │
│  > _                                    (type command, Enter)      │
└────────────────────────────────────────────────────────────────────┘
```

Type a command and press Enter, or use the quick-action keys:

| Key / command | Action |
|---|---|
| `run [N]` / `r` | Launch N iterations → pushes live Dashboard |
| `discover` / `d` | Ranked competition DataTable + s=select |
| `select` | Native DataTable picker → bootstrap selected competition |
| `init <slug>` / `i` | Staged progress screen (✓ per step) while bootstrapping |
| `status` / `s` | Full status screen: leaderboard + recent activity + queue |
| `setup` | Provider wizard with hidden API-key input |
| `doctor` | Diagnostic checklist (Node, pi, auth, smoke test) |
| `clean` | Confirmation modal before wiping memory |
| `compact [file]` | Condense `research_notes.md` when it grows large |
| `q` | Quit |

### Live Dashboard (during a run)

Pushing `run N` from the home screen launches the existing three-panel dashboard over the home screen. `ctrl+c` sends SIGTERM to the subprocess — the harness finishes the current iteration cleanly, then pops back to home.

```bash
automedal tui           # equivalent to bare automedal
automedal tui --demo    # replay a fixture log — no live run required
```

Dashboard panels: phase sprite, val_loss sparkline, top-5 leaderboard with medals, scrollable experiment log, current-experiment card (hypothesis + budget bar + live loss), GPU util/VRAM/temp, session totals, and a phase-colored live log stream. `ctrl+o` flips to a full-screen raw stream.

### Custom sprites

The phase sprite panel shows a generated geometric glyph for each phase (RESEARCH, CODING, EXPERIMENT, SUBMITTING, IDLE, FROZEN). You can replace these with your own pixel art:

**Sprite directory:** `~/.automedal/sprites/<theme>/<phase>/<size>/frame_NN.png`

- `<theme>` — currently `dark` (matches the TUI theme)
- `<phase>` — one of: `research`, `coding`, `experiment`, `submitting`, `idle`, `frozen`
- `<size>` — one of: `16`, `24`, `32` (pixels — the sprite panel is 22 columns wide, so `24` is the default)
- `frame_NN.png` — `frame_00.png` and `frame_01.png` (two frames, cycled every heartbeat)

**Quick recipe:**

```
~/.automedal/sprites/
└── dark/
    ├── research/
    │   └── 24/
    │       ├── frame_00.png    ← idle frame (24×24 RGBA PNG)
    │       └── frame_01.png    ← animated frame
    ├── coding/24/frame_00.png
    ├── experiment/24/frame_00.png
    ├── submitting/24/frame_00.png
    ├── idle/24/frame_00.png
    └── frozen/24/frame_00.png
```

Drop any `frame_00.png` (and optionally `frame_01.png`) into the right folder and restart the TUI — it will use your PNG automatically. If the file doesn't exist, the auto-generated geometric glyph is shown instead. Pillow + rich-pixels must be installed (`uv sync --extra tui`).

**Medal sprites** (shown in the leaderboard) follow the same pattern:

```
~/.automedal/sprites/dark/medal/
├── gold_24.png
├── silver_24.png
└── bronze_24.png
```

## CLI Reference

All commands work from both the TUI command palette and the shell:

| Command | Description |
|---------|-------------|
| `automedal` / `automedal tui` | Open TUI home screen (command palette + dashboard) |
| `automedal setup` | Configure a model provider (first-run) |
| `automedal doctor` | Diagnose pi, provider, and env state |
| `automedal discover` | List and rank active Kaggle competitions |
| `automedal select` | Pick a competition from a DataTable |
| `automedal init <slug>` | Download data, infer schema, wire up a competition |
| `automedal prepare` | Regenerate `.npy` arrays from `data/` |
| `automedal render` | Re-render `AGENTS.md` and `program.md` from templates |
| `automedal run [N]` | Start the three-phase loop (default 50 iterations) |
| `automedal status` | Health check: knowledge, results, latest experiment tags |
| `automedal clean` | Wipe memory files + results (prompts to confirm) |
| `automedal compact [file]` | Condense a memory file when it exceeds 40 KB |

> Legacy: `./am <command>` still works — it's a thin shim that calls `python -m automedal`.

## Providers

Because pi is provider-agnostic, so is AutoMedal. Switch at runtime with a single env var — no code changes:

| Provider | Env var | Example model slug | Notes |
|----------|---------|-------------------|-------|
| **OpenCode Go** (default) | `OPENCODE_API_KEY` | `opencode-go/minimax-m2.7` | One key unlocks GLM-5.1, Kimi K2.5, MiMo, MiniMax M2.5/2.7 |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/<model>` | Free-tier models available; aggregates many providers |
| Ollama (local) | — | `ollama/<local-model>` | No key, no cloud — runs on your own GPU |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-5` | Direct Claude access |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` | Direct GPT access |
| Groq | `GROQ_API_KEY` | `groq/<model>` | Fast inference |
| Google Gemini | `GEMINI_API_KEY` | `google/gemini-*` | |

```bash
MODEL="openrouter/free-model" automedal run 50
```

## Configuration

`run.sh` (and therefore `automedal run`) honors these env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `opencode-go/minimax-m2.7` | Any `<provider>/<model-id>` slug pi recognizes |
| `STAGNATION_K` | `3` | Trigger Researcher + Strategist after K non-improving runs |
| `RESEARCH_EVERY` | `10` | Scheduled Researcher cadence (set `0` to disable) |
| `COOLDOWN_SECS` | `1` | Seconds to pause between iterations (0 to disable) |
| `TRAIN_BUDGET_MINUTES` | `10` | Training wall-clock limit per experiment (minutes) |
| `LOG_FILE` | `agent_loop.log` | Path for the combined loop log |
| `AUTOMEDAL_REGRESSION_GATE` | `warn` | Set to `strict` to auto-revert experiments that regress >1% |

```bash
STAGNATION_K=5 RESEARCH_EVERY=0 MODEL="ollama/qwen3:32b" automedal run 100
AUTOMEDAL_REGRESSION_GATE=strict automedal run 50
```

## Project Structure

```
automedal/               Installed package (entry point + dispatch)
├── cli.py               `automedal` console script entry point
├── dispatch.py          One Python function per subcommand
├── paths.py             Layout class — resolves all paths for dev/user mode
├── pi_runtime.py        ensure_pi() — detects/installs pi into _vendor/
└── _vendor/             pi npm package (auto-populated on first run, gitignored)

run.sh                   Three-phase automation loop (invoked by dispatch)
config_loader.py         Loads configs/competition.yaml

harness/                 Deterministic automation (Python, no LLM)
├── check_stagnation.py  K-run stagnation detector + best_val_loss()
├── next_exp_id.py       Zero-padded NNNN experiment ID allocator
├── init_memory.py       Creates/resets memory files on bootstrap
├── verify_iteration.py  Post-phase invariant checker + regression gate + success_criteria
├── build_trace_trailer.py  Reflective trace builder (last N journals → markdown block)
├── rank_journals.py     Learning-value ranker (score + diversity → top-K summary)
├── compact_memory.py    Memory compaction (size check → pi call → archive)
└── stream_events.py     Pi JSON event stream → terminal output

scout/                   Competition discovery + bootstrap pipeline
├── discover.py          List and rank active Kaggle competitions
├── select.py            Non-interactive pick_and_bootstrap() + CLI fallback
├── bootstrap.py         Download data, infer schema, wire everything up
├── sniff.py             CSV schema inference (target, features, task type)
├── scoring.py           Two-stage competition scoring heuristics
└── render.py            Jinja2 template rendering for AGENTS.md / program.md

prompts/                 Phase contracts (static, competition-agnostic)
├── strategist.md        Planning: KB curation + experiment queue + success_criteria
├── researcher.md        Arxiv research
├── experimenter.md      Implementation: code, train, journal + diff_summary/val_loss_delta
├── experimenter_eval.md Eval: parse result, write journal, commit/revert
└── compactor.md         Memory compaction: distil a large markdown file

tui/                     Textual TUI (home screen + dashboard + per-command screens)
├── app.py               App: sources → bus → state reducer → screens + _custom_screen()
├── state.py             Pure PhaseMachine.reduce((state, event)) → state
├── events.py            Event dataclasses
├── sources/             Async producers: log_tail, journal, results, memory, gpu, demo
├── screens/
│   ├── home.py          Landing page: StatusStrip + RecentActivity + CommandInput
│   ├── dashboard.py     Live monitoring: sprite + sparkline + leaderboard + stream
│   ├── select_competition.py  DataTable picker (replaces input() in select.py)
│   ├── setup_wizard.py  RadioSet + Input(password=True) wizard (replaces getpass)
│   ├── discover.py      Streaming discover + result DataTable + s=select
│   ├── status.py        Status overview: leaderboard + recent activity + queue
│   ├── init_progress.py Staged ✓/✗/⏳ checklist while bootstrap runs
│   ├── confirm_clean.py ModalScreen y/n confirmation
│   ├── doctor.py        Diagnostic checklist with background workers
│   ├── command_output.py  Generic subprocess streamer (fallback)
│   └── raw_stream.py    Full-screen raw log stream
├── widgets/
│   ├── command_input.py   Input + Tab autocomplete
│   ├── status_strip.py    Competition slug + phase + best val_loss + iter counter
│   ├── recent_activity.py Last 5 experiments with status glyph + delta
│   ├── queue_summary.py   Next 5 pending queue entries
│   ├── leaderboard.py     Top-5 by val_loss with medals
│   └── ... (sprite, metric_chart, experiment_log, current_experiment, session_stats, gpu_stats, live_stream)
├── themes/              Palette + dark.tcss
├── assets/sprites/      text_fallback.py (ASCII glyphs when Pillow unavailable)
└── sprite_loader.py     load_sprite() — PNG from ~/.automedal/sprites/ or generated glyph

tests/                   pytest suite (62 tests)
├── test_paths.py        Layout dev/user mode path resolution
├── test_pi_runtime.py   ensure_pi() mock-npm tests
├── test_phase_machine.py  State reducer
├── test_log_parser.py   Log line parser
└── test_sprite_loader.py  Sprite fallback
```

### User's project directory (after `automedal init <slug>`)

```
my-kaggle-project/
├── data/                     raw CSVs + .npy arrays (untracked)
├── submissions/              Kaggle-ready CSVs (untracked, auto-generated)
├── journal/                  NNNN-slug.md per experiment
├── knowledge.md              Curated KB (Strategist rewrites each pass)
├── experiment_queue.md       Next 5 planned experiments with success_criteria
├── research_notes.md         Arxiv findings
├── results.tsv               Flat experiment log
└── .automedal/               Hidden harness files
    ├── agent/
    │   ├── train.py          LLM-editable model code
    │   └── prepare.py        LLM-editable feature pipeline
    ├── configs/competition.yaml
    └── logs/agent_loop.log
```

## Available Libraries

Pre-installed and ready for the agent to use:

| Category | Libraries |
|----------|-----------|
| Gradient Boosting | XGBoost, LightGBM, CatBoost (all GPU-accelerated) |
| Hyperparameter Optimization | Optuna |
| AutoML | FLAML (built-in), AutoGluon (`--extra automl`) |
| Deep Learning | PyTorch, TabNet |
| Feature Engineering | category_encoders, scikit-learn |
| Data Augmentation | imbalanced-learn (SMOTE, ADASYN) |
| Research | arxiv (`--extra research`) |

## Harness Invariants

`harness/verify_iteration.py` runs after every phase and prints `WARN:` lines on violation. Enforcement is soft by default — violations log a warning but don't abort the loop. The regression gate and success_criteria retry are opt-in.

| Phase | Invariants checked |
|-------|--------------------|
| **Strategist** | `knowledge.md` ≤ 80 bullets; every bullet cites an experiment ID; queue has exactly 5 entries; no axis appears more than twice; every entry has Hypothesis/Sketch/Expected/success_criteria |
| **Researcher** | `research_notes.md` grew by one entry with 2-3 paper bullets and a query header |
| **Experimenter** | Journal exists with complete frontmatter (incl. `diff_summary`, `val_loss_delta`); valid status; required sections present; KB entries consulted non-empty when KB is non-empty; optional regression gate + success_criteria near-miss retry |

## Running Without Scout

If you already have the data and just want to run the loop:

```bash
# Place train.csv and test.csv in data/
# Edit configs/competition.yaml by hand (task type, target column, features)
automedal render    # regenerate AGENTS.md + program.md from templates
automedal prepare   # generate .npy arrays
automedal run 10
```

## Switching Competitions

```bash
automedal init spaceship-titanic
```

This wipes `data/` of the old competition's files, pulls the new data, resets the memory, and re-renders `AGENTS.md`. Your code in `.automedal/agent/train.py` and `.automedal/agent/prepare.py` is regenerated from the starter template **only if you delete them first** — otherwise bootstrap preserves your current code. Git history keeps both competitions' progress.

## Design Decisions

- **Two editable files.** The agent edits `agent/train.py` (models, HPO, ensembling) and `agent/prepare.py` (features, encoding, augmentation). Full ML pipeline control, manageable scope.
- **File-based memory over conversational memory.** Every artifact is a git-tracked markdown file. Auto-compaction can't erase `knowledge.md`.
- **Stateless agent calls.** Three separate invocations with small prompts. No single call accumulates enough context to degrade.
- **Deterministic harness, LLM-driven phases.** Stagnation detection, experiment IDs, invariant verification, learning-value ranking — all Python. Planning, curation, research synthesis — all LLM.
- **Fixed time budget.** Each experiment runs for at most 10 minutes, making results directly comparable.
- **GPU-first.** XGBoost `device="cuda"`, LightGBM `device="gpu"`, CatBoost `task_type="GPU"`. 16 GB VRAM utilized aggressively.
- **Automatic submissions.** Every time `val_loss` improves, a Kaggle-ready CSV is written to `submissions/`.
- **TUI-first UX.** `automedal` bare opens a command centre — no flags, no manuals. Every interactive command (`select`, `setup`, `clean`) has a native Textual screen that doesn't block stdin.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `automedal setup` smoke test fails with "unauthorized" | Key didn't persist. Run `automedal doctor` to check `~/.pi/agent/auth.json`, or `export OPENCODE_API_KEY=sk-...` as a fallback |
| `automedal` says "not configured yet" but env var is set | Make sure you exported it in the *same* shell session |
| `scout/bootstrap.py` reports low schema sniff confidence | TUI will prompt; in shell, pass `--yes` to continue or `--abort-on-warning` to abort |
| Strategist queues 5 entries on the same axis | `verify_iteration.py` will warn; fix `experiment_queue.md` by hand or delete it |
| Researcher can't import `arxiv` | `uv sync --extra research`, or `RESEARCH_EVERY=0` to skip |
| `final_val_loss=` line missing from train.py output | Revert `.automedal/agent/train.py`; the next Experimenter will re-add it |
| Agent output invisible in terminal | Ensure `run.sh` uses `--mode json` (default since v3.1) |
| `research_notes.md` is getting very large | Run `automedal compact` — it distils the file to ≤40% its size using the compactor prompt |
| Regression gate is reverting good experiments | Set `AUTOMEDAL_REGRESSION_GATE=warn` (default) or check if `best_before` is being read correctly |

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The three-phase harness, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT
