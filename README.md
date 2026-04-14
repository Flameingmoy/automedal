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
         │  agent/results.tsv  — flat log (untracked)  │
         └─────────────────────────────────────────────┘
```

| Phase | Trigger | What it does |
|-------|---------|--------------|
| **Researcher** | Stagnation (K non-improving runs) or scheduled cadence | Queries arxiv, reads 2-3 abstracts, appends candidate ideas to `research_notes.md` |
| **Strategist** | Empty queue or stagnation | Rewrites `knowledge.md` (capped at 80 cited bullets), plans the next 5 experiments into `experiment_queue.md` with axis-diversity enforcement |
| **Experimenter** | Every iteration | Pops the top pending queue entry, edits `agent/train.py` / `agent/prepare.py`, runs training, writes a journal entry, commits or reverts |

State lives in **git-tracked markdown files**, not in conversation memory. The agent is stateless; the files are state.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4070 Ti Super, 16 GB VRAM)
- **Python**: 3.10 - 3.12
- **[uv](https://docs.astral.sh/uv/)**: Python dependency management
- **[pi coding agent](https://github.com/badlogic/pi-mono)**: `npm install -g @mariozechner/pi-coding-agent`
- **API key** for any [supported provider](#providers) (OpenCode Go recommended)
- **Kaggle credentials** at `~/.kaggle/kaggle.json` ([get one here](https://www.kaggle.com/settings))

```bash
uv sync

# Optional extras:
uv sync --extra research   # arxiv lookups for the Researcher phase
uv sync --extra automl     # AutoGluon for the Experimenter

# Or both:
uv sync --extra research --extra automl
```

> If you skip `--extra research`, set `RESEARCH_EVERY=0` so the Researcher phase doesn't try to import `arxiv`.

## Quick Start

### 1. First-run setup

```bash
./am setup
```

Interactive prompt: defaults to OpenCode Go, takes the API key (hidden input), stores it in pi's auth config at `~/.pi/agent/auth.json`, and runs a smoke test. Any `./am` command besides `setup`, `help`, and `doctor` is gated behind this step.

If you prefer to manage keys yourself, export the env var directly (e.g. `export OPENCODE_API_KEY=sk-...`) and skip `./am setup` — env vars take precedence.

### 2. Find and bootstrap a competition

```bash
./am discover                         # list + rank active Kaggle competitions
./am select                           # pick one interactively
./am bootstrap playground-series-s6e4 # or pass a slug directly
```

### 3. Run the loop

```bash
./am run 50      # 50 iterations of the three-phase loop
```

That's it. Each iteration: check stagnation -> maybe Researcher -> maybe Strategist -> always Experimenter -> verify invariants -> tag `exp/NNNN`. Every improvement auto-generates a Kaggle-ready submission CSV in `submissions/`.

## CLI Reference

All commands go through the `./am` wrapper:

| Command | Description |
|---------|-------------|
| `./am setup` | Configure a model provider (first-run) |
| `./am doctor` | Diagnose pi, provider, and env state |
| `./am discover` | List and rank active Kaggle competitions |
| `./am select` | Pick a competition interactively |
| `./am bootstrap <slug>` | Download data, infer schema, wire up a competition |
| `./am prepare` | Regenerate `.npy` arrays from `data/` |
| `./am render` | Re-render `AGENTS.md` and `program.md` from templates |
| `./am run [N]` | Start the three-phase loop (default 50 iterations) |
| `./am status` | Health check: knowledge, results, latest experiment tags |
| `./am clean` | Wipe memory files + results (prompts to confirm) |

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
MODEL="openrouter/free-model" ./am run 50
```

## Configuration

`run.sh` (and therefore `./am run`) honors these env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `opencode-go/minimax-m2.7` | Any `<provider>/<model-id>` slug pi recognizes |
| `STAGNATION_K` | `3` | Trigger Researcher + Strategist after K non-improving runs |
| `RESEARCH_EVERY` | `10` | Scheduled Researcher cadence (set `0` to disable) |
| `COOLDOWN_SECS` | `1` | Seconds to pause between iterations (0 to disable) |
| `TRAIN_BUDGET_MINUTES` | `10` | Training wall-clock limit per experiment (minutes) |
| `LOG_FILE` | `agent_loop.log` | Path for the combined loop log |

```bash
STAGNATION_K=5 RESEARCH_EVERY=0 MODEL="ollama/qwen3:32b" ./am run 100
```

## Project Structure

```
am                       CLI wrapper (setup, bootstrap, run, status, doctor, clean)
run.sh                   Three-phase automation loop
config_loader.py         Loads configs/competition.yaml (shared by agent scripts)

agent/                   The agent's editable sandbox
├── train.py             Training: models, HPO, ensembling, submission generation
├── prepare.py           Data pipeline: loading, cleaning, features, encoding
├── program.md           Pointer to per-phase prompts (rendered from template)
└── results.tsv          Flat experiment log (appended by train.py, untracked)

prompts/                 Phase contracts (static, competition-agnostic)
├── strategist.md        Planning: KB curation + experiment queue
├── researcher.md        Arxiv research
└── experimenter.md      Implementation: code, train, journal

harness/                 Deterministic automation (Python, no LLM)
├── check_stagnation.py  K-run stagnation detector
├── next_exp_id.py       Zero-padded NNNN experiment ID allocator
├── init_memory.py       Creates/resets memory files on bootstrap
├── verify_iteration.py  Post-phase invariant checker
└── stream_events.py     Pi JSON event stream -> terminal output

scout/                   Competition discovery + bootstrap pipeline
├── discover.py          List and rank active Kaggle competitions
├── select.py            Interactive terminal picker
├── bootstrap.py         Download data, infer schema, wire everything up
├── sniff.py             CSV schema inference (target, features, task type)
├── scoring.py           Two-stage competition scoring heuristics
└── render.py            Jinja2 template rendering for AGENTS.md / program.md

knowledge.md             Curated KB (rewritten by Strategist each planning pass)
experiment_queue.md      Next 5 planned experiments (written by Strategist)
research_notes.md        Arxiv findings (appended by Researcher)
journal/                 One NNNN-slug.md per experiment (written by Experimenter)

configs/competition.yaml Single source of truth for the active competition
templates/               Jinja2 templates (AGENTS.md.j2, program.md.j2, prepare_starter.py.j2)
data/                    Competition inputs: raw CSVs + .npy arrays (untracked)
submissions/             Kaggle-ready CSVs (auto-generated on improvement, untracked)
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

`harness/verify_iteration.py` runs after every phase and prints `WARN:` lines on violation. Enforcement is soft — violations log a warning but don't abort the loop.

| Phase | Invariants checked |
|-------|--------------------|
| **Strategist** | `knowledge.md` <= 80 bullets; every bullet (outside Open Questions) cites an experiment ID; `experiment_queue.md` has exactly 5 entries; no axis appears more than twice; every entry has Hypothesis/Sketch/Expected; valid status transitions |
| **Researcher** | `research_notes.md` grew by one entry with 2-3 paper bullets and a query header |
| **Experimenter** | `journal/NNNN-*.md` exists with complete frontmatter; valid status (`improved`/`no_change`/`worse`/`crashed`); required sections present; KB entries consulted non-empty when KB is non-empty |

## Running Without Scout

If you already have the data and just want to run the loop:

```bash
# Place train.csv and test.csv in data/
# Edit configs/competition.yaml by hand (task type, target column, features)
./am render       # regenerate AGENTS.md + program.md from templates
./am prepare      # generate .npy arrays
./am run 10
```

## Switching Competitions

```bash
./am bootstrap spaceship-titanic
```

This wipes `data/` of the old competition's files, pulls the new data, resets the memory, and re-renders `AGENTS.md`. Your code in `agent/train.py` and `agent/prepare.py` is regenerated from the starter template **only if you delete them first** — otherwise bootstrap preserves your current code. Git history keeps both competitions' progress.

## Design Decisions

- **Two editable files.** The agent edits `agent/train.py` (models, HPO, ensembling) and `agent/prepare.py` (features, encoding, augmentation). Full ML pipeline control, manageable scope.
- **File-based memory over conversational memory.** Every artifact is a git-tracked markdown file. Auto-compaction can't erase `knowledge.md`.
- **Stateless agent calls.** Three separate invocations with small prompts. No single call accumulates enough context to degrade.
- **Deterministic harness, LLM-driven phases.** Stagnation detection, experiment IDs, invariant verification — all Python. Planning, curation, research synthesis — all LLM.
- **Fixed time budget.** Each experiment runs for at most 10 minutes, making results directly comparable.
- **GPU-first.** XGBoost `device="cuda"`, LightGBM `device="gpu"`, CatBoost `task_type="GPU"`. 16 GB VRAM utilized aggressively.
- **Automatic submissions.** Every time `val_loss` improves, a Kaggle-ready CSV is written to `submissions/`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `./am setup` smoke test fails with "unauthorized" | Key didn't persist. Run `./am doctor` to check `~/.pi/agent/auth.json`, or `export OPENCODE_API_KEY=sk-...` as a fallback |
| `./am` says "not configured yet" but env var is set | Make sure you exported it in the *same* shell session |
| `scout/bootstrap.py` reports low schema sniff confidence | Edit `configs/competition.yaml` by hand, set `human_verified: true`, re-run `./am render` |
| Strategist queues 5 entries on the same axis | `verify_iteration.py` will warn; fix `experiment_queue.md` by hand or delete it |
| Researcher can't import `arxiv` | `uv sync --extra research`, or `RESEARCH_EVERY=0` to skip |
| `final_val_loss=` line missing from train.py output | Revert `agent/train.py`; the next Experimenter will re-add it |
| Agent output invisible in terminal | Ensure `run.sh` uses `--mode json` (default since v3.1) |

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The three-phase harness, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT
