# AutoMedal

Autonomous ML research for Kaggle competitions. The harness is agent-agnostic — default stack is [pi coding agent](https://github.com/badlogic/pi-mono) driving MiniMax M2.7 through an [opencode-go](https://opencode.ai) API key, but any provider pi supports (OpenRouter free tier, Ollama local, Anthropic, OpenAI, Groq, Gemini) works with zero code changes. Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — same philosophy (edit, train, check, keep/revert), but targeting tabular ML competitions instead of LLM pretraining.

Point it at any active Kaggle competition, and it will experiment autonomously — trying different models, feature engineering, hyperparameters, ensembles, and literature-inspired ideas — keeping only what improves the score. You wake up to a curated knowledge base, a journal of every experiment, and (hopefully) a leaderboard-climbing submission.

## What's new in v2 — the three-phase harness

v1 was a single-loop agent re-reading `results.tsv` every iteration. It stagnated in ensemble tweaks because context reset every cycle. v2 splits the loop into three distinct phases with a **file-based long-term memory**, so the agent plans ahead, curates what it's learned, and pulls in new ideas from arxiv when it plateaus.

```
             ┌──────────────────────────────────────────────┐
             │              run.sh (loop)                   │
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
         │         File-based memory (git-tracked)     │
         │  knowledge.md       — curated KB            │
         │  experiment_queue.md — next 5 experiments   │
         │  research_notes.md  — arxiv findings        │
         │  journal/NNNN-*.md  — per-experiment record │
         │  results.tsv        — flat log (untracked)  │
         └─────────────────────────────────────────────┘
```

- **Strategist** rewrites `knowledge.md` (capped at 80 bullets, every bullet cites an experiment) and plans the next 5 experiments into `experiment_queue.md` with axis-diversity enforcement (no more than 2 entries on the same axis, so the agent can't spend 7 iterations tuning ensemble weights).
- **Researcher** fires on stagnation (K=3 non-improving runs) or every 10 iterations — queries arxiv, reads 2-3 abstracts, appends candidate ideas to `research_notes.md`.
- **Experimenter** pops the top pending queue entry, edits `train.py` / `prepare.py`, runs training, writes a `journal/NNNN-slug.md` entry, and commits or reverts code.

The three phases run as **separate stateless agent invocations** with small focused prompts (`pi --no-session ...`). No single call lives long enough to hit context auto-compaction, which is how v2+ scales to 100+ experiments without losing state.

Design doc: `docs/superpowers/specs/2026-04-10-automedal-harness-v2-design.md`

## Project Structure

```
prepare.py              — Data pipeline: loading, cleaning, features, encoding (agent-editable)
train.py                — Training sandbox: models, HPO, ensembling, submission gen (agent-editable)
config_loader.py        — Loads configs/competition.yaml (shared by prepare.py and train.py)
run.sh                  — Three-phase automation loop (dispatches the pi coding agent)
am                      — Convenience wrapper: ./am setup / bootstrap / run / status / doctor

prompts/
├── strategist.md       — Planning phase contract (static, competition-agnostic)
├── researcher.md       — Arxiv research phase contract (static)
└── experimenter.md     — Implementation phase contract (static)

harness/
├── check_stagnation.py — Deterministic K-run stagnation detector
├── next_exp_id.py      — Zero-padded NNNN allocator
├── init_memory.py      — Creates/resets memory files on bootstrap
└── verify_iteration.py — Post-phase invariant checker

knowledge.md            — Curated KB (rewritten by Strategist every planning pass)
experiment_queue.md     — Next 5 experiments (written by Strategist)
research_notes.md       — Arxiv findings (appended by Researcher)
journal/                — One NNNN-slug.md per experiment (written by Experimenter)

configs/competition.yaml — Single source of truth for the active competition
templates/              — Jinja2 templates for AGENTS.md, program.md, prepare.py
scout/                  — Competition discovery, selection, and bootstrap pipeline

results.tsv             — Flat experiment log (appended by train.py; untracked)
analysis.ipynb          — Experiment tracking and visualization
pyproject.toml          — Dependencies (managed by uv)

data/                   — Competition inputs: raw CSVs + preprocessed .npy arrays
                          + encoder metadata (per-competition, untracked)
submissions/            — Kaggle-ready submission CSVs (per-competition, untracked;
                          auto-generated by train.py when val_loss improves)
```

## Requirements

- NVIDIA GPU (tested on RTX 4070 Ti Super, 16GB VRAM) — or any hardware XGBoost/LightGBM/CatBoost run on
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- [pi coding agent](https://github.com/badlogic/pi-mono) — `npm install -g @mariozechner/pi-coding-agent`
- An API key for **any** provider pi supports (see Providers table below) — opencode-go recommended for minimax-m2.7
- Kaggle API credentials at `~/.kaggle/kaggle.json` (get one at https://www.kaggle.com/settings)

```bash
uv sync
# Optional extras:
uv sync --extra research   # arxiv lookups for the Researcher phase
uv sync --extra automl     # AutoGluon for the Experimenter
```

You can combine both in one go: `uv sync --extra research --extra automl`. If you skip `--extra research`, also set `RESEARCH_EVERY=0` so the Researcher phase doesn't try to import `arxiv`.

## Providers

Because pi is agent-agnostic about model providers, so is AutoMedal. The default is opencode-go (one `sk-` key gives you access to GLM, Kimi, MiMo, MiniMax), but you can switch to any of these with a single env var change — no code changes needed:

| Provider | env var | Example model slug | Notes |
|---|---|---|---|
| **OpenCode Go** (default) | `OPENCODE_API_KEY` | `opencode-go/minimax-m2.7` | One key unlocks GLM-5.1, Kimi K2.5, MiMo, MiniMax M2.5/2.7 |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/<model>` | Free-tier models available; aggregates many providers |
| Ollama (local) | — | `ollama/<local-model>` | No key, no cloud — runs on your own GPU |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-5` | Direct Claude access |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` | Direct GPT access |
| Groq | `GROQ_API_KEY` | `groq/<model>` | Fast inference |
| Google Gemini | `GEMINI_API_KEY` | `google/gemini-*` | |

Switch at runtime: `MODEL="openrouter/free-model" ./am run 50`.

## First-run setup

```bash
./am setup
```

An interactive prompt asks which provider you want, takes the API key (input is hidden), stores it via pi's auth config at `~/.pi/auth.json`, and runs a smoke test. Any `./am` command besides `setup`, `help`, and `doctor` is gated behind this step — on a fresh clone the wrapper will refuse to run anything else until you've configured a provider.

If you prefer to manage keys yourself, just export the env var from the Providers table (e.g. `export OPENCODE_API_KEY=sk-...`) and skip `./am setup` entirely — `./am` honors env vars first.

## Quick Start — running on a new competition

Once `./am setup` has stored a provider key, the happy path is three commands:

```bash
# 1. Discover active Kaggle competitions and rank them for AutoMedal fit
./am discover

# 2. Pick one interactively (or skip to step 3 if you already know the slug)
./am select

# 3. Bootstrap the chosen competition and start the three-phase loop
./am bootstrap playground-series-s6e4
./am run 50
```

What each step actually does:

| Step | What happens |
|---|---|
| `./am discover` | Lists every active Kaggle competition, scores each with two-stage heuristics (metadata + file listing), writes a ranked shortlist to `scout/outputs/` |
| `./am select` | Terminal picker over the shortlist, then calls bootstrap on your choice |
| `./am bootstrap <slug>` | Downloads competition data into `data/`, infers schema (`target_col`, features, task type), writes `configs/competition.yaml`, re-renders `AGENTS.md` and `program.md` from templates, generates a starter `prepare.py` (only if one doesn't exist), **resets the harness memory** (`knowledge.md`, `experiment_queue.md`, `research_notes.md`, `journal/`), resets `results.tsv`, and runs `prepare.py` to generate `.npy` arrays |
| `./am run 50` | Runs 50 iterations of the three-phase loop. Each iteration: check stagnation → maybe Researcher → maybe Strategist → always Experimenter → verify invariants → tag `exp/NNNN` |

Other `./am` commands worth knowing:

- `./am status` — quick health check: head of `knowledge.md`, tail of `results.tsv`, latest `exp/*` tags
- `./am doctor` — diagnose pi/provider/env state and run a smoke test
- `./am clean` — wipe memory files and `results.tsv` (prompts to confirm)
- `./am prepare` / `./am render` — regenerate `.npy` arrays / re-render `AGENTS.md` from templates

## Running on an existing competition (no scout)

If you already have the data and just want to run the loop:

```bash
# Place train.csv and test.csv in data/
# Edit configs/competition.yaml by hand (task type, target column, features)
./am render                              # regenerate AGENTS.md + program.md
./am prepare                             # generate .npy arrays
uv run python harness/init_memory.py     # create empty memory files
./am run 10
```

## Configuration knobs

`run.sh` (and therefore `./am run`) honors these env vars:

```bash
STAGNATION_K=3 RESEARCH_EVERY=10 MODEL="opencode-go/minimax-m2.7" ./am run 50
```

- `STAGNATION_K` — trigger Researcher + Strategist after K non-improving runs (default 3)
- `RESEARCH_EVERY` — scheduled Researcher cadence regardless of stagnation (default 10; set `0` to disable — required if you skipped `--extra research`)
- `MODEL` — any `<provider>/<model-id>` slug pi recognizes (see Providers table)
- `LOG_FILE` — path for the combined loop log (default `agent_loop.log`)

## Hard invariants enforced by the harness

`harness/verify_iteration.py` runs after every phase and prints `WARN:` lines on violation:

- **Strategist:** `knowledge.md` ≤ 80 bullets; every bullet outside `Open questions` cites at least one experiment ID; `experiment_queue.md` has exactly 5 entries; no axis appears more than twice; every entry has `Hypothesis`/`Sketch`/`Expected`; valid status transitions
- **Researcher:** `research_notes.md` grew by one entry with 2-3 paper bullets and a query header
- **Experimenter:** `journal/NNNN-*.md` exists with complete frontmatter; valid status (`improved`/`no_change`/`worse`/`crashed`); required sections present; `KB entries consulted` non-empty when the KB is non-empty

Enforcement is soft by default — violations log a warning but don't abort the loop, so a single malformed journal doesn't crash an overnight run.

## Available Libraries

Pre-installed, agent can use freely:

| Category | Libraries |
|---|---|
| Gradient Boosting | XGBoost, LightGBM, CatBoost (all GPU-accelerated) |
| Hyperparameter Optimization | Optuna |
| AutoML | FLAML (built-in), AutoGluon (`--extra automl`) |
| Deep Learning | PyTorch, TabNet |
| Feature Engineering | category_encoders, scikit-learn |
| Data Augmentation | imbalanced-learn (SMOTE, ADASYN) |
| Research | arxiv (`--extra research`) |

## Design Choices

- **Two files to modify.** The agent edits both `train.py` (models, HPO, ensembling) and `prepare.py` (features, encoding, augmentation). This gives it full control over the ML pipeline while keeping scope manageable.
- **File-based memory over conversational memory.** Every artifact is a git-tracked markdown file. The agent is stateless; the files are state. Auto-compaction can't erase `knowledge.md`.
- **Two stable per-competition directories.** `data/` holds everything the agent reads (raw CSVs, `.npy` arrays, encoder metadata); `submissions/` holds everything it writes for Kaggle. These are user-facing contracts — harness internals can evolve around them, but the directories and how they're used stay put. Switching competitions is `scout/bootstrap.py <slug>` and both directories get re-seeded.
- **Fixed time budget.** Each experiment runs for at most 10 minutes, making results directly comparable regardless of what the agent changes.
- **GPU-first.** XGBoost uses `device="cuda"` with high `max_bin`, LightGBM uses `device="gpu"`, CatBoost uses `task_type="GPU"`. 16GB VRAM is utilized aggressively.
- **Automatic submissions.** Every time `val_loss` improves, a Kaggle-ready submission CSV is written to `submissions/` with a timestamped filename.
- **Deterministic harness, LLM-driven phases.** Anything that can be a Python check (stagnation, experiment IDs, invariant verification) is Python. Everything requiring judgment (planning, curation, research synthesis) is an LLM call.

## Switching competitions mid-project

```bash
# Current: playground-series-s6e4
# Want: spaceship-titanic
uv run python scout/bootstrap.py spaceship-titanic
```

This wipes `data/` of the old competition's files, pulls the new competition's data, resets the memory, and re-renders `AGENTS.md`. `submissions/` is also per-competition — clear it yourself between competitions if you want a clean output directory. Your code in `train.py` and `prepare.py` is regenerated from the starter template **only if you delete them first** — otherwise bootstrap preserves your current code. Git history keeps both competitions' progress.

## Migrating from v2 (opencode) to v3 (pi)

v2 shipped with opencode as the agent. v3 swaps it for [pi coding agent](https://github.com/badlogic/pi-mono) — lighter, always-YOLO, supports more providers out of the box. The migration is three commands:

```bash
npm install -g @mariozechner/pi-coding-agent   # install pi
./am setup                                     # pick provider + paste key
./am doctor                                    # smoke test
```

Everything else — `prompts/`, `harness/`, `scout/`, your `train.py` and `prepare.py` — is unchanged. The agent swap lives entirely inside `run.sh`'s `run_agent()` function, so any branch that had v2 running will have v3 running after these three commands.

If you want to stay on v2 temporarily, pin to the commit before the v3 swap: `git log --oneline run.sh | head` and check out the last commit whose subject starts with `refactor:` or earlier.

## Troubleshooting

- **`./am setup` smoke test fails with "unauthorized"** — the API key didn't persist. Run `./am doctor` to see whether `~/.pi/auth.json` was written, or just `export OPENCODE_API_KEY=sk-...` in your shell as a fallback
- **`./am` says "not configured yet" but you exported the env var** — make sure you exported it in the *same* shell you're running `./am` from (not a different tab)
- **`scout/bootstrap.py` reports low schema sniff confidence** — edit `configs/competition.yaml` by hand, set `human_verified: true`, re-run `./am render`
- **Strategist queues 5 entries all on the same axis** — the `verify_iteration.py` check will warn; kill the loop, fix `experiment_queue.md` by hand or delete it (next iteration will re-plan)
- **Researcher can't import arxiv** — `uv sync --extra research`, or set `RESEARCH_EVERY=0` to skip the phase entirely
- **`final_val_loss=` line missing from train.py output** — the harness relies on this exact pattern. Agents are prompted to preserve it but may drift; revert `train.py` and the next Experimenter will re-add it

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The three-phase harness, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT
