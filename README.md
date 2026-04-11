# AutoMedal

Autonomous ML research for Kaggle competitions, powered by [OpenCode](https://opencode.ai) + GLM-5.1. Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — same philosophy (edit, train, check, keep/revert), but targeting tabular ML competitions instead of LLM pretraining.

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

The three phases run as **separate `opencode` invocations** with small focused prompts. No single call lives long enough to hit context auto-compaction, which is how v2 scales to 100+ experiments without losing state.

Design doc: `docs/superpowers/specs/2026-04-10-automedal-harness-v2-design.md`

## Project Structure

```
prepare.py              — Data pipeline: loading, cleaning, features, encoding (agent-editable)
train.py                — Training sandbox: models, HPO, ensembling, submission gen (agent-editable)
config_loader.py        — Loads configs/competition.yaml (shared by prepare.py and train.py)
run.sh                  — Three-phase automation loop (dispatches opencode)

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

- NVIDIA GPU (tested on RTX 4070 Ti Super, 16GB VRAM)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [OpenCode](https://opencode.ai) CLI (`opencode run -m opencode-go/glm-5.1 ...`)
- Kaggle API credentials at `~/.kaggle/kaggle.json` (get one at https://www.kaggle.com/settings)

```bash
uv sync
# Optional extras:
uv sync --extra research   # arxiv lookups for the Researcher phase
uv sync --extra automl     # AutoGluon for the Experimenter
```

## Quick Start — running on a new competition

The happy path is four commands. Scout picks a competition, bootstrap downloads data and wires everything up, run starts the three-phase loop.

```bash
# 1. Discover active Kaggle competitions and rank them for AutoMedal fit
uv run python scout/discover.py

# 2. Pick one interactively
uv run python scout/select.py
#   → shows a ranked list, you pick by number or slug, then confirm bootstrap

# 3. (or skip steps 1-2 and go straight to bootstrap if you already know the slug)
uv run python scout/bootstrap.py playground-series-s6e4

# 4. Start the three-phase headless loop
bash run.sh 50
```

What each step actually does:

| Step | What happens |
|---|---|
| `scout/discover.py` | Lists every active Kaggle competition, scores each with two-stage heuristics (metadata + file listing), writes a ranked shortlist to `scout/outputs/` |
| `scout/select.py` | Terminal picker over the shortlist, then calls bootstrap on your choice |
| `scout/bootstrap.py <slug>` | Downloads competition data into `data/`, infers schema (`target_col`, features, task type), writes `configs/competition.yaml`, re-renders `AGENTS.md` and `program.md` from templates, generates a starter `prepare.py` (only if one doesn't exist), **resets the harness memory** (`knowledge.md`, `experiment_queue.md`, `research_notes.md`, `journal/`), resets `results.tsv`, and runs `prepare.py` to generate `.npy` arrays |
| `bash run.sh 50` | Runs 50 iterations of the three-phase loop. Each iteration: check stagnation → maybe Researcher → maybe Strategist → always Experimenter → verify invariants → tag `exp/NNNN` |

## Running on an existing competition (no scout)

If you already have the data and just want to run the loop:

```bash
# Place train.csv and test.csv in data/
# Edit configs/competition.yaml by hand (task type, target column, features)
uv run python scout/render.py          # regenerate AGENTS.md + program.md
uv run python prepare.py               # generate .npy arrays
uv run python harness/init_memory.py   # create empty memory files
bash run.sh 10
```

## Configuration knobs

`run.sh` honors these env vars:

```bash
STAGNATION_K=3 RESEARCH_EVERY=10 MODEL="opencode-go/glm-5.1" bash run.sh 50
```

- `STAGNATION_K` — trigger Researcher + Strategist after K non-improving runs (default 3)
- `RESEARCH_EVERY` — scheduled Researcher cadence regardless of stagnation (default 10; set `0` to disable)
- `MODEL` — any opencode-compatible model id
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

## Troubleshooting

- **`scout/bootstrap.py` reports low schema sniff confidence** — edit `configs/competition.yaml` by hand, set `human_verified: true`, re-run `scout/render.py`
- **Strategist queues 5 entries all on the same axis** — the `verify_iteration.py` check will warn; kill the loop, fix `experiment_queue.md` by hand or delete it (next iteration will re-plan)
- **Researcher can't import arxiv** — `uv sync --extra research`
- **`final_val_loss=` line missing from train.py output** — the harness relies on this exact pattern. Agents are prompted to preserve it but may drift; revert `train.py` and the next Experimenter will re-add it

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The three-phase harness, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT
