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

### One-liner (recommended)

```bash
curl -LsSf https://raw.githubusercontent.com/Flameingmoy/automedal/main/install.sh | bash
```

The installer verifies Go ≥ 1.24, builds the `automedal` binary into `~/.local/bin`, installs the optional `sniff` Python shim (via `pipx`, falling back to `pip --user`), and creates `~/.automedal/` (mode 0700) for credentials and logs.

Upgrade later:

```bash
GOBIN=~/.local/bin go install github.com/Flameingmoy/automedal/cmd/automedal@latest
pipx upgrade automedal-sniff
```

### From source (development)

```bash
git clone https://github.com/Flameingmoy/automedal automedal && cd automedal
go build -o ~/.local/bin/automedal ./cmd/automedal
pipx install ./py-shim/sniff       # or: pip install --user ./py-shim/sniff
automedal version
```

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

Running `automedal` with no arguments opens the [TUI command centre](#tui--command-centre) instead.

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

`automedal` with no arguments execs the Go shell (`tui-go/`) — a single
static binary built on Charmbracelet's Bubbletea stack. Native ANSI,
~10 ms first-frame latency, ~16 MB idle RSS. It tails the same
`agent_loop.events.jsonl` the kernel writes, with no Python re-imports
on every command.

### Build

```bash
cd tui-go
go build -o automedal-tui .          # ~11 MB static binary
cp automedal-tui ~/.local/bin/       # or any dir on $PATH
```

After that, `automedal` just works. Override location with
`AUTOMEDAL_TUI_GO_BIN=/abs/path/automedal-tui`. If the binary is missing,
the CLI prints the build command and exits non-zero — use
`automedal dispatch <cmd>` for headless runs.

### Screens

| Screen | What's on it |
|--------|--------------|
| **Home** | Unicode block-char logo, status pills (phase / iter / advisor), recent journal entries, command palette with Tab autocomplete (incl. `--advisor <TAB>` against the live model cache) |
| **Dashboard** | Braille sparkline of running-best loss, `results.tsv` leaderboard, live JSONL stream, GPU panel (via `nvidia-smi`) |
| **Run** | Subprocess streamer for `automedal run N` — `q` sends SIGTERM and returns home cleanly |
| **Knowledge** | `knowledge.md` rendered via Glamour (scrollable viewport) |
| **Help** | Keybindings reference |

### Commands

Same palette as the dispatch list: `run [N] [--advisor [model]]`, `init <slug>`,
`discover`, `select`, `doctor`, `status`, `clean`, `setup`, `models`, plus
quit aliases (`q`, `quit`, `exit`, `:q`, `:quit`, `:wq`) — these never
spawn a subprocess. Anything else is forwarded to `automedal <cmd>` as a
child process, streamed line-by-line.

## Roadmap — full-Go shell

The kernel (agent loop, advisor, Kaggle SDK) stays Python. The shell is
already Go. We're tracking Charmbracelet's
[crush](https://github.com/charmbracelet/crush) as the reference for how
far that split can go — their `internal/ui/{chat,dialog,common}/` layout
is the next target for us. Concretely:

- **Per-tool renderers.** Crush has one Go file per tool type
  (`ui/chat/{bash,fetch,file,mcp,search,todos,tools,unified_diff}.go`);
  we currently format every tool identically. Adopting their split
  unlocks per-tool visual affordances without ballooning a single
  switch-statement.
- **Status-icon framing.** `●` header with `✓` / `✗` / `⏳` state,
  collapsible result block (default 10 lines, expand on demand),
  syntax-highlighted diff / code output.
- **Dialog layer.** Crush has modals for models, commands, permissions,
  quit, sessions — we have `help` and that's it. A permissions modal
  would let us gate `submit_to_kaggle` behind an approval prompt.
- **Session persistence.** Crush persists sessions under
  `$XDG_DATA_HOME/crush/`; we rely on the journal + `results.tsv` in
  the working directory. A session sidebar + history viewer is within
  reach once we adopt their schema.
- **Gradient styling.** `internal/ui/styles/grad.go` is their signature
  look — one file, cross-cutting. Small lift, big polish.

Not on the roadmap: the MCP transport layer, LSP integration, or
Catwalk-style external model catalogs — those solve problems we don't
have.

## CLI Reference

| Command | Description |
|---------|-------------|
| `automedal` / `automedal tui` | Open TUI home screen |
| `automedal setup` | Configure a provider + API key (first-run) |
| `automedal doctor` | Smoke-test the provider + SDK versions + env state |
| `automedal discover` | List and rank active Kaggle competitions |
| `automedal select` | Pick a competition from a DataTable |
| `automedal init <slug>` | Download data, infer schema, wire up the project |
| `automedal prepare` | Regenerate `.npy` arrays from `data/` |
| `automedal render` | Re-render `AGENTS.md` from the template |
| `automedal run [N]` | Start the loop (default 50 iterations) |
| `automedal status` | Quick health: knowledge head, last 5 results, latest tags |
| `automedal clean` | Wipe memory files + `results.tsv` (confirms first) |
| `automedal version` | Print installed version |

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

```
automedal/                        Installed package
├── cli.py                        `automedal` console script entry point
├── dispatch.py                   One function per subcommand
├── paths.py                      Layout class — dev vs user mode resolution
├── auth.py                       ~/.automedal/.env store
├── run_loop.py                   4-phase orchestrator
├── dedupe.py                     BM25 motivation dedupe
├── quick_reject.py               30s smoke-train guard
└── agent/                        Bespoke agent kernel
    ├── kernel.py                 Async tool-call loop (~250 LOC)
    ├── events.py                 JSONL event emitter + human-log mirror
    ├── providers/
    │   ├── anthropic.py          anthropic SDK (Anthropic + opencode-go)
    │   └── openai.py             openai SDK (OpenAI + Ollama + OpenRouter + Groq)
    ├── tools/
    │   ├── fs.py                 read/write/edit/list/grep (path-guarded)
    │   ├── shell.py              run_shell (cwd-bound, timeout)
    │   ├── cognition.py          BM25 recall tool
    │   ├── arxiv.py              Researcher-only paper search
    │   └── subagent.py           spawn_subagent(prompt, tools, max_steps)
    ├── phases/                   researcher / strategist / experimenter_* / analyzer
    └── prompts/*.md.j2           jinja-templated phase prompts

harness/                          Deterministic automation (no LLM)
├── check_stagnation.py           K-run stagnation detector
├── next_exp_id.py                Experiment ID allocator
├── init_memory.py                Creates memory files on bootstrap
├── verify_iteration.py           Post-phase invariant + regression + success_criteria
├── build_trace_trailer.py        Reflective-trace builder
└── rank_journals.py              Learning-value ranker

scout/                            Competition discovery + bootstrap
├── discover.py / select.py / bootstrap.py / sniff.py / scoring.py / render.py

tui-go/                           Go TUI (Charmbracelet stack) — user-facing shell
├── main.go                       tea.Program + screen router
├── theme/                        Lipgloss palette (Dracula)
├── models/                       home · dashboard · run · help · knowledge screens
├── components/                   logo, statusbar, palette, sparkline, leaderboard, gpu
├── events/                       fsnotify-based JSONL tailer + Format/Reduce
└── proc/                         exec.CommandContext subprocess streamer

tests/                            pytest suite (Python) — Go tests live in tui-go/**/_test.go
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
| TUI shows stale events | Tail `agent_loop.events.jsonl` to confirm the loop is writing events; delete + restart if rotated |
| `automedal-tui (Go binary) not found` | Build it: `cd tui-go && go build -o automedal-tui .`; then copy onto `$PATH` or set `AUTOMEDAL_TUI_GO_BIN` |
| `pipx install` fails on git install | Upgrade pipx: `python3 -m pip install --user --upgrade pipx` |

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via markdown instructions for AI agents — is his. The four-phase loop, dedupe, quick-reject, bespoke kernel, file-based memory, and scout pipeline are AutoMedal-specific extensions for Kaggle-style tabular ML.

## License

MIT. See [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for attribution of bundled third-party software (`anthropic`, `openai`, `rank-bm25`, `jinja2`, `arxiv`, `python-dotenv`, plus the Charmbracelet Go stack — `bubbletea`, `bubbles`, `lipgloss`, `glamour`, `fsnotify` — in `tui-go/`).
