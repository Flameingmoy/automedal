# AutoMedal Harness v2 — Design Spec

**Date:** 2026-04-10
**Status:** Approved (brainstorming phase complete, awaiting implementation plan)
**Addresses GitHub issues:** #3 (improve agent harness), #4 (scientific research), #5 (context management), #7 (knowledge-graph memory), #8 (experiment planning step)
**Explicitly out of scope:** #1 (Kaggle CLI), #2 (multi-agent compatibility), #6 (MLX/AMD/Intel), #9 (alternative coding agents)

## 1. Problem

After 12 experiments the current AutoMedal loop is stagnating. Rows 5–11 of `results.tsv` show seven consecutive runs that all hypothesized variations of "stacking / weighted ensemble / finer grid search → push below 0.0524 val_loss." The `agent_loop.log` tail even contains the correct insight ("stacking LR meta-learner consistently underperforms weighted ensemble, XGB-heavy weights are most reliable") — but that insight dies when the `opencode` process exits and the next iteration re-derives it from scratch.

The root cause is that the harness is **stateless between iterations**. `results.tsv` carries one flat line per run, which is too lossy to function as real memory. Every iteration re-plans from zero and has no mechanism to notice it is repeating itself, to plan more than one step ahead, or to draw on ideas beyond its own pretraining.

All five in-scope issues cluster around this root cause:

- **#3, #5, #7** — long-term memory is missing
- **#8** — multi-step deliberate planning is missing
- **#4** — external knowledge injection is missing

## 2. Goals and non-goals

**Goals**

- Durable structured memory that survives both process restarts and opencode's auto-compaction
- Deliberate multi-step planning that prevents duplicate-hypothesis loops
- Axis diversity enforcement so the agent cannot spend five consecutive runs on ensemble-weight tuning
- External knowledge injection on demand (arxiv) triggered by stagnation or schedule
- Scale to 100+ experiments in a single overnight run without context explosion
- Per-phase LLM cost visibility and logging
- Keep the system agent-swappable later (issue #2) by making memory file-based and agent-agnostic

**Non-goals**

- Real-time training (we keep the 10-minute wall clock per experiment)
- Full knowledge graph with typed edges (we start with flat markdown bullets; upgrade later if the format breaks down)
- Automatic reinforcement learning or per-turn self-improvement
- Multi-agent compatibility in this spec (issue #2 is a downstream refactor)
- Replacing `opencode` or `GLM-5.1`
- Changing the underlying ML stack (xgboost / lightgbm / catboost)

## 3. Architecture overview

AutoMedal v2 is a three-phase loop driven by `run.sh`. Each phase is a separate `opencode run` invocation with its own prompt file, reading and writing to a shared file-based memory. The harness (`run.sh` plus small Python helpers) owns orchestration, stagnation detection, experiment tagging, and TSV logging. The agent owns thinking.

```
            ┌─────────────────────────────────────────────┐
            │               run.sh (loop)                 │
            │  stagnation check · tag · log · dispatch    │
            └──────────┬──────────┬─────────┬─────────────┘
                       │          │         │
              on empty │  on      │  every  │
              queue or │ stagnation│  iter   │
              stagnation         ↓         ↓
                       ↓   ┌──────────┐ ┌───────────────┐
                 ┌─────────┤Researcher│ │ Experimenter  │
                 │Strategist└──────────┘ └───────────────┘
                 └─────────┘     │              │
                       │         │              │
         reads & writes│  reads  │ reads        │ writes
                       ↓         ↓              ↓
        ┌──────────────────────────────────────────────┐
        │          File-based memory (git-tracked)     │
        │  journal/NNNN-slug.md      (per experiment)  │
        │  knowledge.md              (curated index)   │
        │  experiment_queue.md       (pending plans)   │
        │  research_notes.md         (arxiv findings)  │
        │  results.tsv               (flat log, kept)  │
        │  prompts/{strategist,researcher,experimenter}│
        └──────────────────────────────────────────────┘
```

The agent never holds state in context between phases — everything goes through files. Each `opencode run` starts with a small, focused prompt, reads exactly what it needs from files, runs one phase, and exits. No single invocation lives long enough to hit opencode's auto-compaction.

### Key design principles

1. **File-based memory over conversational memory.** Every artifact is a git-tracked file. The agent is stateless; the files are state.
2. **Deterministic harness, LLM-driven phases.** Anything that can be a Python check (stagnation detection, experiment ID assignment, invariant verification) is Python. Everything requiring judgment (planning, curation, research synthesis) is an LLM call.
3. **Hard invariants per phase.** Each prompt declares what it may read, what it must write, and what it must not touch. These are checked by `verify_iteration.py`.
4. **Curated, bounded memory.** `knowledge.md` has a hard 80-bullet cap. The Strategist compacts it on every planning pass. Memory cannot grow without bound.
5. **Axis diversity enforcement.** Every queued hypothesis has a mandatory `axis` tag. No queue may have more than 2 entries on the same axis. This is the mechanical fix for the stacking-loop failure mode observed in the current log.

## 4. File layout

Additions to the repository are marked `+`.

```
automedal/
├── AGENTS.md                     (unchanged)
├── README.md                     (unchanged)
├── pyproject.toml                (add: arxiv package)
├── prepare.py                    (experimenter-editable, unchanged role)
├── train.py                      (experimenter-editable, unchanged role)
├── results.tsv                   (unchanged — still the flat log)
├── run.sh                        (rewritten)
├── program.md                    (kept but shrunk to a pointer at prompts/)
+ prompts/
+   ├── strategist.md             (system prompt for planning phase)
+   ├── researcher.md             (system prompt for arxiv phase)
+   └── experimenter.md           (system prompt for execution phase)
+ journal/
+   └── NNNN-slug.md              (one per experiment, written by experimenter)
+ knowledge.md                    (curated KB, rewritten by strategist)
+ experiment_queue.md             (pending hypotheses, written by strategist)
+ research_notes.md               (arxiv findings, appended by researcher)
+ harness/
+   ├── check_stagnation.py       (deterministic Python helper)
+   ├── next_exp_id.py            (returns next NNNN)
+   └── verify_iteration.py       (post-phase invariant check)
+ docs/superpowers/specs/
+   └── 2026-04-10-automedal-harness-v2-design.md   (this file)
```

`program.md` is kept for backwards compatibility but shrunk to a two-paragraph pointer at `prompts/`. Its old checklist role is now split across the three prompt files.

## 5. Data artifact formats

### 5.1 `journal/NNNN-slug.md`

One file per experiment, written by the Experimenter at the end of its run. This is the lossless per-experiment record.

```markdown
---
id: 0013
slug: lgb-dart-boosting
timestamp: 2026-04-10T14:22:00
git_tag: exp/0013
queue_entry: 2
status: improved | no_change | worse | crashed
val_loss: 0.05231
val_accuracy: 0.9864
best_so_far: 0.05231
---

## Hypothesis
(pulled verbatim from the queue entry this run consumed)

## What I changed
One paragraph. Which file, which part, why.

## Result
Numbers plus one sentence versus baseline.

## What I learned
Two to four bullets. This is the curation input for knowledge.md.
Must be concrete: "X didn't work because Y", not "tried X".

## KB entries consulted
- <bullet text copied from knowledge.md that was relevant to this run>
```

Constraints:

- Filename must match `^\d{4}-[a-z0-9-]+\.md$`
- All frontmatter keys are required; unknown statuses fail `verify_iteration.py`
- `git_tag` must reference an existing tag `exp/NNNN` (created by the harness before the run)
- `queue_entry` is the 1-indexed position of the consumed entry in `experiment_queue.md`
- `KB entries consulted` must have at least one bullet unless `knowledge.md` is empty (the bootstrap case)

### 5.2 `knowledge.md`

Rewritten (not appended) by the Strategist on every planning pass. This is the compacted long-term memory that every iteration reads.

```markdown
# AutoMedal Knowledge Base
_Last curated: exp 0015_

## Models
- LightGBM GPU hard-caps max_bin at 255 (discovered exp 0003)
- XGB-heavy ensemble weights (~0.65) beat balanced; exps 0002, 0008, 0011
- LR meta-learner lost to weighted ensemble by +0.011 log loss across exps 0005, 0007, 0009 — stop trying LR stacking

## Features
- Categorical interactions helped (exp 0004): Moisture_x_Humidity was top-5 importance
- Target encoding smoothing=0.3 was noise, no signal (exp 0006)

## HPO
- Optuna narrower ranges around known-good did not beat wider ranges (exp 0010)

## Open questions
- CatBoost with raw categoricals (not encoded) — untried
- TabNet — untried
```

Constraints:

- Hard cap: 80 bullets total (excluding section headers and "Last curated" line)
- Every bullet must cite at least one experiment ID. Untraceable bullets are dropped on rewrite.
- Sections are free-form but `Open questions` is mandatory and non-empty — the Strategist must always leave at least two untried directions so the next plan has fuel.
- The Strategist is responsible for dedupe and merging; if two bullets say the same thing, they must be combined.

### 5.3 `experiment_queue.md`

Written by the Strategist, popped by the Experimenter one entry per iteration.

```markdown
# Experiment Queue
_Planned: exp 0015 | Runs 0016-0020_

## 1. catboost-native-cats [axis: preprocessing] [STATUS: pending]
**Hypothesis:** CatBoost with raw categorical columns (skip ordinal encoding) may outperform encoded version because CatBoost uses ordered target statistics internally — open question in KB.
**Sketch:** In prepare.py, expose CATEGORICAL_FEATURES as pandas category dtype path; in train.py, pass cat_features to CatBoostClassifier.
**Expected:** 0.002-0.005 improvement if the KB guess is right; null result tells us encoded is optimal.

## 2. tabnet-baseline [axis: new-model] [STATUS: pending]
...
```

Constraints:

- Exactly 5 entries per queue
- Each entry has `slug`, `axis`, `STATUS`, `Hypothesis`, `Sketch`, and `Expected`
- `axis` must be one of: `preprocessing`, `feature-eng`, `HPO`, `new-model`, `ensembling`, `pseudo-label`, `architecture`
- **No axis may appear more than twice in one queue.** This is the diversity enforcement rule and is checked by `verify_iteration.py` after every Strategist run.
- STATUS transitions: `pending` → `running` (set by Experimenter on pop) → `done` (set by Experimenter on completion, with journal reference appended)
- When the Strategist re-plans mid-queue (stagnation trigger), any remaining `pending` entries are discarded and the new queue replaces the file wholesale

### 5.4 `research_notes.md`

Appended (not rewritten) by the Researcher on each trigger.

```markdown
# Research Notes

## exp 0015 · stagnation trigger · query: "tabular gradient boosting ensemble diversity"
- Paper: "Negative Correlation Learning for GBDTs" (arxiv 2503.XXXXX)
  Summary: three bullets
  Applicable idea: ... → candidate hypothesis for strategist
- Paper: "Snapshot Ensembles for Boosted Trees" (arxiv 2401.XXXXX)
  Summary: ...
  Applicable idea: ...
```

Constraints:

- Each entry is headed with the triggering experiment ID, the trigger type (`stagnation` or `scheduled`), and the arxiv search query used
- Exactly 2-3 papers per entry
- Each paper must have a Summary (3 bullets) and an Applicable idea (one sentence framed as a candidate hypothesis)
- Papers older than 3 years are dropped before summarization
- The Strategist marks papers as "consumed" by prefixing the bullet with `[consumed in exp NNNN]` when it uses one in a plan

### 5.5 `results.tsv`

Unchanged. Still the flat, one-line-per-experiment log that `analysis.ipynb` reads. The journal is additive, not a replacement. `results.tsv` remains the canonical source for `check_stagnation.py`.

## 6. Phase contracts

Each prompt file has three things: inputs (files it must read), outputs (files it must write), and invariants (rules it cannot violate). These contracts are what make the system debuggable — when something goes wrong, `verify_iteration.py` reports which invariant was violated.

### 6.1 `prompts/experimenter.md`

**Inputs**

- `experiment_queue.md` — pop the first `pending` entry
- `knowledge.md` — full file
- `train.py`, `prepare.py` — current state
- Last 3 entries of `journal/` (by ID)
- `AGENTS.md` — tech context

**Outputs**

- Edited `train.py` and/or `prepare.py`
- New `journal/NNNN-slug.md`
- Appended `results.tsv` row (via the training script)
- Submission CSV (if val_loss improved)
- Updated `experiment_queue.md` (status flipped on consumed entry)

**Invariants**

- Must pop the top `pending` queue entry, not pick a different one
- Must preserve the `final_val_loss=X.XXXX` print statement — harness parses it
- `KB entries consulted` section of the journal must list at least one bullet (unless KB is empty)
- Must not edit `knowledge.md`, `experiment_queue.md` (except status transitions), `journal/` of prior experiments, or `research_notes.md`
- Hard 10-minute wall clock on the training run
- If the training script crashes, the Experimenter must still write a journal entry with `status: crashed` and a `What I learned` section explaining the failure mode

### 6.2 `prompts/strategist.md`

**Inputs**

- All of `journal/` (agent decides which to read; most-recent-first is advised)
- Current `knowledge.md`
- Current `experiment_queue.md`
- `research_notes.md`
- `results.tsv`

**Outputs**

- Rewritten `knowledge.md`
- Rewritten `experiment_queue.md`

**Invariants**

- `knowledge.md` ≤ 80 bullets
- Every bullet in `knowledge.md` must cite ≥ 1 experiment
- `knowledge.md` must contain a non-empty `Open questions` section
- `experiment_queue.md` must have exactly 5 entries
- Every queue entry must carry a valid `axis` tag from the allowed set
- No axis may appear more than twice in one queue
- Must not touch `train.py`, `prepare.py`, `journal/`, `results.tsv`, or `research_notes.md`
- Must consume at least one `research_notes.md` entry (if any are unconsumed) and mark it as consumed

### 6.3 `prompts/researcher.md`

**Inputs**

- `knowledge.md`
- Last 3 journal entries
- Current bottleneck (passed as a prompt argument from `run.sh` — the `best val_loss` and the dominant axis of recent failures)

**Outputs**

- Appended entry to `research_notes.md`

**Invariants**

- Must query arxiv via the `arxiv` Python package (added to `pyproject.toml`)
- Must read 2-3 abstracts, not full papers
- Must write concrete "Applicable idea → candidate hypothesis" lines
- Must not touch `train.py`, `prepare.py`, `knowledge.md`, `experiment_queue.md`, or `journal/`
- Must include the arxiv search query used in the entry header
- Must skip papers older than 3 years

## 7. Orchestration

### 7.1 `run.sh` (pseudocode)

```bash
#!/usr/bin/env bash
set -euo pipefail

MAX_ITER=${1:-50}
STAGNATION_K=3              # K non-improving runs → research next cycle
RESEARCH_EVERY=10           # ... or every 10 iters regardless
MODEL="opencode-go/glm-5.1"
LOG_FILE="agent_loop.log"

for i in $(seq 1 "$MAX_ITER"); do
    EXP_ID=$(python harness/next_exp_id.py)     # e.g., 0013

    # --- stagnation + scheduled research check (deterministic) ---
    STAGNATING=$(python harness/check_stagnation.py --k "$STAGNATION_K")
    SCHEDULED_RESEARCH=$([ $((i % RESEARCH_EVERY)) -eq 0 ] && echo 1 || echo 0)

    if [ "$STAGNATING" = "1" ] || [ "$SCHEDULED_RESEARCH" = "1" ]; then
        BEST=$(python harness/check_stagnation.py --print-best)
        opencode run -m "$MODEL" --dangerously-skip-permissions \
            "$(cat prompts/researcher.md)

             Trigger: stagnation=$STAGNATING scheduled=$SCHEDULED_RESEARCH
             Current best val_loss: $BEST" \
            2>&1 | tee -a "$LOG_FILE"
        python harness/verify_iteration.py --phase researcher || echo "WARN: researcher invariant violation"
    fi

    # --- strategist check (queue empty OR stagnation) ---
    QUEUE_PENDING=$(grep -c '\[STATUS: pending\]' experiment_queue.md 2>/dev/null || echo 0)
    if [ "$QUEUE_PENDING" = "0" ] || [ "$STAGNATING" = "1" ]; then
        opencode run -m "$MODEL" --dangerously-skip-permissions \
            "$(cat prompts/strategist.md)

             Current iteration: $i / $MAX_ITER
             Stagnating: $STAGNATING" \
            2>&1 | tee -a "$LOG_FILE"
        python harness/verify_iteration.py --phase strategist || echo "WARN: strategist invariant violation"
    fi

    # --- tag before experimenter so journal has a stable ref ---
    git tag "exp/$EXP_ID" HEAD

    # --- experimenter phase ---
    opencode run -m "$MODEL" --dangerously-skip-permissions \
        "$(cat prompts/experimenter.md)

         Experiment ID: $EXP_ID" \
        2>&1 | tee -a "$LOG_FILE"

    python harness/verify_iteration.py --phase experimenter --exp-id "$EXP_ID" \
        || echo "WARN: experimenter invariant violation"
done
```

The loop is deliberately sequential. Strategist runs at most once per iteration. Researcher runs at most once per iteration. Experimenter runs exactly once per iteration. Steady-state cost is ~1.2× the current single-phase loop because Strategist fires ~1/5 iterations and Researcher fires ~1/10 iterations plus stagnation spikes.

### 7.2 `harness/check_stagnation.py`

Small, deterministic Python helper. Reads `results.tsv`, tracks the rolling best `val_loss`, and returns `1` if the best has not improved in the last K runs.

**Interface**

- `python harness/check_stagnation.py --k 3` → prints `0` or `1` to stdout
- `python harness/check_stagnation.py --print-best` → prints the current best val_loss as a float

**Rules**

- "Improvement" means strictly less than previous best (no ties)
- If `results.tsv` has fewer than K rows, returns `0` (no stagnation during bootstrap)
- Parses the TSV with the standard library only (no pandas dependency for the harness)

### 7.3 `harness/next_exp_id.py`

Scans `journal/` for the highest existing `NNNN` and prints `NNNN+1` zero-padded to 4 digits. If the directory is empty, prints `0001`. Used by `run.sh` before the Experimenter phase.

### 7.4 `harness/verify_iteration.py`

Post-phase invariant checker. Called after each phase with `--phase {researcher,strategist,experimenter}` and optionally `--exp-id NNNN`.

**What it checks per phase**

- **researcher** — `research_notes.md` grew by exactly one entry, the new entry has 2-3 paper bullets, query header present
- **strategist** — `knowledge.md` ≤ 80 bullets, every bullet cites ≥1 exp ID, `Open questions` section non-empty; `experiment_queue.md` has exactly 5 entries, each with a valid axis, no axis appears more than twice
- **experimenter** — `journal/NNNN-*.md` exists, frontmatter complete, `results.tsv` grew by exactly one row, the consumed queue entry has its STATUS updated, `KB entries consulted` section has at least one bullet (if KB non-empty)

**On violation:** prints a `WARN:` line to stderr and exits with code 1. `run.sh` logs the warning but does not abort the loop — soft enforcement. (Hard enforcement can be turned on later via an env var.)

## 8. Migration from current state

The repo currently has 12 experiments in `results.tsv` and one win at val_loss 0.0524. Migration steps:

1. **Create the new directories** — `prompts/`, `journal/`, `harness/`, `docs/superpowers/specs/`.
2. **Write the three prompt files** (`strategist.md`, `researcher.md`, `experimenter.md`) referencing `AGENTS.md` for shared tech context and focusing each on its phase-specific invariants.
3. **Write the three Python helpers** (`check_stagnation.py`, `next_exp_id.py`, `verify_iteration.py`). Standard library only.
4. **Add `arxiv` to `pyproject.toml`** under a new optional extra `research` (so default installs stay lean).
5. **Bootstrap `knowledge.md` manually.** Read the existing `results.tsv` and `agent_loop.log`, and write the initial KB with the hindsight we already have: LightGBM max_bin=255, XGB-heavy weights win, LR meta-learner loses, etc. Seed `Open questions` with 2-3 untried directions (CatBoost native categoricals, TabNet).
6. **Leave `results.tsv` alone.** No backfill of `journal/` — old runs simply do not have journal entries, and `verify_iteration.py` only checks new runs.
7. **Replace `run.sh`** with the three-phase version.
8. **Run a single dry-run iteration manually** before enabling the loop:
   - Invoke `opencode run ... prompts/strategist.md` and inspect the generated `experiment_queue.md` for invariant compliance
   - Invoke `opencode run ... prompts/experimenter.md` and confirm the first queue entry is popped correctly, a journal entry is written, and `results.tsv` grows
9. **Shrink `program.md`** to a short pointer file.
10. **Enable the loop.** Start with `bash run.sh 10` to validate multi-iteration behavior before going to 50 or 100.

Rollback is trivial — every change is git-tracked and the old `run.sh` / `program.md` can be restored from history if the new harness misbehaves.

## 9. Traceability to in-scope issues

| Issue | How it is addressed |
|---|---|
| **#3 Improve agent harness** | Entire three-phase architecture; file-based memory replaces single TSV-line memory |
| **#4 Scientific research** | Researcher phase, arxiv API, triggered on stagnation (K=3) and scheduled (every 10 iters) |
| **#5 Context management** | Three separate `opencode` invocations with small focused prompts; file memory survives auto-compaction; `knowledge.md` is pre-compacted context bounded at 80 bullets |
| **#7 Knowledge graph of methods** | `knowledge.md` starts as flat markdown bullets with per-experiment citations; can upgrade to a real graph later if bullets outgrow the format |
| **#8 Experiment planning step** | Strategist phase, queue of 5, mandatory axis tags with diversity enforcement |

Issue #2 (multi-agent compatibility) becomes a small downstream change to `run.sh` later because prompts and memory are agent-agnostic files.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Strategist produces a bad queue (all one axis, vague hypotheses, wrong format) | Invariant check in `verify_iteration.py` + hard "max 2 per axis" rule; malformed queues trigger a warning and can be handled manually |
| `knowledge.md` drifts into noise over time | Hard 80-bullet cap forces compaction on every Strategist pass; every bullet must cite experiments; untraceable bullets dropped on rewrite |
| Researcher phase triggers too often and wastes budget | Stagnation K=3 is deterministic; `RESEARCH_EVERY=10` is configurable (set to 0 to disable scheduled research) |
| Opencode cost inflation from 3 calls per iter | Strategist fires ~1/5 iters, Researcher ~1/10 iters + stagnation; steady-state cost is ~1.2× current, not 3× |
| Experimenter ignores the queue and free-plans | Invariant: journal must reference `queue_entry`; `verify_iteration.py` checks it |
| First Strategist run has no journal entries to learn from | Warm-start with manually-seeded `knowledge.md` from existing `results.tsv` hindsight |
| `opencode` process itself hits auto-compaction mid-phase | Phase prompts are short enough (single digit KB of context) that no phase alone approaches the compaction threshold; this is the architectural answer to issue #5 |
| Stagnation fires on a queue of high-variance experiments that happen to all miss | Stagnation also re-runs Strategist (not just Researcher), giving it a chance to pivot the axis mix |

## 11. Success criteria

The redesign is successful if, after 50 experiments from a clean start:

- `knowledge.md` stays under 80 bullets and contains no duplicates
- No queue ever contains more than 2 entries on the same axis
- The agent never queues a hypothesis whose exact failure mode is already in `knowledge.md` (measurable by spot-check)
- At least 3 distinct model families have been tried (not just XGB/LGB/CatBoost ensemble tweaks)
- At least 2 arxiv-inspired hypotheses have been attempted
- Best val_loss has strictly improved over the current 0.0524 baseline (loose target; the harness change is itself valuable even without score improvement)
- The system has run end-to-end without a hard crash in `run.sh`

## 12. Out of scope (follow-on work)

- **Issue #2 — multi-agent compatibility.** Once the file-based memory is stable, `run.sh` can be refactored to wrap `opencode run` behind a `run_agent <prompt_file>` function, then later behind a config-driven adapter.
- **Issue #1 — Kaggle CLI integration.** Depends on a stable harness; `run.sh` can learn a `--competition` flag later.
- **Issue #6 — MLX/AMD/Intel backends.** Orthogonal to this redesign.
- **Issue #9 — alternative coding agents.** Subsumed by issue #2.
- **Upgrading `knowledge.md` to a structured graph.** Defer until flat markdown demonstrably breaks down (probably >200 experiments).
- **Per-experiment LLM cost tracking.** Useful but not required for the harness to function; can be added as a separate pass later.
