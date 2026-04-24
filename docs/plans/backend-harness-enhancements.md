# AutoMedal — backend harness enhancements (ml-intern adoption)

## Context

I crawled `huggingface/ml-intern` (their autonomous "ML intern" agent that
reads papers, trains models, and ships them) to find polish we can pull
into our own harness. They share our shape — single agent, tool-call loop,
JSONL-style event stream, frontend dashboard — but they've been at it
longer with a bigger team, and several things in their loop are noticeably
more robust than ours.

Repo layout (38.6k LOC):

```
ml-intern/
├── agent/
│   ├── core/
│   │   ├── agent_loop.py          1265 LOC — the main loop
│   │   ├── doom_loop.py            135 LOC — pattern-cycle detector
│   │   ├── effort_probe.py         229 LOC — reasoning-effort cascade
│   │   ├── model_switcher.py       228 LOC — /model command
│   │   ├── llm_params.py           192 LOC — provider param resolver
│   │   ├── session.py              313 LOC — session state
│   │   ├── session_uploader.py     202 LOC — HF dataset upload
│   │   └── tools.py                376 LOC — ToolRouter
│   ├── context_manager/manager.py  406 LOC — compaction at 90% ctx
│   ├── tools/                      ~9 k LOC — research/jobs/sandbox/git/…
│   ├── prompts/system_prompt_v3.yaml
│   └── utils/braille.py, particle_logo.py, crt_boot.py — terminal polish
├── backend/
│   ├── main.py                FastAPI + CORS + static
│   ├── session_manager.py     200-session capacity, per-user 10
│   ├── routes/agent.py        REST + SSE chat endpoint
│   └── routes/auth.py         HF OAuth
├── frontend/                  React 18 + MUI 6 + Zustand 5 + Vite
│   ├── components/Chat/         ToolCallGroup, ActivityStatusBar, …
│   ├── components/CodePanel/    plan/code/output viewer w/ Prism
│   ├── components/SessionSidebar/
│   └── store/                   agentStore, sessionStore, layoutStore
└── configs/                   MCP servers + main config
```

The headline unlock the user called out: **their agent isn't time- or
step-bounded.** Their `max_iterations = -1` lets the agent run until it
decides it's done, while we cap at `max_steps=50` per phase. They earn
that freedom with a doom-loop detector, context compaction, and a stack
of small loop-correctness fixes (truncation handling, dangling tool-call
patching, transient retry) — ours doesn't have any of those.

This plan is the gap analysis + a concrete, prioritized adoption path.

---

## Gap analysis

| Area | ml-intern | AutoMedal today | Adopt? |
|---|---|---|---|
| **Step bound** | `max_iterations=-1` (unbounded) | `max_steps=50` per phase | Tier 1 — make per-phase configurable, default still bounded |
| **Doom-loop detector** | hash tool_call sigs; detects identical-3 OR [A,B,A,B] cycles; injects corrective prompt | none | **Tier 1** |
| **Truncation handler** | `finish_reason=length` + tool_calls → drop them, inject "use heredoc / split edits" hint | tool_calls just fail with malformed JSON | **Tier 1** |
| **Dangling tool_call patcher** | scans last assistant msg for tool_calls without matching tool_results, injects stubs | provider would 400; we'd crash that phase | **Tier 1** |
| **Friendly errors** | auth / credits / rate-limit / model-not-found get clean explanations | bare traceback | **Tier 1** |
| **Transient retry** | 3 attempts × [5, 15, 30]s for 5xx / 429 / timeout / EOF / connection-reset | one shot | **Tier 1** |
| **Plan tool** | pending/in_progress/completed todos; emits `plan_update` event | none | **Tier 1** |
| **Streaming `delta` events** | per-chunk to UI; final `assistant_stream_end` | event type exists, partially wired | Tier 1 — finish wiring |
| **Context compaction** | at 90% of model_max_tokens; preserves system + first user msg + 5 untouched tail | none — each phase is short so we never hit the limit *yet* | Tier 2 — needed once a phase goes unbounded |
| **Effort probe + cascade** | first call to a new model probes max→xhigh→high→… until accepted; caches per-model | hardcoded high for advisor | Tier 2 |
| **Effort heal-and-retry** | mid-call thinking rejection → re-probe and retry once | none | Tier 2 |
| **Cancellation race** | `asyncio.wait([gather, cancel_event], FIRST_COMPLETED)`; kills sandbox + HF jobs | Ctrl-C kills the whole process | Tier 2 |
| **Approval mechanism** | batch approval modal; feedback + edit-before-run for scripts | none — Kaggle submit goes straight through | Tier 2 |
| **Undo / truncate-to-msg** | reduce history to before Nth user msg | none | Tier 3 — only useful in interactive REPL |
| **Session uploader** | separate process uploads trajectories to HF dataset | none | Tier 3 — already have `agent_loop.events.jsonl` locally |
| **SSE backend** | FastAPI + EventBroadcaster fan-out; re-attach via `/events/{id}` | JSONL tail (Go TUI plan) | **Skip** — we picked tail for the Go TUI plan |
| **Title generation** | gpt-oss-120b for chat tab title | none | Tier 3 — only matters if we add session UI |
| **Frontend polish (MUI + Zustand)** | full React dashboard with CodePanel, ToolCallGroup, ActivityStatusBar | textual TUI; Go TUI in plan | Tier 3 — already covered by the Go TUI plan |
| **CRT boot / particle logo** | terminal eye-candy | unicode block-char logo (already shipped) | **Skip** — different aesthetic |

---

## What AutoMedal already has that ml-intern doesn't

So we don't lose ground: keep these.

- Multi-phase orchestration (researcher / strategist / experimenter_edit /
  experimenter_eval / analyzer) — ml-intern is single-phase free-form.
- **Advisor loop** (Kimi K2.6 second-opinion at stagnation + audit
  junctions) — we just shipped this; it's effectively their "model
  switcher" but smarter (deterministic triggers, not user-driven).
- **Stagnation gate** + **knowledge audit** — cross-iteration
  bookkeeping ml-intern doesn't have because it has no concept of
  iterations.
- TSV leaderboard, journal-per-experiment, knowledge.md absorption.
- Kaggle SDK + GPU harness — we run locally on the user's box.

Our orchestration is *better* than theirs at the cross-iteration level.
The gaps are all *inside* one kernel run.

---

## Tier 1 — kernel correctness (the small, contained wins)

These all live inside `automedal/agent/kernel.py` and `providers/*`. No
schema changes, no new event types except `plan_update`. Self-contained
and individually shippable.

### A. Doom-loop detector

**File:** `automedal/agent/kernel.py` (edit, +30 LOC) + new
`automedal/agent/doom_loop.py` (≈80 LOC, ported from
`ml-intern/agent/core/doom_loop.py`).

```python
# automedal/agent/doom_loop.py
def check_for_doom_loop(messages: list[dict]) -> str | None:
    """Returns a corrective system message or None."""
    sigs = _extract_recent_tool_signatures(messages, lookback=30)
    if name := _detect_identical_consecutive(sigs, threshold=3):
        return f"[SYSTEM] You called {name!r} 3+ times with the same args …"
    if pattern := _detect_repeating_sequence(sigs):
        return f"[SYSTEM] Stuck in cycle [{' → '.join(s.name for s in pattern)}] …"
    return None
```

Hook into the kernel right before `provider.chat_stream`:

```python
# automedal/agent/kernel.py:60 — inside the for-step loop, before chat_stream
if doom := check_for_doom_loop(messages):
    messages.append({"role": "user", "content": doom})
    self.events and self.events.error(where="doom_loop", exc=Exception(doom))
```

**Why it matters:** strategist + experimenter_edit both have a known
failure mode where they re-call the same tool with the same args after
getting a "no" answer. Today this just burns steps until `max_steps`;
with the detector we get an explicit corrective injection within 3
calls.

### B. Truncation handler

**File:** `automedal/agent/kernel.py` (edit, +20 LOC).

When `turn.finish_reason == "length"` AND `turn.tool_calls`, the model
ran out of output budget mid-tool-call — the JSON is garbage. Drop the
calls and inject a hint:

```python
if turn.finish_reason == "length" and turn.tool_calls:
    dropped = [tc.name for tc in turn.tool_calls]
    hint = (
        f"Your previous response was truncated by the output token limit. "
        f"Tool calls lost: {dropped}. Do NOT retry with the same large content. "
        f"For 'write': use bash with cat<<'HEREDOC' or split into multiple smaller "
        f"edit calls."
    )
    messages.append({"role": "user", "content": hint})
    continue  # retry this step
```

Requires our `ChatTurn` to expose `finish_reason` (currently doesn't —
small adapter change in `providers/openai_compat.py` and
`providers/anthropic.py`).

**Why it matters:** experimenter_edit writes Python; large `write` calls
truncate today and the kernel returns a `provider_error` — wasting the
whole iteration. With this fix the model retries with smaller content.

### C. Dangling tool_call patcher

**File:** new `automedal/agent/messages.py` (≈40 LOC) called from kernel
before each `chat_stream`.

```python
def patch_dangling_tool_calls(messages: list[dict]) -> None:
    """Add stub tool_results for any tool_call without a matching result.
    Scans backwards for the last assistant msg with tool_calls."""
    answered = {m["tool_use_id"] for m in messages if m["role"] == "tool"}
    for msg in reversed(messages):
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "tool_use" and block["id"] not in answered:
                    messages.append({
                        "role": "tool", "tool_use_id": block["id"],
                        "content": "Tool was not executed (interrupted or error).",
                        "is_error": True,
                    })
            break
        if msg["role"] == "user":
            break
```

**Why it matters:** if a tool raises mid-iteration (e.g. Kaggle SDK 503),
the next chat_stream fails with "tool_use_id has no matching tool_result"
and the phase dies. With the patch it self-heals.

### D. Friendly error messages

**File:** new `automedal/agent/errors.py` (≈60 LOC) called from
`run_loop.py` and the advisor client.

Maps known exception strings to plain-English fixes:

| Error contains | Message |
|---|---|
| `unauthorized`, `invalid x-api-key` | "API key missing/invalid. `export OPENCODE_API_KEY=...`" |
| `insufficient`, `credit` | "Out of credits at opencode.ai" |
| `model_not_found`, `does not exist` | "Model id wrong. Try `automedal models` to list available." |
| `not supported by provider` | "This provider doesn't host that model. Drop the `:provider` suffix." |

**Why it matters:** today an auth failure prints a 12-line traceback that
buries the actual fix. New users hit this constantly.

### E. Transient retry in provider client

**File:** `automedal/agent/providers/openai_compat.py` (edit, ≈30 LOC).

Wrap the `httpx.AsyncClient.post` call in a 3-attempt retry with
`[5, 15, 30]`s backoff for HTTP 429 / 5xx / `asyncio.TimeoutError` /
`httpx.ConnectError`. Emit a `tool_log`-style event on each retry so the
TUI shows "retrying in 5s…".

**Why it matters:** opencode-go has had two minute-long 503 windows in
the last week. Each one killed an iteration. Retry would have ridden
through both.

### F. Plan tool

**File:** new `automedal/agent/tools/plan.py` (≈100 LOC, ported
verbatim from `ml-intern/agent/tools/plan_tool.py`).

In-memory `_current_plan: list[{id, content, status}]`. Each call
replaces the whole plan. Emits `plan_update` event for the TUI.

Wire into the strategist's tool allowlist (it's the natural plan-builder
phase). Emit `plan_update` events that the Go TUI can render in a
side-panel later.

**Why it matters:** strategist currently emits free-form text; the user
can't see at a glance what the agent intends. A structured plan is also
useful for the advisor — gives it concrete structure to critique.

### G. Finish wiring streaming `delta` events

**File:** `automedal/agent/events.py` already declares `delta`. Need to
verify both providers emit it on every chunk and the TUI's
`tui/sources/events_jsonl.py` formats it. ≈10 LOC audit.

**Why it matters:** the Go TUI plan assumes per-chunk events for the log
stream; they exist in the schema but not consistently emitted.

---

## Tier 2 — needed once a phase goes unbounded

Defer until we actually flip a phase's `max_steps` to `-1`. They become
load-bearing the moment the agent might run for hours.

### H. Context compaction

**File:** new `automedal/agent/context.py` (≈200 LOC, port of
`ml-intern/agent/context_manager/manager.py`).

Trigger at 90% of model_max_tokens (from `litellm.get_model_info` or
hardcoded per provider). Preserve:
- System prompt (never touch)
- First user message (the task prompt)
- Last N untouched messages (default 5, walk back to land on a user msg)

Summarize the middle via a `_COMPACT_PROMPT` to a separate model call.
After compact, recount real tokens via `litellm.token_counter` (uses the
correct tokenizer per model).

Their `_RESTORE_PROMPT` (first-person tool-trail summary) is a clever
variant for browser-cached restart — not relevant to us yet.

**Why it matters:** today our researcher can write 30k tokens of journal
context per call; if we ever let it loop unbounded it'll hit 200k fast.
Compaction is what lets `max_iterations=-1` not blow up.

### I. Effort probe + cascade

**File:** new `automedal/agent/effort.py` (≈230 LOC, port of
`ml-intern/agent/core/effort_probe.py`).

Cascade per preference:
```python
{"max": ["max", "xhigh", "high", "medium", "low"],
 "high": ["high", "medium", "low"], …}
```

First time we see a model, fire a 1-token ping with the preferred effort.
On 400 with "thinking not supported" → cache `None` for that model. On
400 with "invalid effort" → walk down the cascade. Cache outcome on the
session.

Useful once we let the user `--advisor` swap to arbitrary models (e.g.
GPT-5 doesn't accept `max`, Claude 4.7 strips `thinking.type.enabled`,
etc.). Today we hardcode `reasoning_effort="high"` for advisor — works
because Kimi K2.6 accepts it, fragile if we widen the model set.

### J. Per-phase configurable max_steps + unbounded mode

**File:** `automedal/agent/phases/_common.py` (edit, ≈10 LOC) +
`automedal/agent/kernel.py` (edit, treat `-1` as infinite).

```python
PHASE_MAX_STEPS = {
    "researcher": 100,        # was 50; needs Tier 2 H to go unbounded
    "strategist": 30,
    "experimenter_edit": 50,
    "experimenter_eval": 20,
    "analyzer": 30,
}
```

ml-intern's `max_iterations` knob becomes our per-phase override. Even
without compaction we benefit from tighter caps on phases that
*shouldn't* take 50 steps (analyzer should be ≤30).

### K. Cancellation race

**File:** `automedal/agent/kernel.py` (edit, ≈40 LOC).

Replace the bare `await self._execute_tools(...)` with the
`asyncio.wait([gather_task, cancel_event], FIRST_COMPLETED)` pattern from
ml-intern's agent_loop.py:768. Plumb a `session.cancel_event` from
`run_loop.py` so SIGINT can interrupt mid-tool, not just between phases.

**Why it matters:** today Ctrl-C during a 5-min Kaggle eval call orphans
the subprocess. With the race, we can SIGTERM the eval and clean up.

### L. Approval mechanism

**File:** new `automedal/agent/approval.py` (≈80 LOC). Tag tools as
`requires_approval=True`. Kernel pauses, emits an `approval_required`
event with the tool args, waits for an `EXEC_APPROVAL` op on a queue.

Useful for `submit_to_kaggle` (counts against daily submission limit) and
any future "spend N USD on inference" tool. **Only worth building when we
add a tool that genuinely needs gating** — no current tool does.

---

## Tier 3 — bigger lifts, lower priority

### M. Undo / truncate-to-message

`run_loop.py` is non-interactive (one-shot iteration loop), so
"truncate to user message N" doesn't apply. Becomes useful only when we
add an interactive REPL — not on the roadmap.

### N. Session uploader

We already write `agent_loop.events.jsonl` locally. Uploading
trajectories to HF (or anywhere) is a future "share runs" feature, not
an enhancement to the harness itself.

### O. Title generation

Useful only with a session list UI. Out of scope until then.

---

## Step-by-step implementation (Tier 1 only)

### Sequence

Order matters because each step's tests rely on earlier ones being
stable.

1. **Day 1 — friendly errors (D)** — pure mapping module; no kernel
   changes; smallest blast radius.
2. **Day 1 — transient retry (E)** — provider-side; isolated.
3. **Day 2 — dangling tool_call patcher (C)** — pure transformation
   over message list; unit-testable without provider.
4. **Day 2 — truncation handler (B)** — needs `ChatTurn.finish_reason`
   plumbed; minor adapter edit.
5. **Day 3 — doom-loop detector (A)** — port verbatim; integrate at top
   of kernel for-loop.
6. **Day 3 — plan tool (F)** — port verbatim; add to strategist
   allowlist.
7. **Day 4 — streaming delta audit (G)** — verify both providers + TUI
   render path.
8. **Day 5 — bench against baseline** — re-run the 10-iter measurement
   loop with all of Tier 1 enabled, confirm no regressions, look for
   wins.

Each step: write the module, add tests (`tests/test_doom_loop.py`,
`tests/test_truncation.py`, etc.), commit. Total ≈ 5 days, ≈ 600 LOC,
≈ 8 commits.

### Critical files

| File | Action | LOC | Purpose |
|---|---|---|---|
| `automedal/agent/doom_loop.py` | new | 80 | port from ml-intern |
| `automedal/agent/messages.py` | new | 40 | dangling tool_call patcher |
| `automedal/agent/errors.py` | new | 60 | friendly error mapper |
| `automedal/agent/tools/plan.py` | new | 100 | plan tool + handler |
| `automedal/agent/kernel.py` | edit | +50 | hook A, B, C; treat max_steps=-1 |
| `automedal/agent/providers/openai_compat.py` | edit | +30 | retry (E) + finish_reason on ChatTurn |
| `automedal/agent/providers/anthropic.py` | edit | +20 | finish_reason on ChatTurn |
| `automedal/agent/providers/base.py` | edit | +5 | add `finish_reason` to `ChatTurn` |
| `automedal/agent/events.py` | reference | — | already declares plan_update + delta |
| `automedal/agent/phases/_common.py` | edit | +10 | wire plan tool into strategist |
| `automedal/run_loop.py` | edit | +15 | catch known errors → friendly messages |
| `automedal/advisor/client.py` | edit | +20 | retry (E) + friendly errors (D) |
| `tui/sources/events_jsonl.py` | edit | +20 | render `plan_update` row |
| `tests/test_doom_loop.py` | new | ~60 | identical-3 + [A,B,A,B] cases |
| `tests/test_truncation.py` | new | ~40 | finish_reason=length drops calls |
| `tests/test_dangling_tool_calls.py` | new | ~50 | patcher injects stubs |
| `tests/test_friendly_errors.py` | new | ~30 | each pattern matched |
| `tests/test_plan_tool.py` | new | ~40 | status validation + event emit |
| `tests/test_provider_retry.py` | new | ~50 | 503-then-200 retry path |

≈ 600 LOC of new code, ≈ 270 LOC of tests. Net ≈ 870 LOC across 19
files.

---

## Verification

1. **Doom-loop unit tests** — feed synthetic message list with 3
   identical `bash(echo hi)` calls → corrective prompt returned. Feed
   `[A, B, A, B]` → cycle detected.
2. **Truncation integration** — monkeypatch provider to return
   `finish_reason="length"` with a tool_call; assert kernel drops the
   call and injects the hint, then continues to next step.
3. **Dangling tool_call** — feed message list with assistant tool_use
   but no tool_result; assert patcher appends stub `is_error=True`.
4. **Friendly errors** — `_friendly_error("401 Unauthorized")` returns
   the API-key message; falls through to `None` on unknown errors.
5. **Provider retry** — mock httpx to return 503 twice then 200; assert
   the call eventually succeeds and 2 `tool_log` events were emitted
   for the retries.
6. **Plan tool** — call with valid + invalid status enum; assert event
   emitted, error returned for invalid.
7. **Streaming delta** — start a fake-provider run, count `delta` events
   in JSONL; should equal the number of chunks emitted.
8. **End-to-end baseline re-run** — `automedal run 10` with all Tier 1
   enabled; compare phase tokens, step counts, and elapsed wall-clock
   to the prior baseline. Acceptable: same final score ±0.1pp, fewer
   `provider_error` stops, fewer 50-step phase ceilings hit.

---

## Risks

| Risk | Mitigation |
|---|---|
| Doom-loop false positives kill productive parallel exploration | Threshold is 3 *identical* (same-args hash); legitimate parallel exploration uses different args. Add an env-var off-switch `AUTOMEDAL_DOOM_LOOP=0` for first week of dogfooding. |
| Truncation hint wastes a step on every long write | Only fires when `finish_reason=length` *and* tool_calls present. Today those calls are dropped silently — net win. |
| Dangling tool_call stubs change advisor behavior | Stubs are visible in the transcript with `is_error=True` so the model can self-correct. Tested in `tests/test_dangling_tool_calls.py`. |
| Friendly-error mapper hides real errors during dev | Always include the original `str(exc)` in the mapped message, prefix the friendly explanation. |
| Retry burns 50s before surfacing a real outage | Cap at 3 attempts × [5,15,30]s = 50s max. Surface as one `tool_log` per attempt so the user sees what's happening. Identical to ml-intern's posture. |
| Plan tool spam from strategist | `plan_update` event is fan-out only — TUI ignores duplicates by `id`. Tool description discourages > 1 update per step. |
| Per-phase max_steps changes break stagnation logic | Tier 2 J only, not Tier 1. When we get there, run the full baseline loop first and compare. |

---

## Out of scope (with rationale)

- **SSE backend / EventBroadcaster** — we already chose JSONL tail in
  the Go TUI plan. SSE would be a parallel approach; not worth two
  transports.
- **Title generation, session list UI** — single-user single-machine
  setup; no UI surface that needs a chat-tab title.
- **Session uploader to HF** — local logs are sufficient. Uploading is
  a separate "share trajectories" product feature.
- **Approval mechanism (Tier 2 L)** — no current tool genuinely needs
  it. Build the day we add a `submit_to_kaggle` tool that risks daily
  submission quota.
- **CRT boot / particle logo** — different aesthetic; we already have
  the unicode block-char logo.
- **MCP server config** — ml-intern uses MCP for some tools; we're
  in-process. Adopt only when we have a third-party tool worth pulling
  in (e.g. a Weights & Biases MCP server).

---

## Inspiration references

- `ml-intern/agent/core/agent_loop.py` — main loop with all the
  patterns above; especially L529-L900 for the run_agent flow.
- `ml-intern/agent/core/doom_loop.py` — verbatim port target.
- `ml-intern/agent/context_manager/manager.py:341-407` — compaction
  algorithm for Tier 2.
- `ml-intern/agent/core/effort_probe.py` — cascade pattern for Tier 2 I.
- `ml-intern/agent/tools/plan_tool.py` — verbatim port target for F.

---

## Recommendations for unbounded phases (when we get there)

Tier 1 doesn't unbound anything — `max_steps` stays a hard ceiling. But
the question "which phases *should* go unbounded once Tier 2 is in" is
worth pinning down now so the Tier 2 work has a target.

### Which phases benefit

| Phase | Unbound? | Why / why not |
|---|---|---|
| **researcher** | **Yes** — first candidate | Open-ended literature/code search where 50 steps is genuinely sometimes too few. The phase already has natural exit conditions (knowledge.md absorption later in the loop). High upside. |
| **experimenter_edit** | Cautious yes (with extra guards) | Code revisions can legitimately need 60-80 tool calls when wrestling with a stubborn bug. But it's also the phase most prone to doom-loop; only unbound *after* Tier 1 A is proven robust on real runs. |
| **strategist** | No | This is a planning phase, not exploration. If it can't decide in 30 steps, the upstream context is broken — adding more steps just papers over it. |
| **analyzer** | No | Compression task with a finite input. > 30 steps means it's hallucinating work. Keep capped; if it needs more, the fix is a better prompt, not more steps. |
| **experimenter_eval** | No | Bounded externally by Kaggle SDK / TSV parsing. |

### Hard backstops (load-bearing once `max_steps=-1`)

Even unbounded phases need an upper bound *somewhere* — otherwise a runaway phase eats wall-clock and tokens until the user notices. Three independent backstops, any of which can stop the phase:

1. **Wall-clock cap** — `AUTOMEDAL_PHASE_MAX_SECONDS=1800` (30 min default). Checked in the kernel for-loop, raises `PhaseTimeout` cleanly. Phase returns `stop="wall_clock"` so the orchestrator knows it didn't naturally complete.
2. **Per-phase token cap** — `AUTOMEDAL_PHASE_MAX_TOKENS=200_000`. Adds usage_total.in_tokens + .out_tokens after each turn; trips the same `stop` field as above with `stop="token_cap"`. Prevents a phase from quietly burning a $40 advisor budget.
3. **No-progress detector** — track a hash of `(knowledge.md size, journal entry count, last tool result preview)` every 10 steps; if unchanged for 3 windows (30 steps), the phase is making no observable progress — abort with `stop="no_progress"`. This is a *coarser* signal than the doom-loop detector (which catches identical tool calls) — it catches the case where the agent calls *different* tools but isn't producing artifacts.

All three are off-by-default flags during dogfooding; flip them on once the unbounded researcher has run cleanly for 5 iters in a row.

### Tier 2 work that becomes load-bearing

When you flip the first `max_steps=-1`, Tier 2 stops being optional:

- **H (compaction)** — non-negotiable. Without it, an unbounded researcher hits `ContextWindowExceededError` within 60 steps and the phase dies messily.
- **I (effort cascade)** — only if you let users swap models mid-run. If the unbounded phase is locked to one model, skip.
- **K (cancellation race)** — non-negotiable. The whole point of "let it run" is that you can let it run *and* trust Ctrl-C to clean up the in-flight tool. Without it, Ctrl-C orphans a Kaggle eval subprocess.
- **L (approval)** — only if you add a tool that genuinely needs gating. Unbounded mode makes the *worst* tools more dangerous (an unbounded experimenter_edit could submit to Kaggle 50 times); pair the unbound flag with an approval requirement on `submit_to_kaggle` if/when that tool exists.

### Explicit "I'm done" affordance

ml-intern relies on the model emitting a turn with no tool_calls to exit. We do too — but in an unbounded phase the model may *forget* it can do this and keep tool-calling forever. Two cheap helpers:

- **System prompt addendum** for unbounded phases: explicitly say "you may end this phase at any time by responding with prose only (no tool calls). Do not feel obligated to keep working if you've answered the user."
- **Optional `done_phase(reason: str)` tool** — if the model prefers an explicit signal, give it one. Maps to `stop="model_done"`. Costs ≈30 LOC.

### Rollout plan

1. Ship Tier 1 (this plan).
2. Run 10-iter baseline with Tier 1 enabled; confirm zero regressions.
3. Implement Tier 2 H + K (compaction + cancellation race). Re-run baseline.
4. Flip researcher to `max_steps=-1` *with all three backstops on*. Run 1 iter, watch closely.
5. If clean: run 10-iter measurement. Compare against baseline (final accuracy, total tokens, wall-clock).
6. If wins, consider experimenter_edit. If losses, the backstop telemetry tells us *why* (token cap vs no-progress vs wall-clock) and we tune.

This is the natural follow-up plan after Tier 1 is merged and battle-tested.
