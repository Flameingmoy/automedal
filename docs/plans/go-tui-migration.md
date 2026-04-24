# AutoMedal — Go TUI migration (Charmbracelet stack)

## Context

The Python/Textual TUI has hit two real walls in the last week:

1. **Cold-start bloat.** Typing `q` at the home prompt routed through
   `spawn_command("q", [])`, which cold-launched `python -m automedal q`.
   That process imports `openai`, `anthropic`, `jinja2`, `httpx`, the whole
   advisor stack, `textual` again — ~1.4 s of import work and ~120 MB of
   transient RSS just to print "Unknown command". The patch
   (`tui/screens/home.py:111`) short-circuits quit aliases inline, but the
   underlying cost is everywhere: every `automedal <verb>` we shell to from
   the TUI pays the same tax.
2. **Heaviness under load.** During the last 10-iter run with advisor on,
   the user reported the system "load[ed] a lot" and the session
   eventually crashed — we have a Python parent running asyncio *and*
   spawning a Python subprocess that re-imports the whole world to stream
   JSONL it then re-parses. RSS pressure on the box is the likely culprit.

Charmbracelet's Go ecosystem — Bubbletea (Elm-architecture TUI runtime),
Lipgloss (CSS-style layout), Bubbles (textinput/table/viewport/spinner/help),
Glamour (markdown→ANSI), Huh (forms), Harmonica (spring physics) — gives us
a single static binary with sub-100 ms cold start, native ANSI rendering,
and zero Python import surface for UI work. The Python side stays the
brain: it owns the agent kernel, advisor consults, Kaggle SDK, prompt
templates, and the JSONL event stream. Go owns the *shell* — the screens
the user actually looks at and types into.

**Goal.** Stand up a `tui-go/` Go module that is a strict drop-in for the
existing Textual TUI's MVP surface (Home + Dashboard + run streaming),
tailing the same `agent_loop.events.jsonl` the Python kernel already
writes. Once it ships, `automedal tui` becomes the canonical TUI entry
point — handing off to the Go binary if present and falling back to the
existing Python TUI otherwise.

**Non-goals.** Rewriting the agent kernel, advisor, Kaggle SDK, or any
Python that doesn't render UI. No Bubble Tea ports of the Setup wizard,
Detail screen, or rare flows yet — those stay Python until the Go shell
is proven on the hot path. No new event types; we tail what
`automedal/agent/events.py` already emits.

---

## Architecture

```
                ┌───────────────────────┐
   keystrokes → │   Go TUI (tui-go/)    │ ← rendered ANSI to terminal
                │  Bubbletea + Lipgloss │
                └───┬────────────┬──────┘
                    │            │
       spawn run    │            │  tail -f
       (exec)       ▼            ▼
                ┌─────────────────────┐
                │  Python automedal   │ writes →  agent_loop.events.jsonl
                │  (kernel + advisor) │           journal/, knowledge.md
                └─────────────────────┘
```

The Go TUI **never** imports Python, never speaks to opencode-go directly,
never touches Kaggle. It does three things only:

1. **Render screens** — Home, Dashboard, command-output stream.
2. **Spawn `automedal run N --advisor X`** as a subprocess and stream its
   stdout into a viewport (mirrors `tui/screens/command_output.py`).
3. **Tail `agent_loop.events.jsonl`** with `fsnotify` + `bufio.Scanner`,
   bridging each parsed event into the Bubbletea message loop via a
   buffered channel + `tea.Cmd`.

The Python `automedal` CLI gets one new subcommand — `automedal tui` —
that `os.execvp`s the Go binary if present, falling back to the existing
Python TUI if not. Existing entry points (`automedal run`, `doctor`, …)
are unchanged.

---

## Repo layout

Monorepo, `tui-go/` lives next to `automedal/` and `tui/`:

```
Autoresearch_for_kaggle/
├── automedal/            (unchanged Python — agent + advisor)
├── tui/                  (unchanged Python — current Textual TUI; fallback)
├── tui-go/               (NEW)
│   ├── go.mod            module github.com/cdharmaraj/automedal-tui
│   ├── main.go           tea.NewProgram + screen router
│   ├── models/
│   │   ├── home.go       HomeModel    — logo, status strip, command input, recent
│   │   ├── dashboard.go  DashModel    — sparkline + leaderboard + log + GPU
│   │   └── command_output.go  RunModel — subprocess streamer
│   ├── components/
│   │   ├── sparkline.go  braille per-iter loss curve
│   │   ├── sprite.go     unicode block-char logo renderer
│   │   ├── statusbar.go  iteration / phase / advisor pills
│   │   └── leaderboard.go  bubbles/table wrapper
│   ├── events/
│   │   ├── tail.go       fsnotify + Scanner → chan Event
│   │   ├── schema.go     Event struct mirroring agent/events.py
│   │   └── reduce.go     PhaseMachine port from tui/state.py
│   ├── proc/
│   │   └── spawn.go      exec.CommandContext + per-line tea.Msg
│   └── theme/
│       └── theme.go      Lipgloss styles (Dracula palette)
├── README.md             (edit: add `automedal tui` section)
└── (everything else unchanged)
```

`tui-go/` has its own `go.mod` so it can be built and released independently
(`go build -o automedal-tui ./tui-go`). The Python project doesn't depend
on it being present.

---

## Library inventory (Charmbracelet)

| Library         | Use                                                          |
|-----------------|--------------------------------------------------------------|
| **bubbletea**   | Elm-architecture runtime (`Init`/`Update`/`View`)            |
| **bubbles**     | `textinput` (command palette), `table` (leaderboard), `viewport` (log stream), `spinner` (advisor consult), `progress` (iteration bar), `help` + `key` (footer hints) |
| **lipgloss**    | All styling (borders, padding, color, layout); `lipgloss/table` for rich grids; `lipgloss/list` for menus |
| **glamour**     | Render `knowledge.md` / journal entries to ANSI in detail panes |
| **huh**         | Setup wizard form (deferred to Phase 2)                      |
| **harmonica**   | Spring-physics easing for sparkline transitions and pill badges |
| **fsnotify**    | (not Charm) inotify wrapper for live JSONL tailing           |

Glow is a CLI app, not a library — we use Glamour for the markdown
rendering it wraps.

---

## Event contract (read-only — already exists)

The Python kernel already writes these to `agent_loop.events.jsonl`. The Go
side only needs to read them. Schema lives in `automedal/agent/events.py`
(authoritative); Go mirror in `tui-go/events/schema.go`:

```go
type Event struct {
    T       float64 `json:"t"`
    Kind    string  `json:"kind"`
    Phase   string  `json:"phase,omitempty"`
    Tool    string  `json:"tool,omitempty"`
    InTok   int     `json:"in,omitempty"`
    OutTok  int     `json:"out,omitempty"`
    Purpose string  `json:"purpose,omitempty"`   // advisor_consult
    Model   string  `json:"model,omitempty"`
    Skipped bool    `json:"skipped,omitempty"`
    Preview string  `json:"preview,omitempty"`
    Step    int     `json:"step,omitempty"`
    Iter    int     `json:"iter,omitempty"`
    Err     string  `json:"err,omitempty"`
    // ...mirrors EventSink methods in agent/events.py
}
```

Event kinds we render in MVP: `phase_start`, `phase_end`, `tool_start`,
`tool_result`, `advisor_consult`, `usage`, `step_advance`, `iter_start`,
`iter_end`, `error`.

---

## Step-by-step build sequence

### Week 1 — Scaffold + Home

**Day 1-2 — module + main**
- `go mod init github.com/cdharmaraj/automedal-tui` in `tui-go/`
- Add deps: `bubbletea`, `bubbles`, `lipgloss`, `glamour`, `harmonica`, `fsnotify`
- `main.go`: parse `--screen home|dashboard` flag, default `home`; init
  `tea.NewProgram(model, tea.WithAltScreen())`
- Theme module: copy the Dracula-ish palette from `tui/screens/home.py`
  DEFAULT_CSS (`#0F111A` bg, `#50FA7B` prompt, `#FFD700` logo, `#6272A4`
  border, `#8BE9FD` accent)

**Day 3-4 — HomeModel**
- Layout (Lipgloss `JoinVertical`): logo | status strip | recent activity |
  command input | footer
- `components/sprite.go`: render the existing PNG logo as Unicode block
  characters (U+2581..U+2588). One-time precompute at build via small Go
  program reading `tui/assets/logo/logo.png` → emits
  `tui-go/components/logo_data.go` with a `[]string` constant. Falls back
  to the same `A U T O M E D A L` text if the file is missing.
- `components/statusbar.go`: pill row (provider | model | run state |
  advisor on/off). Reads from on-disk state, not events (matches current
  Python behavior — `tui/state.py:AppState`).
- `bubbles/textinput` for the command palette; port `normalize()` and the
  COMMANDS list from `tui/widgets/command_input.py:55` verbatim
- Tab autocomplete: same single-match-appends-trailing-space behavior we
  just landed (`tui/widgets/command_input.py:186`); `--advisor <TAB>`
  reads `~/.automedal/models_cache.json` directly (no Python call needed)
- Recent activity: read last 5 entries from `journal/` directory listing,
  same as `tui/widgets/recent_activity.py`

**Day 5 — quit aliases + dispatch**
- Port `_QUIT_ALIASES = {"q","quit","exit",":q",":quit",":wq"}` set —
  these never spawn a subprocess (the bug we just fixed); they call
  `tea.Quit` directly
- `help` opens a help modal; everything else dispatches to either the
  Dashboard model or `proc.Spawn` (for `run`, `doctor`, etc.)

**Verification gate (end of week 1):**
- `time ./automedal-tui --screen home` shows first frame in <100 ms
  (target; Python TUI takes ~1500 ms to first frame on the same box)
- Binary size <20 MB
- `q` / `Ctrl+C` exit cleanly without spawning anything

### Week 2 — JSONL tailer + Dashboard

**Day 1-2 — events/tail.go**
- `fsnotify.NewWatcher`; watch `agent_loop.events.jsonl`'s parent dir for
  CREATE/WRITE
- On open: seek to EOF (don't replay the whole file on attach); on WRITE
  events, `bufio.Scanner` reads complete lines, JSON-unmarshals into
  `Event`, sends on `chan Event`
- Tea integration: a `tea.Cmd` reads one event from the channel and
  returns it as a `tea.Msg`. The model's `Update` switches on event
  kind. After handling, the model's next `Cmd` re-arms the read.
- Handle log-rotation (truncate or rename): if `Scanner.Err()` returns
  EOF or the file shrinks, re-open and seek to 0
- Backpressure: channel buffer = 256; if full, drop oldest (UI is
  cosmetic; correctness lives in the JSONL on disk)

**Day 3 — events/reduce.go**
- Port `PhaseMachine` from `tui/state.py` — it's a small reducer over
  events that yields `(iter, phase, step_count, last_loss, advisor_uses,
  total_tokens)`. ~80 LOC straight translation.

**Day 4-5 — DashModel**
- Layout (`lipgloss.JoinHorizontal`): left column = sparkline + GPU panel;
  right column = leaderboard + log stream
- `components/sparkline.go`: braille rendering of the last N final-loss
  values. Use `harmonica.NewSpring` to ease between data points so new
  values don't jump.
- `components/leaderboard.go`: `bubbles/table` over the parsed
  `results.tsv` (one read on attach, re-read on `iter_end` events)
- Log stream: `bubbles/viewport` containing one styled line per event;
  format mirrors `tui/sources/events_jsonl.py:format_event` (e.g.
  `[10:32:14] strategist phase_end (1.2k/512 tok)`,
  `[advisor:audit] kimi-k2.6 (1.2k/512 tok) "1. Switch from…"`)
- GPU panel: poll `nvidia-smi --query-gpu=utilization.gpu,memory.used,
  memory.total --format=csv,noheader,nounits` every 1 s via
  `tea.Tick`, render a Lipgloss bar

**Verification gate (end of week 2):**
- Replay a recorded `agent_loop.events.${ts}.jsonl` (we kept these from
  the baseline run) by `cp`-ing it to a fresh location and `tail -f`
  appending — Dashboard renders all phases, sparkline animates, GPU
  panel updates.

### Week 3 — Subprocess streaming + polish

**Day 1-2 — proc/spawn.go**
- `exec.CommandContext` for cancellation on `q`/`Ctrl+C`
- Pipe stdout+stderr; goroutine reads lines, sends each as
  `RunOutputMsg{Line string}` on a channel; same channel-bridge pattern
  as the JSONL tailer
- `RunModel` is a viewport + status bar; `q` cancels the context, child
  process is SIGTERM'd, returning to the caller
- `--advisor` flag handling: the Go side simply forwards the user's
  argv; `automedal/run_args.py` parses it Python-side (already shipped)

**Day 3 — `automedal tui` Python entry**
- `automedal/dispatch.py:_cmd_tui` — looks for `automedal-tui` on PATH or
  in `./tui-go/automedal-tui`; `os.execvp`s it if found; otherwise
  falls back to the existing Python TUI launcher
- Add to `_cmd_help` and the README

**Day 4-5 — polish**
- Sprite: pre-render the existing logo PNG to Unicode at build time
  (one-shot Go program reads PNG via `image/png`, downsamples to ~40
  cols × 12 rows, picks block char per cell by luminance). Saves us
  shipping a PNG-decoder runtime.
- Help modal (`?` keybind): `lipgloss.Border` over current screen,
  shows keybindings table
- Glamour pane for `knowledge.md` viewer (bound to `k` from Dashboard)
- Spinner during `advisor_consult` events (`bubbles/spinner` — pulse a
  pill in the status bar while `skipped=false` consult is in flight)

**Verification gate (end of week 3):**
- `automedal tui` → home renders, `run 10 --advisor kimi-k2.6` from the
  prompt spawns the Python kernel, output streams in real time, advisor
  consults light up the status bar, dashboard updates from JSONL.
- E2E run completes; binary size still <20 MB; first-frame latency still
  <100 ms.

---

## Critical files

| File | Action | Why |
|---|---|---|
| `tui-go/go.mod` | new | module root |
| `tui-go/main.go` | new | tea.Program + screen router |
| `tui-go/models/home.go` | new | HomeModel — port of `tui/screens/home.py` |
| `tui-go/models/dashboard.go` | new | DashModel — port of dashboard concept |
| `tui-go/models/command_output.go` | new | RunModel — port of `tui/screens/command_output.py` |
| `tui-go/events/tail.go` | new | fsnotify+Scanner→chan bridge |
| `tui-go/events/schema.go` | new | mirror of `automedal/agent/events.py` Event |
| `tui-go/events/reduce.go` | new | port of `tui/state.py:PhaseMachine` |
| `tui-go/proc/spawn.go` | new | subprocess streamer |
| `tui-go/components/{sparkline,sprite,statusbar,leaderboard}.go` | new | UI primitives |
| `tui-go/theme/theme.go` | new | Lipgloss styles (Dracula palette) |
| `tui-go/components/logo_data.go` | generated | unicode logo from PNG |
| `automedal/dispatch.py` | edit | add `_cmd_tui` (~15 LOC) |
| `automedal/agent/events.py` | reference only | event schema source-of-truth |
| `tui/sources/events_jsonl.py` | reference only | format strings to match |
| `tui/state.py` | reference only | PhaseMachine to port |
| `README.md` | edit | one section: `automedal tui` + how to build |

Reference-only files do **not** change — they're the spec the Go side
ports.

---

## Verification

1. **Cold start** — `time ./automedal-tui --screen home` < 100 ms (vs
   ~1500 ms for the Python TUI on the same box). Measured with
   `hyperfine --warmup 3`.
2. **Binary size** — `ls -lh automedal-tui` < 20 MB stripped.
3. **Quit hot path** — typing `q` at the home prompt: no subprocess
   spawned (verify with `strace -f -e execve` on the Go binary), exits
   in <50 ms. (The Python fix only patched-around this; the Go shell
   never had the import-bloat problem to begin with.)
4. **JSONL tailer** — `cp` a recorded baseline `agent_loop.events.jsonl`
   into place, then `cat baseline.jsonl >> agent_loop.events.jsonl` line
   by line with `pv -L 50` — sparkline animates, log scrolls, no
   dropped lines (the buffered channel has 256 slack).
5. **Subprocess streaming** — `run 1 --advisor kimi-k2.6` from the Go
   prompt spawns Python, streams output, `q` mid-run sends SIGTERM and
   the Python process exits cleanly within 2 s.
6. **Advisor lights up** — during a real iteration, the status bar's
   advisor pill turns green during a `consult_advisor` tool call and
   shows the consult count + token spend.
7. **Fallback path** — rename the Go binary; `automedal tui` still
   launches the Python Textual TUI without error.
8. **No regressions in Python** — all 168 existing pytest tests still
   pass; no Python file in `automedal/` is touched except
   `dispatch.py:_cmd_tui`.
9. **Dashboard accuracy** — leaderboard rows and sparkline values match
   `results.tsv` byte-for-byte.
10. **RSS** — `ps -o rss= -p <pid>` of the Go TUI < 30 MB at idle (vs
    Python's ~95 MB).

---

## Risks

| Risk | Mitigation |
|---|---|
| Bubbletea has no PNG/image rendering | Pre-render PNG to Unicode block-chars at build time; ship as `[]string` constant |
| JSONL tailer races kernel writes (partial line) | `bufio.Scanner` only yields complete `\n`-terminated lines; partial trailing line waits for next inotify event |
| Log rotation / truncation by user | Detect via `Scanner.Err() == io.EOF || file.Size() < lastOffset`; reopen + seek 0 |
| Go binary not on PATH | `automedal tui` falls back to existing Python TUI; documented explicitly in README |
| Two TUIs drift apart visually | Keep the Python TUI as a documented fallback for ≥ 1 release; deprecate only after 2 weeks of dogfooding the Go shell |
| Kaggle SDK / opencode-go calls only exist in Python | Out of scope — Go shell never makes them; it spawns Python for any verb that talks to providers |
| GPU panel needs nvidia-smi | Already a project assumption (RTX 4070 Ti); panel hides if `nvidia-smi` exits non-zero |
| Channel backpressure during burst events | 256-slot buffered channel + drop-oldest policy; UI is cosmetic, correctness lives in JSONL |
| Spring physics overshoot looks weird on loss curves | Use `harmonica.FPS(60)` with a stiff spring (k=20, damping=8); test on noisy data before locking in |

---

## Future scope (NOT in MVP)

- Setup wizard in Huh (Phase 2)
- Detail view for journal entries with Glamour markdown
- SSH-shareable Wish app (one-line wrap of the Bubbletea program)
- Replace Python TUI entirely (after 2-4 weeks of dogfooding)
- LSP-style code preview pane in the Dashboard

---

## Inspiration references (not deps)

- Charmbracelet **crush** — the AI coding TUI; same architectural shape
  (Bubbletea shell + external model providers), proves the pattern at
  scale.
- Charmbracelet **gum** — shell-callable widgets; useful precedent for
  the `automedal tui` → Go binary handoff.
- Charmbracelet **bubbletea** examples — `examples/composable-views` is
  the model split we'll mirror; `examples/realtime` is the channel-bridge
  pattern for the JSONL tailer.
