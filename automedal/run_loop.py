"""AutoMedal three-phase loop — pure-Python replacement for run.sh.

Dispatches Researcher / Strategist / Experimenter-edit / Experimenter-eval
sequentially per iteration. Uses `deepagents` via `automedal.agent_runtime`.

Entry points:
    python -m automedal.run_loop [N] [fast]
    automedal run N           (when AUTOMEDAL_AGENT=deepagents)

Honored env vars (mirror run.sh defaults exactly):
    MAX_ITERATIONS         first positional arg, default 50
    STAGNATION_K           default 3
    RESEARCH_EVERY         default 10 (0 disables scheduled research)
    COOLDOWN_SECS          default 1
    TRAIN_BUDGET_MINUTES   default 10
    MODEL                  default opencode-go/minimax-m2.7
    LOG_FILE               default agent_loop.log
    AUTOMEDAL_REGRESSION_GATE  strict|warn, default warn
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from automedal.auth import load_env


# ── Initialization ───────────────────────────────────────────────────────────

# Load ~/.automedal/.env BEFORE importing agent_runtime so OPENCODE_API_KEY etc.
# are present when build_model() is first called.
load_env()

# Resolve layout up front (agent_runtime.REPO_ROOT reads AUTOMEDAL_CWD at import).
try:
    from automedal.paths import Layout
    _LAYOUT = Layout()
    os.environ.update(_LAYOUT.as_env())
except Exception:
    _LAYOUT = None

from automedal import agent_runtime as ar


# ── Log helpers (format mirrors run.sh so TUI log_tail regex unchanged) ──────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class HarnessLog:
    """Writes bash-shaped markers to the log file + stdout so TUI parser matches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def harness(self, msg: str) -> None:
        self.write(f"  [harness] {msg}")

    def iteration_start(self, i: int, total: int, exp_id: str) -> None:
        self.write("")
        self.write(f"========== Iteration {i} / {total}  exp={exp_id}  [{_now()}] ==========")

    def iteration_end(self, i: int, exp_id: str) -> None:
        self.write(f"--- Iteration {i} complete  exp={exp_id}  [{_now()}] ---")

    def banner(self, lines: list[str]) -> None:
        for line in lines:
            self.write(line)


# ── Environment + paths ──────────────────────────────────────────────────────

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _getpaths() -> dict[str, Path]:
    cwd = ar.REPO_ROOT
    return {
        "cwd":             cwd,
        "log_file":        Path(os.environ.get("AUTOMEDAL_LOG_FILE") or os.environ.get("LOG_FILE") or cwd / "agent_loop.log"),
        "prompts_dir":     Path(os.environ.get("AUTOMEDAL_PROMPTS_DIR", cwd / "prompts")),
        "harness_dir":     Path(os.environ.get("AUTOMEDAL_HARNESS_DIR", cwd / "harness")),
        "train_py":        Path(os.environ.get("AUTOMEDAL_TRAIN_PY", cwd / "agent" / "train.py")),
        "prepare_py":      Path(os.environ.get("AUTOMEDAL_PREPARE_PY", cwd / "agent" / "prepare.py")),
        "last_train_out":  Path(os.environ.get("AUTOMEDAL_LAST_TRAINING_OUTPUT", cwd / "harness" / ".last_training_output")),
        "data_dir":        Path(os.environ.get("AUTOMEDAL_DATA_DIR", cwd / "data")),
        "results_tsv":     Path(os.environ.get("AUTOMEDAL_RESULTS_TSV", cwd / "agent" / "results.tsv")),
        "queue_md":        cwd / "experiment_queue.md",
    }


# ── Harness helpers (shell calls to existing Python scripts — unchanged) ─────

def _next_exp_id(harness_dir: Path) -> str:
    r = subprocess.run([sys.executable, str(harness_dir / "next_exp_id.py")],
                        capture_output=True, text=True, cwd=str(ar.REPO_ROOT))
    return (r.stdout or "").strip() or "0000"


def _check_stagnation(harness_dir: Path, k: int) -> tuple[str, str]:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "check_stagnation.py"), "--k", str(k), "--both"],
        capture_output=True, text=True, cwd=str(ar.REPO_ROOT),
    )
    parts = (r.stdout or "").strip().split()
    stagnating = parts[0] if len(parts) > 0 else "0"
    best = parts[1] if len(parts) > 1 else "nan"
    return stagnating, best


def _count_pending_queue(queue_md: Path) -> int:
    try:
        return queue_md.read_text(encoding="utf-8").count("[STATUS: pending]")
    except OSError:
        return 0


def _build_trace_trailer(harness_dir: Path) -> str:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "build_trace_trailer.py"), "--n", "3"],
        capture_output=True, text=True, cwd=str(ar.REPO_ROOT),
    )
    return (r.stdout or "").strip() or "(trace unavailable)"


def _rank_journals(harness_dir: Path) -> str:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "rank_journals.py"), "--m", "30", "--k", "10"],
        capture_output=True, text=True, cwd=str(ar.REPO_ROOT),
    )
    return (r.stdout or "").strip() or "(ranking unavailable)"


def _verify(harness_dir: Path, *args: str, log: HarnessLog) -> int:
    """Run harness/verify_iteration.py, mirror its output to log, return exit code."""
    r = subprocess.run(
        [sys.executable, str(harness_dir / "verify_iteration.py"), *args],
        capture_output=True, text=True, cwd=str(ar.REPO_ROOT),
    )
    out = (r.stdout or "") + (r.stderr or "")
    if out:
        for line in out.rstrip("\n").splitlines():
            log.write(line)
    return r.returncode


def _tag_exp(exp_id: str, log: HarnessLog) -> None:
    r = subprocess.run(["git", "rev-parse", f"exp/{exp_id}"],
                       capture_output=True, text=True, cwd=str(ar.REPO_ROOT))
    if r.returncode == 0:
        return
    r = subprocess.run(["git", "tag", f"exp/{exp_id}", "HEAD"],
                       capture_output=True, text=True, cwd=str(ar.REPO_ROOT))
    if r.stdout:
        log.write(r.stdout.rstrip("\n"))
    if r.stderr:
        log.write(r.stderr.rstrip("\n"))


def _run_train(train_py: Path, out_path: Path, timeout_s: int, log: HarnessLog) -> tuple[int, str]:
    """Run `python agent/train.py` with a timeout, tee output to both files."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout, open(log.path, "a", encoding="utf-8") as flog:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(train_py)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(ar.REPO_ROOT),
            )
        except FileNotFoundError as e:
            log.write(f"  [harness] train.py not found: {e}")
            return 127, ""
        assert proc.stdout is not None
        buf: list[str] = []
        start = time.time()
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    if time.time() - start > timeout_s:
                        proc.send_signal(signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    time.sleep(0.05)
                    continue
                fout.write(line); fout.flush()
                flog.write(line); flog.flush()
                sys.stdout.write(line); sys.stdout.flush()
                buf.append(line)
        finally:
            rc = proc.poll() if proc.poll() is not None else proc.wait()
        return rc or 0, "".join(buf)


def _parse_final_val_loss(text: str) -> str:
    """Extract `final_val_loss=X.XXXX` from training stdout (matches run.sh grep)."""
    import re
    m = re.search(r"final_val_loss=([0-9.]+)", text or "")
    return m.group(1) if m else "nan"


# ── Phase dispatch ───────────────────────────────────────────────────────────

async def _run_phase(
    phase: str,
    runtime_context: str,
    model,
    log: HarnessLog,
    label: str,
) -> int:
    agent = ar.build_phase_agent(phase, model)
    log.harness(f"dispatching {label}")
    try:
        rc = await ar.invoke_phase(agent, runtime_context, log_path=log.path, echo=False)
    except Exception as e:
        log.write(f"  [agent] UNCAUGHT: {type(e).__name__}: {e}")
        rc = 1
    if rc != 0:
        log.write(f"  [WARN] {label} exited non-zero")
    return rc


# ── Main loop ────────────────────────────────────────────────────────────────

async def _loop(max_iters: int, fast_mode: bool) -> int:
    p = _getpaths()
    log = HarnessLog(p["log_file"])

    stagnation_k = _env_int("STAGNATION_K", 3)
    research_every = _env_int("RESEARCH_EVERY", 10)
    cooldown_secs = _env_int("COOLDOWN_SECS", 1)
    train_budget_min = _env_int("TRAIN_BUDGET_MINUTES", 10)
    model_slug = os.environ.get("MODEL", "opencode-go/minimax-m2.7")
    os.environ["TRAIN_BUDGET_MINUTES"] = str(train_budget_min)
    regression_gate = os.environ.get("AUTOMEDAL_REGRESSION_GATE", "warn")

    log.banner([
        "==================================================",
        "  AutoMedal — Three-Phase Loop (deepagents)",
        f"  Model:          {model_slug}",
        f"  Iterations:     {max_iters}",
        f"  Stagnation K:   {stagnation_k}",
        f"  Research every: {research_every}",
        f"  Cooldown:       {cooldown_secs}s",
        f"  Train budget:   {train_budget_min}m",
        f"  Log:            {p['log_file']}",
        f"  Started:        {datetime.now().isoformat(timespec='seconds')}",
        "==================================================",
    ])

    # Ensure data exists (first run after bootstrap)
    if not (p["data_dir"] / "X_train.npy").exists():
        log.write("Preparing data first...")
        r = subprocess.run([sys.executable, str(p["prepare_py"])],
                            cwd=str(ar.REPO_ROOT), capture_output=True, text=True)
        if r.stdout:
            log.write(r.stdout.rstrip("\n"))
        if r.stderr:
            log.write(r.stderr.rstrip("\n"))

    # Ensure memory files exist
    subprocess.run([sys.executable, str(p["harness_dir"] / "init_memory.py")],
                    cwd=str(ar.REPO_ROOT), capture_output=True, text=True)

    # Cancel flag
    cancel = {"flag": False}

    def _on_term(signum, frame):
        log.harness("SIGTERM received — finishing this iteration then stopping")
        cancel["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    # Build the model once, reuse across phases/iterations
    provider, short_model = ar.parse_slug(model_slug)
    model = ar.build_model(provider, short_model, timeout=120, max_tokens=8192)

    for i in range(1, max_iters + 1):
        if cancel["flag"]:
            log.harness("cancel requested — stopping cleanly")
            break

        exp_id = _next_exp_id(p["harness_dir"])
        log.iteration_start(i, max_iters, exp_id)

        stagnating, best = _check_stagnation(p["harness_dir"], stagnation_k)
        scheduled_research = (research_every > 0 and i % research_every == 0)
        log.harness(
            f"stagnating={stagnating} scheduled_research={1 if scheduled_research else 0} best={best}"
        )

        # ── Researcher ────────────────────────────────────────────────
        if stagnating == "1" or scheduled_research:
            trigger = "stagnation" if stagnating == "1" else "scheduled"
            await _run_phase(
                "researcher",
                f"Triggering experiment: {exp_id}\nTrigger type: {trigger}\n"
                f"Stagnating: {stagnating}\nScheduled research: {1 if scheduled_research else 0}\n"
                f"Current best val_loss: {best}",
                model, log, f"Researcher ({trigger})",
            )
            _verify(p["harness_dir"], "--phase", "researcher", log=log)

        # ── Strategist ───────────────────────────────────────────────
        pending = _count_pending_queue(p["queue_md"])
        if pending == 0 or stagnating == "1":
            reflective = _build_trace_trailer(p["harness_dir"])
            ranked = _rank_journals(p["harness_dir"])
            log.harness(f"dispatching Strategist (queue_pending={pending}, stagnating={stagnating})")
            # Re-use _run_phase but skip its own "dispatching" log by calling directly
            # (we already logged a richer line above)
            agent = ar.build_phase_agent("strategist", model)
            try:
                rc = await ar.invoke_phase(
                    agent,
                    f"Upcoming experiment: {exp_id}\nCurrent iteration: {i} / {max_iters}\n"
                    f"Stagnating: {stagnating}\nCurrent best val_loss: {best}\n"
                    f"Pending queue entries: {pending}\n\n"
                    f"## Reflective trace (last 3 experiments)\n{reflective}\n\n"
                    f"## Top experiments by learning value\n{ranked}",
                    log_path=log.path, echo=False,
                )
            except Exception as e:
                log.write(f"  [agent] UNCAUGHT: {type(e).__name__}: {e}")
                rc = 1
            if rc != 0:
                log.write("  [WARN] Strategist exited non-zero")
            _verify(p["harness_dir"], "--phase", "strategist", log=log)

        # Tag repo BEFORE Experimenter so journal has a stable ref
        _tag_exp(exp_id, log)

        # ── Experimenter (edit) ───────────────────────────────────────
        await _run_phase(
            "experimenter_edit",
            f"Experiment ID: {exp_id}\nCurrent best val_loss: {best}",
            model, log, "Experimenter (edit)",
        )

        # ── Training (harness-managed, no agent) ──────────────────────
        train_timeout = train_budget_min * 60 + 30
        log.harness(f"running training (budget={train_budget_min}m, timeout={train_timeout}s)...")

        # Run prepare.py if modified
        r = subprocess.run(["git", "diff", "--quiet", "--", str(p["prepare_py"])],
                           cwd=str(ar.REPO_ROOT))
        if r.returncode != 0:
            log.harness("prepare.py changed — running prepare first")
            subprocess.run([sys.executable, str(p["prepare_py"])],
                           cwd=str(ar.REPO_ROOT))

        train_rc, train_out = _run_train(p["train_py"], p["last_train_out"], train_timeout, log)
        final_loss = _parse_final_val_loss(train_out)
        log.harness(f"training done: val_loss={final_loss} exit={train_rc}")

        # ── Experimenter (eval) ───────────────────────────────────────
        await _run_phase(
            "experimenter_eval",
            f"Experiment ID: {exp_id}\nCurrent best val_loss: {best}\n"
            f"Training exit code: {train_rc}\nTraining val_loss: {final_loss}",
            model, log, "Experimenter (eval)",
        )

        # Reread best_so_far
        _, best_after = _check_stagnation(p["harness_dir"], stagnation_k)

        # Verify (regression gate + success_criteria)
        if final_loss != "nan":
            verify_rc = _verify(
                p["harness_dir"],
                "--phase", "experimenter", "--exp-id", exp_id,
                "--val-loss", final_loss, "--best-before", best,
                "--best-so-far", best_after or best, log=log,
            )
        else:
            verify_rc = _verify(p["harness_dir"], "--phase", "experimenter", "--exp-id", exp_id, log=log)

        if verify_rc == 2 and regression_gate == "strict":
            log.harness(f"strict regression gate triggered — reverting git tag exp/{exp_id}")
            subprocess.run(["git", "tag", "-d", f"exp/{exp_id}"], cwd=str(ar.REPO_ROOT))

        if verify_rc == 3:
            log.harness("success_criteria near-miss — attempting one retry edit")
            await _run_phase(
                "experimenter_edit",
                f"Experiment ID: {exp_id}\nCurrent best val_loss: {best}\n"
                f"RETRY: Previous attempt val_loss={final_loss} missed success_criteria by ≤1%.\n"
                f"Make a small targeted improvement and re-run training.",
                model, log, "Experimenter (retry)",
            )
            train_rc2, train_out2 = _run_train(p["train_py"], p["last_train_out"], train_timeout, log)
            retry_loss = _parse_final_val_loss(train_out2)
            log.harness(f"retry training done: val_loss={retry_loss}")
            await _run_phase(
                "experimenter_eval",
                f"Experiment ID: {exp_id}\nCurrent best val_loss: {best}\n"
                f"Training exit code: 0\nTraining val_loss: {retry_loss}\n"
                f"NOTE: This is a retry after a near-miss. Evaluate the retry result normally.",
                model, log, "Experimenter (eval-retry)",
            )
            _verify(p["harness_dir"], "--phase", "experimenter", "--exp-id", exp_id, log=log)

        log.iteration_end(i, exp_id)

        if not fast_mode and i < max_iters and cooldown_secs > 0:
            await asyncio.sleep(cooldown_secs)

    log.banner([
        "",
        "==================================================",
        f"  AutoMedal complete — {max_iters} iterations",
        f"  Finished: {datetime.now().isoformat(timespec='seconds')}",
        f"  Results:  cat {p['results_tsv']}",
        "  Memory:   knowledge.md / experiment_queue.md / journal/",
        "==================================================",
    ])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    max_iters = 50
    fast_mode = False
    if args:
        try:
            max_iters = int(args[0])
        except ValueError:
            print(f"invalid MAX_ITERATIONS: {args[0]!r}", file=sys.stderr)
            return 2
    if len(args) > 1 and args[1].lower() == "fast":
        fast_mode = True
    return asyncio.run(_loop(max_iters, fast_mode))


if __name__ == "__main__":
    sys.exit(main())
