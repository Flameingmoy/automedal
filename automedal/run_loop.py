"""AutoMedal four-phase loop — bespoke-agent orchestrator.

Replaces the deepagents-based run_loop with the new bespoke kernel.
Phases per iteration:

    Researcher    (only on stagnation or scheduled cadence)
    Strategist    (only when queue empty or stagnating; followed by dedupe)
    Quick-reject  (optional 30s smoke train guard, opt-in)
    Experimenter  (edit)
    [training]    (subprocess, harness-managed)
    Experimenter  (eval)
    Analyzer      (per-iteration knowledge compression, default ON)
    Verify        (regression gate + success_criteria)

Entry point:
    python -m automedal.run_loop [N] [fast]
    automedal run N

Env vars (with defaults):
    MAX_ITERATIONS          first positional arg, default 50
    STAGNATION_K            3
    RESEARCH_EVERY          10  (0 disables scheduled research)
    COOLDOWN_SECS           1
    TRAIN_BUDGET_MINUTES    10
    AUTOMEDAL_PROVIDER      opencode-go
    AUTOMEDAL_MODEL         minimax-m2.7
    AUTOMEDAL_ANALYZER      1   (set to 0 to disable)
    AUTOMEDAL_QUICK_REJECT  0   (opt-in)
    AUTOMEDAL_DEDUPE        1   (set to 0 to disable post-strategist dedupe)
    AUTOMEDAL_DEDUPE_THRESHOLD  5.0  (BM25 score, higher = stricter)
    AUTOMEDAL_REGRESSION_GATE   warn|strict (default warn)
    LOG_FILE                agent_loop.log
    AUTOMEDAL_EVENTS_FILE   agent_loop.events.jsonl
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from automedal.auth import load_env


# ── Initialization ───────────────────────────────────────────────────────────

for _k, _v in (("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
               ("OPENBLAS_NUM_THREADS", "4"), ("NUMEXPR_NUM_THREADS", "4"),
               ("VECLIB_MAXIMUM_THREADS", "4")):
    os.environ.setdefault(_k, _v)

load_env()

try:
    from automedal.paths import Layout
    _LAYOUT = Layout()
    os.environ.update(_LAYOUT.as_env())
except Exception:
    _LAYOUT = None


from automedal.agent.events import EventSink
from automedal.agent.providers import build_provider
from automedal.agent.tools.base import REPO_ROOT
from automedal.agent.phases import (
    researcher as p_researcher,
    strategist as p_strategist,
    experimenter_edit as p_exp_edit,
    experimenter_eval as p_exp_eval,
    analyzer as p_analyzer,
)
from automedal import dedupe as dedupe_mod
from automedal import quick_reject as quick_reject_mod


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class HarnessLog:
    """Writes harness-shaped marker lines to the same file the EventSink mirrors to."""

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


def _getpaths() -> dict[str, Path]:
    cwd = REPO_ROOT
    return {
        "cwd":             cwd,
        "log_file":        Path(os.environ.get("AUTOMEDAL_LOG_FILE")
                                or os.environ.get("LOG_FILE")
                                or cwd / "agent_loop.log"),
        "events_file":     Path(os.environ.get("AUTOMEDAL_EVENTS_FILE",
                                                cwd / "agent_loop.events.jsonl")),
        "harness_dir":     Path(os.environ.get("AUTOMEDAL_HARNESS_DIR", cwd / "harness")),
        "train_py":        Path(os.environ.get("AUTOMEDAL_TRAIN_PY", cwd / "agent" / "train.py")),
        "prepare_py":      Path(os.environ.get("AUTOMEDAL_PREPARE_PY", cwd / "agent" / "prepare.py")),
        "last_train_out":  Path(os.environ.get("AUTOMEDAL_LAST_TRAINING_OUTPUT",
                                                cwd / "harness" / ".last_training_output")),
        "data_dir":        Path(os.environ.get("AUTOMEDAL_DATA_DIR", cwd / "data")),
        "results_tsv":     Path(os.environ.get("AUTOMEDAL_RESULTS_TSV",
                                                cwd / "agent" / "results.tsv")),
        "queue_md":        cwd / "experiment_queue.md",
        "journal_dir":     cwd / "journal",
    }


# ── Subprocess shellouts to existing harness scripts (unchanged contracts) ──

def _next_exp_id(harness_dir: Path) -> str:
    r = subprocess.run([sys.executable, str(harness_dir / "next_exp_id.py")],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    return (r.stdout or "").strip() or "0000"


def _check_stagnation(harness_dir: Path, k: int) -> tuple[str, str]:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "check_stagnation.py"), "--k", str(k), "--both"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    parts = (r.stdout or "").strip().split()
    return (parts[0] if parts else "0", parts[1] if len(parts) > 1 else "nan")


def _count_pending_queue(queue_md: Path) -> int:
    try:
        return queue_md.read_text(encoding="utf-8").count("[STATUS: pending]")
    except OSError:
        return 0


def _build_trace_trailer(harness_dir: Path) -> str:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "build_trace_trailer.py"), "--n", "3"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    return (r.stdout or "").strip() or "(trace unavailable)"


def _rank_journals(harness_dir: Path) -> str:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "rank_journals.py"), "--m", "30", "--k", "10"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    return (r.stdout or "").strip() or "(ranking unavailable)"


def _verify(harness_dir: Path, *args: str, log: HarnessLog) -> int:
    r = subprocess.run(
        [sys.executable, str(harness_dir / "verify_iteration.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    out = (r.stdout or "") + (r.stderr or "")
    if out:
        for line in out.rstrip("\n").splitlines():
            log.write(line)
    return r.returncode


def _tag_exp(exp_id: str, log: HarnessLog) -> None:
    r = subprocess.run(["git", "rev-parse", f"exp/{exp_id}"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    if r.returncode == 0:
        return
    r = subprocess.run(["git", "tag", f"exp/{exp_id}", "HEAD"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    if r.stdout:
        log.write(r.stdout.rstrip("\n"))
    if r.stderr:
        log.write(r.stderr.rstrip("\n"))


def _run_train(train_py: Path, out_path: Path, timeout_s: int, log: HarnessLog) -> tuple[int, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout, open(log.path, "a", encoding="utf-8") as flog:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(train_py)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(REPO_ROOT),
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
    m = re.search(r"final_val_loss=([0-9.]+)", text or "")
    return m.group(1) if m else "nan"


def _parse_journal_after(journal_dir: Path, exp_id: str) -> dict:
    """Read the journal entry just written for `exp_id`. Returns parsed front-matter fields."""
    matches = sorted(journal_dir.glob(f"{exp_id}-*.md")) if journal_dir.is_dir() else []
    if not matches:
        return {}
    txt = matches[-1].read_text(encoding="utf-8", errors="replace")
    out = {}
    for line in txt.splitlines():
        if line.strip() == "---" and out:
            break
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# ── Main loop ────────────────────────────────────────────────────────────────

async def _loop(max_iters: int, fast_mode: bool) -> int:
    p = _getpaths()
    log = HarnessLog(p["log_file"])

    stagnation_k = _env_int("STAGNATION_K", 3)
    research_every = _env_int("RESEARCH_EVERY", 10)
    cooldown_secs = _env_int("COOLDOWN_SECS", 1)
    train_budget_min = _env_int("TRAIN_BUDGET_MINUTES", 10)
    os.environ["TRAIN_BUDGET_MINUTES"] = str(train_budget_min)

    provider_name = os.environ.get("AUTOMEDAL_PROVIDER", "opencode-go")
    model_name = os.environ.get("AUTOMEDAL_MODEL")
    if not model_name:
        # back-compat with MODEL=provider/model from run.sh days
        slug = os.environ.get("MODEL", "opencode-go/minimax-m2.7")
        if "/" in slug:
            provider_name, model_name = slug.split("/", 1)
        else:
            model_name = slug

    analyzer_on = _env_bool("AUTOMEDAL_ANALYZER", True)
    quick_reject_on = _env_bool("AUTOMEDAL_QUICK_REJECT", False)
    dedupe_on = _env_bool("AUTOMEDAL_DEDUPE", True)
    regression_gate = os.environ.get("AUTOMEDAL_REGRESSION_GATE", "warn")

    log.banner([
        "==================================================",
        "  AutoMedal — Four-Phase Loop (bespoke kernel)",
        f"  Provider:       {provider_name}",
        f"  Model:          {model_name}",
        f"  Iterations:     {max_iters}",
        f"  Stagnation K:   {stagnation_k}",
        f"  Research every: {research_every}",
        f"  Analyzer:       {'on' if analyzer_on else 'off'}",
        f"  Quick-reject:   {'on' if quick_reject_on else 'off'}",
        f"  Dedupe:         {'on' if dedupe_on else 'off'}",
        f"  Cooldown:       {cooldown_secs}s",
        f"  Train budget:   {train_budget_min}m",
        f"  Log:            {p['log_file']}",
        f"  Events:         {p['events_file']}",
        f"  Started:        {datetime.now().isoformat(timespec='seconds')}",
        "==================================================",
    ])

    # Ensure data exists (first run after bootstrap)
    if not (p["data_dir"] / "X_train.npy").exists():
        log.write("Preparing data first...")
        r = subprocess.run([sys.executable, str(p["prepare_py"])],
                           cwd=str(REPO_ROOT), capture_output=True, text=True)
        if r.stdout:
            log.write(r.stdout.rstrip("\n"))
        if r.stderr:
            log.write(r.stderr.rstrip("\n"))

    # Ensure memory files exist
    subprocess.run([sys.executable, str(p["harness_dir"] / "init_memory.py")],
                   cwd=str(REPO_ROOT), capture_output=True, text=True)

    # Cancel flag
    cancel = {"flag": False}

    def _on_term(signum, frame):
        log.harness("SIGTERM received — finishing this iteration then stopping")
        cancel["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    provider = build_provider(provider_name, model_name, max_tokens=8192, timeout=120)

    with EventSink(jsonl_path=p["events_file"], human_path=None) as sink:
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
                log.harness(f"dispatching Researcher ({trigger})")
                rep = await p_researcher.run(
                    provider=provider, events=sink,
                    exp_id=exp_id, trigger=trigger,
                    stagnating=(stagnating == "1"),
                    scheduled_research=scheduled_research,
                    best_val_loss=best,
                )
                if rep.stop != "assistant_done":
                    log.write(f"  [WARN] Researcher stopped {rep.stop}: {rep.error or ''}")
                _verify(p["harness_dir"], "--phase", "researcher", log=log)

            # ── Strategist ───────────────────────────────────────────────
            pending = _count_pending_queue(p["queue_md"])
            if pending == 0 or stagnating == "1":
                reflective = _build_trace_trailer(p["harness_dir"])
                ranked = _rank_journals(p["harness_dir"])
                log.harness(
                    f"dispatching Strategist (queue_pending={pending}, stagnating={stagnating})"
                )
                rep = await p_strategist.run(
                    provider=provider, events=sink,
                    exp_id=exp_id, iteration=i, max_iters=max_iters,
                    stagnating=(stagnating == "1"),
                    best_val_loss=best, pending=pending,
                    reflective=reflective, ranked=ranked,
                )
                if rep.stop != "assistant_done":
                    log.write(f"  [WARN] Strategist stopped {rep.stop}: {rep.error or ''}")
                _verify(p["harness_dir"], "--phase", "strategist", log=log)

                if dedupe_on:
                    summary = dedupe_mod.apply(
                        queue_path=p["queue_md"],
                        journal_path=p["journal_dir"],
                    )
                    log.harness(
                        f"dedupe: scanned={summary['scanned']} marked={summary['marked']} "
                        f"threshold={summary['threshold']:.2f} journal_n={summary['journal_n']}"
                    )

            # Tag repo BEFORE Experimenter so journal has a stable ref
            _tag_exp(exp_id, log)

            # ── Quick-reject (optional pre-train guard) ───────────────────
            if quick_reject_on:
                log.harness("running quick-reject smoke train (budget=30s)")
                ok, reason = await quick_reject_mod.smoke_train(budget_s=30)
                log.harness(f"quick-reject: {'PASS' if ok else 'FAIL'} — {reason}")
                if not ok:
                    log.iteration_end(i, exp_id)
                    if not fast_mode and i < max_iters and cooldown_secs > 0:
                        await asyncio.sleep(cooldown_secs)
                    continue

            # ── Experimenter (edit) ───────────────────────────────────────
            log.harness("dispatching Experimenter (edit)")
            rep = await p_exp_edit.run(
                provider=provider, events=sink,
                exp_id=exp_id, best_val_loss=best,
            )
            if rep.stop != "assistant_done":
                log.write(f"  [WARN] Experimenter (edit) stopped {rep.stop}: {rep.error or ''}")

            # ── Training (harness-managed, no agent) ──────────────────────
            train_timeout = train_budget_min * 60 + 30
            log.harness(f"running training (budget={train_budget_min}m, timeout={train_timeout}s)...")

            r = subprocess.run(["git", "diff", "--quiet", "--", str(p["prepare_py"])],
                               cwd=str(REPO_ROOT))
            if r.returncode != 0:
                log.harness("prepare.py changed — running prepare first")
                subprocess.run([sys.executable, str(p["prepare_py"])], cwd=str(REPO_ROOT))

            train_rc, train_out = _run_train(p["train_py"], p["last_train_out"], train_timeout, log)
            final_loss = _parse_final_val_loss(train_out)
            log.harness(f"training done: val_loss={final_loss} exit={train_rc}")

            # ── Experimenter (eval) ───────────────────────────────────────
            log.harness("dispatching Experimenter (eval)")
            rep = await p_exp_eval.run(
                provider=provider, events=sink,
                exp_id=exp_id, best_val_loss=best,
                train_rc=train_rc, final_loss=final_loss,
            )
            if rep.stop != "assistant_done":
                log.write(f"  [WARN] Experimenter (eval) stopped {rep.stop}: {rep.error or ''}")

            _, best_after = _check_stagnation(p["harness_dir"], stagnation_k)

            if final_loss != "nan":
                verify_rc = _verify(
                    p["harness_dir"],
                    "--phase", "experimenter", "--exp-id", exp_id,
                    "--val-loss", final_loss, "--best-before", best,
                    "--best-so-far", best_after or best, log=log,
                )
            else:
                verify_rc = _verify(
                    p["harness_dir"], "--phase", "experimenter", "--exp-id", exp_id, log=log
                )

            if verify_rc == 2 and regression_gate == "strict":
                log.harness(f"strict regression gate triggered — reverting git tag exp/{exp_id}")
                subprocess.run(["git", "tag", "-d", f"exp/{exp_id}"], cwd=str(REPO_ROOT))

            if verify_rc == 3:
                log.harness("success_criteria near-miss — attempting one retry edit")
                rep = await p_exp_edit.run(
                    provider=provider, events=sink,
                    exp_id=exp_id, best_val_loss=best,
                    retry=True, prev_loss=final_loss,
                )
                if rep.stop != "assistant_done":
                    log.write(f"  [WARN] Experimenter (retry) stopped {rep.stop}: {rep.error or ''}")
                train_rc2, train_out2 = _run_train(
                    p["train_py"], p["last_train_out"], train_timeout, log
                )
                retry_loss = _parse_final_val_loss(train_out2)
                log.harness(f"retry training done: val_loss={retry_loss}")
                rep = await p_exp_eval.run(
                    provider=provider, events=sink,
                    exp_id=exp_id, best_val_loss=best,
                    train_rc=train_rc2, final_loss=retry_loss,
                )
                if rep.stop != "assistant_done":
                    log.write(f"  [WARN] Experimenter (eval-retry) stopped {rep.stop}: {rep.error or ''}")
                _verify(p["harness_dir"], "--phase", "experimenter", "--exp-id", exp_id, log=log)
                final_loss = retry_loss  # for analyzer below

            # ── Analyzer (4th phase) ──────────────────────────────────────
            if analyzer_on:
                front = _parse_journal_after(p["journal_dir"], exp_id)
                slug = front.get("slug", "")
                status = front.get("status", "unknown")
                delta = front.get("val_loss_delta", "0.0")
                log.harness(f"dispatching Analyzer (status={status})")
                rep = await p_analyzer.run(
                    provider=provider, events=sink,
                    exp_id=exp_id, slug=slug, status=status,
                    final_loss=final_loss, best_val_loss=best_after or best,
                    val_loss_delta=delta,
                )
                if rep.stop != "assistant_done":
                    log.write(f"  [WARN] Analyzer stopped {rep.stop}: {rep.error or ''}")

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
