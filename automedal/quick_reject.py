"""Pre-train smoke guard.

Runs `python agent/train.py` for at most `budget_s` seconds with the
`AUTOMEDAL_QUICK_REJECT=1` environment variable set so the script can
short-circuit (e.g., one tiny epoch / one HPO trial) and still print a
final_val_loss line. We accept the run if:
    - the process exited with code 0, AND
    - we found a finite `final_val_loss=` line in stdout, AND
    - the loss is not absurd (configurable upper bound)

If the process is still running at the budget, we kill it and accept
(slow ≠ broken). The guard is intended to catch obvious blunders like
`lr=1e10`, mismatched feature shapes, or import-time exceptions.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from automedal.agent.tools.base import REPO_ROOT


_FINAL_RE = re.compile(r"final_val_loss=([0-9.eE+-]+)")


def _train_py() -> Path:
    return Path(os.environ.get("AUTOMEDAL_TRAIN_PY", REPO_ROOT / "agent" / "train.py"))


async def smoke_train(
    *,
    budget_s: int = 30,
    max_acceptable_loss: float = 100.0,
) -> tuple[bool, str]:
    """Return (accepted, reason). `accepted=False` means the iteration should be aborted."""
    train_py = _train_py()
    if not train_py.exists():
        return True, f"train.py not found at {train_py}; skipping quick-reject"

    env = dict(os.environ)
    env["AUTOMEDAL_QUICK_REJECT"] = "1"
    # Hint to the script: keep it short.
    env.setdefault("TRAIN_BUDGET_MINUTES", "1")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(train_py),
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except OSError as exc:
        return True, f"could not launch train.py: {exc}; skipping quick-reject"

    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=budget_s)
        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    except asyncio.TimeoutError:
        # Process exceeded the budget but isn't necessarily broken.
        try:
            proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        return True, f"smoke train still running at {budget_s}s budget (slow ≠ broken)"

    if proc.returncode != 0:
        tail = "\n".join(stdout.splitlines()[-12:]) if stdout else "(no stdout)"
        return False, f"smoke train exited code={proc.returncode}\n--- last lines ---\n{tail}"

    m = _FINAL_RE.search(stdout)
    if not m:
        return False, "smoke train produced no `final_val_loss=` line"
    try:
        loss = float(m.group(1))
    except ValueError:
        return False, f"smoke train produced unparseable val_loss={m.group(1)!r}"
    if loss != loss:  # NaN check
        return False, "smoke train val_loss is NaN"
    if loss > max_acceptable_loss:
        return False, f"smoke train val_loss={loss:.4f} > {max_acceptable_loss} (likely diverged)"

    return True, f"smoke train ok (val_loss={loss:.4f}, exit=0)"
