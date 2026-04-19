"""Layout — resolves all file paths for dev mode (this repo) and user mode (installed).

Dev mode: cwd has pyproject.toml + an `automedal/` package dir → flat layout as-is.
User mode: everything else → hidden layout under .automedal/ in cwd.

Every module that touches disk should resolve paths through Layout so the same
code works both in the developer's checkout and in a pipx-installed copy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

_PKG_DIR = Path(__file__).parent  # automedal/ inside site-packages or repo root


def _detect_mode(cwd: Path) -> Literal["dev", "user"]:
    """Return 'dev' when running inside the AutoMedal repo, 'user' otherwise."""
    if os.environ.get("AUTOMEDAL_DEV"):
        return "dev"
    # Dev mode = AutoMedal's own checkout (pyproject.toml + the package dir)
    if (cwd / "pyproject.toml").exists() and (cwd / "automedal" / "run_loop.py").exists():
        return "dev"
    return "user"


class Layout:
    """Centralised path resolver for AutoMedal file locations.

    Usage:
        layout = Layout()                    # auto-detects mode from cwd
        layout = Layout(cwd=Path("/proj"))   # explicit cwd
        layout = Layout(mode="user")         # force user mode

    Env var export:
        env = {**os.environ, **layout.as_env()}
        subprocess.run(cmd, env=env)
    """

    def __init__(
        self,
        cwd: Path | None = None,
        mode: Literal["dev", "user"] | None = None,
    ) -> None:
        self.cwd = (cwd or Path.cwd()).resolve()
        self.mode: Literal["dev", "user"] = mode or _detect_mode(self.cwd)

    # ── always visible to the user (same in both modes) ───────────────────

    @property
    def data_dir(self) -> Path:
        return self.cwd / "data"

    @property
    def submissions_dir(self) -> Path:
        return self.cwd / "submissions"

    @property
    def journal_dir(self) -> Path:
        return self.cwd / "journal"

    @property
    def knowledge_md(self) -> Path:
        return self.cwd / "knowledge.md"

    @property
    def queue_md(self) -> Path:
        return self.cwd / "experiment_queue.md"

    @property
    def research_md(self) -> Path:
        return self.cwd / "research_notes.md"

    @property
    def results_tsv(self) -> Path:
        """agent/results.tsv in dev; results.tsv at root in user mode."""
        if self.mode == "dev":
            return self.cwd / "agent" / "results.tsv"
        return self.cwd / "results.tsv"

    # ── hidden in user mode; repo-root in dev mode ────────────────────────

    @property
    def hidden_root(self) -> Path:
        return self.cwd / ".automedal" if self.mode == "user" else self.cwd

    @property
    def agent_dir(self) -> Path:
        return self.hidden_root / "agent" if self.mode == "user" else self.cwd / "agent"

    @property
    def train_py(self) -> Path:
        return self.agent_dir / "train.py"

    @property
    def prepare_py(self) -> Path:
        return self.agent_dir / "prepare.py"

    @property
    def config_yaml(self) -> Path:
        return (
            self.hidden_root / "configs" / "competition.yaml"
            if self.mode == "user"
            else self.cwd / "configs" / "competition.yaml"
        )

    @property
    def agents_md(self) -> Path:
        return (
            self.hidden_root / "AGENTS.md"
            if self.mode == "user"
            else self.cwd / "AGENTS.md"
        )

    @property
    def log_file(self) -> Path:
        if self.mode == "user":
            return self.hidden_root / "logs" / "agent_loop.log"
        # In dev mode respect LOG_FILE env var (same as run.sh does)
        env_log = os.environ.get("AUTOMEDAL_LOG_FILE") or os.environ.get("LOG_FILE")
        if env_log:
            return Path(env_log)
        return self.cwd / "agent_loop.log"

    @property
    def events_file(self) -> Path:
        """JSONL event sink for the bespoke agent kernel."""
        if self.mode == "user":
            return self.hidden_root / "logs" / "agent_loop.events.jsonl"
        env_evt = os.environ.get("AUTOMEDAL_EVENTS_FILE")
        if env_evt:
            return Path(env_evt)
        return self.cwd / "agent_loop.events.jsonl"

    @property
    def last_training_output(self) -> Path:
        return (
            self.hidden_root / "cache" / ".last_training_output"
            if self.mode == "user"
            else self.cwd / "harness" / ".last_training_output"
        )

    # ── always in installed package / repo root ───────────────────────────

    @property
    def package_dir(self) -> Path:
        return _PKG_DIR

    @property
    def prompts_dir(self) -> Path:
        return _PKG_DIR / "prompts" if self.mode == "user" else self.cwd / "prompts"

    @property
    def templates_dir(self) -> Path:
        return _PKG_DIR / "templates" if self.mode == "user" else self.cwd / "templates"

    @property
    def harness_dir(self) -> Path:
        return _PKG_DIR / "harness" if self.mode == "user" else self.cwd / "harness"

    @property
    def scout_dir(self) -> Path:
        return _PKG_DIR / "scout" if self.mode == "user" else self.cwd / "scout"

    # ── convenience ───────────────────────────────────────────────────────

    def as_env(self) -> dict[str, str]:
        """Return a dict of env vars for injecting into subprocesses."""
        d: dict[str, str] = {
            "AUTOMEDAL_CWD": str(self.cwd),
            "AUTOMEDAL_MODE": self.mode,
            "AUTOMEDAL_DATA_DIR": str(self.data_dir),
            "AUTOMEDAL_SUBMISSIONS_DIR": str(self.submissions_dir),
            "AUTOMEDAL_JOURNAL_DIR": str(self.journal_dir),
            "AUTOMEDAL_KNOWLEDGE_MD": str(self.knowledge_md),
            "AUTOMEDAL_QUEUE_MD": str(self.queue_md),
            "AUTOMEDAL_RESEARCH_MD": str(self.research_md),
            "AUTOMEDAL_RESULTS_TSV": str(self.results_tsv),
            "AUTOMEDAL_HIDDEN_ROOT": str(self.hidden_root),
            "AUTOMEDAL_AGENT_DIR": str(self.agent_dir),
            "AUTOMEDAL_TRAIN_PY": str(self.train_py),
            "AUTOMEDAL_PREPARE_PY": str(self.prepare_py),
            "AUTOMEDAL_CONFIG_YAML": str(self.config_yaml),
            "AUTOMEDAL_LOG_FILE": str(self.log_file),
            "AUTOMEDAL_EVENTS_FILE": str(self.events_file),
            "AUTOMEDAL_LAST_TRAINING_OUTPUT": str(self.last_training_output),
            "AUTOMEDAL_PROMPTS_DIR": str(self.prompts_dir),
            "AUTOMEDAL_TEMPLATES_DIR": str(self.templates_dir),
            "AUTOMEDAL_HARNESS_DIR": str(self.harness_dir),
            "AUTOMEDAL_SCOUT_DIR": str(self.scout_dir),
            "LOG_FILE": str(self.log_file),
        }
        return d

    def __repr__(self) -> str:
        return f"Layout(cwd={self.cwd!r}, mode={self.mode!r})"
