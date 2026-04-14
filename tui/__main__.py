"""`python -m tui` / `automedal-tui` entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="automedal-tui", description="AutoMedal observation dashboard")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to agent_loop.log (default: <repo>/agent_loop.log)")
    parser.add_argument("--demo", action="store_true", help="Replay tests/fixtures/demo_agent_loop.log")
    parser.add_argument("--demo-fixture", type=Path, default=None, help="Custom demo fixture path")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repo root (default: cwd)")
    args = parser.parse_args()

    repo_root = args.repo.resolve()
    demo_fixture = None
    if args.demo:
        demo_fixture = args.demo_fixture or (repo_root / "tests" / "fixtures" / "demo_agent_loop.log")
        if not demo_fixture.exists():
            print(f"demo fixture not found: {demo_fixture}", file=sys.stderr)
            return 2

    from tui.app import AutoMedalApp
    app = AutoMedalApp(repo_root=repo_root, log_file=args.log_file, demo_fixture=demo_fixture)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
