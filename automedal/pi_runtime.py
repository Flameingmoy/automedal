"""Detect or auto-install the pi coding agent.

Resolution order:
1. AUTOMEDAL_PI_BIN env var (dev override or explicit path)
2. Package-internal _vendor/node_modules/.bin/pi  (user-mode: installed by us)
3. System `pi` on PATH                           (dev-mode: global npm install)
4. Auto-install into _vendor/ via npm            (first-run, needs Node >=22)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PI_VERSION = "0.66.1"

_PKG_DIR = Path(__file__).parent
_VENDOR_DIR = _PKG_DIR / "_vendor"
_VENDOR_BIN = _VENDOR_DIR / "node_modules" / ".bin" / "pi"


def ensure_pi() -> Path:
    """Return the absolute path to the pi binary, installing it on first run."""
    # 1. Explicit override (dev shortcut or CI)
    override = os.environ.get("AUTOMEDAL_PI_BIN")
    if override:
        p = Path(override)
        if p.exists():
            return p

    # 2. Already installed in vendor dir
    if _VENDOR_BIN.exists():
        return _VENDOR_BIN

    # 3. System pi on PATH (developer with global npm install)
    system_pi = shutil.which("pi")
    if system_pi:
        return Path(system_pi)

    # 4. Auto-install via npm
    _npm_install()
    if _VENDOR_BIN.exists():
        return _VENDOR_BIN

    sys.exit(
        f"npm install succeeded but {_VENDOR_BIN} was not created.\n"
        "Please file a bug at the AutoMedal repo."
    )


def _npm_install() -> None:
    node = shutil.which("node") or shutil.which("nodejs")
    if not node:
        sys.exit(
            "Node.js >=22 is required to run the AutoMedal agent.\n"
            "Install it from https://nodejs.org/ then run 'automedal' again."
        )
    npm = shutil.which("npm")
    if not npm:
        sys.exit("npm not found. Install Node.js >=22 from https://nodejs.org/")

    print(f"Installing pi agent (v{PI_VERSION}) into {_VENDOR_DIR} …", flush=True)
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [npm, "install", "--prefix", str(_VENDOR_DIR),
             f"@mariozechner/pi-coding-agent@{PI_VERSION}"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"npm install failed (exit {exc.returncode}). Check your network and Node version.")


def pi_version() -> str:
    """Return the pi version string, or '(unknown)' on error."""
    try:
        pi = ensure_pi()
        result = subprocess.run([str(pi), "--version"], capture_output=True, text=True, timeout=10)
        return (result.stdout or result.stderr).strip()
    except Exception:
        return "(unknown)"
