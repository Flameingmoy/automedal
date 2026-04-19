#!/usr/bin/env bash
# AutoMedal — one-shot installer.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/OWNER/REPO/main/install.sh | bash
#   # or point AUTOMEDAL_REF at a branch/tag:
#   curl -LsSf .../install.sh | AUTOMEDAL_REF=v1.0.0 bash
#
# What it does:
#   1. Verifies Python ≥ 3.11 is on PATH.
#   2. Installs (or upgrades) `pipx`.
#   3. Installs AutoMedal from GitHub with `pipx`.
#   4. Prints next-step hints.
#
# Upgrading later:        pipx upgrade automedal
# Uninstalling later:     pipx uninstall automedal

set -euo pipefail

REPO="${AUTOMEDAL_REPO:-github.com/OWNER/REPO}"
REF="${AUTOMEDAL_REF:-main}"
PKG_SPEC="git+https://${REPO}@${REF}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ── 1. Python ──────────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH."
PYVER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
python3 -c 'import sys; exit(0 if sys.version_info >= (3,11) and sys.version_info < (3,14) else 1)' \
    || die "Python ${PYVER} is unsupported. AutoMedal requires Python >=3.11,<3.14."
say "Found python3 ${PYVER}"

# ── 2. pipx ────────────────────────────────────────────────────────────────
if ! command -v pipx >/dev/null 2>&1; then
    say "pipx not found — installing with 'pip install --user pipx'"
    python3 -m pip install --user --upgrade pipx >/dev/null
    python3 -m pipx ensurepath >/dev/null
    # Make pipx visible in the current shell session
    export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v pipx >/dev/null 2>&1 || die "pipx still not on PATH — open a new shell and rerun."
say "pipx ready"

# ── 3. AutoMedal ───────────────────────────────────────────────────────────
if pipx list --short 2>/dev/null | grep -q '^automedal'; then
    say "AutoMedal already installed — upgrading from ${PKG_SPEC}"
    pipx upgrade automedal --pip-args="${PKG_SPEC}" 2>/dev/null \
        || pipx install --force "${PKG_SPEC}"
else
    say "Installing AutoMedal from ${PKG_SPEC}"
    pipx install "${PKG_SPEC}"
fi

# ── 4. Directory scaffold ──────────────────────────────────────────────────
mkdir -p "${HOME}/.automedal"
chmod 700 "${HOME}/.automedal"
say "Created ${HOME}/.automedal/  (keys, logs, sprites)"

# ── 5. Smoke-verify ────────────────────────────────────────────────────────
if ! command -v automedal >/dev/null 2>&1; then
    warn "The 'automedal' command is not on PATH in this shell."
    warn "Run:  pipx ensurepath  &&  exec \$SHELL"
    exit 0
fi

VERSION=$(automedal version 2>/dev/null || echo "unknown")
say "Installed: ${VERSION}"

cat <<EOF

Next steps:
  automedal setup                       # paste a provider API key
  automedal discover                    # browse active Kaggle competitions
  automedal init <slug>                 # wire up a competition
  automedal run 50                      # start the loop

Upgrade later:
  pipx upgrade automedal
  # or:  curl -LsSf https://${REPO}/raw/${REF}/install.sh | bash

Docs: https://${REPO}
EOF
