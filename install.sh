#!/usr/bin/env bash
# AutoMedal — one-shot installer.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/Flameingmoy/automedal/main/install.sh | bash
#   # or pin a tag:
#   curl -LsSf .../install.sh | AUTOMEDAL_REF=v2.0.0 bash
#
# Phase 4: AutoMedal is a single Go binary. Only one Python dependency
# remains: the `sniff` shim in py-shim/sniff/ used by `automedal init`
# for pandas-backed CSV schema inference. Installed via pipx (preferred)
# or pip --user (fallback).
#
# What it does:
#   1. Verifies Go ≥ 1.24 and (optional) Python ≥ 3.10 are on PATH.
#   2. Builds and installs the `automedal` binary into ~/.local/bin.
#   3. Installs the `sniff` Python shim via pipx (or skips with a hint).
#   4. Prints next-step hints.

set -euo pipefail

REPO="${AUTOMEDAL_REPO:-github.com/Flameingmoy/automedal}"
REF="${AUTOMEDAL_REF:-main}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ── 1. Go ──────────────────────────────────────────────────────────────────
command -v go >/dev/null 2>&1 || die "go not found on PATH (need Go ≥ 1.24)."
GOVER=$(go version | awk '{print $3}' | sed 's/^go//')
say "Found go ${GOVER}"

# ── 2. Build the binaries ─────────────────────────────────────────────────
# Two binaries ship: `automedal` (CLI / control plane) and `automedal-tui`
# (the v2 Bubbletea UI — Hermes banner, spring nav, drill-down events).
INSTALL_DIR="${AUTOMEDAL_INSTALL_DIR:-${HOME}/.local/bin}"
mkdir -p "${INSTALL_DIR}"

# When piped from curl we won't have the source on disk — fall back to
# `go install`. When run from a checkout, prefer the local tree so the
# user gets exactly what they see.
if [[ -f go.mod && -f cmd/automedal/main.go ]]; then
    say "Building automedal + automedal-tui from local checkout → ${INSTALL_DIR}/"
    GOBIN="${INSTALL_DIR}" go install ./cmd/automedal
    go build -o "${INSTALL_DIR}/automedal-tui" ./internal/ui
else
    TMP="$(mktemp -d)"
    say "Cloning ${REPO}@${REF} → ${TMP}"
    git clone --depth 1 --branch "${REF}" "https://${REPO}.git" "${TMP}" >/dev/null
    say "Building automedal + automedal-tui → ${INSTALL_DIR}/"
    (cd "${TMP}" && \
        GOBIN="${INSTALL_DIR}" go install ./cmd/automedal && \
        go build -o "${INSTALL_DIR}/automedal-tui" ./internal/ui)
    rm -rf "${TMP}"
fi
say "Built ${INSTALL_DIR}/automedal"
say "Built ${INSTALL_DIR}/automedal-tui"

# ── 3. Python shim (optional but recommended) ──────────────────────────────
mkdir -p "${HOME}/.automedal"
chmod 700 "${HOME}/.automedal"

SHIM_DIR=""
if [[ -d py-shim/sniff ]]; then
    SHIM_DIR="$(pwd)/py-shim/sniff"
fi

if command -v python3 >/dev/null 2>&1; then
    PYVER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    say "Found python3 ${PYVER} (used by `automedal init` only)"

    if command -v pipx >/dev/null 2>&1; then
        if [[ -n "${SHIM_DIR}" ]]; then
            say "Installing sniff shim via pipx from ${SHIM_DIR}"
            pipx install --force "${SHIM_DIR}" || warn "pipx install failed — try: pip install --user '${SHIM_DIR}'"
        else
            warn "py-shim/sniff/ not found in this directory; skipping shim install"
            warn "Run from a repo checkout, or:"
            warn "  pipx install 'git+https://${REPO}@${REF}#subdirectory=py-shim/sniff'"
        fi
    else
        warn "pipx not installed — falling back to 'pip install --user'."
        if [[ -n "${SHIM_DIR}" ]]; then
            python3 -m pip install --user --upgrade "${SHIM_DIR}" || warn "pip install failed"
        fi
    fi
else
    warn "python3 not found — `automedal init` will fail until you install Python ≥ 3.10"
    warn "and the sniff shim:"
    warn "  pipx install 'git+https://${REPO}@${REF}#subdirectory=py-shim/sniff'"
fi

# ── 4. PATH check ─────────────────────────────────────────────────────────
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *) warn "${INSTALL_DIR} is not on PATH — add it to your shell rc:"
       warn "  export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

# ── 5. Smoke-verify ───────────────────────────────────────────────────────
if ! command -v automedal >/dev/null 2>&1; then
    warn "The 'automedal' command is not yet on PATH in this shell."
    warn "Run:  export PATH=\"${INSTALL_DIR}:\$PATH\"  &&  automedal version"
    exit 0
fi
VERSION=$(automedal version 2>/dev/null || echo "unknown")
say "Installed: ${VERSION}"

cat <<EOF

Next steps:
  automedal setup                       # paste a provider API key
  automedal discover                    # browse active Kaggle competitions
  automedal init <slug>                 # wire up a competition (uses sniff shim)
  automedal run 50                      # start the loop
  automedal-tui                         # open the live TUI (Hermes home + spring nav)

Upgrade later:
  GOBIN=${INSTALL_DIR} go install ${REPO}/cmd/automedal@${REF}
  # plus rebuild the TUI from a checkout: go build -o ${INSTALL_DIR}/automedal-tui ./internal/ui
  pipx upgrade automedal-sniff

Docs: https://${REPO}
EOF
