"""Tests for automedal.pi_runtime — mock npm so no network calls are made."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import automedal.pi_runtime as pi_runtime


def _make_fake_pi(vendor_dir: Path) -> Path:
    """Create a dummy pi binary in the vendor dir."""
    bin_dir = vendor_dir / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    pi = bin_dir / "pi"
    pi.write_text("#!/bin/bash\necho pi-fake\n")
    pi.chmod(0o755)
    return pi


def test_env_override_takes_priority(tmp_path, monkeypatch):
    """AUTOMEDAL_PI_BIN env var is returned first if the binary exists."""
    fake = tmp_path / "my_pi"
    fake.write_text("#!/bin/bash\necho hi\n")
    fake.chmod(0o755)
    monkeypatch.setenv("AUTOMEDAL_PI_BIN", str(fake))
    result = pi_runtime.ensure_pi()
    assert result == fake


def test_vendor_bin_used_when_present(tmp_path, monkeypatch):
    """Package-internal vendor copy is preferred over system pi."""
    monkeypatch.delenv("AUTOMEDAL_PI_BIN", raising=False)
    fake_pi = _make_fake_pi(tmp_path)
    # Patch _VENDOR_BIN to point at our temp dir
    monkeypatch.setattr(pi_runtime, "_VENDOR_BIN", fake_pi)
    result = pi_runtime.ensure_pi()
    assert result == fake_pi


def test_system_pi_fallback(tmp_path, monkeypatch):
    """Falls back to system `pi` on PATH when vendor copy absent."""
    monkeypatch.delenv("AUTOMEDAL_PI_BIN", raising=False)
    # Vendor bin does not exist
    monkeypatch.setattr(pi_runtime, "_VENDOR_BIN", tmp_path / "no_such_bin")

    fake_system_pi = tmp_path / "pi"
    fake_system_pi.write_text("#!/bin/bash\necho hi\n")
    fake_system_pi.chmod(0o755)

    with patch("automedal.pi_runtime.shutil.which", return_value=str(fake_system_pi)):
        result = pi_runtime.ensure_pi()
    assert result == fake_system_pi


def test_auto_install_called_when_nothing_found(tmp_path, monkeypatch):
    """When neither vendor nor system pi exists, _npm_install is called."""
    monkeypatch.delenv("AUTOMEDAL_PI_BIN", raising=False)
    monkeypatch.setattr(pi_runtime, "_VENDOR_BIN", tmp_path / "no_such_bin")
    monkeypatch.setattr(pi_runtime, "_VENDOR_DIR", tmp_path)

    def fake_which(name):
        return None  # pi not on PATH

    called = []

    def fake_npm_install():
        # Create the vendor bin so ensure_pi can return it
        _make_fake_pi(tmp_path)
        monkeypatch.setattr(pi_runtime, "_VENDOR_BIN", tmp_path / "node_modules" / ".bin" / "pi")
        called.append(True)

    with patch("automedal.pi_runtime.shutil.which", side_effect=fake_which):
        with patch.object(pi_runtime, "_npm_install", side_effect=fake_npm_install):
            result = pi_runtime.ensure_pi()

    assert called, "_npm_install was not invoked"
    assert result.name == "pi"


def test_npm_install_exits_without_node(tmp_path, monkeypatch):
    """_npm_install sys.exits with a clear message when node is missing."""
    monkeypatch.setattr(pi_runtime, "_VENDOR_DIR", tmp_path)
    with patch("automedal.pi_runtime.shutil.which", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            pi_runtime._npm_install()
    assert "Node" in str(exc_info.value) or "node" in str(exc_info.value).lower()
