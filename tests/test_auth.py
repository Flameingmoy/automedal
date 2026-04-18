"""Tests for automedal.auth — .env store + legacy pi auth import."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from automedal import auth


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Isolate ENV_FILE to tmp_path and strip provider keys from os.environ."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr(auth, "ENV_FILE", env_file)
    for var in auth.PROVIDER_ENV.values():
        if var:
            monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return env_file


def test_save_key_writes_mode_0600(tmp_env):
    path = auth.save_key("opencode-go", "sk-test-123", env_file=tmp_env)

    assert path == tmp_env
    assert path.exists()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"
    assert "OPENCODE_API_KEY=sk-test-123" in path.read_text()
    assert os.environ["OPENCODE_API_KEY"] == "sk-test-123"


def test_save_key_rejects_unknown_provider(tmp_env):
    with pytest.raises(ValueError, match="unknown provider"):
        auth.save_key("not-a-real-provider", "sk-x", env_file=tmp_env)


def test_save_key_rejects_ollama(tmp_env):
    """Ollama uses OLLAMA_HOST, not an API key, so save_key should refuse."""
    with pytest.raises(ValueError, match="does not take an API key"):
        auth.save_key("ollama", "anything", env_file=tmp_env)


def test_load_env_populates_os_environ(tmp_env, monkeypatch):
    tmp_env.write_text("ANTHROPIC_API_KEY=sk-ant-abc\n")
    tmp_env.chmod(0o600)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = auth.load_env(env_file=tmp_env)

    assert loaded is True
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc"


def test_load_env_missing_returns_false(tmp_path):
    assert auth.load_env(env_file=tmp_path / "does-not-exist.env") is False


def test_configured_providers_reads_env_only(tmp_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abc")

    active = auth.configured_providers()

    assert "openai" in active
    assert "groq" in active
    assert "anthropic" not in active


def test_configured_providers_ollama_requires_host(tmp_env, monkeypatch):
    # Ollama only counts as configured when OLLAMA_HOST is set
    assert "ollama" not in auth.configured_providers()
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    assert "ollama" in auth.configured_providers()


def test_needs_setup_true_when_empty(tmp_env, monkeypatch):
    # No env keys, no pi auth file → needs setup
    monkeypatch.setattr(auth, "_PI_AUTH", tmp_env.parent / "nonexistent.json")
    assert auth.needs_setup() is True


def test_needs_setup_false_when_env_has_key(tmp_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
    assert auth.needs_setup() is False


def test_import_pi_auth_migrates_keys(tmp_env, tmp_path, monkeypatch):
    pi_auth = tmp_path / "auth.json"
    pi_auth.write_text(json.dumps({
        "opencode-go": {"type": "api_key", "key": "sk-ocg"},
        "anthropic":   {"type": "api_key", "key": "sk-ant-legacy"},
        "malformed":   "not a dict",
        "ollama":      {"type": "api_key", "key": ""},  # empty → skip
    }))
    monkeypatch.setattr(auth, "_PI_AUTH", pi_auth)

    imported = auth.import_pi_auth(env_file=tmp_env)

    assert set(imported) == {"opencode-go", "anthropic"}
    contents = tmp_env.read_text()
    assert "OPENCODE_API_KEY=sk-ocg" in contents
    assert "ANTHROPIC_API_KEY=sk-ant-legacy" in contents
    assert tmp_env.stat().st_mode & 0o777 == 0o600


def test_import_pi_auth_missing_file_returns_empty(tmp_env, monkeypatch):
    monkeypatch.setattr(auth, "_PI_AUTH", tmp_env.parent / "nope.json")
    assert auth.import_pi_auth(env_file=tmp_env) == []
