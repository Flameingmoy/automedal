"""Live-fetch + cache for advisor model autocompletion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from automedal.advisor import models as adv_models


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect the cache file into a tmp dir; clear advisor env vars."""
    monkeypatch.setattr(adv_models, "CACHE_PATH", tmp_path / "models_cache.json")
    monkeypatch.delenv("AUTOMEDAL_ADVISOR_BASE_URL", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    yield


def test_returns_fallback_when_no_cache_and_no_key():
    out = adv_models.list_models()
    assert out == list(adv_models._FALLBACK_MODELS)
    assert "kimi-k2.6" in out


def test_returns_cached_when_no_key_present(monkeypatch, tmp_path):
    cache = {
        "base_url": adv_models.DEFAULT_BASE_URL,
        "fetched_at": time.time(),
        "models": ["alpha", "beta"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    assert adv_models.list_models() == ["alpha", "beta"]


def test_fresh_cache_short_circuits_no_fetch(monkeypatch):
    cache = {
        "base_url": adv_models.DEFAULT_BASE_URL,
        "fetched_at": time.time(),
        "models": ["cached-model"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")

    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")

    def explode(*a, **kw):
        raise AssertionError("fetch should not be called when cache is fresh")

    monkeypatch.setattr(adv_models, "_fetch_remote", explode)
    assert adv_models.list_models() == ["cached-model"]


def test_stale_cache_triggers_fetch(monkeypatch):
    cache = {
        "base_url": adv_models.DEFAULT_BASE_URL,
        "fetched_at": time.time() - (adv_models.CACHE_TTL_SECS + 100),
        "models": ["old"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")
    monkeypatch.setattr(adv_models, "_fetch_remote", lambda b, k: ["new-a", "new-b"])

    out = adv_models.list_models()
    assert out == ["new-a", "new-b"]
    written = json.loads(adv_models.CACHE_PATH.read_text(encoding="utf-8"))
    assert written["models"] == ["new-a", "new-b"]


def test_fetch_failure_falls_back_to_stale_cache(monkeypatch):
    cache = {
        "base_url": adv_models.DEFAULT_BASE_URL,
        "fetched_at": time.time() - (adv_models.CACHE_TTL_SECS + 100),
        "models": ["last-known-good"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(adv_models, "_fetch_remote", boom)
    assert adv_models.list_models() == ["last-known-good"]


def test_force_refresh_ignores_fresh_cache(monkeypatch):
    cache = {
        "base_url": adv_models.DEFAULT_BASE_URL,
        "fetched_at": time.time(),
        "models": ["stale-but-fresh-by-time"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")
    monkeypatch.setattr(adv_models, "_fetch_remote", lambda b, k: ["fresh"])

    assert adv_models.list_models(force_refresh=True) == ["fresh"]


def test_endpoint_change_invalidates_cache(monkeypatch):
    cache = {
        "base_url": "https://old.example.com/v1",
        "fetched_at": time.time(),
        "models": ["old-host"],
    }
    adv_models.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    adv_models.CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")
    monkeypatch.setenv("AUTOMEDAL_ADVISOR_BASE_URL", "https://new.example.com/v1")
    monkeypatch.setattr(adv_models, "_fetch_remote", lambda b, k: ["new-host"])

    assert adv_models.list_models() == ["new-host"]


def test_fetch_failure_with_no_cache_returns_fallback(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")

    def boom(*a, **kw):
        raise RuntimeError("404")

    monkeypatch.setattr(adv_models, "_fetch_remote", boom)
    out = adv_models.list_models()
    assert out == list(adv_models._FALLBACK_MODELS)


def test_refresh_models_no_key_returns_zero():
    n, msg = adv_models.refresh_models()
    assert n == 0
    assert "OPENCODE_API_KEY" in msg


def test_refresh_models_success(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")
    monkeypatch.setattr(adv_models, "_fetch_remote", lambda b, k: ["m1", "m2", "m3"])
    n, where = adv_models.refresh_models()
    assert n == 3
    assert where.startswith("https://")


def test_refresh_models_error_returns_message(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-fake")

    def boom(*a, **kw):
        raise ValueError("bad")

    monkeypatch.setattr(adv_models, "_fetch_remote", boom)
    n, msg = adv_models.refresh_models()
    assert n == 0
    assert "ValueError" in msg
