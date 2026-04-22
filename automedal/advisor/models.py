"""Live model-list fetch for advisor autocompletion.

Pulls `GET <base_url>/models` from the configured advisor endpoint
(default opencode-go) and caches the result on disk so subsequent calls
are free. Returns the cached list (even if stale) on any network failure
so the TUI never blocks.

Public surface:
    list_models(force_refresh=False) -> list[str]   # sorted unique ids
    refresh_models() -> tuple[int, str]             # (count, source_msg)
    cache_path() -> Path
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

CACHE_TTL_SECS = 60 * 60          # 1 hour — fast enough for daily use
CACHE_PATH = Path.home() / ".automedal" / "models_cache.json"
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

# Used only when both (a) the disk cache is empty and (b) the live fetch
# failed (e.g. opencode-go hasn't shipped a `/v1/models` listing yet, or the
# user is offline). Keep this small + advisor-grade — it's the autocomplete
# floor, not a model whitelist. The day opencode-go exposes /v1/models the
# fetcher takes over and this list becomes irrelevant.
_FALLBACK_MODELS: tuple[str, ...] = (
    "kimi-k2.6",
    "minimax-m2.7",
    "glm-4.6",
    "glm-4.5-air",
    "mimo-7b",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
    "gpt-5",
    "gpt-5-mini",
)


def cache_path() -> Path:
    return CACHE_PATH


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_remote(base_url: str, api_key: str, timeout: float = 10.0) -> list[str]:
    """One blocking HTTPS GET. Returns sorted unique model ids."""
    import httpx

    url = base_url.rstrip("/") + "/models"
    r = httpx.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    raw = data.get("data") if isinstance(data, dict) else data
    ids: set[str] = set()
    for entry in raw or []:
        if isinstance(entry, dict) and entry.get("id"):
            ids.add(str(entry["id"]))
        elif isinstance(entry, str):
            ids.add(entry)
    return sorted(ids)


def list_models(force_refresh: bool = False) -> list[str]:
    """Return the cached model list, refreshing if stale or forced.

    Never raises. Falls back through three sources in order:
      1. Fresh cache (or fresh fetch if stale/forced).
      2. Stale cache (when fetch fails or no key).
      3. Built-in `_FALLBACK_MODELS` (when nothing else is available).
    """
    cache = _read_cache()
    cached = cache.get("models") or []
    fetched_at = float(cache.get("fetched_at") or 0)
    base_url = os.environ.get("AUTOMEDAL_ADVISOR_BASE_URL", DEFAULT_BASE_URL)

    fresh = (time.time() - fetched_at) < CACHE_TTL_SECS
    same_endpoint = cache.get("base_url") == base_url
    if cached and fresh and same_endpoint and not force_refresh:
        return list(cached)

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return list(cached) if cached else list(_FALLBACK_MODELS)

    try:
        ids = _fetch_remote(base_url, api_key)
    except Exception:
        return list(cached) if cached else list(_FALLBACK_MODELS)

    _write_cache({
        "base_url": base_url,
        "fetched_at": time.time(),
        "models": ids,
    })
    return ids


def refresh_models() -> tuple[int, str]:
    """Force a fetch. Returns (count, source-or-error-message)."""
    base_url = os.environ.get("AUTOMEDAL_ADVISOR_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return 0, "OPENCODE_API_KEY not set"
    try:
        ids = _fetch_remote(base_url, api_key)
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    _write_cache({
        "base_url": base_url,
        "fetched_at": time.time(),
        "models": ids,
    })
    return len(ids), base_url
