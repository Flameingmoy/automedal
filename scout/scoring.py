"""
AutoMedal — Competition Heuristic Scorer
==========================================
Two-stage scoring system for ranking Kaggle competitions by AutoMedal compatibility.

Stage 1: Metadata-only scoring (from competitions_list() API response)
Stage 2: File listing check (for top-N candidates, lightweight API call)

No LLM calls. No data downloads. Pure deterministic heuristics.
"""

import re
from datetime import datetime, timezone

# ─── KNOWN TABULAR METRICS ─────────────────────────────────────────────
# Case-insensitive matching. Kaggle API uses various formats.
KNOWN_TABULAR_METRICS = {
    "accuracy", "logloss", "log_loss", "categoricalcrossentropy",
    "categoricalaccuracy", "auc", "rmse", "mae", "r2", "f1",
    "rootmeansquarederror", "rootmeansquaredlogerror",
    "meanabsoluteerror", "meansquarederror", "mse",
    "multiclassloss", "mapk", "map@k", "quadraticweightedkappa",
    "weightedlogloss", "binarycrossentropy", "f1score",
    "rmsle", "medianabsoluteerror", "spearman",
}

# ─── NON-TABULAR TAG KEYWORDS ──────────────────────────────────────────
NON_TABULAR_TAGS = {
    "nlp", "computer vision", "image classification", "object detection",
    "text", "audio", "video", "segmentation", "image",
    "natural language processing", "speech", "generative ai",
    "large language models", "diffusion", "gan",
}

# ─── NON-TABULAR DESCRIPTION KEYWORDS ──────────────────────────────────
NON_TABULAR_DESC_KEYWORDS = [
    r"\bimage\b", r"\bnlp\b", r"\btext classification\b",
    r"\bsegmentation\b", r"\bobject detection\b", r"\bcomputer vision\b",
    r"\bspeech\b", r"\baudio\b", r"\bvideo\b", r"\bpixel\b",
    r"\btoken\b", r"\btransformer\b", r"\bbert\b", r"\bgpt\b",
    r"\bllm\b", r"\bdiffusion\b",
]

# ─── NON-TABULAR FILE EXTENSIONS ───────────────────────────────────────
NON_TABULAR_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".mp3", ".wav", ".flac", ".mp4", ".avi", ".mov",
    ".json", ".jsonl", ".txt", ".parquet",
}


def _normalize_metric(metric):
    """Normalize a metric string for comparison."""
    if not metric:
        return ""
    return re.sub(r"[^a-z0-9]", "", metric.lower())


def _days_until_deadline(deadline_str):
    """Calculate days until deadline. Returns None if unparseable."""
    if not deadline_str:
        return None
    try:
        # Kaggle API returns ISO format or similar
        if isinstance(deadline_str, datetime):
            deadline = deadline_str
        else:
            deadline_str = str(deadline_str).replace("Z", "+00:00")
            deadline = datetime.fromisoformat(deadline_str)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (deadline - now).days
    except (ValueError, TypeError):
        return None


def _get_tags_lower(competition):
    """Extract lowercase tag names from competition metadata."""
    tags = competition.get("tags", [])
    if not tags:
        return set()
    result = set()
    for tag in tags:
        if isinstance(tag, str):
            result.add(tag.lower())
        elif isinstance(tag, dict):
            name = tag.get("name", "") or tag.get("ref", "")
            if name:
                result.add(name.lower())
        elif hasattr(tag, "name"):
            result.add(tag.name.lower())
        elif hasattr(tag, "ref"):
            result.add(tag.ref.lower())
    return result


# ─── STAGE 1: METADATA-ONLY SCORING ────────────────────────────────────

def score_stage1(competition):
    """Score a competition using only API metadata (no downloads).

    Args:
        competition: dict-like object from Kaggle API competitions_list().

    Returns:
        (score: int, reasons: list[str], disqualified: bool)
    """
    score = 0
    reasons = []

    # --- Extract fields (handle both dict and object access) ---
    def _get(key, default=None):
        if isinstance(competition, dict):
            return competition.get(key, default)
        return getattr(competition, key, default)

    ref = str(_get("ref", "") or "")
    title = str(_get("title", "") or "")
    description = str(_get("description", "") or "")
    category = str(_get("category", "") or "")
    eval_metric = str(_get("evaluationMetric", "") or "")
    team_count = _get("teamCount", 0) or 0
    kernels_only = _get("isKernelsSubmissionsOnly", False)
    deadline = _get("deadline", None)

    tags_lower = _get_tags_lower(competition)

    # --- HARD DISQUALIFIERS ---
    if kernels_only:
        return (0, ["DISQUALIFIED: kernels-only submissions"], True)

    days_left = _days_until_deadline(deadline)
    if days_left is not None and days_left < 7:
        return (0, [f"DISQUALIFIED: deadline in {days_left} days"], True)

    # --- POSITIVE SIGNALS ---

    # Playground Series slug pattern (+40)
    if "playground-series" in ref.lower():
        score += 40
        reasons.append("+40: Playground Series (slug match)")

    # Category bonuses
    cat_lower = category.lower()
    if cat_lower == "getting started":
        score += 30
        reasons.append("+30: Getting Started category")
    elif cat_lower == "playground":
        score += 25
        reasons.append("+25: Playground category")
    elif cat_lower == "featured":
        score += 5
        reasons.append("+5: Featured category")

    # Known tabular evaluation metric (+20)
    metric_normalized = _normalize_metric(eval_metric)
    if metric_normalized and metric_normalized in {
        _normalize_metric(m) for m in KNOWN_TABULAR_METRICS
    }:
        score += 20
        reasons.append(f"+20: known tabular metric ({eval_metric})")

    # Tags contain "tabular" (+20)
    if "tabular" in tags_lower:
        score += 20
        reasons.append("+20: tagged 'tabular'")

    # Team count signals
    if team_count > 2000:
        score += 15
        reasons.append(f"+15: very active ({team_count} teams)")
    elif team_count > 500:
        score += 10
        reasons.append(f"+10: active ({team_count} teams)")
    elif team_count > 100:
        score += 5
        reasons.append(f"+5: moderate activity ({team_count} teams)")

    # Deadline health
    if days_left is not None and days_left > 30:
        score += 5
        reasons.append(f"+5: comfortable deadline ({days_left} days)")

    # --- NEGATIVE SIGNALS ---

    # Non-tabular tags (-50)
    bad_tags = tags_lower & NON_TABULAR_TAGS
    if bad_tags:
        score -= 50
        reasons.append(f"-50: non-tabular tags ({', '.join(bad_tags)})")

    # Non-tabular description keywords (-30)
    desc_lower = description.lower()
    matched_keywords = [
        kw for kw in NON_TABULAR_DESC_KEYWORDS
        if re.search(kw, desc_lower)
    ]
    if matched_keywords and not bad_tags:
        # Only apply if tags didn't already catch it
        score -= 30
        reasons.append(f"-30: non-tabular description keywords")

    # Unknown/missing evaluation metric (-10)
    if not eval_metric:
        score -= 10
        reasons.append("-10: no evaluation metric specified")
    elif metric_normalized and metric_normalized not in {
        _normalize_metric(m) for m in KNOWN_TABULAR_METRICS
    }:
        score -= 5
        reasons.append(f"-5: unknown metric ({eval_metric})")

    # Research category (-20)
    if cat_lower == "research":
        score -= 20
        reasons.append("-20: Research category (non-standard)")

    return (score, reasons, False)


# ─── STAGE 2: FILE LISTING CHECK ───────────────────────────────────────

def score_stage2(file_list):
    """Score a competition based on its file listing (no data download).

    Args:
        file_list: list of dicts with 'name' and optionally 'totalBytes' keys.
                   Or list of objects with .name and .totalBytes attributes.

    Returns:
        (score: int, reasons: list[str])
    """
    score = 0
    reasons = []

    if not file_list:
        reasons.append("+0: no file listing available")
        return (score, reasons)

    # Extract file info
    filenames = []
    total_bytes = 0
    for f in file_list:
        if isinstance(f, dict):
            name = f.get("name", "")
            size = f.get("totalBytes", 0) or 0
        else:
            name = getattr(f, "name", "")
            size = getattr(f, "totalBytes", 0) or 0
        filenames.append(name.lower())
        total_bytes += size

    extensions = {
        "." + fn.rsplit(".", 1)[-1] if "." in fn else ""
        for fn in filenames
    }

    # train.csv present (+25)
    has_train = any(
        fn in ("train.csv", "train.csv.zip", "train.csv.gz")
        for fn in filenames
    )
    if has_train:
        score += 25
        reasons.append("+25: train.csv found")

    # test.csv present (+15)
    has_test = any(
        fn in ("test.csv", "test.csv.zip", "test.csv.gz")
        for fn in filenames
    )
    if has_test:
        score += 15
        reasons.append("+15: test.csv found")

    # sample_submission.csv present (+10)
    has_sample = any(
        "sample" in fn and "submission" in fn
        for fn in filenames
    )
    if has_sample:
        score += 10
        reasons.append("+10: sample_submission.csv found")

    # All CSV files (+10)
    csv_exts = {".csv", ".csv.zip", ".csv.gz", ".zip"}
    non_csv = extensions - csv_exts - {""}
    if not non_csv and filenames:
        score += 10
        reasons.append("+10: all files are CSV/zip")

    # Non-tabular file types (-50)
    bad_exts = extensions & NON_TABULAR_EXTENSIONS
    if bad_exts:
        score -= 50
        reasons.append(f"-50: non-tabular files ({', '.join(bad_exts)})")

    # Total size > 5GB (-15)
    size_gb = total_bytes / (1024 ** 3)
    if size_gb > 5:
        score -= 15
        reasons.append(f"-15: large dataset ({size_gb:.1f}GB)")

    return (score, reasons)


# ─── COMBINED SCORING ───────────────────────────────────────────────────

def compute_final_score(stage1_score, stage2_score):
    """Clamp combined score to [0, 100]."""
    return max(0, min(100, stage1_score + stage2_score))
