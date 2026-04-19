"""
AutoMedal — Competition Bootstrap
====================================
Phase 3: Fully automated setup after competition selection.

1. Download competition data via Kaggle API
2. Sniff CSV schema to infer target, features, task type
3. Write configs/competition.yaml
4. Render AGENTS.md and program.md from templates
5. Generate starter prepare.py (if not existing)
6. Reset results.tsv
7. Run prepare.py to generate .npy arrays
8. Optional smoke test

Usage:
    python scout/bootstrap.py <competition-slug>
    python scout/bootstrap.py playground-series-s6e4
"""

import os
import sys
import yaml
import datetime
import subprocess

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scout.sniff import sniff_schema
from scout.render import render_templates, render_prepare_starter
from harness.init_memory import init_memory

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "agent", "results.tsv")

# Objective lookup: task_type → {library: objective_string}
OBJECTIVE_MAP = {
    "multiclass": {
        "xgboost": "multi:softprob",
        "xgboost_eval": "mlogloss",
        "lightgbm": "multiclass",
        "catboost": "MultiClass",
    },
    "binary": {
        "xgboost": "binary:logistic",
        "xgboost_eval": "logloss",
        "lightgbm": "binary",
        "catboost": "Logloss",
    },
    "regression": {
        "xgboost": "reg:squarederror",
        "xgboost_eval": "rmse",
        "lightgbm": "regression",
        "catboost": "RMSE",
    },
}

# Metric mapping: task_type → (kaggle_metric, proxy_metric)
METRIC_MAP = {
    "multiclass": ("accuracy", "log_loss"),
    "binary": ("auc", "log_loss"),
    "regression": ("rmse", "rmse"),
}


def download_competition_data(slug):
    """Download competition files via Kaggle API into data/."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    print(f"\n  Downloading data for '{slug}'...")
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(slug, path=DATA_DIR, quiet=False)

    # Unzip if needed
    zip_path = os.path.join(DATA_DIR, f"{slug}.zip")
    if os.path.exists(zip_path):
        print(f"  Extracting {zip_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print(f"  Removed {zip_path}")

    # List downloaded files
    files = os.listdir(DATA_DIR)
    print(f"  Downloaded {len(files)} files: {', '.join(files)}")
    return files


def _get_competition_metadata(slug):
    """Fetch competition title and deadline from Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        competitions = api.competitions_list(search=slug)
        for comp in competitions:
            if getattr(comp, "ref", "") == slug:
                title = getattr(comp, "title", slug)
                deadline = getattr(comp, "deadline", None)
                eval_metric = getattr(comp, "evaluationMetric", None)
                if isinstance(deadline, datetime.datetime):
                    deadline = deadline.strftime("%Y-%m-%d")
                return {
                    "title": title,
                    "deadline": deadline or "unknown",
                    "eval_metric_kaggle": eval_metric,
                }
    except Exception as e:
        print(f"  [WARN] Could not fetch metadata: {e}")

    return {
        "title": slug,
        "deadline": "unknown",
        "eval_metric_kaggle": None,
    }


def build_config(slug, schema, metadata):
    """Build competition.yaml content from sniffed schema and metadata."""
    task_type = schema["task_type"]

    # Determine metrics
    kaggle_metric = metadata.get("eval_metric_kaggle")
    if kaggle_metric:
        proxy = METRIC_MAP.get(task_type, ("accuracy", "log_loss"))[1]
        eval_kaggle = kaggle_metric
    else:
        eval_kaggle, proxy = METRIC_MAP.get(task_type, ("accuracy", "log_loss"))

    # Determine objectives
    objectives = OBJECTIVE_MAP.get(task_type, OBJECTIVE_MAP["multiclass"])

    # Parse title into title + subtitle
    title = metadata.get("title", slug)
    # Try to split on " - " or " — "
    if " - " in title:
        parts = title.split(" - ", 1)
        comp_title, subtitle = parts[0].strip(), parts[1].strip()
    elif " — " in title:
        parts = title.split(" — ", 1)
        comp_title, subtitle = parts[0].strip(), parts[1].strip()
    else:
        comp_title = title
        subtitle = title

    config = {
        "competition": {
            "slug": slug,
            "title": comp_title,
            "subtitle": subtitle,
            "url": f"https://www.kaggle.com/competitions/{slug}",
            "deadline": metadata.get("deadline", "unknown"),
        },
        "task": {
            "type": task_type,
            "target_col": schema["target_col"],
            "id_col": schema["id_col"],
            "class_names": schema["class_names"],
            "num_classes": schema["num_classes"],
            "eval_metric_kaggle": eval_kaggle,
            "eval_metric_proxy": proxy,
        },
        "dataset": {
            "train_rows": schema["train_rows"],
            "test_rows": schema["test_rows"],
            "numeric_features": schema["numeric_features"],
            "categorical_features": schema["categorical_features"],
        },
        "submission": schema["submission"],
        "objectives": objectives,
        "meta": {
            "bootstrapped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sniff_confidence": schema["confidence"],
            "human_verified": False,
        },
    }

    return config


def write_config(config, path=None):
    """Write competition.yaml."""
    if path is None:
        path = os.path.join(CONFIGS_DIR, "competition.yaml")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# configs/competition.yaml — Single source of truth for active competition\n")
        f.write(f"# Bootstrapped: {config['meta']['bootstrapped_at']}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Config written: {path}")


def reset_results():
    """Reset results.tsv to header only."""
    with open(RESULTS_FILE, "w") as f:
        f.write("timestamp\tmethod\ttrials\tval_loss\tval_accuracy\tsubmission\tnotes\n")
    print(f"  Reset: {RESULTS_FILE}")


def run_prepare():
    """Run prepare.py to generate .npy arrays."""
    print("\n  Running prepare.py...")
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "agent", "prepare.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  [ERROR] prepare.py failed:\n{result.stderr}")
        return False
    return True


def bootstrap(slug, skip_download=False, smoke_test=False, yes=False, abort_on_warning=False, extra_args=None):
    """Full bootstrap pipeline for a competition.

    Args:
        slug: Kaggle competition slug (e.g. 'playground-series-s6e4')
        skip_download: If True, skip download (data already in data/)
        smoke_test: If True, run one bespoke-kernel iteration after bootstrap
        yes: If True, skip the low-confidence "continue anyway?" prompt (auto-yes)
        abort_on_warning: If True, abort instead of prompting when confidence < 0.7
        extra_args: Ignored; accepted for forward-compat with CLI passthrough
    """
    print("=" * 60)
    print("AutoMedal — Competition Bootstrap")
    print(f"  Competition: {slug}")
    print("=" * 60)

    # Step 1: Download data
    if not skip_download:
        download_competition_data(slug)
    else:
        print("\n  Skipping download (--skip-download)")

    # Step 2: Sniff schema
    print("\n  Sniffing CSV schema...")
    schema = sniff_schema(DATA_DIR)

    if "error" in schema:
        print(f"\n  [ERROR] {schema['error']}")
        sys.exit(1)

    # Check confidence
    if schema["confidence"] < 0.7:
        print(f"\n  WARNING: Low sniff confidence ({schema['confidence']:.0%})")
        print("  Warnings:")
        for w in schema["warnings"]:
            print(f"    - {w}")
        print()
        if abort_on_warning:
            print("  Aborted (low confidence).")
            sys.exit(0)
        if not yes:
            response = input("  Continue anyway? [y/N] ").strip().lower()
            if response != "y":
                print("  Aborted.")
                sys.exit(0)
        else:
            print("  Continuing (--yes).")

    # Step 3: Fetch competition metadata
    print("\n  Fetching competition metadata...")
    metadata = _get_competition_metadata(slug)

    # Step 4: Build and write config
    print("\n  Building config...")
    config = build_config(slug, schema, metadata)
    write_config(config)

    # Invalidate config cache so subsequent imports pick up the new config
    import config_loader
    config_loader._config_cache = None

    # Step 5: Render templates
    print("\n  Rendering templates...")
    render_templates(config)

    # Step 6: Generate starter prepare.py (only if not existing)
    print("\n  Checking prepare.py...")
    render_prepare_starter(config)

    # Step 7: Reset results.tsv
    print("\n  Resetting experiment log...")
    reset_results()

    # Step 7b: Initialize harness memory (knowledge.md, queue, research_notes, journal/)
    print("\n  Initializing harness memory files...")
    memory_state = init_memory(project_root=PROJECT_ROOT, force=True)
    for artifact, state in memory_state.items():
        print(f"    {state:>7}  {artifact}")

    # Step 8: Run prepare.py
    success = run_prepare()

    # Summary
    print("\n" + "=" * 60)
    print("Bootstrap Summary")
    print("=" * 60)
    print(f"  Competition:  {slug}")
    print(f"  Task:         {schema['task_type']}")
    print(f"  Target:       {schema['target_col']}")
    print(f"  Features:     {len(schema['numeric_features'])} numeric + {len(schema['categorical_features'])} categorical")
    print(f"  Train/Test:   {schema['train_rows']:,} / {schema['test_rows']:,} rows")
    print(f"  Confidence:   {schema['confidence']:.0%}")
    print(f"  Config:       configs/competition.yaml")
    print(f"  Prepare:      {'OK' if success else 'FAILED'}")

    if schema["warnings"]:
        print(f"\n  Warnings:")
        for w in schema["warnings"]:
            print(f"    - {w}")

    if not config["meta"]["human_verified"]:
        print(f"\n  REVIEW: Check configs/competition.yaml and set human_verified: true")

    # Optional smoke test
    if smoke_test and success:
        print("\n  Starting smoke test (1 iteration)...")
        subprocess.Popen(
            [sys.executable, "-m", "automedal.run_loop", "1"],
            cwd=PROJECT_ROOT,
        )

    print("\nDone.")
    return config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bootstrap a Kaggle competition for AutoMedal")
    parser.add_argument("slug", help="Kaggle competition slug (e.g. playground-series-s6e4)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (data already in data/)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single experiment after bootstrap")
    parser.add_argument("--yes", action="store_true",
                        help="Auto-yes for low-confidence schema warnings")
    parser.add_argument("--abort-on-warning", action="store_true",
                        help="Abort instead of prompting when schema confidence < 0.7")
    args = parser.parse_args()

    bootstrap(
        args.slug,
        skip_download=args.skip_download,
        smoke_test=args.smoke_test,
        yes=args.yes,
        abort_on_warning=args.abort_on_warning,
    )


if __name__ == "__main__":
    main()
