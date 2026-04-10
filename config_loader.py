"""
AutoMedal — Competition Config Loader
======================================
Loads configs/competition.yaml and caches it at module level.
Used by prepare.py and train.py to read competition-specific values.
"""

import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "competition.yaml")
_config_cache = None


def load_config(path=None):
    """Load and cache competition config from YAML.

    Args:
        path: Optional override path. Defaults to configs/competition.yaml.

    Returns:
        dict with full competition configuration.
    """
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Competition config not found at {config_path}. "
            "Run 'python scout/bootstrap.py' to generate one, "
            "or create configs/competition.yaml manually."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if path is None:
        _config_cache = config
    return config


# Convenience accessors
def get_task():
    """Return the task section of the config."""
    return load_config()["task"]


def get_dataset():
    """Return the dataset section of the config."""
    return load_config()["dataset"]


def get_submission():
    """Return the submission section of the config."""
    return load_config()["submission"]


def get_objectives():
    """Return the objectives section of the config."""
    return load_config()["objectives"]


def get_competition():
    """Return the competition identity section of the config."""
    return load_config()["competition"]
