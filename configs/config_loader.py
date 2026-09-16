import os
import yaml
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "config.yaml",
)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load research configuration YAML file with fallback to default."""
    if config_path is None or not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged
