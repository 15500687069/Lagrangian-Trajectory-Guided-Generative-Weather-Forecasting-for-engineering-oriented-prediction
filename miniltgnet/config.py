from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base_config", None)
    if base is None:
        return cfg

    base_path = (path.parent / base).resolve()
    if not base_path.exists():
        candidate = Path(base).resolve()
        if candidate.exists():
            base_path = candidate
        else:
            raise FileNotFoundError(f"Base config not found: {base}")
    base_cfg = load_config(base_path)
    return _deep_merge(base_cfg, cfg)
