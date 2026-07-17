"""Configuration helpers for the refrigerator category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache the category-owned rules and metadata mapping."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid refrigerator config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    """Resolve missing core fields using deterministic category rules."""
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "budget":
            resolved = profile.get("budget_max_vnd") is not None or bool(
                profile.get("budget_segment")
            )
        else:
            resolved = bool(profile.get(field))
        if not resolved:
            missing.append(field)
    return missing


__all__ = ["get_missing_profile_fields", "load_config"]
