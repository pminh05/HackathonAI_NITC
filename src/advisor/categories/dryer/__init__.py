"""Configuration and behavior contract for the clothes-dryer category."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")

USAGE_SEARCH_LABELS = {
    "rainy_season": "sấy ổn định trong mùa mưa và thời tiết ẩm",
    "frequent_drying": "sấy quần áo thường xuyên hằng ngày",
    "bulky_items": "sấy chăn mền và đồ cồng kềnh",
    "delicate_care": "chăm sóc đồ mỏng và vải dễ hư",
    "energy_saving": "ưu tiên tiết kiệm điện",
    "quick_dry": "ưu tiên sấy nhanh và hạn chế nhăn",
}
DRYER_TYPE_LABELS = {
    "heat_pump": "sấy bơm nhiệt",
    "condenser": "sấy ngưng tụ",
    "vented": "sấy thông hơi",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid clothes-dryer config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
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


def build_search_text(profile: dict[str, Any], user_query: str) -> str:
    parts: list[Any] = [user_query]
    if profile.get("household_size") is not None:
        parts.append(f"phù hợp gia đình {profile['household_size']} người")
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(
        USAGE_SEARCH_LABELS.get(value, value)
        for value in profile.get("usage_preferences", [])
    )
    parts.extend(profile.get("soft_preferences", []))
    parts.extend(profile.get("implicit_needs", []))
    hard = profile.get("hard_constraints") or {}
    if hard:
        parts.append(json.dumps(hard, ensure_ascii=False, default=str))
    return ". ".join(str(part) for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    if profile.get("household_size") is not None:
        constraints.append(f"nhu cầu cho {profile['household_size']} người")
    if profile.get("budget_max_vnd") is not None:
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("min_dry_capacity_kg") is not None:
        constraints.append(f"tải sấy từ {hard['min_dry_capacity_kg']:g} kg")
    if hard.get("max_dry_capacity_kg") is not None:
        constraints.append(f"tải sấy tối đa {hard['max_dry_capacity_kg']:g} kg")
    dryer_types = [
        DRYER_TYPE_LABELS.get(value, str(value))
        for value in hard.get("dryer_types", [])
    ]
    if dryer_types:
        constraints.append("loại " + ", ".join(dryer_types))
    if hard.get("max_power_w") is not None:
        constraints.append(f"công suất không quá {hard['max_power_w']:,} W")
    if hard.get("inverter") is True:
        constraints.append("bắt buộc có inverter")
    if hard.get("sensor") is True:
        constraints.append("bắt buộc có cảm biến")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy máy sấy quần áo đáp ứng đầy đủ {text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.dryer.filter_builder import build_filter
    from advisor.categories.dryer.normalizer import normalize_candidate
    from advisor.categories.dryer.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.dryer.schemas import DryerCustomAnswer, DryerNeedProfile

    valid_paths = frozenset(
        {
            "household_size",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.dryer_types",
            "hard_constraints.min_dry_capacity_kg",
            "hard_constraints.max_dry_capacity_kg",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.max_power_w",
            "hard_constraints.inverter",
            "hard_constraints.sensor",
        }
    )
    return CategorySpec(
        name="dryer",
        display_name="Máy sấy quần áo",
        config=load_config(),
        profile_model=DryerNeedProfile,
        custom_answer_model=DryerCustomAnswer,
        build_need_extraction_prompt=build_need_extraction_prompt,
        build_custom_answer_prompt=build_custom_answer_prompt,
        build_ranking_prompt=build_ranking_prompt,
        build_response_prompt=build_response_prompt,
        build_filter=build_filter,
        build_search_text=build_search_text,
        no_match_answer=no_match_answer,
        normalize_candidate=normalize_candidate,
        get_missing_profile_fields=get_missing_profile_fields,
        valid_patch_paths=valid_paths,
        list_patch_paths=frozenset(
            {
                "usage_preferences",
                "soft_preferences",
                "implicit_needs",
                "hard_constraints.brands",
                "hard_constraints.dryer_types",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "household_size",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.dryer_types",
                "hard_constraints.min_dry_capacity_kg",
                "hard_constraints.max_dry_capacity_kg",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.max_power_w",
                "hard_constraints.inverter",
                "hard_constraints.sensor",
            }
        ),
        question_profile_paths={
            "household_size": frozenset({"household_size"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "usage_preferences": frozenset(
                {"usage_preferences", "soft_preferences", "implicit_needs"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.dryer.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
