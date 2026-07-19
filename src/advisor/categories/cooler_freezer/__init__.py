"""Configuration and behavior contract for the cooler/freezer category."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")

FAMILY_LABELS = {
    "cooler": "tủ mát",
    "freezer": "tủ đông",
    "open": "cả tủ mát và tủ đông",
}
USAGE_SEARCH_LABELS = {
    "display_drinks": "trưng bày và làm mát đồ uống",
    "fresh_food_cooling": "giữ thực phẩm tươi ở nhiệt độ mát",
    "bulk_frozen_storage": "trữ đông thịt cá dài ngày",
    "commercial_storage": "kinh doanh và cần sức chứa lớn",
    "convertible_use": "chuyển đổi linh hoạt giữa chế độ mát và đông",
    "energy_saving": "ưu tiên tiết kiệm điện và vận hành lâu dài",
}
FEATURE_LABELS = {
    "glass_door": "cửa kính",
    "convertible_mode": "chuyển đổi chế độ mát/đông",
    "fast_freeze": "làm đông nhanh",
    "lock": "khóa cửa",
    "wheels": "bánh xe",
    "external_temperature_control": "điều khiển nhiệt độ bên ngoài",
    "led_light": "đèn LED",
    "drain": "lỗ thoát nước",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid cooler/freezer config at {CONFIG_PATH}")
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
        elif field == "product_family":
            resolved = profile.get("product_family") in FAMILY_LABELS
        else:
            resolved = bool(profile.get(field))
        if not resolved:
            missing.append(field)
    return missing


def build_search_text(profile: dict[str, Any], user_query: str) -> str:
    parts: list[Any] = [user_query]
    family = profile.get("product_family")
    if family in FAMILY_LABELS:
        parts.append(FAMILY_LABELS[family])
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
    family = profile.get("product_family")
    if family in {"cooler", "freezer"}:
        constraints.append(FAMILY_LABELS[family])
    if profile.get("budget_max_vnd") is not None:
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("size_variants") == ["mini"]:
        constraints.append("kích thước mini")
    if hard.get("min_capacity_lit") is not None:
        constraints.append(f"dung tích từ {hard['min_capacity_lit']:,} lít")
    if hard.get("max_capacity_lit") is not None:
        constraints.append(f"dung tích tối đa {hard['max_capacity_lit']:,} lít")
    if hard.get("required_temperature_c") is not None:
        constraints.append(
            f"nhiệt độ cần đạt {hard['required_temperature_c']:g}°C hoặc thấp hơn"
        )
    for field, label in {
        "max_width_cm": "ngang",
        "max_height_cm": "cao",
        "max_depth_cm": "sâu",
    }.items():
        if hard.get(field) is not None:
            constraints.append(f"{label} không quá {hard[field]:g} cm")
    if hard.get("inverter") is True:
        constraints.append("bắt buộc có Inverter")
    features = [
        FEATURE_LABELS.get(value, str(value))
        for value in hard.get("required_features", [])
    ]
    if features:
        constraints.append("bắt buộc có " + ", ".join(features))
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy tủ mát hoặc tủ đông đáp ứng đầy đủ {text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.cooler_freezer.filter_builder import build_filter
    from advisor.categories.cooler_freezer.normalizer import normalize_candidate
    from advisor.categories.cooler_freezer.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.cooler_freezer.schemas import (
        CoolerFreezerCustomAnswer,
        CoolerFreezerNeedProfile,
    )

    valid_paths = frozenset(
        {
            "product_family",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.size_variants",
            "hard_constraints.min_capacity_lit",
            "hard_constraints.max_capacity_lit",
            "hard_constraints.required_temperature_c",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.inverter",
            "hard_constraints.gas_types",
            "hard_constraints.required_features",
        }
    )
    return CategorySpec(
        name="cooler_freezer",
        display_name="Tủ mát, tủ đông",
        config=load_config(),
        profile_model=CoolerFreezerNeedProfile,
        custom_answer_model=CoolerFreezerCustomAnswer,
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
                "hard_constraints.size_variants",
                "hard_constraints.gas_types",
                "hard_constraints.required_features",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "product_family",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.size_variants",
                "hard_constraints.min_capacity_lit",
                "hard_constraints.max_capacity_lit",
                "hard_constraints.required_temperature_c",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.inverter",
                "hard_constraints.gas_types",
                "hard_constraints.required_features",
            }
        ),
        question_profile_paths={
            "product_family": frozenset({"product_family"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "usage_preferences": frozenset(
                {"usage_preferences", "soft_preferences", "implicit_needs"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.cooler_freezer.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
