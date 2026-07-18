"""Configuration and behavior contract for the water-heater category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
PRODUCT_TYPE_LABELS = {
    "direct": "làm nóng trực tiếp, nóng liền",
    "indirect": "làm nóng gián tiếp, có bình chứa",
    "solar": "làm nóng bằng năng lượng mặt trời",
    "direct_multipoint": "làm nóng trực tiếp cho nhiều điểm",
}
WATER_SUPPLY_LABELS = {
    "stable": "áp lực nước ổn định tại một phòng tắm",
    "low_pressure": "nguồn nước yếu hoặc áp lực thấp",
    "multi_outlet": "cấp nước nóng cho nhiều điểm",
    "open": "chưa xác định điều kiện nguồn nước",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid water-heater config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    hard = profile.get("hard_constraints") or {}
    soft = profile.get("soft_preferences") or []
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "heater_type":
            resolved = bool(hard.get("product_types")) or (
                "heater_type_flexible" in soft
            )
        elif field == "water_supply":
            resolved = bool(profile.get("water_supply"))
        elif field == "budget":
            resolved = profile.get("budget_max_vnd") is not None or bool(
                profile.get("budget_segment")
            )
        else:
            resolved = bool(profile.get(field))
        if not resolved:
            missing.append(field)
    return missing


def build_search_text(profile: dict[str, Any], user_query: str) -> str:
    """Build concise Vietnamese semantic text; strict filters remain separate."""
    parts: list[str] = [user_query]
    household_size = profile.get("household_size")
    if household_size is not None:
        parts.append(f"phù hợp nhu cầu {household_size} người")
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        parts.append(f"ngân sách tối đa {budget} đồng")
    water_supply = profile.get("water_supply")
    if water_supply in WATER_SUPPLY_LABELS:
        parts.append(WATER_SUPPLY_LABELS[water_supply])

    hard = profile.get("hard_constraints") or {}
    product_types = [
        PRODUCT_TYPE_LABELS.get(value, str(value))
        for value in hard.get("product_types", [])
    ]
    if product_types:
        parts.append("kiểu máy " + ", ".join(product_types))
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("min_capacity_lit") is not None:
        parts.append(f"dung tích từ {hard['min_capacity_lit']} lít")
    if hard.get("max_capacity_lit") is not None:
        parts.append(f"dung tích không quá {hard['max_capacity_lit']} lít")
    if hard.get("max_power_w") is not None:
        parts.append(f"công suất không quá {hard['max_power_w']} W")
    if hard.get("max_heating_time_minutes") is not None:
        parts.append(
            f"thời gian làm nóng không quá {hard['max_heating_time_minutes']} phút"
        )
    if hard.get("required_features"):
        parts.extend(str(value) for value in hard["required_features"])
    if hard.get("required_safety_features"):
        parts.extend(str(value) for value in hard["required_safety_features"])
    parts.extend(str(value) for value in profile.get("usage_preferences", []))
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    hard = profile.get("hard_constraints") or {}
    product_types = [
        PRODUCT_TYPE_LABELS.get(value, str(value))
        for value in hard.get("product_types", [])
    ]
    if product_types:
        constraints.append("kiểu " + ", ".join(product_types))
    if hard.get("min_capacity_lit") is not None:
        constraints.append(f"dung tích từ {hard['min_capacity_lit']} lít")
    if hard.get("required_features"):
        constraints.append(
            "tính năng bắt buộc "
            + ", ".join(str(value) for value in hard["required_features"])
        )
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có dữ liệu giá không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy máy nước nóng đáp ứng đầy đủ {text} trong dữ liệu "
        f"hiện có.{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình "
        "chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.water_heater.filter_builder import build_filter
    from advisor.categories.water_heater.normalizer import normalize_candidate
    from advisor.categories.water_heater.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.water_heater.schemas import (
        WaterHeaterCustomAnswer,
        WaterHeaterNeedProfile,
    )

    valid_paths = frozenset(
        {
            "household_size",
            "budget_max_vnd",
            "budget_segment",
            "water_supply",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.product_types",
            "hard_constraints.min_capacity_lit",
            "hard_constraints.max_capacity_lit",
            "hard_constraints.max_power_w",
            "hard_constraints.max_heating_time_minutes",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.required_features",
            "hard_constraints.required_safety_features",
            "hard_constraints.ip_ratings",
        }
    )
    return CategorySpec(
        name="water_heater",
        display_name="Máy nước nóng",
        config=load_config(),
        profile_model=WaterHeaterNeedProfile,
        custom_answer_model=WaterHeaterCustomAnswer,
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
                "hard_constraints.product_types",
                "hard_constraints.required_features",
                "hard_constraints.required_safety_features",
                "hard_constraints.ip_ratings",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "hard_constraints.brands",
                "hard_constraints.product_types",
                "hard_constraints.min_capacity_lit",
                "hard_constraints.max_capacity_lit",
                "hard_constraints.max_power_w",
                "hard_constraints.max_heating_time_minutes",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.required_features",
                "hard_constraints.required_safety_features",
                "hard_constraints.ip_ratings",
            }
        ),
        question_profile_paths={
            "heater_type": frozenset(
                {"hard_constraints.product_types", "soft_preferences"}
            ),
            "water_supply": frozenset(
                {"water_supply", "usage_preferences", "soft_preferences"}
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.water_heater.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
