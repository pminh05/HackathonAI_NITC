"""Configuration and behavior contract for the dishwasher category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
PRODUCT_TYPE_LABELS = {
    "freestanding": "máy độc lập",
    "built_in": "máy âm tủ",
    "semi_integrated": "máy bán âm",
    "mini": "máy mini hoặc để bàn",
}
CAPACITY_LABELS = {
    "compact": "sức chứa mục tiêu 6–8 bộ châu Âu",
    "standard": "sức chứa mục tiêu 11–13 bộ châu Âu",
    "large": "sức chứa mục tiêu từ 14 bộ châu Âu",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid dishwasher config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    hard = profile.get("hard_constraints") or {}
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "installation":
            resolved = bool(hard.get("product_types")) or (
                "installation_flexible" in (profile.get("soft_preferences") or [])
            )
        elif field == "capacity":
            resolved = bool(profile.get("capacity_segment")) or any(
                hard.get(path) is not None
                for path in ("min_place_settings", "min_vietnamese_meals")
            )
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
    """Build concise Vietnamese semantic text; hard filters remain separate."""
    parts: list[str] = [user_query]
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    segment = profile.get("capacity_segment")
    if segment in CAPACITY_LABELS:
        parts.append(CAPACITY_LABELS[segment])
    hard = profile.get("hard_constraints") or {}
    product_types = [
        PRODUCT_TYPE_LABELS.get(value, str(value))
        for value in hard.get("product_types", [])
    ]
    if product_types:
        parts.append("kiểu lắp đặt " + ", ".join(product_types))
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("min_place_settings") is not None:
        parts.append(f"ít nhất {hard['min_place_settings']} bộ châu Âu")
    if hard.get("min_vietnamese_meals") is not None:
        parts.append(f"ít nhất {hard['min_vietnamese_meals']} bữa ăn Việt")
    if hard.get("max_noise_db") is not None:
        parts.append(f"độ ồn không quá {hard['max_noise_db']} dB")
    if hard.get("max_water_l_per_cycle") is not None:
        parts.append(
            f"lượng nước không quá {hard['max_water_l_per_cycle']} lít mỗi lần"
        )
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
    if hard.get("min_place_settings") is not None:
        constraints.append(f"sức chứa từ {hard['min_place_settings']} bộ châu Âu")
    if hard.get("min_vietnamese_meals") is not None:
        constraints.append(f"sức chứa từ {hard['min_vietnamese_meals']} bữa ăn Việt")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có dữ liệu giá không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy máy rửa chén đáp ứng đầy đủ {text} trong dữ liệu hiện có."
        f"{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý "
        "bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.dishwasher.filter_builder import build_filter
    from advisor.categories.dishwasher.normalizer import normalize_candidate
    from advisor.categories.dishwasher.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.dishwasher.schemas import (
        DishwasherCustomAnswer,
        DishwasherNeedProfile,
    )

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "capacity_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.product_types",
            "hard_constraints.min_place_settings",
            "hard_constraints.min_vietnamese_meals",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.max_noise_db",
            "hard_constraints.max_water_l_per_cycle",
        }
    )
    return CategorySpec(
        name="dishwasher",
        display_name="Máy rửa chén",
        config=load_config(),
        profile_model=DishwasherNeedProfile,
        custom_answer_model=DishwasherCustomAnswer,
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
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.product_types",
                "hard_constraints.min_place_settings",
                "hard_constraints.min_vietnamese_meals",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.max_noise_db",
                "hard_constraints.max_water_l_per_cycle",
            }
        ),
        question_profile_paths={
            "installation": frozenset(
                {"hard_constraints.product_types", "soft_preferences"}
            ),
            "capacity": frozenset(
                {
                    "capacity_segment",
                    "hard_constraints.min_place_settings",
                    "hard_constraints.min_vietnamese_meals",
                }
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.dishwasher.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
