"""Configuration and behavior contract for the smartwatch category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
USE_LABELS = {
    "health_monitoring": "theo dõi sức khỏe hằng ngày",
    "fitness_sports": "luyện tập và thể thao",
    "outdoor_navigation": "hoạt động ngoài trời cần GPS",
    "calls_notifications": "nghe gọi và nhận thông báo trên cổ tay",
    "children_safety": "liên lạc và định vị cho trẻ em",
    "everyday_style": "đeo hằng ngày và ưu tiên thiết kế",
}
PLATFORM_LABELS = {
    "ios": "tương thích iPhone/iOS",
    "android": "tương thích Android",
    "flexible": "không giới hạn hệ điện thoại",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid smartwatch config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "primary_usage":
            resolved = bool(profile.get("usage_preferences"))
        elif field == "budget":
            resolved = profile.get("budget_max_vnd") is not None or bool(
                profile.get("budget_segment")
            )
        elif field == "phone_platform":
            resolved = bool(profile.get("phone_platform"))
        else:
            resolved = bool(profile.get(field))
        if not resolved:
            missing.append(field)
    return missing


def build_search_text(profile: dict[str, Any], user_query: str) -> str:
    parts: list[str] = [user_query]
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(
        USE_LABELS.get(value, str(value))
        for value in profile.get("usage_preferences", [])
    )
    platform = profile.get("phone_platform")
    if platform in PLATFORM_LABELS:
        parts.append(PLATFORM_LABELS[platform])

    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("display_families"):
        parts.append("màn hình " + ", ".join(hard["display_families"]))
    if hard.get("strap_material_families"):
        parts.append("dây " + ", ".join(hard["strap_material_families"]))
    numeric_labels = {
        "min_screen_size_inch": "màn hình tối thiểu {} inch",
        "max_screen_size_inch": "màn hình tối đa {} inch",
        "max_case_width_mm": "mặt rộng tối đa {} mm",
        "max_weight_g": "khối lượng tối đa {} gram",
        "wrist_circumference_cm": "phù hợp cổ tay {} cm",
        "min_typical_battery_hours": "pin sử dụng thường ít nhất {} giờ",
        "min_water_resistance_atm": "chống nước tối thiểu {} ATM",
    }
    for key, label in numeric_labels.items():
        if hard.get(key) is not None:
            parts.append(label.format(hard[key]))
    if hard.get("call_requirement"):
        parts.append(f"kiểu cuộc gọi {hard['call_requirement']}")
    capability_labels = {
        "requires_cellular": "có SIM hoặc eSIM",
        "requires_gps": "có GPS",
        "requires_notifications": "có thông báo",
        "requires_swimming": "phù hợp khi bơi",
    }
    for key, label in capability_labels.items():
        if hard.get(key) is True:
            parts.append(label)
    parts.extend(str(value) for value in hard.get("required_health_features", []))
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    platform = profile.get("phone_platform")
    if platform and platform != "flexible":
        constraints.append(PLATFORM_LABELS.get(platform, platform))
    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        constraints.append("hãng " + ", ".join(hard["brands"]))
    if hard.get("wrist_circumference_cm") is not None:
        constraints.append(f"cổ tay {hard['wrist_circumference_cm']} cm")
    if hard.get("call_requirement") == "standalone":
        constraints.append("nghe gọi độc lập")
    if hard.get("requires_gps") is True:
        constraints.append("có GPS")
    if hard.get("required_health_features"):
        constraints.append(
            "tính năng sức khỏe " + ", ".join(hard["required_health_features"])
        )
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    missing_note = (
        " Các mẫu thiếu giá hoặc thông tin tương thích không được coi là đạt "
        "điều kiện tương ứng."
        if budget is not None or platform in {"ios", "android"}
        else ""
    )
    return (
        f"Mình chưa tìm thấy đồng hồ thông minh đáp ứng đầy đủ {text} trong dữ "
        f"liệu hiện có.{missing_note} Bạn có thể cân nhắc nới một điều kiện, "
        "nhưng mình chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.smartwatch.filter_builder import build_filter
    from advisor.categories.smartwatch.normalizer import normalize_candidate
    from advisor.categories.smartwatch.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.smartwatch.schemas import (
        SmartwatchCustomAnswer,
        SmartwatchNeedProfile,
    )

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "phone_platform",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.display_families",
            "hard_constraints.strap_material_families",
            "hard_constraints.min_screen_size_inch",
            "hard_constraints.max_screen_size_inch",
            "hard_constraints.max_case_width_mm",
            "hard_constraints.max_weight_g",
            "hard_constraints.wrist_circumference_cm",
            "hard_constraints.min_typical_battery_hours",
            "hard_constraints.min_water_resistance_atm",
            "hard_constraints.call_requirement",
            "hard_constraints.requires_cellular",
            "hard_constraints.requires_gps",
            "hard_constraints.requires_notifications",
            "hard_constraints.requires_swimming",
            "hard_constraints.required_health_features",
        }
    )
    return CategorySpec(
        name="smartwatch",
        display_name="Đồng hồ thông minh",
        config=load_config(),
        profile_model=SmartwatchNeedProfile,
        custom_answer_model=SmartwatchCustomAnswer,
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
                "hard_constraints.display_families",
                "hard_constraints.strap_material_families",
                "hard_constraints.required_health_features",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "budget_segment",
                "phone_platform",
                "hard_constraints.brands",
                "hard_constraints.display_families",
                "hard_constraints.strap_material_families",
                "hard_constraints.min_screen_size_inch",
                "hard_constraints.max_screen_size_inch",
                "hard_constraints.max_case_width_mm",
                "hard_constraints.max_weight_g",
                "hard_constraints.wrist_circumference_cm",
                "hard_constraints.min_typical_battery_hours",
                "hard_constraints.min_water_resistance_atm",
                "hard_constraints.call_requirement",
                "hard_constraints.requires_cellular",
                "hard_constraints.requires_gps",
                "hard_constraints.requires_notifications",
                "hard_constraints.requires_swimming",
                "hard_constraints.required_health_features",
            }
        ),
        question_profile_paths={
            "primary_usage": frozenset(
                {
                    "usage_preferences",
                    "soft_preferences",
                    "hard_constraints.call_requirement",
                    "hard_constraints.requires_gps",
                    "hard_constraints.requires_notifications",
                }
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "phone_platform": frozenset({"phone_platform"}),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.smartwatch.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
