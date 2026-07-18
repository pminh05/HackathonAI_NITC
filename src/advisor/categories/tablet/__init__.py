"""Configuration and behavior contract for the tablet category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
USE_LABELS = {
    "study_work": "học tập và làm việc",
    "entertainment": "xem phim và giải trí",
    "gaming": "chơi game",
    "drawing_notes": "vẽ và ghi chú bằng bút",
    "children": "cho trẻ em sử dụng",
    "general": "dùng đa mục đích",
}
CONNECTIVITY_LABELS = {
    "wifi_only": "chỉ Wi-Fi",
    "cellular_4g": "kết nối di động từ 4G",
    "cellular_5g": "kết nối 5G",
    "flexible": "kết nối linh hoạt",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid tablet config at {CONFIG_PATH}")
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
        elif field == "connectivity":
            resolved = bool(profile.get("connectivity_segment"))
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
    connectivity = profile.get("connectivity_segment")
    if connectivity in CONNECTIVITY_LABELS:
        parts.append(CONNECTIVITY_LABELS[connectivity])
    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("os_families"):
        parts.append("hệ điều hành " + ", ".join(hard["os_families"]))
    if hard.get("min_ram_gb") is not None:
        parts.append(f"RAM ít nhất {hard['min_ram_gb']} GB")
    if hard.get("min_storage_gb") is not None:
        parts.append(f"bộ nhớ ít nhất {hard['min_storage_gb']} GB")
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    connectivity = profile.get("connectivity_segment")
    if connectivity and connectivity != "flexible":
        constraints.append(CONNECTIVITY_LABELS.get(connectivity, connectivity))
    hard = profile.get("hard_constraints") or {}
    if hard.get("os_families"):
        constraints.append("hệ điều hành " + ", ".join(hard["os_families"]))
    if hard.get("min_ram_gb") is not None:
        constraints.append(f"RAM từ {hard['min_ram_gb']} GB")
    if hard.get("min_storage_gb") is not None:
        constraints.append(f"bộ nhớ từ {hard['min_storage_gb']} GB")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có dữ liệu giá không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy máy tính bảng đáp ứng đầy đủ {text} trong dữ liệu hiện có."
        f"{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý "
        "bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.tablet.filter_builder import build_filter
    from advisor.categories.tablet.normalizer import normalize_candidate
    from advisor.categories.tablet.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.tablet.schemas import TabletCustomAnswer, TabletNeedProfile

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "connectivity_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.os_families",
            "hard_constraints.display_families",
            "hard_constraints.min_ram_gb",
            "hard_constraints.min_storage_gb",
            "hard_constraints.min_screen_size_inch",
            "hard_constraints.max_screen_size_inch",
            "hard_constraints.max_weight_g",
            "hard_constraints.requires_calls",
            "hard_constraints.requires_memory_card",
        }
    )
    return CategorySpec(
        name="tablet",
        display_name="Máy tính bảng",
        config=load_config(),
        profile_model=TabletNeedProfile,
        custom_answer_model=TabletCustomAnswer,
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
                "hard_constraints.os_families",
                "hard_constraints.display_families",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "budget_segment",
                "connectivity_segment",
                "hard_constraints.brands",
                "hard_constraints.os_families",
                "hard_constraints.display_families",
                "hard_constraints.min_ram_gb",
                "hard_constraints.min_storage_gb",
                "hard_constraints.min_screen_size_inch",
                "hard_constraints.max_screen_size_inch",
                "hard_constraints.max_weight_g",
                "hard_constraints.requires_calls",
                "hard_constraints.requires_memory_card",
            }
        ),
        question_profile_paths={
            "primary_usage": frozenset({"usage_preferences"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "connectivity": frozenset({"connectivity_segment"}),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.tablet.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
