"""Configuration and behavior contract for the computer-monitor category."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
USE_LABELS = {
    "office_study": "văn phòng và học tập",
    "programming_multitasking": "lập trình và đa nhiệm",
    "gaming": "chơi game",
    "creative_color": "thiết kế và chỉnh sửa hình ảnh",
    "entertainment": "giải trí và xem phim",
    "general": "dùng đa mục đích",
}
SIZE_LABELS = {
    "compact": "màn hình nhỏ gọn tối đa 24 inch",
    "standard": "màn hình khoảng 24–27 inch",
    "large": "màn hình khoảng 27–32 inch",
    "ultrawide": "màn hình ultrawide hoặc từ 34 inch",
    "flexible": "không giới hạn kích thước",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid monitor config at {CONFIG_PATH}")
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
        elif field == "screen_size":
            resolved = bool(profile.get("screen_size_preference")) or (
                profile.get("preferred_screen_size_inch") is not None
            )
        else:
            resolved = bool(profile.get(field))
        if not resolved:
            missing.append(field)
    return missing


def build_search_text(profile: dict[str, Any], user_query: str) -> str:
    parts: list[Any] = [user_query]
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(
        USE_LABELS.get(value, str(value))
        for value in profile.get("usage_preferences", [])
    )
    size = profile.get("screen_size_preference")
    if size in SIZE_LABELS:
        parts.append(SIZE_LABELS[size])
    if profile.get("preferred_screen_size_inch") is not None:
        parts.append(f"ưu tiên {profile['preferred_screen_size_inch']:g} inch")
    parts.extend(profile.get("soft_preferences", []))
    parts.extend(profile.get("implicit_needs", []))
    hard = profile.get("hard_constraints") or {}
    if hard:
        parts.append(json.dumps(hard, ensure_ascii=False, default=str))
    return ". ".join(str(part) for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("panel_families"):
        constraints.append("tấm nền " + ", ".join(hard["panel_families"]))
    if hard.get("min_screen_size_inch") is not None:
        constraints.append(f"kích thước từ {hard['min_screen_size_inch']:g} inch")
    if hard.get("max_screen_size_inch") is not None:
        constraints.append(f"kích thước tối đa {hard['max_screen_size_inch']:g} inch")
    if hard.get("required_connections"):
        constraints.append("kết nối " + ", ".join(hard["required_connections"]))
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có giá xác minh không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy màn hình máy tính đáp ứng đầy đủ {text} trong catalog "
        f"hiện tại.{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình "
        "chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.monitor.filter_builder import build_filter
    from advisor.categories.monitor.normalizer import normalize_candidate
    from advisor.categories.monitor.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.monitor.schemas import MonitorCustomAnswer, MonitorNeedProfile

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "screen_size_preference",
            "preferred_screen_size_inch",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.panel_families",
            "hard_constraints.screen_shapes",
            "hard_constraints.resolution_keys",
            "hard_constraints.required_connections",
            "hard_constraints.required_features",
            "hard_constraints.response_time_metrics",
            "hard_constraints.min_screen_size_inch",
            "hard_constraints.max_screen_size_inch",
            "hard_constraints.min_resolution_width_px",
            "hard_constraints.min_resolution_height_px",
            "hard_constraints.max_response_time_ms",
            "hard_constraints.min_brightness_nits",
            "hard_constraints.min_srgb_coverage_pct",
            "hard_constraints.min_dci_p3_coverage_pct",
            "hard_constraints.requires_speakers",
            "hard_constraints.requires_vesa",
            "hard_constraints.requires_touch",
            "hard_constraints.max_width_mm",
        }
    )
    list_paths = frozenset(
        {
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.panel_families",
            "hard_constraints.screen_shapes",
            "hard_constraints.resolution_keys",
            "hard_constraints.required_connections",
            "hard_constraints.required_features",
            "hard_constraints.response_time_metrics",
        }
    )
    hard_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "hard_constraints.brands",
            "hard_constraints.panel_families",
            "hard_constraints.screen_shapes",
            "hard_constraints.resolution_keys",
            "hard_constraints.required_connections",
            "hard_constraints.required_features",
            "hard_constraints.response_time_metrics",
            "hard_constraints.min_screen_size_inch",
            "hard_constraints.max_screen_size_inch",
            "hard_constraints.min_resolution_width_px",
            "hard_constraints.min_resolution_height_px",
            "hard_constraints.max_response_time_ms",
            "hard_constraints.min_brightness_nits",
            "hard_constraints.min_srgb_coverage_pct",
            "hard_constraints.min_dci_p3_coverage_pct",
            "hard_constraints.requires_speakers",
            "hard_constraints.requires_vesa",
            "hard_constraints.requires_touch",
            "hard_constraints.max_width_mm",
        }
    )
    return CategorySpec(
        name="monitor",
        display_name="Màn hình máy tính",
        config=load_config(),
        profile_model=MonitorNeedProfile,
        custom_answer_model=MonitorCustomAnswer,
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
        list_patch_paths=list_paths,
        hard_retrieval_paths=hard_paths,
        question_profile_paths={
            "primary_usage": frozenset(
                {"usage_preferences", "soft_preferences"}
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "screen_size": frozenset(
                {
                    "screen_size_preference",
                    "preferred_screen_size_inch",
                    "soft_preferences",
                }
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.monitor.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
