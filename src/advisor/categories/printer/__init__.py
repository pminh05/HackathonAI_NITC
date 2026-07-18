"""Configuration and behavior contract for the printer category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
PURPOSE_LABELS = {
    "mono_documents": "in văn bản đen trắng",
    "color_documents": "in tài liệu màu",
    "photo": "in ảnh màu",
    "receipt_label": "in hóa đơn hoặc nhãn",
    "general": "in đa mục đích",
}
VOLUME_LABELS = {
    "light": "tối đa khoảng 200 trang mỗi tháng",
    "regular": "khoảng 200–500 trang mỗi tháng",
    "office": "khoảng 500–1.500 trang mỗi tháng",
    "high": "trên 1.500 trang mỗi tháng",
    "open": "chưa xác định khối lượng in",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid printer config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "print_purpose":
            resolved = bool(profile.get("usage_preferences"))
        elif field == "monthly_volume":
            resolved = profile.get("monthly_pages_estimate") is not None or bool(
                profile.get("monthly_volume_segment")
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
    parts: list[str] = [user_query]
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(
        PURPOSE_LABELS.get(value, str(value))
        for value in profile.get("usage_preferences", [])
    )
    segment = profile.get("monthly_volume_segment")
    if segment in VOLUME_LABELS:
        parts.append(VOLUME_LABELS[segment])
    if profile.get("monthly_pages_estimate") is not None:
        parts.append(f"khoảng {profile['monthly_pages_estimate']} trang mỗi tháng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("technologies"):
        parts.append("công nghệ " + ", ".join(hard["technologies"]))
    if hard.get("color_modes"):
        parts.append("chế độ màu " + ", ".join(hard["color_modes"]))
    if hard.get("required_connections"):
        parts.append("kết nối " + ", ".join(hard["required_connections"]))
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    if profile.get("monthly_pages_estimate") is not None:
        constraints.append(
            f"công suất đáp ứng {profile['monthly_pages_estimate']:,} trang mỗi tháng"
        )
    hard = profile.get("hard_constraints") or {}
    if hard.get("technologies"):
        constraints.append("công nghệ " + ", ".join(hard["technologies"]))
    if hard.get("color_modes"):
        constraints.append("chế độ " + ", ".join(hard["color_modes"]))
    if hard.get("required_connections"):
        constraints.append("kết nối " + ", ".join(hard["required_connections"]))
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có dữ liệu giá không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy máy in đáp ứng đầy đủ {text} trong dữ liệu hiện có."
        f"{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý "
        "bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.printer.filter_builder import build_filter
    from advisor.categories.printer.normalizer import normalize_candidate
    from advisor.categories.printer.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.printer.schemas import PrinterCustomAnswer, PrinterNeedProfile

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "monthly_volume_segment",
            "monthly_pages_estimate",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.technologies",
            "hard_constraints.color_modes",
            "hard_constraints.min_print_speed_ppm",
            "hard_constraints.required_connections",
            "hard_constraints.required_paper_sizes",
            "hard_constraints.requires_duplex",
            "hard_constraints.max_width_mm",
            "hard_constraints.max_height_mm",
            "hard_constraints.max_depth_mm",
        }
    )
    return CategorySpec(
        name="printer",
        display_name="Máy in",
        config=load_config(),
        profile_model=PrinterNeedProfile,
        custom_answer_model=PrinterCustomAnswer,
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
                "hard_constraints.technologies",
                "hard_constraints.color_modes",
                "hard_constraints.required_connections",
                "hard_constraints.required_paper_sizes",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "budget_segment",
                "monthly_pages_estimate",
                "hard_constraints.brands",
                "hard_constraints.technologies",
                "hard_constraints.color_modes",
                "hard_constraints.min_print_speed_ppm",
                "hard_constraints.required_connections",
                "hard_constraints.required_paper_sizes",
                "hard_constraints.requires_duplex",
                "hard_constraints.max_width_mm",
                "hard_constraints.max_height_mm",
                "hard_constraints.max_depth_mm",
            }
        ),
        question_profile_paths={
            "print_purpose": frozenset(
                {
                    "usage_preferences",
                    "hard_constraints.color_modes",
                    "hard_constraints.technologies",
                }
            ),
            "monthly_volume": frozenset(
                {"monthly_volume_segment", "monthly_pages_estimate"}
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.printer.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
