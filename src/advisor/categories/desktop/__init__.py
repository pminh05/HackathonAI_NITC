"""Configuration and behavior contract for the desktop-computer category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
USE_LABELS = {
    "office_study": "văn phòng và học tập",
    "programming_multitasking": "lập trình và đa nhiệm",
    "gaming": "chơi game",
    "creative_content": "thiết kế và dựng nội dung",
    "engineering_workstation": "kỹ thuật và workstation",
    "general": "dùng đa mục đích",
}
FORM_LABELS = {
    "all_in_one": "máy all-in-one liền màn hình",
    "separate_unit": "bộ máy dùng màn hình riêng",
    "flexible": "không giới hạn kiểu máy",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid desktop config at {CONFIG_PATH}")
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
        elif field == "desktop_form":
            resolved = bool(profile.get("form_preference"))
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
    form = profile.get("form_preference")
    if form in FORM_LABELS:
        parts.append(FORM_LABELS[form])
    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("cpu_vendors"):
        parts.append("CPU " + ", ".join(hard["cpu_vendors"]))
    if hard.get("os_families"):
        parts.append("hệ điều hành " + ", ".join(hard["os_families"]))
    if hard.get("min_ram_gb") is not None:
        parts.append(f"RAM ít nhất {hard['min_ram_gb']} GB")
    if hard.get("min_storage_gb") is not None:
        parts.append(f"lưu trữ ít nhất {hard['min_storage_gb']} GB")
    if hard.get("gpu_types"):
        parts.append("GPU " + ", ".join(hard["gpu_types"]))
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    form = profile.get("form_preference")
    if form and form != "flexible":
        constraints.append(FORM_LABELS.get(form, form))
    hard = profile.get("hard_constraints") or {}
    if hard.get("cpu_vendors"):
        constraints.append("CPU " + ", ".join(hard["cpu_vendors"]))
    if hard.get("os_families"):
        constraints.append("hệ điều hành " + ", ".join(hard["os_families"]))
    if hard.get("min_ram_gb") is not None:
        constraints.append(f"RAM từ {hard['min_ram_gb']} GB")
    if hard.get("min_storage_gb") is not None:
        constraints.append(f"lưu trữ từ {hard['min_storage_gb']} GB")
    if "discrete" in hard.get("gpu_types", []):
        constraints.append("card đồ họa rời")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có dữ liệu giá không được coi là đạt trần ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy máy tính để bàn đáp ứng đầy đủ {text} trong catalog "
        f"hiện tại.{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình "
        "chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.desktop.filter_builder import build_filter
    from advisor.categories.desktop.normalizer import normalize_candidate
    from advisor.categories.desktop.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.desktop.schemas import (
        DesktopCustomAnswer,
        DesktopNeedProfile,
    )

    valid_paths = frozenset(
        {
            "budget_max_vnd",
            "budget_segment",
            "form_preference",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.cpu_vendors",
            "hard_constraints.os_families",
            "hard_constraints.storage_types",
            "hard_constraints.gpu_types",
            "hard_constraints.min_ram_gb",
            "hard_constraints.min_supported_ram_gb",
            "hard_constraints.min_storage_gb",
            "hard_constraints.min_screen_size_inch",
            "hard_constraints.max_screen_size_inch",
            "hard_constraints.requires_wifi",
        }
    )
    list_paths = frozenset(
        {
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.cpu_vendors",
            "hard_constraints.os_families",
            "hard_constraints.storage_types",
            "hard_constraints.gpu_types",
        }
    )
    return CategorySpec(
        name="desktop",
        display_name="Máy tính để bàn",
        config=load_config(),
        profile_model=DesktopNeedProfile,
        custom_answer_model=DesktopCustomAnswer,
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
        hard_retrieval_paths=frozenset(
            {
                "budget_max_vnd",
                "budget_segment",
                "form_preference",
                "hard_constraints.brands",
                "hard_constraints.cpu_vendors",
                "hard_constraints.os_families",
                "hard_constraints.storage_types",
                "hard_constraints.gpu_types",
                "hard_constraints.min_ram_gb",
                "hard_constraints.min_supported_ram_gb",
                "hard_constraints.min_storage_gb",
                "hard_constraints.min_screen_size_inch",
                "hard_constraints.max_screen_size_inch",
                "hard_constraints.requires_wifi",
            }
        ),
        question_profile_paths={
            "primary_usage": frozenset(
                {"usage_preferences", "soft_preferences"}
            ),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "desktop_form": frozenset({"form_preference"}),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.desktop.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
