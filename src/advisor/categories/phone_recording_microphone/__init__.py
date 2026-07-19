"""Configuration and behavior contract for the recording-microphone category."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
SETUP_LABELS = {
    "iphone_lightning": "iPhone hoặc iPad cổng Lightning",
    "iphone_usb_c": "iPhone hoặc iPad cổng USB-C",
    "android_usb_c": "điện thoại Android cổng USB-C",
    "camera_3_5mm": "camera cổng 3.5 mm",
    "computer_usb": "máy tính qua USB",
    "open": "chưa xác định thiết bị hoặc cổng",
}
USAGE_LABELS = {
    "solo_content": "quay video hoặc vlog một người",
    "two_person_interview": "phỏng vấn hoặc quay hai người",
    "outdoor_mobile": "thu âm ngoài trời và thường xuyên di chuyển",
    "podcast_livestream": "podcast, livestream hoặc gaming tại bàn",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid recording-microphone config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    hard = profile.get("hard_constraints") or {}
    for field in category_config["required_profile_fields"]:
        if field == "recording_setup":
            resolved = bool(profile.get("recording_setup")) or (
                bool(hard.get("required_compatibility_tags"))
                and bool(hard.get("connector_types"))
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
    parts: list[Any] = [user_query]
    setup = profile.get("recording_setup")
    if setup in SETUP_LABELS:
        parts.append(SETUP_LABELS[setup])
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    if profile.get("budget_segment") == "premium":
        parts.append("phân khúc cao cấp")
    parts.extend(
        USAGE_LABELS.get(value, str(value))
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
    setup = profile.get("recording_setup")
    if setup in SETUP_LABELS and setup != "open":
        constraints.append(SETUP_LABELS[setup])
    if profile.get("budget_max_vnd") is not None:
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("min_transmitter_count") is not None:
        constraints.append(f"ít nhất {hard['min_transmitter_count']} bộ phát")
    if hard.get("min_runtime_hours") is not None:
        constraints.append(f"thời lượng từ {hard['min_runtime_hours']:g} giờ")
    if hard.get("min_transmission_range_m") is not None:
        constraints.append(
            f"phạm vi truyền từ {hard['min_transmission_range_m']:g} m"
        )
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu thiếu giá xác minh không được coi là đạt trần ngân sách."
        if profile.get("budget_max_vnd") is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy micro thu âm đáp ứng đầy đủ {text} trong dữ liệu hiện có."
        f"{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý "
        "bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.phone_recording_microphone.filter_builder import (
        build_filter,
    )
    from advisor.categories.phone_recording_microphone.normalizer import (
        normalize_candidate,
    )
    from advisor.categories.phone_recording_microphone.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.phone_recording_microphone.schemas import (
        PhoneRecordingMicrophoneCustomAnswer,
        PhoneRecordingMicrophoneNeedProfile,
    )

    valid_paths = frozenset(
        {
            "recording_setup",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.product_types",
            "hard_constraints.required_compatibility_tags",
            "hard_constraints.connector_types",
            "hard_constraints.min_transmitter_count",
            "hard_constraints.min_runtime_hours",
            "hard_constraints.min_transmission_range_m",
            "hard_constraints.pickup_patterns",
            "hard_constraints.wireless_bands",
            "hard_constraints.required_features",
        }
    )
    return CategorySpec(
        name="phone_recording_microphone",
        display_name="Micro thu âm",
        config=load_config(),
        profile_model=PhoneRecordingMicrophoneNeedProfile,
        custom_answer_model=PhoneRecordingMicrophoneCustomAnswer,
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
                "hard_constraints.required_compatibility_tags",
                "hard_constraints.connector_types",
                "hard_constraints.pickup_patterns",
                "hard_constraints.wireless_bands",
                "hard_constraints.required_features",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "recording_setup",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.product_types",
                "hard_constraints.required_compatibility_tags",
                "hard_constraints.connector_types",
                "hard_constraints.min_transmitter_count",
                "hard_constraints.min_runtime_hours",
                "hard_constraints.min_transmission_range_m",
                "hard_constraints.pickup_patterns",
                "hard_constraints.wireless_bands",
                "hard_constraints.required_features",
            }
        ),
        question_profile_paths={
            "recording_setup": frozenset({"recording_setup"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "usage_preferences": frozenset(
                {
                    "usage_preferences",
                    "soft_preferences",
                    "implicit_needs",
                    "hard_constraints.min_transmitter_count",
                }
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.phone_recording_microphone.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
