"""Configuration and behavior contract for the micro-karaoke category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")
USAGE_LABELS = {
    "home_family": "hát karaoke gia đình",
    "karaoke_room": "dùng trong phòng karaoke cố định",
    "stage_event": "dùng cho sân khấu hoặc sự kiện",
    "portable": "thường xuyên mang theo",
}
CONNECTION_LABELS = {
    "wired": "micro có dây",
    "wireless": "micro không dây",
    "open": "chưa ưu tiên micro có dây hay không dây",
}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid karaoke microphone config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "usage_context":
            resolved = bool(profile.get("usage_preferences"))
        elif field == "connection_preference":
            hard = profile.get("hard_constraints") or {}
            resolved = bool(profile.get("connection_preference")) or bool(
                hard.get("microphone_types")
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
    parts.extend(
        USAGE_LABELS.get(value, str(value))
        for value in profile.get("usage_preferences", [])
    )
    connection = profile.get("connection_preference")
    if connection in CONNECTION_LABELS:
        parts.append(CONNECTION_LABELS[connection])
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("brands"):
        parts.append("hãng " + ", ".join(str(value) for value in hard["brands"]))
    if hard.get("microphone_types"):
        parts.append("loại " + ", ".join(hard["microphone_types"]))
    if hard.get("wireless_bands"):
        parts.append("băng tần " + ", ".join(hard["wireless_bands"]))
    parts.extend(str(value) for value in profile.get("soft_preferences", []))
    parts.extend(str(value) for value in profile.get("implicit_needs", []))
    return ". ".join(part for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    budget = profile.get("budget_max_vnd")
    if budget is not None:
        constraints.append(f"ngân sách tối đa {budget:,} đồng")
    hard = profile.get("hard_constraints") or {}
    types = hard.get("microphone_types") or []
    if not types and profile.get("connection_preference") in {"wired", "wireless"}:
        types = [profile["connection_preference"]]
    if types:
        constraints.append(
            "loại "
            + ", ".join(CONNECTION_LABELS.get(value, str(value)) for value in types)
        )
    if hard.get("brands"):
        constraints.append("hãng " + ", ".join(hard["brands"]))
    if hard.get("wireless_bands"):
        constraints.append("băng tần " + ", ".join(hard["wireless_bands"]))
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    price_note = (
        " Các mẫu chưa có giá xác minh chỉ được dùng để tham khảo và không thể "
        "khẳng định là nằm trong ngân sách."
        if budget is not None
        else ""
    )
    return (
        f"Mình chưa tìm thấy micro karaoke đáp ứng đầy đủ {text} trong dữ liệu hiện có."
        f"{price_note} Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý "
        "bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.karaoke_microphone.filter_builder import build_filter
    from advisor.categories.karaoke_microphone.normalizer import normalize_candidate
    from advisor.categories.karaoke_microphone.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.karaoke_microphone.schemas import (
        KaraokeMicrophoneCustomAnswer,
        KaraokeMicrophoneNeedProfile,
    )

    valid_paths = frozenset(
        {
            "connection_preference",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.microphone_types",
            "hard_constraints.wireless_bands",
        }
    )
    return CategorySpec(
        name="karaoke_microphone",
        display_name="Micro karaoke",
        config=load_config(),
        profile_model=KaraokeMicrophoneNeedProfile,
        custom_answer_model=KaraokeMicrophoneCustomAnswer,
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
                "hard_constraints.microphone_types",
                "hard_constraints.wireless_bands",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "connection_preference",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.microphone_types",
                "hard_constraints.wireless_bands",
            }
        ),
        question_profile_paths={
            "usage_context": frozenset({"usage_preferences", "implicit_needs"}),
            "connection_preference": frozenset({"connection_preference"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.karaoke_microphone.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
