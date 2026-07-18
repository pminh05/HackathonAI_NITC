"""Configuration and behavior contract for the air-conditioner category."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid air-conditioner config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "room_size":
            resolved = profile.get("room_area_m2") is not None or profile.get(
                "room_volume_m3"
            ) is not None
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
    if profile.get("room_area_m2") is not None:
        parts.append(f"phòng {profile['room_area_m2']} mét vuông")
    if profile.get("room_volume_m3") is not None:
        parts.append(f"thể tích phòng {profile['room_volume_m3']} mét khối")
    if profile.get("room_type"):
        parts.append(f"loại phòng {profile['room_type']}")
    if profile.get("budget_max_vnd") is not None:
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(profile.get("usage_preferences", []))
    parts.extend(profile.get("soft_preferences", []))
    parts.extend(profile.get("implicit_needs", []))
    hard = profile.get("hard_constraints") or {}
    if hard:
        parts.append(json.dumps(hard, ensure_ascii=False, default=str))
    return ". ".join(str(part) for part in parts if part)


def no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    if profile.get("room_area_m2") is not None:
        constraints.append(f"phòng {profile['room_area_m2']:g} m²")
    if profile.get("room_volume_m3") is not None:
        constraints.append(f"thể tích {profile['room_volume_m3']:g} m³")
    if profile.get("budget_max_vnd") is not None:
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy máy lạnh đáp ứng đầy đủ {text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.air_conditioner.filter_builder import build_filter
    from advisor.categories.air_conditioner.normalizer import normalize_candidate
    from advisor.categories.air_conditioner.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.air_conditioner.schemas import (
        AirConditionerCustomAnswer,
        AirConditionerNeedProfile,
    )
    from advisor.categories.base import CategorySpec

    valid_paths = frozenset(
        {
            "room_area_m2",
            "room_volume_m3",
            "room_type",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.machine_types",
            "hard_constraints.inverter",
            "hard_constraints.gas_types",
        }
    )
    return CategorySpec(
        name="air_conditioner",
        display_name="Máy lạnh",
        config=load_config(),
        profile_model=AirConditionerNeedProfile,
        custom_answer_model=AirConditionerCustomAnswer,
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
                "hard_constraints.machine_types",
                "hard_constraints.gas_types",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "room_area_m2",
                "room_volume_m3",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.machine_types",
                "hard_constraints.inverter",
                "hard_constraints.gas_types",
            }
        ),
        question_profile_paths={
            "room_size": frozenset({"room_area_m2", "room_volume_m3"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "usage_preferences": frozenset(
                {
                    "room_type",
                    "usage_preferences",
                    "soft_preferences",
                    "implicit_needs",
                    "hard_constraints.machine_types",
                }
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.air_conditioner.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]

