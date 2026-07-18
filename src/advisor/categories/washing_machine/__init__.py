"""Configuration and behavior contract for the washing-machine category."""

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
        raise ValueError(f"Invalid washing-machine config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    category_config = config or load_config()
    missing: list[str] = []
    for field in category_config["required_profile_fields"]:
        if field == "budget":
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
    if profile.get("household_size") is not None:
        parts.append(f"phù hợp gia đình {profile['household_size']} người")
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
    if profile.get("household_size") is not None:
        constraints.append(f"nhu cầu cho {profile['household_size']} người")
    if profile.get("budget_max_vnd") is not None:
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    hard = profile.get("hard_constraints") or {}
    if hard.get("min_wash_capacity_kg") is not None:
        constraints.append(
            f"khối lượng giặt từ {hard['min_wash_capacity_kg']:g} kg"
        )
    if hard.get("dryer") is True:
        constraints.append("có chức năng sấy")
    text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy máy giặt đáp ứng đầy đủ {text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào."
    )


def get_category_spec():
    from advisor.categories.base import CategorySpec
    from advisor.categories.washing_machine.filter_builder import build_filter
    from advisor.categories.washing_machine.normalizer import normalize_candidate
    from advisor.categories.washing_machine.prompts import (
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_ranking_prompt,
        build_response_prompt,
    )
    from advisor.categories.washing_machine.schemas import (
        WashingMachineCustomAnswer,
        WashingMachineNeedProfile,
    )

    valid_paths = frozenset(
        {
            "household_size",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.product_types",
            "hard_constraints.drum_types",
            "hard_constraints.min_wash_capacity_kg",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.inverter",
            "hard_constraints.dryer",
        }
    )
    return CategorySpec(
        name="washing_machine",
        display_name="Máy giặt",
        config=load_config(),
        profile_model=WashingMachineNeedProfile,
        custom_answer_model=WashingMachineCustomAnswer,
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
                "hard_constraints.drum_types",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "household_size",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.product_types",
                "hard_constraints.drum_types",
                "hard_constraints.min_wash_capacity_kg",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.inverter",
                "hard_constraints.dryer",
            }
        ),
        question_profile_paths={
            "household_size": frozenset({"household_size"}),
            "budget": frozenset(
                {"budget_max_vnd", "budget_segment", "soft_preferences"}
            ),
            "usage_preferences": frozenset(
                {
                    "usage_preferences",
                    "soft_preferences",
                    "hard_constraints.dryer",
                }
            ),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.washing_machine.setup_indexes --apply"
        ),
    )


__all__ = [
    "build_search_text",
    "get_category_spec",
    "get_missing_profile_fields",
    "load_config",
    "no_match_answer",
]
