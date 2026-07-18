"""Configuration helpers for the refrigerator category."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache the category-owned rules and metadata mapping."""
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid refrigerator config at {CONFIG_PATH}")
    return config


def get_missing_profile_fields(
    profile: dict[str, Any], config: dict[str, Any] | None = None
) -> list[str]:
    """Resolve missing core fields using deterministic category rules."""
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


def _build_search_text(profile: dict[str, Any], user_query: str) -> str:
    """Build category-specific semantic text; hard filters remain separate."""
    import json

    parts: list[Any] = [user_query]
    if profile.get("household_size"):
        parts.append(f"phù hợp gia đình {profile['household_size']} người")
    if profile.get("budget_max_vnd"):
        parts.append(f"ngân sách tối đa {profile['budget_max_vnd']} đồng")
    parts.extend(profile.get("usage_preferences", []))
    parts.extend(profile.get("soft_preferences", []))
    parts.extend(profile.get("implicit_needs", []))
    hard = profile.get("hard_constraints") or {}
    if hard:
        parts.append(json.dumps(hard, ensure_ascii=False, default=str))
    return ". ".join(str(part) for part in parts if part)


def _no_match_answer(profile: dict[str, Any]) -> str:
    constraints: list[str] = []
    if profile.get("budget_max_vnd"):
        constraints.append(f"ngân sách tối đa {profile['budget_max_vnd']:,} đồng")
    if profile.get("household_size"):
        constraints.append(f"nhu cầu cho {profile['household_size']} người")
    constraint_text = ", ".join(constraints) or "các điều kiện hiện tại"
    return (
        f"Mình chưa tìm thấy tủ lạnh đáp ứng đầy đủ {constraint_text}. "
        "Bạn có thể cân nhắc nới một điều kiện, nhưng mình chưa tự ý bỏ yêu cầu nào của bạn."
    )


def get_category_spec():
    """Return the refrigerator implementation of the shared category contract."""
    from advisor.categories.base import CategorySpec
    from advisor.categories.refrigerator.filter_builder import build_filter
    from advisor.categories.refrigerator.prompts import (
        build_advisory_prompt,
        build_custom_answer_prompt,
        build_need_extraction_prompt,
        build_response_prompt,
    )
    from advisor.retrieval.qdrant import normalize_candidate
    from advisor.schemas import CustomAnswerInterpretation, RefrigeratorNeedExtraction

    valid_paths = frozenset(
        {
            "household_size",
            "budget_max_vnd",
            "budget_segment",
            "usage_preferences",
            "soft_preferences",
            "implicit_needs",
            "hard_constraints.brands",
            "hard_constraints.styles",
            "hard_constraints.min_capacity_lit",
            "hard_constraints.max_capacity_lit",
            "hard_constraints.max_width_cm",
            "hard_constraints.max_height_cm",
            "hard_constraints.max_depth_cm",
            "hard_constraints.required_features",
        }
    )
    return CategorySpec(
        name="refrigerator",
        display_name="Tủ Lạnh",
        config=load_config(),
        profile_model=RefrigeratorNeedExtraction,
        custom_answer_model=CustomAnswerInterpretation,
        build_need_extraction_prompt=build_need_extraction_prompt,
        build_custom_answer_prompt=build_custom_answer_prompt,
        build_ranking_prompt=build_advisory_prompt,
        build_response_prompt=build_response_prompt,
        build_filter=build_filter,
        build_search_text=_build_search_text,
        no_match_answer=_no_match_answer,
        normalize_candidate=normalize_candidate,
        get_missing_profile_fields=get_missing_profile_fields,
        valid_patch_paths=valid_paths,
        list_patch_paths=frozenset(
            {
                "usage_preferences",
                "soft_preferences",
                "implicit_needs",
                "hard_constraints.brands",
                "hard_constraints.styles",
                "hard_constraints.required_features",
            }
        ),
        hard_retrieval_paths=frozenset(
            {
                "household_size",
                "budget_max_vnd",
                "budget_segment",
                "hard_constraints.brands",
                "hard_constraints.styles",
                "hard_constraints.min_capacity_lit",
                "hard_constraints.max_capacity_lit",
                "hard_constraints.max_width_cm",
                "hard_constraints.max_height_cm",
                "hard_constraints.max_depth_cm",
                "hard_constraints.required_features",
            }
        ),
        question_profile_paths={
            "household_size": frozenset({"household_size"}),
            "budget": frozenset({"budget_max_vnd", "budget_segment"}),
            "usage_preferences": frozenset({"usage_preferences"}),
        },
        setup_indexes_command=(
            "PYTHONPATH=src .venv/bin/python -m "
            "advisor.categories.refrigerator.setup_indexes --apply"
        ),
    )


__all__ = ["get_category_spec", "get_missing_profile_fields", "load_config"]
