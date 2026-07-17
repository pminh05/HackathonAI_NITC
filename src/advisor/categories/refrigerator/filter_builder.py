"""Translate refrigerator needs into indexed Qdrant payload filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.refrigerator import load_config


FEATURE_PROFILE_KEYS = {
    "inverter": "inverter",
    "external_water": "external_water",
    "automatic_mode": "automatic_mode",
}


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter | None:
    """Build strict filters only from explicit hard constraints.

    Soft and implicit preferences deliberately stay in the semantic query and
    final selection prompt. Missing metadata never silently satisfies a hard
    constraint, with the sole exception of an open-ended people range.
    """
    fields = payload_fields or load_config()["payload_fields"]
    must: list[models.Condition] = []

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        must.append(
            models.Filter(
                should=[
                    _range_condition(fields["promotional_price"], lte=budget_max),
                    _range_condition(fields["original_price"], lte=budget_max),
                ]
            )
        )

    household_size = need_profile.get("household_size")
    if household_size is not None:
        must.extend(
            [
                _range_condition(fields["people_min"], lte=household_size),
                models.Filter(
                    should=[
                        _range_condition(fields["people_max"], gte=household_size),
                        models.IsEmptyCondition(
                            is_empty=models.PayloadField(key=fields["people_max"])
                        ),
                    ]
                ),
            ]
        )

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value) for value in hard.get("brands", []) if value]
    if brands:
        must.append(
            models.FieldCondition(
                key=fields["brand"], match=models.MatchAny(any=brands)
            )
        )

    styles = [str(value) for value in hard.get("styles", []) if value]
    if styles:
        must.append(
            models.FieldCondition(
                key=fields["style"], match=models.MatchAny(any=styles)
            )
        )

    if hard.get("min_capacity_lit") is not None:
        must.append(
            _range_condition(fields["capacity"], gte=hard["min_capacity_lit"])
        )
    if hard.get("max_capacity_lit") is not None:
        must.append(
            _range_condition(fields["capacity"], lte=hard["max_capacity_lit"])
        )

    dimension_limits = {
        "max_width_cm": "width",
        "max_height_cm": "height",
        "max_depth_cm": "depth",
    }
    for profile_key, payload_key in dimension_limits.items():
        if hard.get(profile_key) is not None:
            must.append(
                _range_condition(fields[payload_key], lte=hard[profile_key])
            )

    for feature in hard.get("required_features", []):
        mapped_key = FEATURE_PROFILE_KEYS.get(feature)
        if mapped_key:
            must.append(
                models.FieldCondition(
                    key=fields[mapped_key], match=models.MatchValue(value=True)
                )
            )

    return models.Filter(must=must) if must else None


def serialize_filter(query_filter: models.Filter | None) -> dict[str, Any] | None:
    if query_filter is None:
        return None
    return query_filter.model_dump(mode="json", exclude_none=True)


def deserialize_filter(data: dict[str, Any] | None) -> models.Filter | None:
    return models.Filter.model_validate(data) if data else None
