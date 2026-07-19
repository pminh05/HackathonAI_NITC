"""Translate smartwatch needs into indexed Qdrant payload filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.smartwatch import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand_key",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.compatible_platforms",
    "metadata.call_mode",
    "metadata.has_cellular",
    "metadata.has_gps",
    "metadata.has_notifications",
    "metadata.swim_ready",
    "metadata.health_feature_tags",
    "metadata.display_family",
    "metadata.strap_material_family",
    "metadata.screen_size_inch",
    "metadata.case_width_mm",
    "metadata.weight_g",
    "metadata.wrist_min_cm",
    "metadata.wrist_max_cm",
    "metadata.typical_battery_hours",
    "metadata.water_resistance_atm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported donghothongminh metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        _match_value(fields["category_scope"], "smartwatch")
    ]

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        must.append(
            models.Filter(
                should=[
                    _range_condition(fields["promotional_price"], lte=budget_max),
                    models.Filter(
                        must=[
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(
                                    key=fields["promotional_price"]
                                )
                            ),
                            _range_condition(fields["original_price"], lte=budget_max),
                        ]
                    ),
                ]
            )
        )

    platform = need_profile.get("phone_platform")
    if platform in {"ios", "android"}:
        must.append(_match_value(fields["compatible_platforms"], platform))

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value).strip().casefold() for value in hard.get("brands", []) if value]
    if brands:
        must.append(
            models.FieldCondition(
                key=fields["brand_key"], match=models.MatchAny(any=brands)
            )
        )
    for profile_key, payload_key in {
        "display_families": "display_family",
        "strap_material_families": "strap_material_family",
    }.items():
        values = [str(value) for value in hard.get(profile_key, []) if value]
        if values:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key], match=models.MatchAny(any=values)
                )
            )

    range_rules = {
        "min_screen_size_inch": ("screen_size", "gte"),
        "max_screen_size_inch": ("screen_size", "lte"),
        "max_case_width_mm": ("case_width", "lte"),
        "max_weight_g": ("weight", "lte"),
        "min_typical_battery_hours": ("typical_battery_hours", "gte"),
        "min_water_resistance_atm": ("water_resistance_atm", "gte"),
    }
    for profile_key, (payload_key, operator) in range_rules.items():
        value = hard.get(profile_key)
        if value is not None:
            must.append(_range_condition(fields[payload_key], **{operator: value}))

    wrist = hard.get("wrist_circumference_cm")
    if wrist is not None:
        must.extend(
            [
                _range_condition(fields["wrist_min"], lte=wrist),
                _range_condition(fields["wrist_max"], gte=wrist),
            ]
        )

    call_requirement = hard.get("call_requirement")
    if call_requirement == "on_wrist":
        must.append(
            models.FieldCondition(
                key=fields["call_mode"],
                match=models.MatchAny(any=["on_wrist", "standalone"]),
            )
        )
    elif call_requirement == "standalone":
        must.append(_match_value(fields["call_mode"], "standalone"))

    for profile_key, payload_key in {
        "requires_cellular": "cellular",
        "requires_gps": "gps",
        "requires_notifications": "notifications",
        "requires_swimming": "swim_ready",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_match_value(fields[payload_key], hard[profile_key]))

    for feature in hard.get("required_health_features", []):
        if feature:
            must.append(_match_value(fields["health_features"], str(feature)))
    return models.Filter(must=must)
