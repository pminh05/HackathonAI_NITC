"""Translate water-heater needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.water_heater import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand",
    "metadata.product_type",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.capacity_l",
    "metadata.power_w",
    "metadata.heating_time_max_minutes",
    "metadata.has_booster_pump",
    "metadata.includes_shower",
    "metadata.width_cm",
    "metadata.height_cm",
    "metadata.depth_cm",
    "metadata.ip_rating",
    "metadata.safety_tags",
}

FEATURE_PROFILE_KEYS = {
    "booster_pump": "booster_pump",
    "included_shower": "included_shower",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported maynuocnong metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    """Build strict filters and enforce water-heater collection eligibility."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        models.FieldCondition(
            key=fields["category_scope"],
            match=models.MatchValue(value="water_heater"),
        )
    ]

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        # A promotion is the effective price. Fall back to the original only
        # when no promotion exists; unknown prices fail an explicit cap.
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

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value) for value in hard.get("brands", []) if value]
    if brands:
        must.append(
            models.FieldCondition(
                key=fields["brand"], match=models.MatchAny(any=brands)
            )
        )

    product_types = [str(value) for value in hard.get("product_types", []) if value]
    if product_types:
        must.append(
            models.FieldCondition(
                key=fields["product_type"],
                match=models.MatchAny(any=product_types),
            )
        )

    for profile_key, operator, payload_key in (
        ("min_capacity_lit", "gte", "capacity"),
        ("max_capacity_lit", "lte", "capacity"),
        ("max_power_w", "lte", "power"),
        ("max_heating_time_minutes", "lte", "heating_time_max"),
        ("max_width_cm", "lte", "width"),
        ("max_height_cm", "lte", "height"),
        ("max_depth_cm", "lte", "depth"),
    ):
        value = hard.get(profile_key)
        if value is not None:
            must.append(_range_condition(fields[payload_key], **{operator: value}))

    for feature in hard.get("required_features", []):
        payload_key = FEATURE_PROFILE_KEYS.get(str(feature))
        if payload_key:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key],
                    match=models.MatchValue(value=True),
                )
            )

    safety_features = [
        str(value) for value in hard.get("required_safety_features", []) if value
    ]
    for feature in safety_features:
        must.append(
            models.FieldCondition(
                key=fields["safety_tags"],
                match=models.MatchValue(value=feature),
            )
        )

    ip_ratings = [str(value) for value in hard.get("ip_ratings", []) if value]
    if ip_ratings:
        must.append(
            models.FieldCondition(
                key=fields["ip_rating"],
                match=models.MatchAny(any=ip_ratings),
            )
        )

    return models.Filter(must=must)
