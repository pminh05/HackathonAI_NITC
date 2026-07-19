"""Translate cooler/freezer needs into indexed Qdrant payload filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.cooler_freezer import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.product_family",
    "metadata.product_type",
    "metadata.is_mini",
    "metadata.brand",
    "metadata.effective_price_vnd",
    "metadata.total_capacity_lit",
    "metadata.temperature_min_c",
    "metadata.has_inverter",
    "metadata.gas_type",
    "metadata.feature_tags",
    "metadata.width_cm",
    "metadata.height_cm",
    "metadata.depth_cm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported tumattudong metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter | None:
    """Build strict filters only from explicit, indexed requirements."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = []

    product_family = need_profile.get("product_family")
    if product_family in {"cooler", "freezer"}:
        must.append(
            models.FieldCondition(
                key=fields["product_family"],
                match=models.MatchValue(value=product_family),
            )
        )

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        must.append(_range_condition(fields["effective_price"], lte=budget_max))

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value) for value in hard.get("brands", []) if value]
    if brands:
        must.append(
            models.FieldCondition(
                key=fields["brand"], match=models.MatchAny(any=brands)
            )
        )

    size_variants = set(hard.get("size_variants", []))
    if size_variants == {"mini"}:
        must.append(
            models.FieldCondition(
                key=fields["is_mini"], match=models.MatchValue(value=True)
            )
        )
    elif size_variants == {"standard"}:
        must.append(
            models.FieldCondition(
                key=fields["is_mini"], match=models.MatchValue(value=False)
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

    if hard.get("required_temperature_c") is not None:
        # To reach -24°C, a product's coldest declared temperature must be
        # numerically less than or equal to -24.
        must.append(
            _range_condition(
                fields["temperature_min"],
                lte=hard["required_temperature_c"],
            )
        )

    for profile_key, payload_key in {
        "max_width_cm": "width",
        "max_height_cm": "height",
        "max_depth_cm": "depth",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(
                _range_condition(fields[payload_key], lte=hard[profile_key])
            )

    if hard.get("inverter") is not None:
        must.append(
            models.FieldCondition(
                key=fields["inverter"],
                match=models.MatchValue(value=hard["inverter"]),
            )
        )

    gas_types = [str(value) for value in hard.get("gas_types", []) if value]
    if gas_types:
        must.append(
            models.FieldCondition(
                key=fields["gas_type"], match=models.MatchAny(any=gas_types)
            )
        )

    for feature in hard.get("required_features", []):
        must.append(
            models.FieldCondition(
                key=fields["feature_tags"],
                match=models.MatchValue(value=feature),
            )
        )

    return models.Filter(must=must) if must else None
