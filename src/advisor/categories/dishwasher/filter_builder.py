"""Translate dishwasher needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.dishwasher import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand",
    "metadata.product_type",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.vietnamese_meals_max",
    "metadata.place_settings_max",
    "metadata.water_max_l_per_cycle",
    "metadata.noise_db",
    "metadata.width_cm",
    "metadata.height_cm",
    "metadata.depth_cm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported mayruachen metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    """Build strict filters and enforce dishwasher-only collection eligibility."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        models.FieldCondition(
            key=fields["category_scope"],
            match=models.MatchValue(value="dishwasher"),
        )
    ]

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        # Promotion is the effective price. Original price is only considered
        # when the promotion field is missing; unknown prices fail a hard cap.
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

    if hard.get("min_place_settings") is not None:
        must.append(
            _range_condition(
                fields["place_settings_max"], gte=hard["min_place_settings"]
            )
        )
    if hard.get("min_vietnamese_meals") is not None:
        must.append(
            _range_condition(
                fields["vietnamese_meals_max"],
                gte=hard["min_vietnamese_meals"],
            )
        )
    if hard.get("max_water_l_per_cycle") is not None:
        must.append(
            _range_condition(
                fields["water_max"], lte=hard["max_water_l_per_cycle"]
            )
        )
    if hard.get("max_noise_db") is not None:
        must.append(_range_condition(fields["noise"], lte=hard["max_noise_db"]))

    for profile_key, payload_key in {
        "max_width_cm": "width",
        "max_height_cm": "height",
        "max_depth_cm": "depth",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(
                _range_condition(fields[payload_key], lte=hard[profile_key])
            )

    return models.Filter(must=must)
