"""Translate washing-machine needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.washing_machine import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.brand",
    "metadata.product_type",
    "metadata.drum_type",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.wash_capacity_kg",
    "metadata.people_min",
    "metadata.people_max",
    "metadata.has_inverter",
    "metadata.has_dryer",
    "metadata.width_cm",
    "metadata.height_cm",
    "metadata.depth_cm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported maygiat metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    """Build strict filters plus the washing-machine collection eligibility rule."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = []
    must_not: list[models.Condition] = [
        models.FieldCondition(
            key=fields["product_type"],
            match=models.MatchValue(value="clothing_care"),
        )
    ]

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        # A promotion is the effective price. Fall back to original price only
        # when no promotion is available for the product.
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

    product_types = [str(value) for value in hard.get("product_types", []) if value]
    if product_types:
        must.append(
            models.FieldCondition(
                key=fields["product_type"],
                match=models.MatchAny(any=product_types),
            )
        )

    drum_types = [str(value) for value in hard.get("drum_types", []) if value]
    if drum_types:
        must.append(
            models.FieldCondition(
                key=fields["drum_type"], match=models.MatchAny(any=drum_types)
            )
        )

    if hard.get("min_wash_capacity_kg") is not None:
        must.append(
            _range_condition(
                fields["wash_capacity"], gte=hard["min_wash_capacity_kg"]
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

    for profile_key, payload_key in {
        "inverter": "inverter",
        "dryer": "dryer",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key],
                    match=models.MatchValue(value=hard[profile_key]),
                )
            )

    return models.Filter(must=must or None, must_not=must_not)
