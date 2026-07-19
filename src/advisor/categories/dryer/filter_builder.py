"""Translate clothes-dryer needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.dryer import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.brand",
    "metadata.dryer_type",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.dry_capacity_kg",
    "metadata.people_min",
    "metadata.people_max",
    "metadata.has_inverter",
    "metadata.has_sensor",
    "metadata.power_max_w",
    "metadata.width_cm",
    "metadata.height_cm",
    "metadata.depth_cm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported maysayquanao metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter | None:
    """Build strict filters only from explicit, indexed constraints."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = []

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

    dryer_types = [str(value) for value in hard.get("dryer_types", []) if value]
    if dryer_types:
        must.append(
            models.FieldCondition(
                key=fields["dryer_type"],
                match=models.MatchAny(any=dryer_types),
            )
        )

    if hard.get("min_dry_capacity_kg") is not None:
        must.append(
            _range_condition(
                fields["dry_capacity"], gte=hard["min_dry_capacity_kg"]
            )
        )
    if hard.get("max_dry_capacity_kg") is not None:
        must.append(
            _range_condition(
                fields["dry_capacity"], lte=hard["max_dry_capacity_kg"]
            )
        )

    for profile_key, payload_key in {
        "max_width_cm": "width",
        "max_height_cm": "height",
        "max_depth_cm": "depth",
        "max_power_w": "power_max",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(
                _range_condition(fields[payload_key], lte=hard[profile_key])
            )

    for profile_key, payload_key in {
        "inverter": "inverter",
        "sensor": "sensor",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key],
                    match=models.MatchValue(value=hard[profile_key]),
                )
            )

    return models.Filter(must=must) if must else None
