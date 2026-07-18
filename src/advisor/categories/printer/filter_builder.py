"""Translate printer needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.printer import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.print_technology",
    "metadata.color_mode",
    "metadata.monthly_volume_max_pages",
    "metadata.print_speed_ppm",
    "metadata.connection_tags",
    "metadata.paper_size_tags",
    "metadata.supports_duplex",
    "metadata.width_mm",
    "metadata.height_mm",
    "metadata.depth_mm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported mayin metadata filter field(s): " + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        _match_value(fields["category_scope"], "printer")
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
    if need_profile.get("monthly_pages_estimate") is not None:
        must.append(
            _range_condition(
                fields["monthly_volume_max"],
                gte=need_profile["monthly_pages_estimate"],
            )
        )

    hard = need_profile.get("hard_constraints") or {}
    for profile_key, payload_key in {
        "brands": "brand",
        "technologies": "print_technology",
        "color_modes": "color_mode",
    }.items():
        values = [str(value) for value in hard.get(profile_key, []) if value]
        if values:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key], match=models.MatchAny(any=values)
                )
            )
    if hard.get("min_print_speed_ppm") is not None:
        must.append(
            _range_condition(
                fields["print_speed"], gte=hard["min_print_speed_ppm"]
            )
        )
    # Each required tag is a separate condition so requirements are ANDed.
    for value in hard.get("required_connections", []):
        must.append(_match_value(fields["connection_tags"], value))
    for value in hard.get("required_paper_sizes", []):
        must.append(_match_value(fields["paper_size_tags"], value))
    if hard.get("requires_duplex") is not None:
        must.append(_match_value(fields["supports_duplex"], hard["requires_duplex"]))
    for profile_key, payload_key in {
        "max_width_mm": "width",
        "max_height_mm": "height",
        "max_depth_mm": "depth",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], lte=hard[profile_key]))
    return models.Filter(must=must)
