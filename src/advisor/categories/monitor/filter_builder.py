"""Translate monitor needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.monitor import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand_key",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.screen_size_inch",
    "metadata.resolution_key",
    "metadata.resolution_width_px",
    "metadata.resolution_height_px",
    "metadata.panel_family",
    "metadata.screen_shape",
    "metadata.response_time_ms",
    "metadata.response_time_metric",
    "metadata.brightness_nits",
    "metadata.srgb_coverage_pct",
    "metadata.dci_p3_coverage_pct",
    "metadata.connection_tags",
    "metadata.feature_tags",
    "metadata.has_speakers",
    "metadata.has_vesa_mount",
    "metadata.has_touch",
    "metadata.width_mm",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported manhinhmaytinh metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: list[Any]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    """Build strict filters from only monitor fields that are indexed."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        _match_value(fields["category_scope"], "monitor")
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

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value).strip().casefold() for value in hard.get("brands", []) if value]
    if brands:
        must.append(_match_any(fields["brand"], brands))

    for profile_key, payload_key in {
        "panel_families": "panel_family",
        "screen_shapes": "screen_shape",
        "resolution_keys": "resolution_key",
        "response_time_metrics": "response_metric",
    }.items():
        values = [str(value) for value in hard.get(profile_key, []) if value]
        if values:
            must.append(_match_any(fields[payload_key], values))

    # Each requested capability is mandatory; MatchAny would incorrectly apply OR.
    for value in hard.get("required_connections", []):
        if value:
            must.append(_match_value(fields["connections"], str(value)))
    for value in hard.get("required_features", []):
        if value:
            must.append(_match_value(fields["features"], str(value)))

    for profile_key, payload_key in {
        "min_screen_size_inch": "screen_size",
        "min_resolution_width_px": "resolution_width",
        "min_resolution_height_px": "resolution_height",
        "min_brightness_nits": "brightness",
        "min_srgb_coverage_pct": "srgb_coverage",
        "min_dci_p3_coverage_pct": "dci_p3_coverage",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], gte=hard[profile_key]))

    for profile_key, payload_key in {
        "max_screen_size_inch": "screen_size",
        "max_response_time_ms": "response_time",
        "max_width_mm": "width",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], lte=hard[profile_key]))

    for profile_key, payload_key in {
        "requires_speakers": "speakers",
        "requires_vesa": "vesa",
        "requires_touch": "touch",
    }.items():
        if hard.get(profile_key) is True:
            must.append(_match_value(fields[payload_key], True))

    return models.Filter(must=must)
