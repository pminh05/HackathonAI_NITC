"""Translate recording-microphone needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.phone_recording_microphone import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.brand_key",
    "metadata.product_type",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.compatibility_tags",
    "metadata.connector_tags",
    "metadata.feature_tags",
    "metadata.pickup_pattern",
    "metadata.wireless_band",
    "metadata.transmitter_count",
    "metadata.runtime_min_hours",
    "metadata.transmission_range_m",
}

SETUP_REQUIREMENTS = {
    "iphone_lightning": ({"ios"}, {"lightning"}),
    "iphone_usb_c": ({"ios", "ipados"}, {"usb_c"}),
    "android_usb_c": ({"android"}, {"usb_c"}),
    "camera_3_5mm": ({"camera"}, {"3_5mm"}),
    "computer_usb": ({"windows", "macos"}, {"usb_c", "micro_usb"}),
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported microthuamdienthoai metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def _match_any(key: str, values: list[str] | set[str]) -> models.FieldCondition:
    return models.FieldCondition(
        key=key, match=models.MatchAny(any=sorted(set(values)))
    )


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter | None:
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

    setup = need_profile.get("recording_setup")
    if setup in SETUP_REQUIREMENTS:
        compatibility, connectors = SETUP_REQUIREMENTS[setup]
        must.append(_match_any(fields["compatibility"], compatibility))
        must.append(_match_any(fields["connector"], connectors))

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value).strip().casefold() for value in hard.get("brands", []) if value]
    if brands:
        must.append(_match_any(fields["brand"], brands))
    product_types = [str(value) for value in hard.get("product_types", []) if value]
    if product_types:
        must.append(_match_any(fields["product_type"], product_types))

    # Each required platform or feature must be present in an array payload.
    for tag in hard.get("required_compatibility_tags", []):
        must.append(
            models.FieldCondition(
                key=fields["compatibility"], match=models.MatchValue(value=tag)
            )
        )
    connector_types = [str(value) for value in hard.get("connector_types", []) if value]
    if connector_types:
        must.append(_match_any(fields["connector"], connector_types))
    for feature in hard.get("required_features", []):
        must.append(
            models.FieldCondition(
                key=fields["features"], match=models.MatchValue(value=feature)
            )
        )

    for profile_key, payload_key in {
        "min_transmitter_count": "transmitter_count",
        "min_runtime_hours": "runtime_min",
        "min_transmission_range_m": "transmission_range",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], gte=hard[profile_key]))

    pickup_patterns = [str(value) for value in hard.get("pickup_patterns", []) if value]
    if pickup_patterns:
        must.append(_match_any(fields["pickup_pattern"], pickup_patterns))
    wireless_bands = [str(value) for value in hard.get("wireless_bands", []) if value]
    if wireless_bands:
        must.append(_match_any(fields["wireless_band"], wireless_bands))

    return models.Filter(must=must) if must else None
