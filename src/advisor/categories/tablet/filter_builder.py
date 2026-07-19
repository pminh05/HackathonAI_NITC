"""Translate tablet needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.tablet import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.os_family",
    "metadata.ram_gb",
    "metadata.storage_gb",
    "metadata.screen_size_inch",
    "metadata.weight_g",
    "metadata.connectivity_class",
    "metadata.max_mobile_generation",
    "metadata.supports_calls",
    "metadata.supports_memory_card",
    "metadata.display_family",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported maytinhbang metadata filter field(s): "
            + ", ".join(sorted(invalid))
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
        _match_value(fields["category_scope"], "tablet")
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

    connectivity = need_profile.get("connectivity_segment")
    if connectivity == "wifi_only":
        must.append(_match_value(fields["connectivity_class"], "wifi_only"))
    elif connectivity in {"cellular_4g", "cellular_5g"}:
        must.append(_match_value(fields["connectivity_class"], "cellular"))
        must.append(
            _range_condition(
                fields["mobile_generation"],
                gte=5 if connectivity == "cellular_5g" else 4,
            )
        )

    hard = need_profile.get("hard_constraints") or {}
    for profile_key, payload_key in {
        "brands": "brand",
        "os_families": "os_family",
        "display_families": "display_family",
    }.items():
        values = [str(value) for value in hard.get(profile_key, []) if value]
        if values:
            must.append(
                models.FieldCondition(
                    key=fields[payload_key], match=models.MatchAny(any=values)
                )
            )

    for profile_key, payload_key in {
        "min_ram_gb": "ram",
        "min_storage_gb": "storage",
        "min_screen_size_inch": "screen_size",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], gte=hard[profile_key]))
    for profile_key, payload_key in {
        "max_screen_size_inch": "screen_size",
        "max_weight_g": "weight",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], lte=hard[profile_key]))
    for profile_key, payload_key in {
        "requires_calls": "supports_calls",
        "requires_memory_card": "supports_memory_card",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_match_value(fields[payload_key], hard[profile_key]))
    return models.Filter(must=must)
