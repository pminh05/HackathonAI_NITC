"""Translate desktop needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.desktop import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand_key",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.desktop_form",
    "metadata.cpu_vendor",
    "metadata.os_family_tags",
    "metadata.ram_gb",
    "metadata.ram_max_gb",
    "metadata.storage_total_gb",
    "metadata.storage_type_tags",
    "metadata.gpu_type",
    "metadata.has_wifi",
    "metadata.screen_size_inch",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported maytinhdeban metadata filter field(s): "
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
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        _match_value(fields["category_scope"], "desktop")
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

    form = need_profile.get("form_preference")
    if form in {"all_in_one", "separate_unit"}:
        must.append(_match_value(fields["desktop_form"], form))

    hard = need_profile.get("hard_constraints") or {}
    keyword_fields = {
        "cpu_vendors": "cpu_vendor",
        "os_families": "os_families",
        "storage_types": "storage_types",
        "gpu_types": "gpu_type",
    }
    for profile_key, payload_key in keyword_fields.items():
        values = [str(value) for value in hard.get(profile_key, []) if value]
        if values:
            must.append(_match_any(fields[payload_key], values))

    brands = [str(value).strip().casefold() for value in hard.get("brands", []) if value]
    if brands:
        must.append(_match_any(fields["brand"], brands))

    for profile_key, payload_key in {
        "min_ram_gb": "ram",
        "min_supported_ram_gb": "ram_max",
        "min_storage_gb": "storage",
        "min_screen_size_inch": "screen_size",
    }.items():
        if hard.get(profile_key) is not None:
            must.append(_range_condition(fields[payload_key], gte=hard[profile_key]))
    if hard.get("max_screen_size_inch") is not None:
        must.append(
            _range_condition(
                fields["screen_size"], lte=hard["max_screen_size_inch"]
            )
        )
    if hard.get("requires_wifi") is True:
        must.append(_match_value(fields["wifi"], True))
    return models.Filter(must=must)
