"""Translate micro-karaoke needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.karaoke_microphone import load_config


STANDARDIZED_METADATA_PATHS = {
    "metadata.category_scope",
    "metadata.brand_key",
    "metadata.original_price_vnd",
    "metadata.promotional_price_vnd",
    "metadata.microphone_type",
    "metadata.wireless_band",
}


def _validate_payload_fields(fields: dict[str, str]) -> None:
    invalid = {path for path in fields.values() if path not in STANDARDIZED_METADATA_PATHS}
    if invalid:
        raise ValueError(
            "Unsupported microkaraoke metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _match_value(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter:
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = [
        _match_value(fields["category_scope"], "karaoke_microphone")
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
                    models.Filter(
                        must=[
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(
                                    key=fields["promotional_price"]
                                )
                            ),
                            models.IsEmptyCondition(
                                is_empty=models.PayloadField(
                                    key=fields["original_price"]
                                )
                            ),
                        ]
                    ),
                ]
            )
        )

    hard = need_profile.get("hard_constraints") or {}
    brands = [str(value).strip().casefold() for value in hard.get("brands", []) if value]
    if brands:
        must.append(
            models.FieldCondition(
                key=fields["brand_key"], match=models.MatchAny(any=brands)
            )
        )

    microphone_types = [
        str(value) for value in hard.get("microphone_types", []) if value
    ]
    if not microphone_types:
        preference = need_profile.get("connection_preference")
        if preference in {"wired", "wireless"}:
            microphone_types = [preference]
    if microphone_types:
        must.append(
            models.FieldCondition(
                key=fields["microphone_type"],
                match=models.MatchAny(any=microphone_types),
            )
        )

    bands = [str(value) for value in hard.get("wireless_bands", []) if value]
    if bands:
        must.append(
            models.FieldCondition(
                key=fields["wireless_band"], match=models.MatchAny(any=bands)
            )
        )
    return models.Filter(must=must)
