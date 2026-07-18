"""Translate air-conditioner needs into indexed Qdrant filters."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from advisor.categories.air_conditioner import load_config


STANDARDIZED_METADATA_KEYS = {
    "Giá gốc vnd",
    "Giá khuyến mãi vnd",
    "Diện tích min m2",
    "Diện tích max m2",
    "Thể tích min m3",
    "Thể tích max m3",
    "Loại máy",
    "Loại Inverter",
    "Loại Gas",
}
PASSTHROUGH_METADATA_PATHS = {"metadata.brand"}

MACHINE_TYPE_VALUES = {
    "one_way": "Máy lạnh 1 chiều (chỉ làm lạnh)",
    "two_way": "Máy lạnh 2 chiều (có sưởi ấm)",
    "multi_indoor": "Dàn lạnh multi",
    "multi_outdoor": "Dàn nóng multi",
}
GAS_TYPE_VALUES = {"r32": "R-32", "r410a": "R-410A", "r22": "R-22"}


def _metadata_path(key: str) -> str:
    return f'metadata."{key}"'


def _validate_payload_fields(fields: dict[str, str]) -> None:
    supported = {
        _metadata_path(key) for key in STANDARDIZED_METADATA_KEYS
    } | PASSTHROUGH_METADATA_PATHS
    invalid = {path for path in fields.values() if path not in supported}
    if invalid:
        raise ValueError(
            "Unsupported maylanh metadata filter field(s): "
            + ", ".join(sorted(invalid))
        )


def _range_condition(key: str, **limits: float) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(**limits))


def build_filter(
    need_profile: dict[str, Any], payload_fields: dict[str, str] | None = None
) -> models.Filter | None:
    """Build strict filters; preferences remain in semantic search and ranking."""
    fields = payload_fields or load_config()["payload_fields"]
    _validate_payload_fields(fields)
    must: list[models.Condition] = []

    budget_max = need_profile.get("budget_max_vnd")
    if budget_max is not None:
        must.append(
            models.Filter(
                should=[
                    _range_condition(fields["promotional_price"], lte=budget_max),
                    _range_condition(fields["original_price"], lte=budget_max),
                ]
            )
        )

    room_area = need_profile.get("room_area_m2")
    if room_area is not None:
        must.extend(
            [
                _range_condition(fields["area_min"], lte=room_area),
                _range_condition(fields["area_max"], gte=room_area),
            ]
        )

    room_volume = need_profile.get("room_volume_m3")
    if room_volume is not None:
        must.extend(
            [
                _range_condition(fields["volume_min"], lte=room_volume),
                _range_condition(fields["volume_max"], gte=room_volume),
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

    machine_types = [
        MACHINE_TYPE_VALUES[value]
        for value in hard.get("machine_types", [])
        if value in MACHINE_TYPE_VALUES
    ]
    if machine_types:
        must.append(
            models.FieldCondition(
                key=fields["machine_type"],
                match=models.MatchAny(any=machine_types),
            )
        )

    inverter = hard.get("inverter")
    if inverter is not None:
        must.append(
            models.FieldCondition(
                key=fields["inverter"],
                match=models.MatchValue(
                    value=(
                        "Máy lạnh Inverter"
                        if inverter
                        else "Máy lạnh không Inverter"
                    )
                ),
            )
        )

    gas_types = [
        GAS_TYPE_VALUES[value]
        for value in hard.get("gas_types", [])
        if value in GAS_TYPE_VALUES
    ]
    if gas_types:
        must.append(
            models.FieldCondition(
                key=fields["gas_type"], match=models.MatchAny(any=gas_types)
            )
        )

    return models.Filter(must=must) if must else None

