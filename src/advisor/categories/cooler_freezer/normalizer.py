"""Whitelist Qdrant cooler/freezer payload fields for downstream use."""

from __future__ import annotations

from typing import Any


PRODUCT_TYPE_LABELS = {
    "cooler": "Tủ mát",
    "cooler_mini": "Tủ mát mini",
    "freezer": "Tủ đông",
    "freezer_mini": "Tủ đông mini",
}


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Project one raw point without guessing missing prices or capabilities."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    effective_price = _integer(metadata.get("effective_price_vnd"))
    if effective_price is None:
        effective_price = (
            promotional_price if promotional_price is not None else original_price
        )
    product_type = metadata.get("product_type")
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Sản phẩm chưa có tên"),
        "qdrant_score": float(point.score),
        "brand": metadata.get("brand"),
        "model_code": metadata.get("model_code"),
        "image_path": payload.get("image_path") or metadata.get("image_path"),
        "effective_price_vnd": effective_price,
        "original_price_vnd": original_price,
        "promotional_price_vnd": promotional_price,
        "description": payload.get("text"),
        "product_family": metadata.get("product_family"),
        "product_type": product_type,
        "product_type_label": PRODUCT_TYPE_LABELS.get(product_type),
        "is_mini": _boolean(metadata.get("is_mini")),
        "total_capacity_lit": _integer(metadata.get("total_capacity_lit")),
        "compartments": {
            "total": _integer(metadata.get("compartment_count")),
            "freezer": _integer(metadata.get("freezer_compartment_count")),
            "cooler": _integer(metadata.get("cooler_compartment_count")),
        },
        "temperature_range_c": {
            "min": _number(metadata.get("temperature_min_c")),
            "max": _number(metadata.get("temperature_max_c")),
        },
        "has_inverter": _boolean(metadata.get("has_inverter")),
        "energy_kwh_day": _number(metadata.get("energy_kwh_day")),
        "energy_kwh_year": _number(metadata.get("energy_kwh_year")),
        "noise_range_db": {
            "min": _number(metadata.get("noise_min_db")),
            "max": _number(metadata.get("noise_max_db")),
        },
        "door_count": _integer(metadata.get("door_count")),
        "dimensions_cm": {
            "width": _number(metadata.get("width_cm")),
            "height": _number(metadata.get("height_cm")),
            "depth": _number(metadata.get("depth_cm")),
        },
        "weight_kg": _number(metadata.get("weight_kg")),
        "gas_type": metadata.get("gas_type"),
        "feature_tags": _string_list(metadata.get("feature_tags")),
        "technology": metadata.get("Công nghệ"),
        "energy_saving_technology": metadata.get("Công nghệ tiết kiệm điện"),
        "utilities": metadata.get("Tiện ích"),
        "face_material": metadata.get("Chất liệu mặt"),
        "inner_material": metadata.get("Chất liệu ruột"),
        "origin": metadata.get("Sản xuất tại"),
        "product_year": _integer(metadata.get("nam_ra_mat")),
    }
