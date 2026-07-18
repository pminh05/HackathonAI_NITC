"""Whitelist water-heater Qdrant payload fields for downstream use."""

from __future__ import annotations

from typing import Any


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


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Project a water-heater point without guessing missing facts."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    effective_price = (
        promotional_price if promotional_price is not None else original_price
    )
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
        "product_type": metadata.get("product_type"),
        "product_type_label": metadata.get("Loại máy"),
        "capacity_l": _integer(metadata.get("capacity_l")),
        "power_w": _integer(metadata.get("power_w")),
        "heating_time_minutes": {
            "min": _number(metadata.get("heating_time_min_minutes")),
            "max": _number(metadata.get("heating_time_max_minutes")),
        },
        "max_temperature_c": _integer(metadata.get("max_temperature_c")),
        "has_booster_pump": _boolean(metadata.get("has_booster_pump")),
        "includes_shower": _boolean(metadata.get("includes_shower")),
        "water_pressure_mpa": {
            "min": _number(metadata.get("water_pressure_min_mpa")),
            "max": _number(metadata.get("water_pressure_max_mpa")),
        },
        "ip_rating": metadata.get("ip_rating"),
        "safety_tags": list(metadata.get("safety_tags") or []),
        "safety_description": metadata.get("Tính năng an toàn"),
        "feature_tags": list(metadata.get("feature_tags") or []),
        "temperature_control": metadata.get("Tùy chỉnh nhiệt độ"),
        "utilities": metadata.get("Tiện ích"),
        "shower_description": metadata.get("Vòi sen"),
        "suitable_people": {
            "min": _integer(metadata.get("people_min")),
            "max": _integer(metadata.get("people_max")),
        },
        "dimensions_cm": {
            "width": _number(metadata.get("width_cm")),
            "height": _number(metadata.get("height_cm")),
            "depth": _number(metadata.get("depth_cm")),
        },
        "weight_kg": _number(metadata.get("weight_kg")),
        "origin": metadata.get("Sản xuất tại"),
    }
