"""Whitelist dishwasher Qdrant payload fields for downstream use."""

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


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Project a dishwasher point without guessing missing prices or features."""
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
        "model_code": metadata.get("model_code") or metadata.get("Model code"),
        "image_path": payload.get("image_path") or metadata.get("image_path"),
        "effective_price_vnd": effective_price,
        "original_price_vnd": original_price,
        "promotional_price_vnd": promotional_price,
        "description": payload.get("text"),
        "product_type": metadata.get("product_type"),
        "product_type_label": metadata.get("Loại sản phẩm"),
        "capacity": {
            "vietnamese_meals_min": _integer(
                metadata.get("vietnamese_meals_min")
            ),
            "vietnamese_meals_max": _integer(
                metadata.get("vietnamese_meals_max")
            ),
            "place_settings_min": _integer(metadata.get("place_settings_min")),
            "place_settings_max": _integer(metadata.get("place_settings_max")),
        },
        "water_l_per_cycle": {
            "min": _number(metadata.get("water_min_l_per_cycle")),
            "max": _number(metadata.get("water_max_l_per_cycle")),
        },
        "noise_db": _number(metadata.get("noise_db")),
        "power_w": {
            "min": _integer(metadata.get("power_min_w")),
            "max": _integer(metadata.get("power_max_w")),
        },
        "program_count": _integer(metadata.get("program_count")),
        "programs": metadata.get("Chương trình"),
        "washing_technology": metadata.get("Công nghệ"),
        "drying_technology": metadata.get("Công nghệ sấy"),
        "utilities": metadata.get("Tiện ích"),
        "racks": metadata.get("Khay chén"),
        "control_panel": metadata.get("Bảng điều khiển"),
        "dimensions_cm": {
            "width": _number(metadata.get("width_cm")),
            "height": _number(metadata.get("height_cm")),
            "depth": _number(metadata.get("depth_cm")),
        },
        "body_material": metadata.get("Chất liệu thân vỏ"),
        "door_material": metadata.get("Chất liệu cửa"),
        "origin": metadata.get("Sản xuất tại"),
    }
