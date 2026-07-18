"""Whitelist smartwatch Qdrant payload fields for downstream use."""

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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Project one point without guessing missing prices or capabilities."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Đồng hồ thông minh chưa có tên"),
        "qdrant_score": float(point.score),
        "brand": metadata.get("brand"),
        "model_code": metadata.get("model_code"),
        "image_path": payload.get("image_path"),
        "effective_price_vnd": (
            promotional_price if promotional_price is not None else original_price
        ),
        "original_price_vnd": original_price,
        "promotional_price_vnd": promotional_price,
        "description": payload.get("text"),
        "compatible_platforms": _string_list(metadata.get("compatible_platforms")),
        "compatibility_description": metadata.get("Tương thích"),
        "call_mode": metadata.get("call_mode"),
        "calling_description": metadata.get("Thực hiện cuộc gọi"),
        "has_cellular": _boolean(metadata.get("has_cellular")),
        "sim": metadata.get("SIM"),
        "has_gps": _boolean(metadata.get("has_gps")),
        "positioning": metadata.get("Định vị"),
        "has_notifications": _boolean(metadata.get("has_notifications")),
        "notifications": metadata.get("Hiển thị thông báo"),
        "swim_ready": _boolean(metadata.get("swim_ready")),
        "has_sos": _boolean(metadata.get("has_sos")),
        "health_feature_tags": _string_list(metadata.get("health_feature_tags")),
        "health_features": metadata.get("Theo dõi sức khoẻ"),
        "sport_modes": metadata.get("Môn thể thao")
        or metadata.get("Chế độ luyện tập"),
        "display_family": metadata.get("display_family"),
        "display_technology": metadata.get("Màn hình hiển thị"),
        "screen_size_inch": _number(metadata.get("screen_size_inch")),
        "resolution": metadata.get("Độ phân giải"),
        "case_dimensions_mm": {
            "length": _number(metadata.get("case_length_mm")),
            "width": _number(metadata.get("case_width_mm")),
            "thickness": _number(metadata.get("case_thickness_mm")),
        },
        "weight_g": _number(metadata.get("weight_g")),
        "wrist_range_cm": {
            "min": _number(metadata.get("wrist_min_cm")),
            "max": _number(metadata.get("wrist_max_cm")),
        },
        "strap_material_family": metadata.get("strap_material_family"),
        "strap_material": metadata.get("Chất liệu dây"),
        "typical_battery_hours": _number(metadata.get("typical_battery_hours")),
        "battery_mah": _integer(metadata.get("battery_mah")),
        "battery_life_description": metadata.get("Thời gian sử dụng"),
        "charging_time_hours": _number(metadata.get("charging_time_hours")),
        "water_resistance_atm": _number(metadata.get("water_resistance_atm")),
        "water_resistance_description": metadata.get("Chuẩn chống nước, bụi"),
        "operating_system": metadata.get("Hệ điều hành"),
        "utilities": metadata.get("Tiện ích khác") or metadata.get("Tiện ích"),
    }
