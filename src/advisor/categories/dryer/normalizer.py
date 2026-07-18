"""Whitelist Qdrant clothes-dryer payload fields for downstream use."""

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
    """Project one raw point without guessing missing prices or capabilities."""
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
        "image_path": payload.get("image_path") or metadata.get("image_path"),
        "effective_price_vnd": effective_price,
        "original_price_vnd": original_price,
        "promotional_price_vnd": promotional_price,
        "description": payload.get("text"),
        "dryer_type": metadata.get("dryer_type"),
        "dryer_type_label": metadata.get("Loại sản phẩm"),
        "dry_capacity_kg": _number(metadata.get("dry_capacity_kg")),
        "suitable_for": metadata.get("Số người sử dụng"),
        "people_range": {
            "min": _integer(metadata.get("people_min")),
            "max": _integer(metadata.get("people_max")),
        },
        "has_inverter": _boolean(metadata.get("has_inverter")),
        "has_sensor": _boolean(metadata.get("has_sensor")),
        "power_range_w": {
            "min": _integer(metadata.get("power_min_w")),
            "max": _integer(metadata.get("power_max_w")),
        },
        "temperature_range_c": {
            "min": _number(metadata.get("temperature_min_c")),
            "max": _number(metadata.get("temperature_max_c")),
        },
        "dimensions_cm": {
            "width": _number(metadata.get("width_cm")),
            "height": _number(metadata.get("height_cm")),
            "depth": _number(metadata.get("depth_cm")),
        },
        "drying_technology": metadata.get("Công nghệ")
        or metadata.get("Công nghệ sấy"),
        "energy_saving_technology": metadata.get("Công nghệ tiết kiệm điện"),
        "motor": metadata.get("Động cơ"),
        "utilities": metadata.get("Tiện ích"),
        "control_panel": metadata.get("Bảng điều khiển"),
        "drum_material": metadata.get("Chất liệu ruột"),
        "body_material": metadata.get("Chất liệu thân vỏ"),
        "origin": metadata.get("Sản xuất tại"),
        "product_year": _integer(metadata.get("nam_dong_san_pham")),
        "drain_hose_length_cm": _number(metadata.get("dai_ong_xa_nuoc_cm")),
        "vent_hose_length_cm": _number(metadata.get("dai_ong_thoat_khi_cm")),
    }
