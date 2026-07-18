"""Whitelist Qdrant washing-machine payload fields for downstream use."""

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
        "product_type": metadata.get("product_type"),
        "product_type_label": metadata.get("Loại sản phẩm"),
        "drum_type": metadata.get("drum_type"),
        "drum_type_label": metadata.get("Lồng giặt"),
        "wash_capacity_kg": _number(metadata.get("wash_capacity_kg")),
        "dry_capacity_kg": _number(metadata.get("dry_capacity_kg")),
        "suitable_for": metadata.get("Số người sử dụng"),
        "people_range": {
            "min": _integer(metadata.get("people_min")),
            "max": _integer(metadata.get("people_max")),
        },
        "has_inverter": metadata.get("has_inverter")
        if isinstance(metadata.get("has_inverter"), bool)
        else None,
        "inverter_type": metadata.get("Loại Inverter"),
        "has_dryer": metadata.get("has_dryer")
        if isinstance(metadata.get("has_dryer"), bool)
        else None,
        "spin_speed_rpm": _integer(metadata.get("spin_speed_rpm")),
        "energy_kwh_per_kg": _number(metadata.get("energy_kwh_per_kg")),
        "program_count": _integer(metadata.get("program_count")),
        "programs": metadata.get("Chương trình"),
        "washing_technology": metadata.get("Công nghệ"),
        "drying_technology": metadata.get("Công nghệ sấy"),
        "motor": metadata.get("Động cơ"),
        "utilities": metadata.get("Tiện ích"),
        "control_panel": metadata.get("Bảng điều khiển"),
        "dimensions_cm": {
            "width": _number(metadata.get("width_cm")),
            "height": _number(metadata.get("height_cm")),
            "depth": _number(metadata.get("depth_cm")),
        },
        "tub_material": metadata.get("Chất liệu ruột"),
        "body_material": metadata.get("Chất liệu thân vỏ"),
        "motor_warranty": metadata.get("Bảo hành động cơ"),
        "origin": metadata.get("Sản xuất tại"),
    }
