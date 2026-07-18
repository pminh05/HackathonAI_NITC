"""Whitelist printer Qdrant payload fields for downstream use."""

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
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    duplex = metadata.get("supports_duplex")
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Máy in chưa có tên"),
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
        "print_technology": metadata.get("print_technology"),
        "color_mode": metadata.get("color_mode"),
        "product_type_label": metadata.get("Loại sản phẩm"),
        "print_speed_ppm": _number(metadata.get("print_speed_ppm")),
        "thermal_speed_mm_s": _number(metadata.get("thermal_speed_mm_s")),
        "monthly_volume_pages": {
            "min": _integer(metadata.get("monthly_volume_min_pages")),
            "max": _integer(metadata.get("monthly_volume_max_pages")),
        },
        "resolution_dpi": {
            "horizontal": _integer(metadata.get("resolution_horizontal_dpi")),
            "vertical": _integer(metadata.get("resolution_vertical_dpi")),
        },
        "connection_tags": metadata.get("connection_tags") or [],
        "paper_size_tags": metadata.get("paper_size_tags") or [],
        "os_tags": metadata.get("os_tags") or [],
        "supports_duplex": duplex if isinstance(duplex, bool) else None,
        "paper_input_sheets": _integer(metadata.get("paper_input_sheets")),
        "toner_yield_pages": {
            "min": _integer(metadata.get("toner_yield_min_pages")),
            "max": _integer(metadata.get("toner_yield_max_pages")),
        },
        "dimensions_mm": {
            "width": _integer(metadata.get("width_mm")),
            "height": _integer(metadata.get("height_mm")),
            "depth": _integer(metadata.get("depth_mm")),
        },
        "ink_type": metadata.get("Loại mực in"),
        "compatible_systems": metadata.get("Tương thích"),
    }
