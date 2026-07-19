"""Whitelist Qdrant monitor payload fields for downstream use."""

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


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def normalize_candidate(point: Any) -> dict[str, Any]:
    """Project one raw point without guessing absent monitor capabilities."""
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
        "screen_size_inch": _number(metadata.get("screen_size_inch")),
        "resolution": metadata.get("Độ phân giải"),
        "resolution_key": metadata.get("resolution_key"),
        "resolution_px": {
            "width": _integer(metadata.get("resolution_width_px")),
            "height": _integer(metadata.get("resolution_height_px")),
        },
        "panel_family": metadata.get("panel_family"),
        "panel_variant": metadata.get("panel_variant"),
        "screen_shape": metadata.get("screen_shape"),
        "response_time_ms": _number(metadata.get("response_time_ms")),
        "response_time_metric": metadata.get("response_time_metric"),
        "brightness_nits": _number(metadata.get("brightness_nits")),
        "srgb_coverage_pct": _number(metadata.get("srgb_coverage_pct")),
        "dci_p3_coverage_pct": _number(metadata.get("dci_p3_coverage_pct")),
        "connection_tags": _strings(metadata.get("connection_tags")),
        "connections": metadata.get("Kết nối"),
        "feature_tags": _strings(metadata.get("feature_tags")),
        "display_features": metadata.get("Màn hình hiển thị"),
        "utilities": metadata.get("Tiện ích"),
        "has_speakers": _boolean(metadata.get("has_speakers")),
        "has_vesa_mount": _boolean(metadata.get("has_vesa_mount")),
        "has_touch": _boolean(metadata.get("has_touch")),
        "dimensions_mm": {
            "width": _number(metadata.get("width_mm")),
            "height_min": _number(metadata.get("height_min_mm")),
            "height_max": _number(metadata.get("height_max_mm")),
            "depth": _number(metadata.get("depth_mm")),
        },
        "weight_kg": _number(metadata.get("weight_kg")),
        "power_consumption_w": _number(metadata.get("power_consumption_w")),
        "data_quality_flags": _strings(metadata.get("data_quality_flags")),
    }
