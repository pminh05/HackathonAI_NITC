"""Whitelist tablet Qdrant payload fields for downstream use."""

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
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Máy tính bảng chưa có tên"),
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
        "os_family": metadata.get("os_family"),
        "operating_system": metadata.get("Hệ điều hành"),
        "ram_gb": _integer(metadata.get("ram_gb")),
        "storage_gb": _integer(metadata.get("storage_gb")),
        "available_storage_gb": _number(metadata.get("available_storage_gb")),
        "screen_size_inch": _number(metadata.get("screen_size_inch")),
        "display_family": metadata.get("display_family"),
        "display_technology": metadata.get("Màn hình hiển thị"),
        "weight_g": _integer(metadata.get("weight_g")),
        "connectivity_class": metadata.get("connectivity_class"),
        "max_mobile_generation": _integer(metadata.get("max_mobile_generation")),
        "supports_calls": metadata.get("supports_calls")
        if isinstance(metadata.get("supports_calls"), bool)
        else None,
        "supports_memory_card": metadata.get("supports_memory_card")
        if isinstance(metadata.get("supports_memory_card"), bool)
        else None,
        "battery_mah": _integer(metadata.get("battery_mah")),
        "battery_wh": _number(metadata.get("battery_wh")),
        "max_charging_w": _number(metadata.get("max_charging_w")),
        "cpu": metadata.get("Chip xử lý (CPU)"),
        "gpu": metadata.get("Chip đồ họa (GPU)"),
        "feature_tags": metadata.get("feature_tags") or [],
        "special_features": metadata.get("Tính năng đặc biệt"),
    }
