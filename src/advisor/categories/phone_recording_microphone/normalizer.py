"""Whitelist recording-microphone Qdrant payload fields for downstream use."""

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


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def normalize_candidate(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Micro thu âm chưa có tên"),
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
        "product_type": metadata.get("product_type"),
        "compatibility_tags": _strings(metadata.get("compatibility_tags")),
        "connector_tags": _strings(metadata.get("connector_tags")),
        "feature_tags": _strings(metadata.get("feature_tags")),
        "pickup_pattern": metadata.get("pickup_pattern"),
        "wireless_band": metadata.get("wireless_band"),
        "transmitter_count": _integer(metadata.get("transmitter_count")),
        "receiver_count": _integer(metadata.get("receiver_count")),
        "runtime_hours": {
            "min": _number(metadata.get("runtime_min_hours")),
            "max": _number(metadata.get("runtime_max_hours")),
        },
        "transmission_range_m": _number(metadata.get("transmission_range_m")),
        "audio_frequency_hz": {
            "min": _integer(metadata.get("audio_frequency_min_hz")),
            "max": _integer(metadata.get("audio_frequency_max_hz")),
        },
        "max_spl_db": _integer(metadata.get("max_spl_db")),
        "total_weight_g": _number(metadata.get("total_weight_g")),
        "manufacture_year": _integer(metadata.get("manufacture_year")),
        "origin": metadata.get("origin"),
        "data_quality_flags": _strings(metadata.get("data_quality_flags")),
    }
