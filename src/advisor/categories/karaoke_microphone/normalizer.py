"""Whitelist micro-karaoke Qdrant payload fields for downstream use."""

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


def _string_list(value: Any) -> list[str]:
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
        "name": str(payload.get("name") or "Micro karaoke chưa có tên"),
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
        "microphone_type": metadata.get("microphone_type"),
        "wireless_band": metadata.get("wireless_band"),
        "rf_frequency_mhz": {
            "min": _number(metadata.get("rf_frequency_min_mhz")),
            "max": _number(metadata.get("rf_frequency_max_mhz")),
        },
        "audio_frequency_hz": {
            "min": _number(metadata.get("audio_frequency_min_hz")),
            "max": _number(metadata.get("audio_frequency_max_hz")),
        },
        "distortion_pct": _number(metadata.get("distortion_pct")),
        "distortion_operator": metadata.get("distortion_operator"),
        "manufacture_year": _integer(metadata.get("manufacture_year")),
        "origin": metadata.get("origin"),
        "data_quality_flags": _string_list(metadata.get("data_quality_flags")),
    }
