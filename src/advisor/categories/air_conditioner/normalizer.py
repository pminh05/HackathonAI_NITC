"""Whitelist Qdrant air-conditioner payload fields for downstream use."""

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
    """Project one raw point without guessing missing price, BTU, or features."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("Giá khuyến mãi vnd"))
    original_price = _integer(metadata.get("Giá gốc vnd"))
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
        "machine_type": metadata.get("Loại máy"),
        "inverter_type": metadata.get("Loại Inverter"),
        "gas_type": metadata.get("Loại Gas"),
        "room_area_range_m2": {
            "min": _number(metadata.get("Diện tích min m2")),
            "max": _number(metadata.get("Diện tích max m2")),
        },
        "room_volume_range_m3": {
            "min": _number(metadata.get("Thể tích min m3")),
            "max": _number(metadata.get("Thể tích max m3")),
        },
        "cooling_capacity_btu_h": _integer(
            metadata.get("Công suất lạnh BTU/h")
        ),
        "energy_stars": _integer(metadata.get("Số sao năng lượng")),
        "energy_efficiency": _number(metadata.get("Hiệu suất năng lượng")),
        "noise_db": {
            "min": _number(metadata.get("Độ ồn min dB")),
            "max": _number(metadata.get("Độ ồn max dB")),
            "indoor": _number(metadata.get("Độ ồn dàn lạnh dB")),
            "outdoor": _number(metadata.get("Độ ồn dàn nóng dB")),
        },
        "cooling_technology": metadata.get("Công nghệ làm lạnh"),
        "energy_saving_technology": metadata.get("Công nghệ tiết kiệm điện"),
        "airflow_mode": metadata.get("Chế độ gió"),
        "utilities": metadata.get("Tiện ích"),
        "radiator_material": metadata.get("Chất liệu dàn tản nhiệt"),
        "component_warranty_years": _integer(
            metadata.get("Bảo hành bộ phận năm")
        ),
        "compressor_warranty_years": _number(
            metadata.get("Bảo hành máy nén năm")
        ),
        "maximum_pipe_length_m": _number(metadata.get("Ống đồng max m")),
        "maximum_installation_height_m": _number(metadata.get("Cao lắp đặt m")),
    }

