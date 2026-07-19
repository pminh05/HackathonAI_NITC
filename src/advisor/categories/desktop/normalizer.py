"""Whitelist desktop Qdrant payload fields for downstream use."""

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
    """Project one desktop point without guessing missing capabilities."""
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    promotional_price = _integer(metadata.get("promotional_price_vnd"))
    original_price = _integer(metadata.get("original_price_vnd"))
    return {
        "product_id": str(payload.get("product_id") or point.id),
        "name": str(payload.get("name") or "Máy tính để bàn chưa có tên"),
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
        "desktop_form": metadata.get("desktop_form"),
        "cpu_vendor": metadata.get("cpu_vendor"),
        "cpu_technology": metadata.get("Công nghệ CPU"),
        "cpu_model": metadata.get("Loại CPU"),
        "cpu_base_ghz": _number(metadata.get("toc_do_cpu_co_ban_ghz")),
        "cpu_max_ghz": _number(metadata.get("toc_do_cpu_toi_da_ghz")),
        "cpu_core_count": _integer(metadata.get("so_nhan_cpu")),
        "cpu_thread_count": _integer(metadata.get("so_luong_cpu_luong")),
        "ram_gb": _integer(metadata.get("ram_gb")),
        "ram_max_gb": _integer(metadata.get("ram_max_gb")),
        "ram_type": metadata.get("Loại RAM"),
        "ram_slots": _integer(metadata.get("ram_slots")),
        "storage_total_gb": _integer(metadata.get("storage_total_gb")),
        "storage_type_tags": metadata.get("storage_type_tags") or [],
        "storage_description": metadata.get("Ổ cứng"),
        "storage_expansion": metadata.get("Khe cắm mở rộng")
        or metadata.get("Chuẩn kết nối ổ cứng"),
        "gpu_type": metadata.get("gpu_type"),
        "gpu_model": metadata.get("Chip đồ họa (GPU)"),
        "gpu_vram_gb": _integer(metadata.get("vram_gpu_gb")),
        "os_family_tags": metadata.get("os_family_tags") or [],
        "operating_system": metadata.get("Hệ điều hành"),
        "has_wifi": _boolean(metadata.get("has_wifi")),
        "has_bluetooth": _boolean(metadata.get("has_bluetooth")),
        "wireless_connectivity": metadata.get("Wifi"),
        "ports": metadata.get("Cổng kết nối")
        or metadata.get("Cổng giao tiếp")
        or metadata.get("Cổng I/O mặt sau"),
        "screen_size_inch": _number(metadata.get("screen_size_inch")),
        "display": metadata.get("Màn hình hiển thị"),
        "resolution": metadata.get("Độ phân giải"),
        "touchscreen": _boolean(metadata.get("co_man_hinh_cam_ung")),
        "dimensions_mm": {
            "length": _number(metadata.get("dai_mm")),
            "width": _number(metadata.get("rong_mm")),
            "thickness": _number(metadata.get("day_mm")),
        },
        "weight_kg": _number(metadata.get("weight_kg")),
        "power_w": _number(metadata.get("nguon_dien_max_w")),
        "release_year": _integer(metadata.get("release_year")),
        "data_quality_flags": metadata.get("data_quality_flags") or [],
    }
