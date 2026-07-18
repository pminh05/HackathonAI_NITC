"""Normalize tablet metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_tinh_bang"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def parse_vnd(value: Any) -> int | None:
    """Parse displayed VND values instead of faulty distributor columns."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        return int(parsed) if parsed > 0 else None
    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip().replace(" ", "")
    if not text:
        return None
    try:
        if re.fullmatch(r"\d+(?:\.0+)?", text.replace(",", "")):
            parsed = Decimal(text.replace(",", ""))
        else:
            parsed = Decimal(re.sub(r"[.,]", "", text))
    except InvalidOperation:
        return None
    return int(parsed) if parsed > 0 else None


def normalize_os_family(value: Any) -> str | None:
    text = str(value or "").casefold()
    if "android" in text or "hyperos" in text:
        return "android"
    if "ipados" in text or re.search(r"\bios\b", text):
        return "ipados"
    if "harmonyos" in text:
        return "harmonyos"
    return None


def normalize_display_family(value: Any) -> str | None:
    text = str(value or "").casefold()
    if not text:
        return None
    if "ultra retina xdr" in text or "amoled" in text or "oled" in text:
        return "oled_amoled"
    if "mini-led" in text or "mini led" in text:
        return "mini_led"
    if "ips" in text or "liquid retina" in text or "retina" in text:
        return "ips_lcd"
    if "tft" in text:
        return "tft_lcd"
    if "lcd" in text or "led" in text or "ltps" in text:
        return "lcd"
    return "other"


def _boolean_from_text(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text.startswith("không"):
        return False
    if text.startswith("có"):
        return True
    return None


def normalize_mobile_generation(metadata: dict[str, Any]) -> int | None:
    generation = _integer(metadata.get("the_he_mang_cao_nhat"))
    if generation in {3, 4, 5}:
        return generation
    match = re.search(r"\b([345])g\b", str(metadata.get("Mạng di động") or "").casefold())
    return int(match.group(1)) if match else None


def normalize_connectivity_class(
    metadata: dict[str, Any], generation: int | None
) -> str | None:
    if generation is not None:
        return "cellular"
    sim = str(metadata.get("SIM") or "").strip().casefold()
    mobile = str(metadata.get("Mạng di động") or "").strip().casefold()
    if sim and not sim.startswith("không"):
        return "cellular"
    if (sim and sim.startswith("không")) or (mobile and mobile.startswith("không")):
        return "wifi_only"
    return None


def normalize_memory_card(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text.startswith("không"):
        return False
    return True


def feature_tags(metadata: dict[str, Any]) -> list[str]:
    text = " | ".join(
        str(metadata.get(field) or "")
        for field in ("Tính năng đặc biệt", "Phụ kiện đi kèm", "Chuẩn chống nước, bụi")
    ).casefold()
    tags: list[str] = []
    checks = {
        "stylus": ("pencil", "s pen", "bút cảm ứng", "bút thông minh"),
        "keyboard": ("bàn phím", "keyboard"),
        "face_unlock": ("khuôn mặt", "face id"),
        "fingerprint": ("vân tay", "touch id"),
    }
    for tag, needles in checks.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    protection = str(metadata.get("Chuẩn chống nước, bụi") or "").strip().casefold()
    if protection and not protection.startswith("không"):
        tags.append("water_dust_resistance")
    return tags


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add stable typed fields while retaining useful source descriptions."""
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if _clean(value) is not None
    }
    # These generated fields are consistently 10x larger than displayed prices.
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    generation = normalize_mobile_generation(metadata)
    _set_if_present(normalized, "original_price_vnd", parse_vnd(metadata.get("giá gốc")))
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi")),
    )
    _set_if_present(normalized, "os_family", normalize_os_family(metadata.get("Hệ điều hành")))
    _set_if_present(
        normalized,
        "display_family",
        normalize_display_family(metadata.get("Màn hình hiển thị")),
    )
    _set_if_present(normalized, "ram_gb", _integer(metadata.get("ram_gb")))
    _set_if_present(
        normalized, "storage_gb", _integer(metadata.get("bo_nho_luu_tru_gb"))
    )
    _set_if_present(
        normalized,
        "available_storage_gb",
        _number(metadata.get("bo_nho_kha_dung_gb")),
    )
    _set_if_present(
        normalized,
        "screen_size_inch",
        _number(metadata.get("kich_thuoc_man_hinh_inch")),
    )
    _set_if_present(normalized, "weight_g", _integer(metadata.get("khoi_luong_g")))
    _set_if_present(normalized, "battery_mah", _integer(metadata.get("dung_luong_pin_mah")))
    _set_if_present(normalized, "battery_wh", _number(metadata.get("dung_luong_pin_wh")))
    _set_if_present(
        normalized,
        "max_charging_w",
        _number(metadata.get("cong_suat_sac_toi_da_w")),
    )
    _set_if_present(normalized, "cpu_core_count", _integer(metadata.get("so_nhan_cpu")))
    _set_if_present(
        normalized, "cpu_max_ghz", _number(metadata.get("toc_do_cpu_max_ghz"))
    )
    _set_if_present(normalized, "max_mobile_generation", generation)
    _set_if_present(
        normalized,
        "connectivity_class",
        normalize_connectivity_class(metadata, generation),
    )
    calling = _boolean_from_text(metadata.get("co_goi_dien"))
    if calling is None:
        calling = _boolean_from_text(metadata.get("Thực hiện cuộc gọi"))
    _set_if_present(normalized, "supports_calls", calling)
    _set_if_present(
        normalized,
        "supports_memory_card",
        normalize_memory_card(metadata.get("Thẻ nhớ")),
    )
    tags = feature_tags(metadata)
    if tags:
        normalized["feature_tags"] = tags
    normalized["category_scope"] = "tablet"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy tính bảng đã xử lý")
    for item in data:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = normalize_metadata(metadata)
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã chuẩn hóa metadata cho {len(data)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
