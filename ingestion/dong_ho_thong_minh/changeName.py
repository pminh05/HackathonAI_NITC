"""Normalize smartwatch metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "dong_ho_thong_minh"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

NUMERIC_FIELDS = {
    "screen_size_inch",
    "case_length_mm",
    "case_width_mm",
    "case_thickness_mm",
    "weight_g",
    "wrist_min_cm",
    "wrist_max_cm",
    "battery_mah",
    "charging_time_hours",
    "typical_battery_hours",
    "water_resistance_atm",
}
BOOLEAN_FIELDS = {
    "has_cellular",
    "has_gps",
    "has_notifications",
    "swim_ready",
    "has_sos",
}
CALL_MODES = {"none", "on_wrist", "standalone"}


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


def _bounded_number(
    value: Any, *, minimum: float, maximum: float
) -> float | None:
    number = _number(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return round(number, 2)


def _integer(value: Any, *, minimum: int = 1, maximum: int = 1_000_000) -> int | None:
    number = _number(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return int(number)


def parse_vnd(value: Any) -> int | None:
    """Parse a positive VND amount without guessing a missing currency."""
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


def normalize_display_family(value: Any) -> str | None:
    text = str(value or "").casefold()
    if not text:
        return None
    if "amoled" in text or "oled" in text:
        return "amoled_oled"
    if "mip" in text or "memory-in-pixel" in text:
        return "mip"
    if "tft" in text:
        return "tft_lcd"
    if "ips" in text:
        return "ips_lcd"
    if "lcd" in text or "led" in text:
        return "lcd"
    return "other"


def normalize_strap_material(value: Any) -> str | None:
    text = str(value or "").casefold()
    if not text:
        return None
    if "silicone" in text or "silicon" in text:
        return "silicone"
    if "cao su" in text or "tpu" in text or "fluoroelastomer" in text:
        return "rubber_tpu"
    if "titanium" in text:
        return "titanium"
    if "thép" in text or "kim loại" in text or "stainless" in text:
        return "metal"
    if "da" in text:
        return "leather"
    if "vải" in text or "nylon" in text or "dệt" in text:
        return "fabric_nylon"
    if "composite" in text:
        return "composite"
    return "other"


def compatible_platforms(value: Any) -> list[str]:
    text = str(value or "").casefold()
    platforms: list[str] = []
    if "ios" in text or "iphone" in text:
        platforms.append("ios")
    if "android" in text:
        platforms.append("android")
    return platforms


def _explicit_presence(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text.startswith("không") or text in {"none", "null"}:
        return False
    return True


def normalize_call_mode(value: Any) -> str | None:
    present = _explicit_presence(value)
    if present is None:
        return None
    if present is False:
        return "none"
    text = str(value).casefold()
    return "standalone" if "độc lập" in text else "on_wrist"


def normalize_has_cellular(value: Any) -> bool | None:
    return _explicit_presence(value)


def normalize_has_gps(value: Any) -> bool | None:
    present = _explicit_presence(value)
    if present is None or present is False:
        return present
    text = str(value).casefold()
    return any(
        token in text
        for token in ("gps", "gnss", "galileo", "glonass", "beidou", "qzss")
    )


def normalize_has_notifications(value: Any) -> bool | None:
    return _explicit_presence(value)


def normalize_swim_ready(value: Any) -> bool | None:
    present = _explicit_presence(value)
    if present is None or present is False:
        return present
    text = str(value).casefold()
    return "bơi" in text or "lặn" in text


def normalize_has_sos(metadata: dict[str, Any]) -> bool | None:
    values = [metadata.get("Tiện ích"), metadata.get("Tiện ích khác")]
    present = [_explicit_presence(value) for value in values]
    if all(value is None for value in present):
        return None
    text = " | ".join(str(value or "") for value in values).casefold()
    return "sos" in text or "khẩn cấp" in text


def normalize_water_resistance_atm(value: Any) -> float | None:
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*atm\b", str(value or ""), re.I)
    return float(match.group(1).replace(",", ".")) if match else None


def _duration_hours(segment: str) -> float | None:
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(ngày|day|giờ|tiếng|hour)",
        segment,
        re.I,
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).casefold()
    return round(value * 24 if unit in {"ngày", "day"} else value, 2)


def normalize_typical_battery_hours(value: Any) -> float | None:
    """Extract normal runtime and ignore GPS, saver, AOD and solar modes."""
    text = str(value or "").strip()
    if not text:
        return None
    segments = [part.strip() for part in text.split("|") if part.strip()]
    preferred_markers = ("chế độ đồng hồ thông minh", "sử dụng thông thường")
    for segment in segments:
        folded = segment.casefold()
        if any(marker in folded for marker in preferred_markers):
            return _duration_hours(segment)

    excluded_markers = (
        "gps",
        "định vị",
        "tiết kiệm pin",
        "always-on",
        "always on",
        "aod",
        "năng lượng mặt trời",
        "nghe gọi",
    )
    for segment in segments:
        folded = segment.casefold()
        if not any(marker in folded for marker in excluded_markers):
            duration = _duration_hours(segment)
            if duration is not None:
                return duration
    return None


def health_feature_tags(value: Any) -> list[str]:
    text = str(value or "").casefold()
    checks = [
        ("heart_rate", ("nhịp tim",)),
        ("spo2", ("spo2", "oxy trong máu")),
        ("sleep", ("giấc ngủ",)),
        ("stress", ("stress", "căng thẳng")),
        ("ecg", ("điện tâm đồ", "ecg")),
        ("blood_pressure", ("huyết áp",)),
        ("step_count", ("bước chân",)),
        ("menstrual_cycle", ("chu kỳ kinh nguyệt",)),
        ("vo2_max", ("vo2 max", "vo2max", "tiêu thụ oxy tối đa")),
        ("body_composition", ("thành phần cơ thể",)),
    ]
    return [tag for tag, needles in checks if any(needle in text for needle in needles)]


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
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)
    normalized.pop("feature_tags", None)

    brand = str(metadata.get("brand") or "").strip()
    original_price = parse_vnd(metadata.get("giá gốc")) or parse_vnd(
        metadata.get("gia_goc_vnd")
    )
    promotional_price = parse_vnd(metadata.get("giá khuyến mãi")) or parse_vnd(
        metadata.get("gia_khuyen_mai_vnd")
    )

    _set_if_present(normalized, "brand_key", brand.casefold() or None)
    _set_if_present(normalized, "original_price_vnd", original_price)
    _set_if_present(normalized, "promotional_price_vnd", promotional_price)
    normalized["compatible_platforms"] = compatible_platforms(
        metadata.get("Tương thích")
    )
    _set_if_present(
        normalized, "call_mode", normalize_call_mode(metadata.get("Thực hiện cuộc gọi"))
    )
    _set_if_present(normalized, "has_cellular", normalize_has_cellular(metadata.get("SIM")))
    _set_if_present(normalized, "has_gps", normalize_has_gps(metadata.get("Định vị")))
    _set_if_present(
        normalized,
        "has_notifications",
        normalize_has_notifications(metadata.get("Hiển thị thông báo")),
    )
    _set_if_present(
        normalized,
        "swim_ready",
        normalize_swim_ready(metadata.get("Chuẩn chống nước, bụi")),
    )
    _set_if_present(normalized, "has_sos", normalize_has_sos(metadata))
    normalized["health_feature_tags"] = health_feature_tags(
        metadata.get("Theo dõi sức khoẻ")
    )
    _set_if_present(
        normalized,
        "display_family",
        normalize_display_family(metadata.get("Màn hình hiển thị")),
    )
    _set_if_present(
        normalized,
        "strap_material_family",
        normalize_strap_material(metadata.get("Chất liệu dây")),
    )
    _set_if_present(
        normalized,
        "screen_size_inch",
        _bounded_number(
            metadata.get("kich_thuoc_man_hinh_inch"), minimum=0.3, maximum=5
        ),
    )
    _set_if_present(
        normalized,
        "case_length_mm",
        _bounded_number(metadata.get("dai_mm"), minimum=10, maximum=100),
    )
    _set_if_present(
        normalized,
        "case_width_mm",
        _bounded_number(
            metadata.get("ngang_mm") or metadata.get("kich_thuoc_mat_mm"),
            minimum=10,
            maximum=100,
        ),
    )
    _set_if_present(
        normalized,
        "case_thickness_mm",
        _bounded_number(metadata.get("day_mm"), minimum=1, maximum=50),
    )
    _set_if_present(
        normalized,
        "weight_g",
        _bounded_number(metadata.get("khoi_luong_g"), minimum=1, maximum=1000),
    )
    _set_if_present(
        normalized,
        "wrist_min_cm",
        _bounded_number(metadata.get("chu_vi_co_tay_min_cm"), minimum=5, maximum=50),
    )
    _set_if_present(
        normalized,
        "wrist_max_cm",
        _bounded_number(metadata.get("chu_vi_co_tay_max_cm"), minimum=5, maximum=50),
    )
    _set_if_present(
        normalized,
        "battery_mah",
        _integer(metadata.get("dung_luong_pin_mah"), maximum=100_000),
    )
    _set_if_present(
        normalized,
        "charging_time_hours",
        _bounded_number(metadata.get("thoi_gian_sac_gio"), minimum=0.05, maximum=48),
    )
    _set_if_present(
        normalized,
        "typical_battery_hours",
        normalize_typical_battery_hours(metadata.get("Thời gian sử dụng")),
    )
    _set_if_present(
        normalized,
        "water_resistance_atm",
        normalize_water_resistance_atm(metadata.get("Chuẩn chống nước, bụi")),
    )
    normalized["category_scope"] = "smartwatch"
    return normalized


def validate_canonical_types(metadata: dict[str, Any]) -> None:
    for field in NUMERIC_FIELDS:
        value = metadata.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"Metadata {field} phải là số")
    for field in BOOLEAN_FIELDS:
        value = metadata.get(field)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Metadata {field} phải là bool")
    for field in ("compatible_platforms", "health_feature_tags"):
        value = metadata.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Metadata {field} phải là list[str]")
    call_mode = metadata.get("call_mode")
    if call_mode is not None and call_mode not in CALL_MODES:
        raise ValueError("Metadata call_mode không hợp lệ")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu đồng hồ thông minh đã xử lý")
    for index, item in enumerate(data):
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Sản phẩm index {index} thiếu metadata")
        normalized = normalize_metadata(metadata)
        validate_canonical_types(normalized)
        item["metadata"] = normalized
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã chuẩn hóa metadata cho {len(data)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
