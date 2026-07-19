"""Normalize water-heater metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_nuoc_nong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

PRODUCT_TYPE_VALUES = {
    "làm nóng trực tiếp": "direct",
    "làm nóng gián tiếp": "indirect",
    "làm nóng bằng năng lượng mặt trời": "solar",
    "làm nóng trực tiếp đa điểm": "direct_multipoint",
}

GENERATED_FIELDS = {
    "dung_tich_lit",
    "cong_suat_w",
    "thoi_gian_lam_nong_min_phut",
    "thoi_gian_lam_nong_max_phut",
    "nhiet_do_toi_da_c",
    "co_bom_tro_luc",
    "co_kem_voi_sen",
    "cao_cm",
    "rong_cm",
    "dai_cm",
    "day_cm",
    "khoi_luong_may_kg",
    "ap_luc_nuoc_min_mpa",
    "ap_luc_nuoc_max_mpa",
    "chi_so_chong_nuoc_ip",
    "so_nguoi_min",
    "so_nguoi_max",
    "gia_goc_vnd",
    "gia_khuyen_mai_vnd",
    "phan_tram_giam_gia",
    "kiem_tra_gia",
    "kiem_tra_du_lieu",
}

PRESSURE_PATTERN = re.compile(
    r"(tối\s*thiểu|min|tối\s*đa|max)\s*:?\s*"
    r"(\d+(?:[.,]\d+)?)\s*(mpa|kpa|bar)",
    flags=re.IGNORECASE,
)


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
    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def parse_vnd(value: Any) -> int | None:
    """Parse displayed VND values instead of trusting faulty generated columns."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        return int(parsed) if parsed > 0 else None

    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip().replace(" ", "")
    if not text:
        return None
    try:
        # Spreadsheet values such as ``4790000.0`` use a decimal suffix.
        if re.fullmatch(r"\d+(?:\.0+)?", text.replace(",", "")):
            parsed = Decimal(text.replace(",", ""))
        else:
            parsed = Decimal(re.sub(r"[.,]", "", text))
    except InvalidOperation:
        return None
    return int(parsed) if parsed > 0 else None


def normalize_product_type(value: Any) -> str | None:
    return PRODUCT_TYPE_VALUES.get(str(value or "").strip().casefold())


def _pressure_to_mpa(value: str, unit: str) -> float:
    number = float(value.replace(",", "."))
    normalized_unit = unit.casefold()
    if normalized_unit == "bar":
        number /= 10
    elif normalized_unit == "kpa":
        number /= 1000
    return round(number, 6)


def normalize_water_pressure(value: Any) -> tuple[float | None, float | None, list[str]]:
    """Parse min/max water pressure from the source phrase and normalize to MPa."""
    text = str(value or "").strip()
    if not text or text.casefold() in {"không", "không có", "none"}:
        return None, None, []

    minimum: float | None = None
    maximum: float | None = None
    for label, raw_number, unit in PRESSURE_PATTERN.findall(text):
        number = _pressure_to_mpa(raw_number, unit)
        if label.casefold().replace(" ", "") in {"tốithiểu", "min"}:
            minimum = number
        else:
            maximum = number

    flags: list[str] = []
    if minimum is None and maximum is None:
        flags.append("unparsed_water_pressure")
    elif minimum is not None and maximum is not None and minimum > maximum:
        flags.append("invalid_water_pressure_range")
        minimum = maximum = None
    return minimum, maximum, flags


def normalize_safety_tags(value: Any) -> list[str]:
    text = str(value or "").casefold()
    if not text or text in {"không", "không có", "none"}:
        return []
    rules = {
        "elcb": ("elcb",),
        "rcd": ("rcd",),
        "overheat_cutoff": ("quá nhiệt", "nhiệt độ quá cao"),
        "flow_sensor": ("cảm biến lưu lượng", "công tắc dòng chảy"),
        "pressure_relief_valve": ("van xả áp", "van an toàn"),
        "waterproof": ("chống thấm nước", "chống nước ip"),
        "anti_scald": ("chống bỏng",),
        "thermal_stabilizer": ("ổn định nhiệt", "thermostat"),
    }
    return [tag for tag, phrases in rules.items() if any(item in text for item in phrases)]


def normalize_feature_tags(metadata: dict[str, Any]) -> list[str]:
    text = " | ".join(
        str(metadata.get(key) or "")
        for key in ("Tiện ích", "Vòi sen", "Tùy chỉnh nhiệt độ")
    ).casefold()
    tags: list[str] = []
    rules = {
        "low_pressure_compatible": ("áp lực nước thấp", "nước yếu"),
        "multipoint_supply": ("đa điểm", "nhiều điểm"),
        "temperature_display": ("hiển thị nhiệt độ", "màn hình"),
        "antibacterial": ("kháng khuẩn", "ag+"),
        "copper_heating_element": ("thanh nhiệt bằng đồng",),
    }
    for tag, phrases in rules.items():
        if any(phrase in text for phrase in phrases):
            tags.append(tag)
    return tags


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add stable typed fields while retaining useful Vietnamese descriptions."""
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if key not in GENERATED_FIELDS and _clean(value) is not None
    }

    product_type = normalize_product_type(metadata.get("Loại máy"))
    pressure_min, pressure_max, quality_flags = normalize_water_pressure(
        metadata.get("Áp lực nước hoạt động")
    )
    safety_tags = normalize_safety_tags(metadata.get("Tính năng an toàn"))
    feature_tags = normalize_feature_tags(metadata)

    normalized["category_scope"] = "water_heater"
    _set_if_present(normalized, "product_type", product_type)
    _set_if_present(normalized, "original_price_vnd", parse_vnd(metadata.get("giá gốc")))
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi")),
    )
    _set_if_present(normalized, "capacity_l", _integer(metadata.get("dung_tich_lit")))
    _set_if_present(normalized, "power_w", _integer(metadata.get("cong_suat_w")))
    _set_if_present(
        normalized,
        "heating_time_min_minutes",
        _number(metadata.get("thoi_gian_lam_nong_min_phut")),
    )
    _set_if_present(
        normalized,
        "heating_time_max_minutes",
        _number(metadata.get("thoi_gian_lam_nong_max_phut")),
    )
    _set_if_present(
        normalized, "max_temperature_c", _integer(metadata.get("nhiet_do_toi_da_c"))
    )
    _set_if_present(normalized, "has_booster_pump", metadata.get("co_bom_tro_luc"))
    _set_if_present(normalized, "includes_shower", metadata.get("co_kem_voi_sen"))
    _set_if_present(normalized, "height_cm", _number(metadata.get("cao_cm")))
    _set_if_present(normalized, "width_cm", _number(metadata.get("rong_cm")))
    _set_if_present(normalized, "depth_cm", _number(metadata.get("day_cm")))
    _set_if_present(normalized, "weight_kg", _number(metadata.get("khoi_luong_may_kg")))
    _set_if_present(normalized, "water_pressure_min_mpa", pressure_min)
    _set_if_present(normalized, "water_pressure_max_mpa", pressure_max)
    _set_if_present(normalized, "ip_rating", _clean(metadata.get("chi_so_chong_nuoc_ip")))
    _set_if_present(normalized, "people_min", _integer(metadata.get("so_nguoi_min")))
    _set_if_present(normalized, "people_max", _integer(metadata.get("so_nguoi_max")))
    if safety_tags:
        normalized["safety_tags"] = safety_tags
    if feature_tags:
        normalized["feature_tags"] = feature_tags
    if quality_flags:
        normalized["data_quality_flags"] = quality_flags
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy nước nóng đã xử lý")
    for item in data:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            item["metadata"] = normalize_metadata(metadata)

    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã chuẩn hóa metadata cho {len(data)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
