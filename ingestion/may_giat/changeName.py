"""Normalize washing-machine metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_giat"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

PRODUCT_TYPE_VALUES = {
    "cửa trên": "top_load",
    "cửa trước": "front_load",
    "máy giặt sấy": "washer_dryer",
    "máy giặt mini": "mini",
    "tháp giặt sấy": "wash_tower",
    "tủ chăm sóc quần áo": "clothing_care",
}
DRUM_TYPE_VALUES = {
    "lồng đứng": "vertical",
    "lồng ngang": "horizontal",
}


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


def parse_vnd(value: Any) -> int | None:
    """Parse displayed VND values, including spreadsheet strings ending in ``.0``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        return int(parsed) if parsed > 0 else None

    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip().replace(" ", "")
    if not text:
        return None
    try:
        # CSV/Excel exports in this dataset use values such as ``18640000.0``.
        if re.fullmatch(r"\d+(?:\.0+)?", text.replace(",", "")):
            parsed = Decimal(text.replace(",", ""))
        else:
            parsed = Decimal(re.sub(r"[.,]", "", text))
    except InvalidOperation:
        return None
    return int(parsed) if parsed > 0 else None


def normalize_dimension_cm(value: Any) -> float | None:
    """Normalize washing-machine dimensions to cm without trusting ``*_mm`` names."""
    number = _number(value)
    if number is None:
        return None
    text = str(value).strip().lower()
    if "mm" in text:
        number /= 10
    elif re.search(r"(?<!c)(?<!m)\bm\b", text):
        number *= 100
    elif "cm" not in text:
        # Source values are mixed: 59 means 59 cm, while 600 means 600 mm.
        if 20 <= number <= 200:
            pass
        elif 200 < number <= 3000:
            number /= 10
        else:
            return None
    if 20 <= number <= 300:
        return round(number, 2)
    return None


def normalize_people_range(value: Any) -> tuple[int | None, int | None]:
    """Read only the people phrase, never the load range inside parentheses."""
    if value is None:
        return None, None
    text = str(value).split("(", 1)[0].strip().casefold()
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if not numbers:
        return None, None
    if "trên" in text:
        return numbers[0] + 1, None
    if "dưới" in text:
        return None, max(numbers[0] - 1, 1)
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def normalize_product_type(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return PRODUCT_TYPE_VALUES.get(text)


def normalize_drum_type(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    return DRUM_TYPE_VALUES.get(text)


def normalize_inverter(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text in {"không", "không có", "none"}:
        return False
    return True if "inverter" in text else None


def normalize_dryer(metadata: dict[str, Any], product_type: str | None) -> bool | None:
    dry_capacity = _number(metadata.get("khoi_luong_say_kg"))
    if dry_capacity is not None and dry_capacity > 0:
        return True
    if product_type in {"washer_dryer", "wash_tower"}:
        return True
    technology = str(metadata.get("Công nghệ sấy") or "").strip().casefold()
    if technology:
        return technology not in {"không", "không có", "none"}
    if product_type in {"top_load", "front_load", "mini"}:
        return False
    return None


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add stable, typed fields while retaining useful source descriptions."""
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if _clean(value) is not None
    }

    product_type = normalize_product_type(metadata.get("Loại sản phẩm"))
    drum_type = normalize_drum_type(metadata.get("Lồng giặt"))
    people_min, people_max = normalize_people_range(metadata.get("Số người sử dụng"))

    original_price = parse_vnd(metadata.get("giá gốc"))
    if original_price is None:
        original_price = parse_vnd(metadata.get("gia_goc_vnd"))
    promotional_price = parse_vnd(metadata.get("giá khuyến mãi"))
    if promotional_price is None:
        promotional_price = parse_vnd(metadata.get("gia_khuyen_mai_vnd"))

    _set_if_present(normalized, "product_type", product_type)
    _set_if_present(normalized, "drum_type", drum_type)
    _set_if_present(normalized, "original_price_vnd", original_price)
    _set_if_present(normalized, "promotional_price_vnd", promotional_price)
    _set_if_present(normalized, "wash_capacity_kg", _number(metadata.get("khoi_luong_giat_kg")))
    _set_if_present(normalized, "dry_capacity_kg", _number(metadata.get("khoi_luong_say_kg")))
    _set_if_present(normalized, "people_min", people_min)
    _set_if_present(normalized, "people_max", people_max)
    _set_if_present(normalized, "has_inverter", normalize_inverter(metadata.get("Loại Inverter")))
    _set_if_present(normalized, "has_dryer", normalize_dryer(metadata, product_type))
    _set_if_present(normalized, "width_cm", normalize_dimension_cm(metadata.get("Ngang")))
    _set_if_present(normalized, "height_cm", normalize_dimension_cm(metadata.get("Cao")))
    _set_if_present(normalized, "depth_cm", normalize_dimension_cm(metadata.get("Sâu")))
    _set_if_present(normalized, "spin_speed_rpm", _number(metadata.get("toc_do_vat_rpm")))
    _set_if_present(normalized, "program_count", _number(metadata.get("so_chuong_trinh")))
    _set_if_present(normalized, "energy_kwh_per_kg", _number(metadata.get("dien_nang_kwh")))
    normalized["category_scope"] = "washing_machine"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy giặt đã xử lý")
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
