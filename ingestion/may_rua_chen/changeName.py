"""Normalize dishwasher metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_rua_chen"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

PRODUCT_TYPE_VALUES = {
    "máy rửa chén độc lập": "freestanding",
    "máy rửa chén mini": "mini",
    "máy rửa chén bán âm": "semi_integrated",
    "máy rửa chén âm tủ": "built_in",
    "máy sấy chén": "dish_dryer",
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


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def parse_vnd(value: Any) -> int | None:
    """Parse displayed VND values without trusting faulty generated columns."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        return int(parsed) if parsed > 0 else None

    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip().replace(" ", "")
    if not text:
        return None
    try:
        # Spreadsheet exports use values such as ``11772000.0``. Treat that
        # suffix as a decimal, not as a thousands separator.
        if re.fullmatch(r"\d+(?:\.0+)?", text.replace(",", "")):
            parsed = Decimal(text.replace(",", ""))
        else:
            parsed = Decimal(re.sub(r"[.,]", "", text))
    except InvalidOperation:
        return None
    return int(parsed) if parsed > 0 else None


def normalize_product_type(value: Any) -> str | None:
    return PRODUCT_TYPE_VALUES.get(str(value or "").strip().casefold())


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
    # The distributor-generated price fields are consistently 10x too large.
    # Only the values parsed from the displayed source columns may be filtered.
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    product_type = normalize_product_type(metadata.get("Loại sản phẩm"))
    original_price = parse_vnd(metadata.get("giá gốc"))
    promotional_price = parse_vnd(metadata.get("giá khuyến mãi"))

    _set_if_present(normalized, "product_type", product_type)
    _set_if_present(normalized, "original_price_vnd", original_price)
    _set_if_present(normalized, "promotional_price_vnd", promotional_price)
    _set_if_present(
        normalized, "vietnamese_meals_min", _integer(metadata.get("bua_an_viet_min"))
    )
    _set_if_present(
        normalized, "vietnamese_meals_max", _integer(metadata.get("bua_an_viet_max"))
    )
    _set_if_present(
        normalized, "place_settings_min", _integer(metadata.get("bo_chau_au_min"))
    )
    _set_if_present(
        normalized, "place_settings_max", _integer(metadata.get("bo_chau_au_max"))
    )
    _set_if_present(
        normalized,
        "water_min_l_per_cycle",
        _number(metadata.get("tieu_thu_nuoc_min_lit_lan")),
    )
    _set_if_present(
        normalized,
        "water_max_l_per_cycle",
        _number(metadata.get("tieu_thu_nuoc_max_lit_lan")),
    )
    _set_if_present(normalized, "noise_db", _number(metadata.get("do_on_db")))
    _set_if_present(normalized, "width_cm", _number(metadata.get("ngang_cm")))
    _set_if_present(normalized, "height_cm", _number(metadata.get("cao_cm")))
    _set_if_present(normalized, "depth_cm", _number(metadata.get("sau_cm")))
    _set_if_present(
        normalized, "power_min_w", _integer(metadata.get("cong_suat_min_w"))
    )
    _set_if_present(
        normalized, "power_max_w", _integer(metadata.get("cong_suat_max_w"))
    )
    _set_if_present(
        normalized, "program_count", _integer(metadata.get("so_chuong_trinh"))
    )
    normalized["category_scope"] = (
        "dish_dryer" if product_type == "dish_dryer" else "dishwasher"
    )
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy rửa chén đã xử lý")
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
