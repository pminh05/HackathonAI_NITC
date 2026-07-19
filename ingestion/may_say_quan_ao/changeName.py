"""Normalize clothes-dryer metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_say_quan_ao"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

DRYER_TYPE_VALUES = {
    "sấy bơm nhiệt": "heat_pump",
    "sấy ngưng tụ": "condenser",
    "sấy thông hơi": "vented",
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
    """Parse displayed VND values without trusting faulty precomputed columns."""
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


def normalize_dimension_cm(value: Any) -> float | None:
    number = _number(value)
    if number is None or not 10 <= number <= 300:
        return None
    return round(number, 2)


def normalize_people_range(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    direct_min = _integer(metadata.get("so_nguoi_min"))
    direct_max = _integer(metadata.get("so_nguoi_max"))
    if direct_min is not None or direct_max is not None:
        return direct_min, direct_max

    text = str(metadata.get("Số người sử dụng") or "").split("(", 1)[0]
    folded = text.strip().casefold()
    numbers = [int(item) for item in re.findall(r"\d+", folded)]
    if not numbers:
        return None, None
    if "trên" in folded:
        return numbers[0] + 1, None
    if "dưới" in folded:
        return None, max(numbers[0] - 1, 1)
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def normalize_dryer_type(value: Any) -> str | None:
    return DRYER_TYPE_VALUES.get(str(value or "").strip().casefold())


def normalize_inverter(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text in {"không", "không có", "none"}:
        return False
    return True if "inverter" in text else None


def normalize_sensor(metadata: dict[str, Any]) -> bool | None:
    direct = metadata.get("co_cam_bien")
    if isinstance(direct, bool):
        return direct
    text = str(metadata.get("Cảm biến") or "").strip().casefold()
    if text in {"có", "yes", "true"}:
        return True
    if text in {"không", "không có", "no", "false"}:
        return False
    return None


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
    # These distributor-generated fields are consistently 10x too large in the
    # current dryer dataset. Only the validated canonical values below may be used.
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    people_min, people_max = normalize_people_range(metadata)
    original_price = parse_vnd(metadata.get("giá gốc"))
    promotional_price = parse_vnd(metadata.get("giá khuyến mãi"))

    _set_if_present(normalized, "dryer_type", normalize_dryer_type(metadata.get("Loại sản phẩm")))
    _set_if_present(normalized, "original_price_vnd", original_price)
    _set_if_present(normalized, "promotional_price_vnd", promotional_price)
    _set_if_present(normalized, "dry_capacity_kg", _number(metadata.get("khoi_luong_say_kg")))
    _set_if_present(normalized, "people_min", people_min)
    _set_if_present(normalized, "people_max", people_max)
    _set_if_present(normalized, "width_cm", normalize_dimension_cm(metadata.get("ngang_cm")))
    _set_if_present(normalized, "height_cm", normalize_dimension_cm(metadata.get("cao_cm")))
    _set_if_present(normalized, "depth_cm", normalize_dimension_cm(metadata.get("sau_cm")))
    _set_if_present(normalized, "power_min_w", _integer(metadata.get("cong_suat_min_w")))
    _set_if_present(normalized, "power_max_w", _integer(metadata.get("cong_suat_max_w")))
    _set_if_present(normalized, "temperature_min_c", _number(metadata.get("nhiet_do_min_c")))
    _set_if_present(normalized, "temperature_max_c", _number(metadata.get("nhiet_do_max_c")))
    _set_if_present(
        normalized,
        "has_inverter",
        normalize_inverter(metadata.get("Công nghệ tiết kiệm điện")),
    )
    _set_if_present(normalized, "has_sensor", normalize_sensor(metadata))
    normalized["category_scope"] = "dryer"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy sấy đã xử lý")
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
