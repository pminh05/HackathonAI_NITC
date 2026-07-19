"""Normalize cooler/freezer metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "tu_mat_tu_dong"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

PRODUCT_TYPE_VALUES = {
    "tủ mát": ("cooler", "cooler", False),
    "tủ mát mini": ("cooler_mini", "cooler", True),
    "tủ đông": ("freezer", "freezer", False),
    "tủ đông mini": ("freezer_mini", "freezer", True),
}

FEATURE_PATTERNS = {
    "convertible_mode": ("chuyển đổi", "3 chế độ"),
    "fast_freeze": ("làm lạnh nhanh", "làm đông nhanh", "extra freezing"),
    "lock": ("khóa cửa", "khoá cửa"),
    "wheels": ("bánh xe",),
    "external_temperature_control": (
        "điều khiển nhiệt độ bên ngoài",
        "điều chỉnh nhiệt độ bên ngoài",
    ),
    "led_light": ("đèn led", "hộp đèn led"),
    "drain": ("thoát nước",),
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
    text = str(value).strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def parse_vnd(value: Any) -> int | None:
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
    if number is None or not 10 <= number <= 500:
        return None
    return round(number, 2)


def normalize_product_type(value: Any) -> tuple[str, str, bool]:
    normalized = PRODUCT_TYPE_VALUES.get(str(value or "").strip().casefold())
    if normalized is None:
        raise ValueError(f"Loại sản phẩm tủ mát/tủ đông không hợp lệ: {value!r}")
    return normalized


def normalize_temperature_range(
    metadata: dict[str, Any], product_family: str
) -> tuple[float | None, float | None]:
    """Parse only signed/credible temperature values from the source text."""
    raw = _clean(metadata.get("Nhiệt độ ngăn đông (độ C)"))
    if raw is not None and str(raw).casefold() not in {"không", "không có"}:
        text = str(raw).replace("−", "-").replace("–", "-")
        # Collapse a spaced unary minus ("≤ - 18", "- 18 độ") without
        # turning the range separator in "0 - 10" into a negative sign.
        text = re.sub(
            r"(^|[≤<>~=]|đến)\s*-\s+(?=\d)",
            lambda match: f"{match.group(1)}-",
            text,
            flags=re.IGNORECASE,
        )
        matches = re.findall(r"[+-]?\d+(?:[.,]\d+)?", text)
        numbers = [float(value.replace(",", ".")) for value in matches]
        if numbers:
            # A freezer source such as "Dưới 18℃" has lost its sign. Do not
            # silently turn that ambiguous value into either +18 or -18.
            if product_family == "freezer" and all(value > 0 for value in numbers):
                return None, None
            return min(numbers), max(numbers)

    direct_min = _number(metadata.get("nhiet_do_min_c"))
    direct_max = _number(metadata.get("nhiet_do_max_c"))
    direct = [value for value in (direct_min, direct_max) if value is not None]
    if product_family == "freezer" and direct and all(value > 0 for value in direct):
        return None, None
    if direct:
        return min(direct), max(direct)
    return None, None


def normalize_inverter(metadata: dict[str, Any]) -> bool | None:
    direct = metadata.get("co_inverter")
    if isinstance(direct, bool):
        return direct
    text = str(metadata.get("Công nghệ tiết kiệm điện") or "").strip().casefold()
    if not text:
        return None
    if text in {"không", "không có", "none"}:
        return False
    return True if "inverter" in text else None


def normalize_feature_tags(metadata: dict[str, Any]) -> list[str]:
    fields = (
        metadata.get("Tiện ích"),
        metadata.get("Công nghệ"),
        metadata.get("Chất liệu mặt"),
    )
    text = " | ".join(str(value) for value in fields if _clean(value)).casefold()
    tags = {
        tag
        for tag, patterns in FEATURE_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    }
    if (
        "kính" in text
        and "không có kính" not in text
        and "không kính" not in text
    ):
        tags.add("glass_door")
    return sorted(tags)


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if _clean(value) is not None
    }

    raw_type = metadata.get("loai_san_pham_chuan") or metadata.get("Loại sản phẩm")
    product_type, product_family, is_mini = normalize_product_type(raw_type)
    original_price = parse_vnd(
        metadata.get("giá gốc") or metadata.get("gia_goc_vnd")
    )
    promotional_price = parse_vnd(
        metadata.get("giá khuyến mãi") or metadata.get("gia_khuyen_mai_vnd")
    )
    effective_price = (
        promotional_price if promotional_price is not None else original_price
    )
    temperature_min, temperature_max = normalize_temperature_range(
        metadata, product_family
    )

    normalized["category_scope"] = "cooler_freezer"
    normalized["product_type"] = product_type
    normalized["product_family"] = product_family
    normalized["is_mini"] = is_mini
    normalized["feature_tags"] = normalize_feature_tags(metadata)

    _set_if_present(normalized, "brand", _clean(metadata.get("brand")))
    _set_if_present(normalized, "model_code", _clean(metadata.get("model_code")))
    _set_if_present(normalized, "original_price_vnd", original_price)
    _set_if_present(normalized, "promotional_price_vnd", promotional_price)
    _set_if_present(normalized, "effective_price_vnd", effective_price)
    _set_if_present(
        normalized, "total_capacity_lit", _integer(metadata.get("dung_tich_tong_lit"))
    )
    _set_if_present(
        normalized, "compartment_count", _integer(metadata.get("tong_so_ngan"))
    )
    _set_if_present(
        normalized,
        "freezer_compartment_count",
        _integer(metadata.get("so_ngan_dong")),
    )
    _set_if_present(
        normalized,
        "cooler_compartment_count",
        _integer(metadata.get("so_ngan_mat")),
    )
    _set_if_present(normalized, "temperature_min_c", temperature_min)
    _set_if_present(normalized, "temperature_max_c", temperature_max)
    _set_if_present(normalized, "has_inverter", normalize_inverter(metadata))
    _set_if_present(
        normalized, "energy_kwh_day", _number(metadata.get("dien_nang_kwh_ngay"))
    )
    _set_if_present(
        normalized, "energy_kwh_year", _number(metadata.get("dien_nang_kwh_nam"))
    )
    _set_if_present(
        normalized, "noise_min_db", _number(metadata.get("do_on_min_db"))
    )
    _set_if_present(
        normalized, "noise_max_db", _number(metadata.get("do_on_max_db"))
    )
    _set_if_present(normalized, "door_count", _integer(metadata.get("so_cua_chuan")))
    _set_if_present(
        normalized, "width_cm", normalize_dimension_cm(metadata.get("ngang_cm"))
    )
    _set_if_present(
        normalized, "height_cm", normalize_dimension_cm(metadata.get("cao_cm"))
    )
    _set_if_present(
        normalized, "depth_cm", normalize_dimension_cm(metadata.get("sau_cm"))
    )
    _set_if_present(
        normalized, "weight_kg", _number(metadata.get("khoi_luong_may_kg"))
    )
    _set_if_present(
        normalized,
        "gas_type",
        _clean(metadata.get("loai_gas_chuan") or metadata.get("Loại Gas")),
    )
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu tủ mát, tủ đông đã xử lý")
    for item in data:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Sản phẩm {item.get('id')} thiếu metadata")
        item["metadata"] = normalize_metadata(metadata)

    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Đã chuẩn hóa metadata cho {len(data)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
