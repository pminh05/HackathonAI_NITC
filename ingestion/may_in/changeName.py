"""Normalize printer metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_in"
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


def normalize_print_technology(value: Any) -> str | None:
    text = str(value or "").casefold()
    if "laser" in text:
        return "laser"
    if "phun" in text:
        return "inkjet"
    if "nhiệt" in text:
        return "thermal"
    return None


def normalize_color_mode(value: Any) -> str | None:
    text = str(value or "").casefold()
    if "trắng đen" in text or "đen trắng" in text:
        return "monochrome"
    if "màu" in text:
        return "color"
    return None


def normalize_connection_tags(metadata: dict[str, Any]) -> list[str]:
    text = " | ".join(
        str(metadata.get(field) or "")
        for field in ("Kết nối", "Cổng kết nối", "Công nghệ")
    ).casefold()
    tags: list[str] = []
    checks = {
        "wifi": ("wifi", "wi-fi"),
        "wifi_direct": ("wifi direct", "wi-fi direct"),
        "lan": ("lan", "ethernet"),
        "usb": ("usb",),
        "bluetooth": ("bluetooth",),
        "mobile_print": (
            "in bằng điện thoại",
            "airprint",
            "mopria",
            "smart app",
            "iprint",
            "ứng dụng",
        ),
    }
    for tag, needles in checks.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    return tags


def normalize_paper_size_tags(metadata: dict[str, Any]) -> list[str]:
    text = " | ".join(
        str(metadata.get(field) or "")
        for field in ("Kích thước phụ kiện", "Khổ giấy")
    ).casefold()
    checks = {
        "a3": (r"\ba3\b",),
        "a4": (r"\ba4\b",),
        "a5": (r"\ba5\b",),
        "a6": (r"\ba6\b",),
        "b5": (r"\bb5\b",),
        "f4": (r"\bf4a?\b", r"foolscap"),
        "letter": (r"\bletter\b",),
        "legal": (r"\blegal\b",),
    }
    return [
        tag
        for tag, patterns in checks.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]


def normalize_os_tags(value: Any) -> list[str]:
    text = str(value or "").casefold()
    checks = {
        "windows": ("windows",),
        "macos": ("macos", "os x"),
        "linux": ("linux",),
        "android": ("android",),
        "ios": ("ios",),
    }
    return [
        tag for tag, needles in checks.items() if any(needle in text for needle in needles)
    ]


def normalize_duplex(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    return not text.startswith("không")


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if _clean(value) is not None
    }
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    product_type = metadata.get("Loại sản phẩm")
    _set_if_present(normalized, "original_price_vnd", parse_vnd(metadata.get("giá gốc")))
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi")),
    )
    _set_if_present(
        normalized, "print_technology", normalize_print_technology(product_type)
    )
    _set_if_present(normalized, "color_mode", normalize_color_mode(product_type))
    _set_if_present(
        normalized,
        "monthly_volume_min_pages",
        _integer(metadata.get("cong_suat_thang_min_trang")),
    )
    _set_if_present(
        normalized,
        "monthly_volume_max_pages",
        _integer(metadata.get("cong_suat_thang_max_trang")),
    )
    _set_if_present(
        normalized, "print_speed_ppm", _number(metadata.get("toc_do_in_trang_phut"))
    )
    _set_if_present(
        normalized,
        "thermal_speed_mm_s",
        _number(metadata.get("toc_do_in_nhiet_mm_s")),
    )
    _set_if_present(
        normalized,
        "resolution_horizontal_dpi",
        _integer(metadata.get("do_phan_giai_ngang_dpi")),
    )
    _set_if_present(
        normalized,
        "resolution_vertical_dpi",
        _integer(metadata.get("do_phan_giai_doc_dpi")),
    )
    _set_if_present(
        normalized, "paper_input_sheets", _integer(metadata.get("khay_nap_giay_so_to"))
    )
    _set_if_present(
        normalized,
        "toner_yield_min_pages",
        _integer(metadata.get("so_trang_muc_min")),
    )
    _set_if_present(
        normalized,
        "toner_yield_max_pages",
        _integer(metadata.get("so_trang_muc_max")),
    )
    # Source dimensions use Dài x Rộng x Cao. Runtime exposes W x D x H.
    _set_if_present(normalized, "width_mm", _integer(metadata.get("dai_mm")))
    _set_if_present(normalized, "depth_mm", _integer(metadata.get("rong_mm")))
    _set_if_present(normalized, "height_mm", _integer(metadata.get("cao_mm")))
    connection_tags = normalize_connection_tags(metadata)
    if connection_tags:
        normalized["connection_tags"] = connection_tags
    paper_tags = normalize_paper_size_tags(metadata)
    if paper_tags:
        normalized["paper_size_tags"] = paper_tags
    os_tags = normalize_os_tags(metadata.get("Tương thích"))
    if os_tags:
        normalized["os_tags"] = os_tags
    _set_if_present(
        normalized, "supports_duplex", normalize_duplex(metadata.get("Loại giấy in 2 mặt"))
    )
    normalized["category_scope"] = "printer"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy in đã xử lý")
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
