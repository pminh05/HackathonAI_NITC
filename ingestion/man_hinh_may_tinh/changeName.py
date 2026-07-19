"""Normalize computer-monitor metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "man_hinh_may_tinh"
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
        number = float(value)
        return number if number >= 0 else None
    text = str(value).strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group())
    return number if number >= 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold()).replace("đ", "d")
    return "".join(character for character in text if unicodedata.category(character) != "Mn")


def parse_vnd(value: Any) -> int | None:
    """Parse displayed VND values instead of the generated columns that are x10."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        return int(parsed) if parsed > 0 else None
    text = re.sub(r"(?i)(vnd|₫|đ)", "", str(value)).strip().replace(" ", "")
    if not text:
        return None
    try:
        # Spreadsheet exports use values such as ``3490000.0``.
        if re.fullmatch(r"\d+(?:\.0+)?", text.replace(",", "")):
            parsed = Decimal(text.replace(",", ""))
        else:
            parsed = Decimal(re.sub(r"[.,]", "", text))
    except InvalidOperation:
        return None
    return int(parsed) if parsed > 0 else None


def normalize_panel_family(value: Any) -> str | None:
    text = _fold(value)
    if "oled" in text:
        return "oled"
    if "ips" in text:
        return "ips"
    if re.search(r"\bva\b", text):
        return "va"
    if re.search(r"\btn\b", text):
        return "tn"
    return None


def normalize_screen_shape(value: Any) -> str | None:
    text = _fold(value)
    if "phang" in text:
        return "flat"
    if "cong" in text:
        return "curved"
    return None


def normalize_response_metric(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    for metric in ("gtg", "mprt", "prt"):
        if metric in text:
            return metric
    return None


def normalize_presence(value: Any) -> bool | None:
    text = _fold(value).strip()
    if not text:
        return None
    if text in {"khong", "khong co", "khong cam ung", "none", "no"} or text.startswith("khong "):
        return False
    if text == "co" or text.startswith("co "):
        return True
    return None


def normalize_connection_tags(value: Any) -> list[str]:
    text = _fold(value)
    checks = (
        ("hdmi", ("hdmi",)),
        ("displayport", ("displayport", "display port")),
        ("usb_c", ("usb-c", "usb c", "usb type-c", "usb type c", "type-c")),
        ("thunderbolt", ("thunderbolt",)),
        ("vga", ("vga", "d-sub", "d/sub")),
        ("dvi", ("dvi",)),
        ("usb_a", ("usb-a", "usb a", "usb type-a", "usb type a")),
        ("ethernet", ("rj45", "rj-45", "lan")),
        (
            "audio_out",
            ("audio out", "audio-out", "headphone", "earphone", "mini-jack", "jack tai nghe"),
        ),
    )
    return [tag for tag, needles in checks if any(needle in text for needle in needles)]


def normalize_feature_tags(metadata: dict[str, Any]) -> list[str]:
    text = _fold(
        " | ".join(
            str(metadata.get(field) or "")
            for field in ("Màn hình hiển thị", "Tiện ích")
        )
    )
    checks = (
        ("freesync", ("freesync", "free sync")),
        ("gsync", ("g-sync", "g sync", "gsync")),
        ("adaptive_sync", ("adaptive-sync", "adaptive sync")),
        ("flicker_free", ("flicker-free", "anti-flicker", "chong nhay")),
        (
            "low_blue_light",
            ("anh sang xanh", "blue light", "bluelight", "eye saver"),
        ),
        ("anti_glare", ("chong choi", "anti-glare", "anti glare")),
        ("height_adjust", ("gap man hinh len xuong", "dieu chinh do cao")),
        ("pivot", ("xoay doc", "pivot")),
        ("swivel", ("quay trai - phai", "quay trai-phai", "swivel")),
        ("webcam", ("webcam", "camera slimfit")),
        ("smart_monitor", ("smart tv", "tizen")),
    )
    tags = [tag for tag, needles in checks if any(needle in text for needle in needles)]
    if re.search(r"\b(?:hdr|hdr10|displayhdr)\b", text):
        tags.append("hdr")
    return tags


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
    quality_flags: list[str] = []
    if "gia_goc_vnd" in normalized or "gia_khuyen_mai_vnd" in normalized:
        quality_flags.append("generated_price_ignored")
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    brand = str(metadata.get("brand") or "").strip().casefold()
    resolution_width = _integer(metadata.get("do_phan_giai_ngang_px"))
    resolution_height = _integer(metadata.get("do_phan_giai_doc_px"))
    resolution_key = (
        f"{resolution_width}x{resolution_height}"
        if resolution_width is not None and resolution_height is not None
        else None
    )
    panel_variant = str(metadata.get("Tấm nền") or "").strip() or None
    response_metric = normalize_response_metric(
        metadata.get("loai_thoi_gian_dap_ung")
        or metadata.get("Thời gian đáp ứng")
    )

    _set_if_present(normalized, "brand_key", brand or None)
    _set_if_present(normalized, "original_price_vnd", parse_vnd(metadata.get("giá gốc")))
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi")),
    )
    _set_if_present(
        normalized,
        "screen_size_inch",
        _number(metadata.get("kich_thuoc_man_hinh_inch")),
    )
    _set_if_present(normalized, "resolution_key", resolution_key)
    _set_if_present(normalized, "resolution_width_px", resolution_width)
    _set_if_present(normalized, "resolution_height_px", resolution_height)
    _set_if_present(normalized, "panel_family", normalize_panel_family(panel_variant))
    _set_if_present(normalized, "panel_variant", panel_variant)
    _set_if_present(
        normalized,
        "screen_shape",
        normalize_screen_shape(metadata.get("Loại màn hình")),
    )
    _set_if_present(
        normalized,
        "response_time_ms",
        _number(metadata.get("thoi_gian_dap_ung_ms")),
    )
    _set_if_present(normalized, "response_time_metric", response_metric)
    _set_if_present(normalized, "brightness_nits", _number(metadata.get("do_sang_cd_m2")))
    _set_if_present(
        normalized,
        "srgb_coverage_pct",
        _number(metadata.get("do_phu_srgb_pct")),
    )
    _set_if_present(
        normalized,
        "dci_p3_coverage_pct",
        _number(metadata.get("do_phu_dci_p3_pct")),
    )
    connection_tags = normalize_connection_tags(metadata.get("Kết nối"))
    if connection_tags:
        normalized["connection_tags"] = connection_tags
    feature_tags = normalize_feature_tags(metadata)
    if feature_tags:
        normalized["feature_tags"] = feature_tags
    _set_if_present(normalized, "has_speakers", normalize_presence(metadata.get("Loa")))
    _set_if_present(normalized, "has_vesa_mount", normalize_presence(metadata.get("Vesa")))
    _set_if_present(normalized, "has_touch", normalize_presence(metadata.get("Màn hình cảm ứng")))
    _set_if_present(normalized, "width_mm", _number(metadata.get("ngang_mm")))
    _set_if_present(normalized, "height_min_mm", _number(metadata.get("cao_min_mm")))
    _set_if_present(normalized, "height_max_mm", _number(metadata.get("cao_max_mm")))
    _set_if_present(normalized, "depth_mm", _number(metadata.get("day_mm")))
    _set_if_present(normalized, "weight_kg", _number(metadata.get("khoi_luong_kg")))
    _set_if_present(
        normalized,
        "power_consumption_w",
        _number(metadata.get("dien_nang_tieu_thu_w")),
    )
    if quality_flags:
        normalized["data_quality_flags"] = quality_flags
    normalized["category_scope"] = "monitor"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu màn hình máy tính đã xử lý")
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
