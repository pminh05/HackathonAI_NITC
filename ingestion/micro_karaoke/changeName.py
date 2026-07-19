"""Normalize micro-karaoke metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "micro_karaoke"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed.json"
OUTPUT_FILE = BASE_DIR / "data" / f"{DATASET}_processed_vi.json"

GENERATED_FIELDS = {
    "loai_san_pham_chuan",
    "bang_tan_chuan",
    "loai_du_lieu_tan_so",
    "tan_so_song_min_mhz",
    "tan_so_song_max_mhz",
    "tan_so_am_thanh_min_hz",
    "tan_so_am_thanh_max_hz",
    "do_meo_tieng_pct",
    "toan_tu_do_meo",
    "nam_san_xuat_chuan",
    "gia_goc_vnd",
    "gia_khuyen_mai_vnd",
    "kiem_tra_du_lieu",
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


def normalize_microphone_type(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if "không dây" in text:
        return "wireless"
    if "có dây" in text:
        return "wired"
    return None


def normalize_wireless_band(value: Any, microphone_type: str | None) -> str | None:
    if microphone_type == "wired":
        return None
    text = str(value or "").strip().casefold()
    if "uhf" in text:
        return "uhf"
    if "2.4" in text or "2,4" in text:
        return "2_4_ghz"
    return None


def normalize_frequency_data_type(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if "gồm cả" in text:
        return "rf_and_audio"
    if "sóng" in text:
        return "rf"
    if "âm thanh" in text:
        return "audio"
    return None


def _raw_mhz_range(value: Any) -> tuple[float, float] | None:
    text = str(value or "")
    if "mhz" not in text.casefold():
        return None
    numbers = [
        float(item.replace(",", "."))
        for item in re.findall(r"\d+(?:[.,]\d+)?", text)
    ]
    if not numbers:
        return None
    return min(numbers), max(numbers)


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return filter-safe canonical fields while preserving useful raw labels."""
    normalized = {
        key: _clean(value)
        for key, value in metadata.items()
        if key not in GENERATED_FIELDS and _clean(value) is not None
    }
    brand = str(_clean(metadata.get("brand")) or "").strip()
    microphone_type = normalize_microphone_type(
        metadata.get("loai_san_pham_chuan") or metadata.get("Loại sản phẩm")
    )
    band = normalize_wireless_band(
        metadata.get("bang_tan_chuan") or metadata.get("Băng tần"), microphone_type
    )
    flags: list[str] = []
    source_warning = str(_clean(metadata.get("kiem_tra_du_lieu")) or "")
    warning_folded = source_warning.casefold()
    if source_warning:
        flags.append(source_warning)
    if not microphone_type:
        flags.append("missing_microphone_type")

    _set_if_present(normalized, "brand", brand or None)
    _set_if_present(normalized, "brand_key", brand.casefold() or None)
    _set_if_present(normalized, "microphone_type", microphone_type)
    _set_if_present(normalized, "wireless_band", band)
    _set_if_present(
        normalized,
        "frequency_data_type",
        normalize_frequency_data_type(metadata.get("loai_du_lieu_tan_so")),
    )
    _set_if_present(
        normalized,
        "original_price_vnd",
        parse_vnd(metadata.get("giá gốc")) or parse_vnd(metadata.get("gia_goc_vnd")),
    )
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi"))
        or parse_vnd(metadata.get("gia_khuyen_mai_vnd")),
    )

    rf_min = _number(metadata.get("tan_so_song_min_mhz"))
    rf_max = _number(metadata.get("tan_so_song_max_mhz"))
    raw_range = _raw_mhz_range(metadata.get("Tần số hoạt động"))
    if raw_range is not None:
        rf_min, rf_max = raw_range
    rf_is_unsafe = (
        microphone_type == "wired"
        or "chưa tách được tần số sóng" in warning_folded
    )
    if (
        not rf_is_unsafe
        and rf_min is not None
        and rf_max is not None
        and rf_min <= rf_max
    ):
        normalized["rf_frequency_min_mhz"] = rf_min
        normalized["rf_frequency_max_mhz"] = rf_max

    audio_min = _number(metadata.get("tan_so_am_thanh_min_hz"))
    audio_max = _number(metadata.get("tan_so_am_thanh_max_hz"))
    audio_is_unsafe = "dải tần âm thanh bất thường" in warning_folded
    if (
        not audio_is_unsafe
        and audio_min is not None
        and audio_max is not None
        and audio_min <= audio_max
    ):
        normalized["audio_frequency_min_hz"] = audio_min
        normalized["audio_frequency_max_hz"] = audio_max

    _set_if_present(
        normalized, "distortion_pct", _number(metadata.get("do_meo_tieng_pct"))
    )
    _set_if_present(
        normalized, "distortion_operator", _clean(metadata.get("toan_tu_do_meo"))
    )
    _set_if_present(
        normalized, "manufacture_year", _integer(metadata.get("nam_san_xuat_chuan"))
    )
    _set_if_present(normalized, "origin", _clean(metadata.get("Sản xuất tại")))
    if flags:
        normalized["data_quality_flags"] = list(dict.fromkeys(flags))
    normalized["category_scope"] = "karaoke_microphone"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu micro karaoke đã xử lý")
    for item in data:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Sản phẩm thiếu metadata")
        item["metadata"] = normalize_metadata(metadata)
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã chuẩn hóa metadata cho {len(data)} sản phẩm -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
