"""Normalize desktop metadata into the Qdrant runtime contract."""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET = "may_tinh_de_ban"
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
    """Parse source display prices instead of faulty generated price columns."""
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


def normalize_cpu_vendor(value: Any) -> str | None:
    text = str(value or "").casefold()
    if not text:
        return None
    if "intel" in text or any(token in text for token in ("pentium", "celeron", "xeon")):
        return "intel"
    if "amd" in text or "ryzen" in text or "athlon" in text:
        return "amd"
    if "apple m" in text:
        return "apple"
    return None


def normalize_os_family_tags(value: Any) -> list[str]:
    text = str(value or "").strip().casefold()
    tags: list[str] = []
    checks = (
        ("windows", ("windows",)),
        ("macos", ("mac os", "macos")),
        ("linux", ("ubuntu", "linux")),
        ("freedos", ("freedos", "free dos")),
    )
    for tag, needles in checks:
        if any(needle in text for needle in needles):
            tags.append(tag)
    if text in {"không", "không có", "none"}:
        tags.append("no_os")
    return tags


def normalize_gpu_type(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if "đồ hoạ rời" in text or "đồ họa rời" in text:
        return "discrete"
    if "đồ hoạ tích hợp" in text or "đồ họa tích hợp" in text:
        return "integrated"
    return None


def normalize_wifi(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if text.startswith("không") or text in {"none", "no"}:
        return False
    if "wifi" in text or "wi-fi" in text or "802.11" in text:
        return True
    return None


def normalize_bluetooth(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if "bluetooth" in text:
        return True
    return None


def normalize_desktop_form(metadata: dict[str, Any]) -> str:
    display_evidence = any(
        _clean(metadata.get(field)) is not None
        for field in (
            "Kích thước màn hình",
            "kich_thuoc_man_hinh_inch",
            "Màn hình hiển thị",
            "Độ phân giải",
            "do_phan_giai_chuan",
        )
    )
    return "all_in_one" if display_evidence else "separate_unit"


def storage_type_tags(value: Any, normalized_type: Any = None) -> list[str]:
    text = f"{value or ''} | {normalized_type or ''}".casefold()
    tags: list[str] = []
    if "nvme" in text or "nmve" in text:
        tags.extend(["nvme", "ssd"])
    elif "ssd" in text:
        tags.append("ssd")
    if "hdd" in text:
        tags.append("hdd")
    return list(dict.fromkeys(tags))


def parse_installed_storage_gb(value: Any) -> int | None:
    """Sum installed drives while ignoring upgrade/support capacities."""
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"\([^)]*\)", "", text)
    total = 0.0
    found = False
    for segment in re.split(r"\||\+", text):
        folded = segment.casefold()
        if any(
            marker in folded
            for marker in ("hỗ trợ", "nâng cấp", "tối đa", "có thể lắp", "khe cắm")
        ):
            continue
        for number_text, unit in re.findall(
            r"(\d+(?:[.,]\d+)?)\s*(tb|gb)\b", segment, flags=re.IGNORECASE
        ):
            number = float(number_text.replace(",", "."))
            total += number * 1024 if unit.casefold() == "tb" else number
            found = True
    return int(round(total)) if found and total > 0 else None


def normalize_ram_max(metadata: dict[str, Any]) -> int | None:
    values = [
        _integer(metadata.get("ram_toi_da_gb")),
        _integer(metadata.get("ram_mainboard_toi_da_gb")),
    ]
    values = [value for value in values if value is not None and value > 0]
    return max(values) if values else None


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
    quality_flags: list[str] = []
    if "gia_goc_vnd" in normalized or "gia_khuyen_mai_vnd" in normalized:
        quality_flags.append("generated_price_ignored")
    normalized.pop("gia_goc_vnd", None)
    normalized.pop("gia_khuyen_mai_vnd", None)

    storage_total = parse_installed_storage_gb(metadata.get("Ổ cứng"))
    if storage_total is None:
        storage_total = _integer(metadata.get("dung_luong_o_cung_gb"))
    elif metadata.get("Ổ cứng"):
        generated_storage = _integer(metadata.get("dung_luong_o_cung_gb"))
        if generated_storage is not None and generated_storage != storage_total:
            quality_flags.append("storage_total_recomputed")

    desktop_form = normalize_desktop_form(metadata)
    if desktop_form == "separate_unit":
        quality_flags.append("desktop_form_inferred_from_missing_display")

    cpu_source = " | ".join(
        str(metadata.get(field) or "") for field in ("Công nghệ CPU", "Loại CPU")
    )
    _set_if_present(normalized, "brand_key", str(metadata.get("brand") or "").strip().casefold() or None)
    _set_if_present(normalized, "original_price_vnd", parse_vnd(metadata.get("giá gốc")))
    _set_if_present(
        normalized,
        "promotional_price_vnd",
        parse_vnd(metadata.get("giá khuyến mãi")),
    )
    normalized["desktop_form"] = desktop_form
    normalized["has_integrated_display"] = desktop_form == "all_in_one"
    _set_if_present(normalized, "cpu_vendor", normalize_cpu_vendor(cpu_source))
    os_tags = normalize_os_family_tags(metadata.get("Hệ điều hành"))
    if os_tags:
        normalized["os_family_tags"] = os_tags
    _set_if_present(normalized, "ram_gb", _integer(metadata.get("ram_gb")))
    _set_if_present(normalized, "ram_max_gb", normalize_ram_max(metadata))
    _set_if_present(normalized, "ram_slots", _integer(metadata.get("so_khe_ram_chuan")))
    _set_if_present(normalized, "storage_total_gb", storage_total)
    storage_tags = storage_type_tags(
        metadata.get("Ổ cứng"), metadata.get("loai_o_cung_chuan")
    )
    if storage_tags:
        normalized["storage_type_tags"] = storage_tags
    _set_if_present(normalized, "gpu_type", normalize_gpu_type(metadata.get("Thiết kế card")))
    _set_if_present(normalized, "has_wifi", normalize_wifi(metadata.get("Wifi")))
    _set_if_present(normalized, "has_bluetooth", normalize_bluetooth(metadata.get("Wifi")))
    _set_if_present(
        normalized,
        "screen_size_inch",
        _number(metadata.get("kich_thuoc_man_hinh_inch")),
    )
    _set_if_present(normalized, "release_year", _integer(metadata.get("nam_ra_mat")))
    _set_if_present(normalized, "weight_kg", _number(metadata.get("khoi_luong_kg")))
    if quality_flags:
        normalized["data_quality_flags"] = quality_flags
    normalized["category_scope"] = "desktop"
    return normalized


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Không có dữ liệu máy tính để bàn đã xử lý")
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
